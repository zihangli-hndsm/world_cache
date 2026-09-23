"""Evaluate instance/semantic correspondence on an annotated 3RScan pair."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2
from worldcache.geometry.projection import backproject_pixels
from worldcache.reuse.descriptor import load_descriptor_projection
from worldcache.reuse.ranking import fuse_descriptor_geometry_ranks, fuse_descriptor_geometry_scores
from worldcache.reuse.world_field import encode_world_receptive_field


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image - mean) / std).unsqueeze(0)


def patch_world_points(observation: dict[str, torch.Tensor], patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = observation["depth"].shape
    yy, xx = torch.meshgrid(
        torch.arange(patch_size / 2, height, patch_size),
        torch.arange(patch_size / 2, width, patch_size),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
    depth = observation["depth"][pixels[:, 1].long(), pixels[:, 0].long()]
    valid = torch.isfinite(depth) & (depth > 0)
    return backproject_pixels(pixels[valid], depth[valid], observation["intrinsics"], observation["camera_to_world"]), valid


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def read_annotated_mesh(source: Path | zipfile.ZipFile, scan_id: str) -> tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    if isinstance(source, zipfile.ZipFile):
        payload = source.read(f"{scan_id}/labels.instances.annotated.v2.ply")
        semseg_payload = source.read(f"{scan_id}/semseg.v2.json")
    else:
        scan_dir = source / scan_id
        payload = (scan_dir / "labels.instances.annotated.v2.ply").read_bytes()
        semseg_payload = (scan_dir / "semseg.v2.json").read_bytes()
    header_end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:header_end].decode("ascii")
    vertex_count = int(re.search(r"element vertex (\d+)", header).group(1))
    values = np.fromstring(payload[header_end:].decode("ascii"), sep=" ", dtype=np.float64)
    vertices = values[: vertex_count * 11].reshape(vertex_count, 11)
    points = vertices[:, :3].astype(np.float32)
    global_ids = vertices[:, 7].astype(np.int64)
    semantic_ids = vertices[:, 8].astype(np.int64)  # official NYU40 category

    semseg = json.loads(semseg_payload)
    object_to_label = {int(group["objectId"]): str(group["label"]) for group in semseg["segGroups"]}
    object_ids = vertices[:, 6].astype(np.int64)
    return cKDTree(points), object_ids, global_ids, semantic_ids, object_to_label


def global_labels(object_ids: np.ndarray, global_ids: np.ndarray, object_to_label: dict[int, str]) -> dict[int, str]:
    """Map 3RScan cross-scan global IDs to semantic labels."""
    result: dict[int, str] = {}
    for object_id, global_id in zip(object_ids, global_ids):
        object_id = int(object_id)
        global_id = int(global_id)
        if global_id == 0 or object_id == 0 or global_id in result:
            continue
        label = object_to_label.get(object_id)
        if label is not None:
            result[global_id] = label
    return result


def field_features(points: torch.Tensor, tokens: torch.Tensor, radius: float) -> dict[str, torch.Tensor]:
    raw = torch.nn.functional.normalize(tokens, dim=1)
    return {"raw": raw, f"world_r{radius:g}": encode_world_receptive_field(points, tokens, radius, radius * 0.5)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--mesh-archive", type=Path)
    parser.add_argument(
        "--annotation-root",
        type=Path,
        help="directory containing one annotation subdirectory per scan ID",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("small", "base", "large", "giant"), default="base")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--field-radius", type=float, default=0.30)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=0.15)
    parser.add_argument("--candidate-scope", choices=("local", "global"), default="local")
    parser.add_argument("--projection", type=Path, help="optional trained descriptor projection checkpoint")
    parser.add_argument(
        "--fusion-alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        help="distance penalties in score = cosine - alpha * distance",
    )
    parser.add_argument(
        "--rank-fusion-betas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="within-candidate rank penalties for rank fusion",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("pair manifest must store native scan coordinates; run migrate_3rscan_rescan_coordinates.py")
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    if (args.mesh_archive is None) == (args.annotation_root is None):
        raise ValueError("provide exactly one of --mesh-archive or --annotation-root")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2(args.model_size, config["resolution"], device)
    projection = load_descriptor_projection(str(args.projection), device) if args.projection else None
    if args.annotation_root is not None:
        annotation_source = args.annotation_root
        reference_tree, reference_object_ids, reference_global_ids, reference_semantic_ids, reference_labels = read_annotated_mesh(annotation_source, config["reference"])
        rescan_tree, rescan_object_ids, rescan_global_ids, rescan_semantic_ids, rescan_labels = read_annotated_mesh(annotation_source, config["rescan"])
    else:
        with zipfile.ZipFile(args.mesh_archive) as archive:
            reference_tree, reference_object_ids, reference_global_ids, reference_semantic_ids, reference_labels = read_annotated_mesh(archive, config["reference"])
            rescan_tree, rescan_object_ids, rescan_global_ids, rescan_semantic_ids, rescan_labels = read_annotated_mesh(archive, config["rescan"])
    reference_global_labels = global_labels(reference_object_ids, reference_global_ids, reference_labels)
    rescan_global_labels = global_labels(rescan_object_ids, rescan_global_ids, rescan_labels)
    field_names = ["raw", f"world_r{args.field_radius:g}"]
    stores: dict[str, dict[tuple[int, int, int], dict[str, object]]] = {field: {} for field in field_names}

    for entry in manifest["reference"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        tokens = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()[valid]
        if projection is not None:
            tokens = projection(tokens.to(device)).cpu()
        _, nearest = reference_tree.query(points.numpy(), k=1)
        labels = reference_global_ids[nearest]
        semantic_labels = reference_semantic_ids[nearest]
        features = field_features(points, tokens, args.field_radius)
        for field, descriptors in features.items():
            for point, feature, label, semantic_label in zip(points, descriptors, labels, semantic_labels):
                label = int(label)
                if label == 0:
                    continue
                semantic_label = int(semantic_label)
                key = tuple(torch.floor(point / args.voxel_size).to(torch.int64).tolist())
                state = stores[field].get(key)
                if state is None:
                    stores[field][key] = {
                        "sum": feature.clone(),
                        "count": 1,
                        "point_sum": point.clone(),
                        "labels": Counter({label: 1}),
                        "semantic_labels": Counter({semantic_label: 1}),
                    }
                else:
                    state["sum"] += feature
                    state["count"] += 1
                    state["point_sum"] += point
                    state["labels"].update([label])
                    state["semantic_labels"].update([semantic_label])

    caches = {}
    for field, store in stores.items():
        points = torch.stack([state["point_sum"] / state["count"] for state in store.values()])
        descriptors = torch.nn.functional.normalize(torch.stack([state["sum"] / state["count"] for state in store.values()]), dim=1)
        labels = np.asarray([state["labels"].most_common(1)[0][0] for state in store.values()], dtype=np.int64)
        semantic_labels = np.asarray([state["semantic_labels"].most_common(1)[0][0] for state in store.values()], dtype=np.int64)
        spatial: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, key in enumerate(store):
            spatial[key].append(index)
        caches[field] = points, descriptors, labels, semantic_labels, spatial

    rows: list[dict[str, object]] = []
    for frame_index, entry in enumerate(manifest["rescan"]):
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        tokens = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()[valid]
        if projection is not None:
            tokens = projection(tokens.to(device)).cpu()
        world_points = points.numpy()
        _, nearest = rescan_tree.query(world_points, k=1)
        query_ids = rescan_global_ids[nearest]
        query_semantic_ids = rescan_semantic_ids[nearest]
        features = field_features(points, tokens, args.field_radius)
        for query_index, (point, query_id, query_semantic_id) in enumerate(zip(points, query_ids, query_semantic_ids)):
            query_id = int(query_id)
            if query_id == 0:
                continue
            query_semantic_id = int(query_semantic_id)
            if query_semantic_id == 0:
                continue
            for field in field_names:
                cache_points, cache_descriptors, cache_ids, cache_semantic_ids, spatial = caches[field]
                if args.candidate_scope == "global":
                    candidates = list(range(len(cache_points)))
                else:
                    key = tuple(torch.floor(point / args.voxel_size).to(torch.int64).tolist())
                    candidates = []
                    for dx in range(-args.neighbor_radius, args.neighbor_radius + 1):
                        for dy in range(-args.neighbor_radius, args.neighbor_radius + 1):
                            for dz in range(-args.neighbor_radius, args.neighbor_radius + 1):
                                candidates.extend(spatial.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
                    candidates = [item for item in candidates if float(torch.linalg.vector_norm(cache_points[item] - point)) <= args.max_distance]
                if not candidates:
                    continue
                local = torch.tensor(candidates, dtype=torch.long)
                distances = torch.linalg.vector_norm(cache_points[local] - point, dim=1)
                candidate_global_ids = cache_ids[local.numpy()]
                candidate_semantic_ids = cache_semantic_ids[local.numpy()]
                same_instance_candidates = int(np.sum(candidate_global_ids == query_id))
                same_semantic_candidates = int(np.sum(candidate_semantic_ids == query_semantic_id))
                same_semantic_instances = len({
                    int(candidate_id)
                    for candidate_id, candidate_semantic_id in zip(candidate_global_ids, candidate_semantic_ids)
                    if candidate_semantic_id == query_semantic_id
                })
                target = int(torch.argmin(distances))
                query = features[field][query_index]
                scores = cache_descriptors[local] @ query
                generator = torch.Generator().manual_seed(1_000_000 * frame_index + 10_000 * (field_names.index(field) + 1) + query_index)
                selections = {
                    "geometry": target,
                    "descriptor": int(torch.argmax(scores)),
                    "random": int(torch.randperm(len(local), generator=generator)[0]),
                }
                for alpha in args.fusion_alphas:
                    selections[f"fused_a{alpha:g}"] = int(
                        torch.argmax(fuse_descriptor_geometry_scores(scores, distances, alpha))
                    )
                for beta in args.rank_fusion_betas:
                    selections[f"rank_fused_b{beta:g}"] = int(
                        torch.argmax(fuse_descriptor_geometry_ranks(scores, distances, beta))
                    )
                for method, selected in selections.items():
                    predicted_id = int(cache_ids[local[selected]])
                    rows.append({
                        "frame_index": frame_index,
                        "query_index": query_index,
                        "query_point_ref_x": float(point[0]),
                        "query_point_ref_y": float(point[1]),
                        "query_point_ref_z": float(point[2]),
                        "selected_point_ref_x": float(cache_points[local[selected], 0]),
                        "selected_point_ref_y": float(cache_points[local[selected], 1]),
                        "selected_point_ref_z": float(cache_points[local[selected], 2]),
                        "field": field,
                        "method": method,
                        "candidate_count": len(candidates),
                        "same_instance_candidates": same_instance_candidates,
                        "same_semantic_candidates": same_semantic_candidates,
                        "same_semantic_instances": same_semantic_instances,
                        "query_instance": query_id,
                        "predicted_instance": predicted_id,
                        "instance_correct": int(predicted_id == query_id),
                        "semantic_correct": int(cache_semantic_ids[local[selected]] == query_semantic_id),
                        "selected_distance_m": float(distances[selected]),
                    })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no annotated rescan patches passed the geometric cache gate")
    frame.to_parquet(args.output_dir / "query_metrics.parquet", index=False)
    summary = frame.groupby(["field", "method"], as_index=False).agg(
        queries=("instance_correct", "size"),
        candidate_count_median=("candidate_count", "median"),
        instance_accuracy=("instance_correct", "mean"),
        semantic_accuracy=("semantic_correct", "mean"),
        selected_distance_median_m=("selected_distance_m", "median"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    status = {
        "status": "complete",
        "annotated_queries": int(frame["query_index"].nunique()),
        "rows": len(frame),
        "config": vars(args),
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, default=str, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
