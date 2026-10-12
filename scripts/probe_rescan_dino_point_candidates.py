#!/usr/bin/env python3
"""Feasibility probe for independent Rescan object retrieval with DINOv2.

The public Rescan archive stores colored, instance-labeled reconstructions.
This probe renders each reconstructed point cloud from eight deterministic
synthetic views, assigns DINO patch tokens back to visible 3-D samples, and
measures cross-rescan semantic/physical candidate Recall@K. It does not use
target labels for descriptor extraction or ranking. The synthetic-view input
is an explicit protocol shift from the project's native RGB-D frame pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2
from rerank_rescan_fpfh_by_color import read_scene_ply


LAYERS = {"small": (2, 6, 12), "base": (2, 6, 12), "large": (2, 10, 24), "giant": (2, 20, 40)}
VIEWS = 8
RESOLUTION = 224
PATCH = 14
VOXEL_M = 0.05
MAX_POINTS = 25_000
MAX_QUERIES = 3_000
RNG_SEED = 20261006


def voxel_sample(points: np.ndarray, voxel_m: float, cap: int, seed: int) -> np.ndarray:
    voxel = np.floor(points / voxel_m).astype(np.int32)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    _, first = np.unique(voxel[order], axis=0, return_index=True)
    selected = order[first]
    selected.sort()
    if len(selected) > cap:
        selected = np.sort(rng.choice(selected, size=cap, replace=False))
    return selected.astype(np.int64)


def render_view(points: np.ndarray, colors: np.ndarray, center: np.ndarray, azimuth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render one colored point-cloud view and return a z-buffer visibility map."""
    # Rescan reconstructions use Y as the vertical axis.
    forward = np.asarray([np.cos(azimuth), 0.04, np.sin(azimuth)], dtype=np.float32)
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = points - center[None, :]
    z = delta @ forward
    x = delta @ right
    y = delta @ up
    valid = z > 0.1
    focal = RESOLUTION * 0.32
    px = np.rint((RESOLUTION - 1) * 0.5 + focal * x / np.maximum(z, 0.1)).astype(np.int32)
    py = np.rint((RESOLUTION - 1) * 0.5 - focal * y / np.maximum(z, 0.1)).astype(np.int32)
    valid &= (px >= 0) & (px < RESOLUTION) & (py >= 0) & (py < RESOLUTION)

    flat = py[valid] * RESOLUTION + px[valid]
    depth = z[valid]
    color_values = np.clip(colors[valid] * 255.0, 0, 255).astype(np.uint8)
    if not len(flat):
        return np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8), np.full((RESOLUTION, RESOLUTION), np.inf, dtype=np.float32), px, py, z
    order = np.lexsort((depth, flat))
    sorted_flat = flat[order]
    first = np.r_[True, sorted_flat[1:] != sorted_flat[:-1]]
    chosen = order[first]
    pixel_ids = flat[chosen]
    depth_buffer = np.full(RESOLUTION * RESOLUTION, np.inf, dtype=np.float32)
    rgb_buffer = np.zeros((RESOLUTION * RESOLUTION, 3), dtype=np.uint8)
    depth_buffer[pixel_ids] = depth[chosen]
    rgb_buffer[pixel_ids] = color_values[chosen]
    depth_buffer = depth_buffer.reshape(RESOLUTION, RESOLUTION)
    rgb_buffer = rgb_buffer.reshape(RESOLUTION, RESOLUTION, 3)

    # Fill only small holes between observed surfels; keep distant background black.
    missing = ~np.isfinite(depth_buffer)
    if missing.any():
        distance, indices = distance_transform_edt(missing, return_indices=True)
        near = missing & (distance <= 3.0)
        rgb_buffer[near] = rgb_buffer[indices[0][near], indices[1][near]]
    rgb_buffer[missing & (distance_transform_edt(missing) > 3.0)] = 0
    return rgb_buffer, depth_buffer, px, py, z


def preprocess(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0)


@torch.inference_mode()
def scan_features(path: Path, model: torch.nn.Module, layers: tuple[int, ...], device: torch.device, seed: int) -> dict[str, np.ndarray]:
    points, colors, ids, classes = read_scene_ply(path)
    valid_label = (ids > 0) & (classes > 0)
    all_instance_class_pairs = np.unique(
        np.column_stack((ids[valid_label], classes[valid_label])), axis=0
    ).astype(np.int32)
    sample_idx = voxel_sample(points, VOXEL_M, MAX_POINTS, seed)
    sample_points = points[sample_idx]
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    feature_sum = None
    feature_count = np.zeros(len(sample_idx), dtype=np.int16)
    for view in range(VIEWS):
        azimuth = (2.0 * np.pi * view) / VIEWS
        image, z_buffer, raw_px, raw_py, raw_z = render_view(points, colors, center, azimuth)
        features = extract_block_tokens(model, preprocess(image, device), layers)
        blocks = []
        for layer in layers:
            token = features[layer][0, 1:, :]
            token = F.normalize(token, dim=-1)
            blocks.append(token)
        patch_features = torch.cat(blocks, dim=-1).squeeze(0).numpy().astype(np.float32)

        # Read the DINO feature for each sampled point that is visible in this view.
        px, py, z = raw_px[sample_idx], raw_py[sample_idx], raw_z[sample_idx]
        inside = (z > 0.1) & (px >= 0) & (px < RESOLUTION) & (py >= 0) & (py < RESOLUTION)
        visible = np.zeros(len(sample_idx), dtype=bool)
        in_view = np.flatnonzero(inside)
        visible[in_view] = z[in_view] <= z_buffer[py[in_view], px[in_view]] + 0.03
        inside = visible
        patch_index = (py[inside] // PATCH) * (RESOLUTION // PATCH) + (px[inside] // PATCH)
        values = patch_features[patch_index]
        if feature_sum is None:
            feature_sum = np.zeros((len(sample_idx), values.shape[1]), dtype=np.float32)
        selected = np.flatnonzero(inside)
        feature_sum[selected] += values
        feature_count[selected] += 1
    valid = feature_count > 0
    features = np.zeros_like(feature_sum)
    features[valid] = feature_sum[valid] / feature_count[valid, None]
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    valid &= norm[:, 0] > 0
    features[valid] /= norm[valid]
    return {
        "points": sample_points.astype(np.float32),
        "colors": colors[sample_idx].astype(np.float32),
        "ids": ids[sample_idx].astype(np.int32),
        "all_instance_ids": np.unique(ids[ids > 0]).astype(np.int32),
        "all_instance_class_pairs": all_instance_class_pairs,
        "classes": classes[sample_idx].astype(np.int32),
        "features": features.astype(np.float32),
        "valid": valid,
        "point_count_raw": np.asarray([len(points)], dtype=np.int64),
    }


def topk_ids(query: np.ndarray, candidates: np.ndarray, query_mask: np.ndarray, candidate_mask: np.ndarray, k: int, device: torch.device, *, cosine: bool = True, return_scores: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    qidx = np.flatnonzero(query_mask)
    cidx = np.flatnonzero(candidate_mask)
    if not len(qidx) or not len(cidx):
        indices = np.empty((len(qidx), 0), dtype=np.int64)
        return (indices, np.empty((len(qidx), 0), dtype=np.float32)) if return_scores else indices
    q = torch.from_numpy(query[qidx]).to(device)
    c = torch.from_numpy(candidates[cidx]).to(device)
    result_indices = []
    result_scores = []
    chunk = max(1, min(64, int(1_500_000_000 / max(c.shape[0] * c.shape[1] * 4, 1))))
    if cosine:
        q = F.normalize(q, dim=1)
        c = F.normalize(c, dim=1)
    candidate_norm = (c * c).sum(dim=1) if not cosine else None
    for start in range(0, len(q), chunk):
        scores = q[start:start + chunk] @ c.T
        if not cosine:
            scores = 2.0 * scores - (q[start:start + chunk] ** 2).sum(dim=1, keepdim=True) - candidate_norm[None, :]
        values, indices = torch.topk(scores, k=min(k, len(cidx)), dim=1, largest=True, sorted=True)
        result_indices.append(indices.cpu().numpy())
        if return_scores:
            result_scores.append(values.cpu().numpy())
    local = np.concatenate(result_indices, axis=0)
    global_indices = cidx[local]
    if return_scores:
        return global_indices, np.concatenate(result_scores, axis=0).astype(np.float32)
    return global_indices


def score_pair(scene: str, source_scan: str, target_scan: str, source: dict[str, np.ndarray], target: dict[str, np.ndarray], device: torch.device, query_details: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED + ord(scene[-1]) * 31 + int(target_scan.rsplit("_", 1)[1]))
    # Define target presence from the full annotated source scan, not the
    # downsampled candidate pool. Instances omitted by voxel sampling count as
    # candidate misses, as they should for a fixed candidate budget.
    source_ids = set(source["all_instance_ids"].tolist())
    # Both label types must be valid so semantic and physical recall share a
    # well-defined query population (class 0 is unlabelled in this dataset).
    eligible = target["valid"] & (target["ids"] > 0) & (target["classes"] > 0)
    eligible_idx = np.flatnonzero(eligible)
    if len(eligible_idx) > MAX_QUERIES:
        # Balance queries by object so large surfaces do not dominate the probe.
        buckets = []
        for object_id in np.unique(target["ids"][eligible_idx]):
            members = eligible_idx[target["ids"][eligible_idx] == object_id]
            buckets.extend(rng.choice(members, size=min(96, len(members)), replace=False).tolist())
        eligible_idx = np.asarray(buckets, dtype=np.int64)
        if len(eligible_idx) > MAX_QUERIES:
            eligible_idx = np.sort(rng.choice(eligible_idx, size=MAX_QUERIES, replace=False))
    all_mask = target["valid"] & (target["ids"] > 0)
    qidx = np.intersect1d(eligible_idx, np.flatnonzero(all_mask), assume_unique=True)
    dino_candidates, dino_scores = topk_ids(target["features"], source["features"], np.isin(np.arange(len(target["ids"])), qidx), source["valid"], 20, device, return_scores=True)
    rgb_query_mask = np.zeros(len(target["ids"]), dtype=bool)
    rgb_query_mask[qidx] = True
    rgb_candidates, rgb_scores = topk_ids(target["colors"], source["colors"], rgb_query_mask, source["valid"], 20, device, cosine=False, return_scores=True)
    candidate_qidx = qidx
    expected_id = target["ids"][candidate_qidx]
    expected_class = target["classes"][candidate_qidx]
    present = np.isin(expected_id, np.fromiter(source_ids, dtype=np.int32))
    full_class_pairs = source["all_instance_class_pairs"]
    full_class_counts = {
        int(class_id): int(np.sum(full_class_pairs[:, 1] == class_id))
        for class_id in np.unique(full_class_pairs[:, 1])
    }
    candidate_label = source["valid"] & (source["ids"] > 0) & (source["classes"] > 0)
    candidate_class_pairs = np.unique(
        np.column_stack((source["ids"][candidate_label], source["classes"][candidate_label])), axis=0
    )
    candidate_class_counts = {
        int(class_id): int(np.sum(candidate_class_pairs[:, 1] == class_id))
        for class_id in np.unique(candidate_class_pairs[:, 1])
    }
    rows = []
    for method, candidates, candidate_scores in (
        ("dino_synthetic_views", dino_candidates, dino_scores),
        ("rgb_color", rgb_candidates, rgb_scores),
    ):
        top20_ids = source["ids"][candidates[:, :20]]
        top20_classes = source["classes"][candidates[:, :20]]
        top20_hit_id = top20_ids == expected_id[:, None]
        top20_hit_sem = top20_classes == expected_class[:, None]
        if query_details is not None:
            semantic_rank = np.where(top20_hit_sem.any(axis=1), top20_hit_sem.argmax(axis=1) + 1, 0)
            physical_rank = np.where(top20_hit_id.any(axis=1), top20_hit_id.argmax(axis=1) + 1, 0)
            full_multiplicity = np.asarray([full_class_counts.get(int(class_id), 0) for class_id in expected_class])
            candidate_multiplicity = np.asarray([candidate_class_counts.get(int(class_id), 0) for class_id in expected_class])
            for row_index, sample_index in enumerate(candidate_qidx):
                same_class_ids = top20_ids[row_index][top20_hit_sem[row_index]]
                same_class_ids = same_class_ids[same_class_ids > 0]
                query_details.append({
                    "scene": scene, "source_scan": source_scan, "target_scan": target_scan,
                    "method": method, "target_sample_index": int(sample_index),
                    "target_instance_id": int(expected_id[row_index]),
                    "target_class_id": int(expected_class[row_index]),
                    "target_present_in_full_source": bool(present[row_index]),
                    "source_class_instance_count_full": int(full_multiplicity[row_index]),
                    "source_class_instance_count_candidates": int(candidate_multiplicity[row_index]),
                    "top20_semantic_hits": int(top20_hit_sem[row_index].sum()),
                    "top20_distinct_semantic_instances": int(len(np.unique(same_class_ids))),
                    "semantic_rank_top20": int(semantic_rank[row_index]),
                    "physical_rank_top20": int(physical_rank[row_index]),
                    "semantic_hit_top20": bool(semantic_rank[row_index] > 0),
                    "physical_hit_top20": bool(physical_rank[row_index] > 0),
                    "semantic_only_hit_top20": bool(semantic_rank[row_index] > 0 and physical_rank[row_index] == 0),
                    "top1_similarity": float(candidate_scores[row_index, 0]),
                    "top1_top2_margin": float(candidate_scores[row_index, 0] - candidate_scores[row_index, 1]) if candidate_scores.shape[1] > 1 else 0.0,
                })
        for k in (1, 5, 20):
            top = candidates[:, :k]
            hit_id = source["ids"][top] == expected_id[:, None]
            hit_sem = source["classes"][top] == expected_class[:, None]
            physical = hit_id.any(axis=1)
            semantic = hit_sem.any(axis=1)
            rows.append({
                "scene": scene, "source_scan": source_scan, "target_scan": target_scan,
                "method": method, "k": k, "queries": int(len(candidate_qidx)),
                "target_present_queries": int(present.sum()),
                "physical_correct": int((physical & present).sum()),
                "semantic_correct_present": int((semantic & present).sum()),
                "semantic_correct": int(semantic.sum()),
                "physical_recall_present": float((physical & present).sum() / present.sum()) if present.any() else None,
                "semantic_recall_present": float((semantic & present).sum() / present.sum()) if present.any() else None,
                "semantic_recall_labeled": float(semantic.mean()) if len(semantic) else None,
                "feature_valid_source": int(source["valid"].sum()),
                "feature_valid_target": int(target["valid"].sum()),
                "candidate_points": int(source["valid"].sum()),
                "raw_source_points": int(source["point_count_raw"][0]),
                "raw_target_points": int(target["point_count_raw"][0]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=(Path.home() / 'data/worldcache_datasets/rescan_subset'))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_rescan_dino_external_pilot"))
    parser.add_argument("--scenes", nargs="+", default=["scene_b", "scene_c"])
    parser.add_argument("--model-size", choices=("small", "base", "large", "giant"), default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-query-metrics", action="store_true", help="write query-level top-20 rank diagnostics for ambiguity analysis")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_dinov2(args.model_size, RESOLUTION, device)
    layers = LAYERS[args.model_size]
    feature_cache: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    query_details: list[dict[str, object]] | None = [] if args.save_query_metrics else None
    pair_count = 0
    for scene in args.scenes:
        scan_ids = sorted(p.stem for p in (args.data_root / scene / "gt_segmentation").glob(f"{scene}_*.ply"))
        if len(scan_ids) < 2:
            raise FileNotFoundError(f"scene {scene} needs at least two local labeled scans")
        for target_scan in scan_ids[1:]:
            source_scan = scan_ids[0]
            for scan in (source_scan, target_scan):
                if scan not in feature_cache:
                    path = args.data_root / scene / "gt_segmentation" / f"{scan}.ply"
                    print(f"extracting {scan} with {args.model_size}; views={VIEWS}", flush=True)
                    feature_cache[scan] = scan_features(path, model, layers, device, RNG_SEED + len(feature_cache))
            rows.extend(score_pair(scene, source_scan, target_scan, feature_cache[source_scan], feature_cache[target_scan], device, query_details))
            pair_count += 1
            print(f"scored {source_scan}->{target_scan}", flush=True)
    with (args.output_dir / "pair_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if query_details is not None:
        with (args.output_dir / "query_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(query_details[0]))
            writer.writeheader()
            writer.writerows(query_details)
    summary = {
        "status": "complete", "scenes": args.scenes, "pairs": pair_count,
        "model_size": args.model_size, "layers": layers, "render_views_per_scan": VIEWS,
        "resolution": RESOLUTION, "voxel_m": VOXEL_M, "max_points": MAX_POINTS,
        "max_queries_per_pair": MAX_QUERIES, "seed": RNG_SEED,
        "input_protocol": "synthetic wide-FOV RGB views rendered from colored Rescan point-cloud annotations; not raw RGB-D frames",
        "validation_labels_used_for_features_or_ranking": False,
        "query_metrics_written": query_details is not None,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'pair_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
