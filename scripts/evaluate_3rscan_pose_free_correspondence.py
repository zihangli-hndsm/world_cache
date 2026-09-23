"""Evaluate pose-free global DINO correspondence with reciprocal filtering.

This evaluator never uses query 3-D coordinates to select a reference voxel.
It builds a reference cache from raw DINO tokens, matches every rescan token
against the complete cache, and optionally keeps only reciprocal matches or
matches with a cosine-margin gap. Native scan coordinates are used for the
RANSAC registration; the known transform is used only for scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_3rscan_instance_correspondence import (
    load,
    patch_world_points,
    preprocess,
    read_annotated_mesh,
    transform_points,
)
from evaluate_3rscan_pose_free_registration import ransac, transform
from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2


def add_reference_observation(
    observation: dict[str, torch.Tensor],
    backbone: torch.nn.Module,
    layer: int,
    device: torch.device,
    reference_tree,
    reference_global_ids: np.ndarray,
    reference_semantic_ids: np.ndarray,
    voxel_size: float,
    store: dict[tuple[int, int, int], dict[str, object]],
) -> None:
    points, valid = patch_world_points(observation, patch_size=14)
    tokens = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [layer])[layer][
        0, backbone.num_prefix_tokens:
    ].cpu()[valid]
    descriptors = torch.nn.functional.normalize(tokens, dim=1)
    _, nearest = reference_tree.query(points.numpy(), k=1)
    for point, descriptor, nearest_index in zip(points, descriptors, nearest):
        global_id = int(reference_global_ids[nearest_index])
        semantic_id = int(reference_semantic_ids[nearest_index])
        if global_id == 0 or semantic_id == 0:
            continue
        key = tuple(torch.floor(point / voxel_size).to(torch.int64).tolist())
        state = store.get(key)
        if state is None:
            store[key] = {
                "descriptor_sum": descriptor.clone(),
                "point_sum": point.clone(),
                "count": 1,
                "global_ids": Counter({global_id: 1}),
                "semantic_ids": Counter({semantic_id: 1}),
            }
        else:
            state["descriptor_sum"] += descriptor
            state["point_sum"] += point
            state["count"] += 1
            state["global_ids"].update([global_id])
            state["semantic_ids"].update([semantic_id])


def finalize_cache(store: dict[tuple[int, int, int], dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    points = torch.stack([state["point_sum"] / state["count"] for state in store.values()])
    descriptors = torch.stack([
        state["descriptor_sum"] / state["count"] for state in store.values()
    ])
    descriptors = torch.nn.functional.normalize(descriptors, dim=1)
    global_ids = np.asarray([state["global_ids"].most_common(1)[0][0] for state in store.values()], dtype=np.int64)
    semantic_ids = np.asarray([state["semantic_ids"].most_common(1)[0][0] for state in store.values()], dtype=np.int64)
    return points, descriptors, global_ids, semantic_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("small", "base", "large", "giant"), default="large")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--margin-thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--ransac-threshold", type=float, default=0.10)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("pair manifest must store native scan coordinates; run migrate_3rscan_rescan_coordinates.py")
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2(args.model_size, config["resolution"], device)
    reference_tree, _, reference_global_ids, reference_semantic_ids, _ = read_annotated_mesh(
        args.annotation_root, config["reference"]
    )
    rescan_tree, _, rescan_global_ids, rescan_semantic_ids, _ = read_annotated_mesh(
        args.annotation_root, config["rescan"]
    )

    cache_store: dict[tuple[int, int, int], dict[str, object]] = {}
    for entry in manifest["reference"]:
        add_reference_observation(
            load(args.pair_run / entry["path"]),
            backbone,
            args.layer,
            device,
            reference_tree,
            reference_global_ids,
            reference_semantic_ids,
            args.voxel_size,
            cache_store,
        )
    cache_points, cache_descriptors, cache_global_ids, cache_semantic_ids = finalize_cache(cache_store)

    query_points_ref = []
    query_points_native = []
    query_descriptors = []
    query_global_ids = []
    query_semantic_ids = []
    rescan_to_reference = np.asarray(config["rescan_to_reference"], dtype=np.float64)
    for entry in manifest["rescan"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        tokens = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][
            0, backbone.num_prefix_tokens:
        ].cpu()[valid]
        descriptors = torch.nn.functional.normalize(tokens, dim=1)
        points_native = points.numpy().astype(np.float64)
        _, nearest = rescan_tree.query(points_native, k=1)
        ids = rescan_global_ids[nearest]
        semantics = rescan_semantic_ids[nearest]
        keep = (ids != 0) & (semantics != 0)
        query_points_ref.append(points_native[keep])
        query_points_native.append(points_native[keep])
        query_descriptors.append(descriptors[torch.from_numpy(keep)])
        query_global_ids.append(ids[keep])
        query_semantic_ids.append(semantics[keep])

    points_ref = np.concatenate(query_points_ref, axis=0)
    points_native = np.concatenate(query_points_native, axis=0)
    descriptors = torch.cat(query_descriptors, dim=0)
    query_ids = np.concatenate(query_global_ids, axis=0)
    query_semantics = np.concatenate(query_semantic_ids, axis=0)
    scores = descriptors @ cache_descriptors.T
    top_scores, top_indices = torch.topk(scores, k=min(2, scores.shape[1]), dim=1)
    selected = top_indices[:, 0].numpy()
    margins = (top_scores[:, 0] - top_scores[:, 1]).numpy() if scores.shape[1] > 1 else np.full(len(selected), np.inf)
    cache_to_query = torch.argmax(scores, dim=0).numpy()
    reciprocal = cache_to_query[selected] == np.arange(len(selected))

    masks = {"one_way": np.ones(len(selected), dtype=bool), "mutual": reciprocal}
    for margin in args.margin_thresholds:
        masks[f"mutual_margin_{margin:g}"] = reciprocal & (margins >= margin)

    rows = []
    for method, mask in masks.items():
        count = int(mask.sum())
        if count >= 3:
            estimated, inliers = ransac(
                points_native[mask],
                cache_points[selected[mask]].numpy(),
                seed=17,
                threshold=args.ransac_threshold,
                iterations=args.ransac_iterations,
            )
            residuals = np.linalg.norm(transform(points_native[mask], estimated) - cache_points[selected[mask]].numpy(), axis=1)
            from scipy.spatial.transform import Rotation

            rotation_error = Rotation.from_matrix(estimated[:3, :3].T @ rescan_to_reference[:3, :3]).magnitude()
            translation_error = np.linalg.norm(estimated[:3, 3] - rescan_to_reference[:3, 3])
            inlier_rate = float(inliers.mean())
            median_residual = float(np.median(residuals))
        else:
            inliers = np.zeros(count, dtype=bool)
            rotation_error = float("nan")
            translation_error = float("nan")
            inlier_rate = float("nan")
            median_residual = float("nan")
        rows.append({
            "method": method,
            "matches": count,
            "coverage": float(count / len(selected)),
            "instance_accuracy": float(np.mean(cache_global_ids[selected[mask]] == query_ids[mask])) if count else float("nan"),
            "semantic_accuracy": float(np.mean(cache_semantic_ids[selected[mask]] == query_semantics[mask])) if count else float("nan"),
            "ransac_inlier_rate": inlier_rate,
            "median_registration_residual_m": median_residual,
            "rotation_error_deg": float(np.degrees(rotation_error)),
            "translation_error_m": float(translation_error),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame({"margin": margins, "reciprocal": reciprocal}).to_csv(args.output_dir / "match_diagnostics.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "cache_voxels": len(cache_points),
        "query_patches": len(selected),
        "candidate_selection": "global raw DINO descriptor only",
        "known_alignment_used_for_selection": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
