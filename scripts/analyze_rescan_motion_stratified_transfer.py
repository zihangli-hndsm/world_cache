#!/usr/bin/env python3
"""Stratify Rescan object-pose transfer by annotated centroid displacement."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from rerank_rescan_fpfh_by_color import read_scene_ply


def metric(source_points: np.ndarray, target_points: np.ndarray, target_ids: np.ndarray,
           instance_id: int, target_tree: cKDTree, max_distance: float) -> tuple[float, int]:
    distances, nearest = target_tree.query(source_points, k=1, workers=1)
    predicted = np.unique(nearest[distances <= max_distance])
    target_mask = target_ids == instance_id
    tp = int(np.count_nonzero(target_mask[predicted]))
    fp = len(predicted) - tp
    fn = int(np.count_nonzero(target_mask)) - tp
    union = tp + fp + fn
    return (tp / union if union else 0.0), len(predicted)


def bootstrap_ci(values: np.ndarray, scenes: list[str], samples: int,
                 rng: np.random.Generator) -> list[float]:
    unique = sorted(set(scenes))
    index = {scene: np.flatnonzero(np.asarray(scenes) == scene) for scene in unique}
    draws = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        picked = rng.integers(0, len(unique), size=len(unique))
        selected = np.concatenate([index[unique[j]] for j in picked])
        draws[sample] = float(np.mean(values[selected]))
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--eligible-candidates-csv", type=Path, required=True)
    parser.add_argument("--selected-transfer-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_032_026)
    args = parser.parse_args()

    with args.eligible_candidates_csv.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    with args.selected_transfer_csv.open(newline="", encoding="utf-8") as stream:
        transfer_rows = list(csv.DictReader(stream))
    candidates: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        candidates[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)
    transfer = {
        (row["scene"], row["target_scan"], row["source_instance_id"]): row
        for row in transfer_rows
    }
    if set(candidates) != set(transfer):
        raise ValueError("candidate and transfer files do not contain the same queries")

    result_rows: list[dict[str, object]] = []
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    for key in sorted(candidates):
        scene, target_scan, raw_id = key
        instance_id = int(raw_id)
        if scene not in source_cache:
            points, _, ids, _ = read_scene_ply(
                args.data_root / scene / "gt_segmentation" / f"{scene}_0.ply"
            )
            source_cache[scene] = (points, ids)
        target_key = (scene, target_scan)
        if target_key not in target_cache:
            points, _, ids, _ = read_scene_ply(
                args.data_root / scene / "gt_segmentation" / f"{target_scan}.ply"
            )
            target_cache[target_key] = (points, ids, cKDTree(points))
        source_all, source_ids = source_cache[scene]
        target_points, target_ids, target_tree = target_cache[target_key]
        source_points = source_all[source_ids == instance_id]
        target_mask = target_ids == instance_id
        if len(source_points) == 0 or not np.any(target_mask):
            raise ValueError(f"query {key} is not target-present in its annotations")

        identity_iou, identity_predicted = metric(
            source_points, target_points, target_ids, instance_id, target_tree, args.max_distance
        )
        transfer_row = transfer[key]
        color_iou = float(transfer_row["color_instance_iou"])
        source_center = source_points.mean(axis=0)
        target_center = target_points[target_mask].mean(axis=0)
        displacement = float(np.linalg.norm(target_center - source_center))
        if displacement < 0.10:
            stratum = "<0.10m"
        elif displacement < 0.30:
            stratum = "0.10-0.30m"
        else:
            stratum = ">=0.30m"
        result_rows.append({
            "scene": scene,
            "target_scan": target_scan,
            "source_instance_id": raw_id,
            "source_points": len(source_points),
            "target_instance_points": int(np.count_nonzero(target_mask)),
            "observed_centroid_displacement_m": displacement,
            "displacement_stratum": stratum,
            "identity_instance_iou": identity_iou,
            "identity_predicted_target_points": identity_predicted,
            "color_instance_iou": color_iou,
            "color_minus_identity_iou": color_iou - identity_iou,
            "fitness_instance_iou": float(transfer_row["fitness_instance_iou"]),
            "spatial_instance_iou": float(transfer_row["spatial_instance_iou"]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "motion_stratified_objects.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    strata = ("<0.10m", "0.10-0.30m", ">=0.30m")
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "queries": len(result_rows),
        "scenes": len({row["scene"] for row in result_rows}),
        "target_scans": len({(row["scene"], row["target_scan"]) for row in result_rows}),
        "max_distance_m": args.max_distance,
        "displacement_definition": "Euclidean distance between source/target annotated instance point-cloud centroids; post-hoc diagnostic, not a pure object-motion measurement",
        "prediction": "identity or frozen spatial+RGB top-1 object proposal; target point cloud remains unsegmented for nearest-neighbor assignment",
        "target_labels_used_for_prediction": False,
        "target_labels_used_for_stratification_and_scoring": True,
        "bootstrap_unit": "scene, query-weighted within each resampled scene set",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "strata": {},
    }
    all_deltas = np.asarray([float(row["color_minus_identity_iou"]) for row in result_rows])
    all_scenes = [str(row["scene"]) for row in result_rows]
    summary["overall"] = {
        "identity_mean_iou": float(np.mean([float(row["identity_instance_iou"]) for row in result_rows])),
        "color_mean_iou": float(np.mean([float(row["color_instance_iou"]) for row in result_rows])),
        "color_minus_identity_iou": float(np.mean(all_deltas)),
        "color_minus_identity_scene_cluster_bootstrap_95pct": bootstrap_ci(
            all_deltas, all_scenes, args.bootstrap_samples, rng
        ),
    }
    for stratum in strata:
        subset = [row for row in result_rows if row["displacement_stratum"] == stratum]
        if not subset:
            continue
        deltas = np.asarray([float(row["color_minus_identity_iou"]) for row in subset])
        scenes = [str(row["scene"]) for row in subset]
        summary["strata"][stratum] = {
            "queries": len(subset),
            "scenes": len(set(scenes)),
            "mean_observed_displacement_m": float(np.mean([float(row["observed_centroid_displacement_m"]) for row in subset])),
            "identity_mean_iou": float(np.mean([float(row["identity_instance_iou"]) for row in subset])),
            "color_mean_iou": float(np.mean([float(row["color_instance_iou"]) for row in subset])),
            "color_minus_identity_iou": float(np.mean(deltas)),
            "scene_cluster_bootstrap_95pct": bootstrap_ci(
                deltas, scenes, args.bootstrap_samples, rng
            ) if len(set(scenes)) > 1 else None,
        }
    per_scene_strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in result_rows:
        per_scene_strata[str(row["scene"])][str(row["displacement_stratum"])].append(
            float(row["color_minus_identity_iou"])
        )
    complete_scenes = sorted(per_scene_strata)
    scene_interactions = np.asarray([
        np.mean(per_scene_strata[scene][">=0.30m"])
        - np.mean(per_scene_strata[scene]["<0.10m"])
        for scene in complete_scenes
    ])
    interaction_boot = np.empty(args.bootstrap_samples, dtype=np.float64)
    for sample in range(args.bootstrap_samples):
        chosen_scenes = rng.integers(0, len(complete_scenes), size=len(complete_scenes))
        interaction_boot[sample] = float(np.mean(scene_interactions[chosen_scenes]))
    summary["movement_interaction_high_minus_low"] = {
        "scene_macro_delta_iou": float(np.mean(scene_interactions)),
        "scene_cluster_bootstrap_95pct": [
            float(v) for v in np.quantile(interaction_boot, [0.025, 0.975])
        ],
        "scenes_with_positive_interaction": int(np.count_nonzero(scene_interactions > 0)),
    }
    (args.output_dir / "motion_stratified_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Rescan motion-stratified per-query instance transfer",
        "",
        f"For {len(result_rows)} target-present object/scan queries in {summary['scenes']} scenes, compare identity transfer with the frozen spatial+RGB-selected object pose. Displacement is the observed Euclidean shift between annotated instance centroids in the source and target reconstructed point clouds. It can include reconstruction/viewpoint effects and is only a post-hoc stratum; target masks are not used during prediction.",
        "",
        "| Observed centroid shift | Queries | Scenes | Identity IoU | Spatial+RGB IoU | Paired delta | Scene-cluster 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stratum, row in summary["strata"].items():
        ci = row["scene_cluster_bootstrap_95pct"]
        interval = "not estimable" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(
            f"| {stratum} | {row['queries']} | {row['scenes']} | {row['identity_mean_iou']:.3f} | {row['color_mean_iou']:.3f} | {row['color_minus_identity_iou']:+.3f} | {interval} |"
        )
    overall = summary["overall"]
    ci = overall["color_minus_identity_scene_cluster_bootstrap_95pct"]
    interaction = summary["movement_interaction_high_minus_low"]
    interaction_ci = interaction["scene_cluster_bootstrap_95pct"]
    lines.extend([
        "",
        f"Overall, identity IoU is {overall['identity_mean_iou']:.3f} and spatial+RGB IoU is {overall['color_mean_iou']:.3f}; paired delta {overall['color_minus_identity_iou']:+.3f} (scene-cluster 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]).",
        f"",
        f"The high-minus-low displacement interaction in the color-minus-identity delta is {interaction['scene_macro_delta_iou']:+.3f} (scene-cluster 95% CI [{interaction_ci[0]:+.3f}, {interaction_ci[1]:+.3f}]); it is positive in {interaction['scenes_with_positive_interaction']}/{len(complete_scenes)} scenes.",
        "",
        "These are independent per-object query masks, not globally conflict-resolved full-scene Instance Transfer mIoU. The observed centroid shift is not a ground-truth rigid-motion magnitude because partial visibility and reconstruction differences can move centroids.",
    ])
    (args.output_dir / "motion_stratified_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
