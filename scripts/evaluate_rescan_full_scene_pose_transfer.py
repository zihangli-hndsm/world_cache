#!/usr/bin/env python3
"""Evaluate full-scene Rescan label transfer from independently posed source objects."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_rescan_xyz_transfer import (
    MAX_INSTANCE_ID,
    UNKNOWN_ID,
    instance_transfer_miou,
    read_annotated_ply,
    rescan_eval_instance_ids,
)


def rank_group(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    if arm == "fitness":
        key = lambda r: (-float(r["fitness"]), int(r["repeat"]))
    elif arm == "spatial":
        key = lambda r: (-float(r["spatial_coverage"]), -float(r["fitness"]), int(r["repeat"]))
    elif arm == "color":
        key = lambda r: (
            -float(r["joint_spatial_rgb_inlier_fraction"]),
            -float(r["spatial_coverage"]),
            float(r["median_rgb_distance_on_spatial_inliers"]),
            -float(r["fitness"]),
            int(r["repeat"]),
        )
    else:
        raise ValueError(arm)
    return sorted(rows, key=key)[0]


def get_transform(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row[f"transform_{r}{c}"]) for r in range(4) for c in range(4)],
        dtype=np.float64,
    ).reshape(4, 4)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_global_prediction(
    target_points: np.ndarray,
    groups: dict[tuple[str, str, str], list[dict[str, str]]],
    pair_keys: list[tuple[str, str, str]],
    source_cache: dict[str, np.ndarray],
    arm: str,
    max_distance: float,
) -> tuple[np.ndarray, float, int]:
    transformed_parts = []
    label_parts = []
    for key in pair_keys:
        source_id = int(key[2])
        source_points = source_cache.get(str(source_id))
        if source_points is None or len(source_points) == 0:
            continue
        matrix = get_transform(rank_group(groups[key], arm))
        transformed_parts.append(source_points @ matrix[:3, :3].T + matrix[:3, 3])
        label_parts.append(np.full(len(source_points), source_id, dtype=np.int32))
    if not transformed_parts:
        pred = np.full(len(target_points), UNKNOWN_ID, dtype=np.int32)
        return pred, 0.0, 0

    transformed_all = np.concatenate(transformed_parts, axis=0)
    labels_all = np.concatenate(label_parts)
    tree = cKDTree(transformed_all)
    distances, nearest = tree.query(target_points, k=1, workers=1)
    accepted = distances <= max_distance
    prediction = np.full(len(target_points), UNKNOWN_ID, dtype=np.int32)
    labels = labels_all[nearest[accepted]]
    valid_label = (labels >= 0) & (labels < MAX_INSTANCE_ID)
    accepted_indices = np.flatnonzero(accepted)
    prediction[accepted_indices[valid_label]] = labels[valid_label] + 1
    coverage = float(np.mean(accepted))
    predicted_instances = int(len(np.unique(prediction[prediction <= MAX_INSTANCE_ID])))
    return prediction, coverage, predicted_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_042_026)
    args = parser.parse_args()
    if args.max_distance <= 0 or args.bootstrap_samples < 1:
        raise ValueError("max distance and bootstrap sample count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["scene"], row["target_scan"], str(row["source_instance_id"]))].append(row)
    if not groups or any(len(rows) != 5 for rows in groups.values()):
        raise ValueError("expected five saved proposals for every source-object/target-scan pair")
    if not all("target_instance_present" in row for row in candidate_rows):
        raise ValueError("candidate file must record whether target-absent source objects were included")
    target_absent_pairs = sum(
        row["target_instance_present"].lower() == "false"
        for key, rows in groups.items()
        for row in rows[:1]
    )

    scans = sorted({(key[0], key[1]) for key in groups})
    partial_path = args.output_dir / "full_scene_transfer_per_scan.partial.csv"
    scan_rows: list[dict[str, str]] = []
    if partial_path.exists():
        with partial_path.open(newline="", encoding="utf-8") as stream:
            scan_rows = list(csv.DictReader(stream))
        print(f"resuming from {len(scan_rows)} completed scans", flush=True)
    completed = {(row["scene"], row["target_scan"]) for row in scan_rows}

    source_cache: dict[str, np.ndarray] = {}
    cached_scene = ""
    for scene, target_scan in scans:
        if (scene, target_scan) in completed:
            continue
        annotation_dir = args.data_root / scene / "gt_segmentation"
        if cached_scene != scene:
            source_points, source_ids, _ = read_annotated_ply(annotation_dir / f"{scene}_0.ply")
            source_cache = {
                str(instance_id): source_points[source_ids == instance_id]
                for instance_id in np.unique(source_ids)
                if 0 <= int(instance_id) < 1024
            }
            cached_scene = scene
        target_points, target_ids, _ = read_annotated_ply(annotation_dir / f"{target_scan}.ply")
        ground_truth = rescan_eval_instance_ids(target_ids)
        pair_keys = sorted(key for key in groups if key[:2] == (scene, target_scan))
        permutation_file = annotation_dir / f"{target_scan}.txt"
        row: dict[str, object] = {
            "scene": scene,
            "target_scan": target_scan,
            "source_objects_with_candidates": len(pair_keys),
            "target_points": len(target_points),
            "target_instances_scored": int(len(np.unique(ground_truth[(ground_truth > 0) & (ground_truth <= MAX_INSTANCE_ID)]))),
        }
        for arm in ("fitness", "spatial", "color"):
            prediction, coverage, instance_count = make_global_prediction(
                target_points,
                groups,
                pair_keys,
                source_cache,
                arm,
                args.max_distance,
            )
            miou, scored_instances = instance_transfer_miou(ground_truth, prediction, permutation_file)
            row[f"{arm}_instance_transfer_miou"] = miou
            row[f"{arm}_point_coverage"] = coverage
            row[f"{arm}_predicted_instances"] = instance_count
            row[f"{arm}_instances_scored"] = scored_instances
        scan_rows.append(row)
        write_csv(partial_path, scan_rows)
        print(f"completed {target_scan}; scans={len(scan_rows)}", flush=True)

    if len(scan_rows) != len(scans):
        raise ValueError(f"incomplete scan rows: expected {len(scans)}, got {len(scan_rows)}")
    write_csv(args.output_dir / "full_scene_transfer_per_scan.csv", scan_rows)
    partial_path.unlink(missing_ok=True)
    scenes = sorted({row["scene"] for row in scan_rows})
    by_scene = {scene: [row for row in scan_rows if row["scene"] == scene] for scene in scenes}
    arms = ("fitness", "spatial", "color")
    summary: dict[str, object] = {
        "scenes": len(scenes),
        "target_scans": len(scan_rows),
        "source_object_target_pairs": len(groups),
        "target_absent_source_object_pairs": target_absent_pairs,
        "target_absent_pairs_included": target_absent_pairs > 0,
        "max_distance_m": args.max_distance,
        "prediction": "globally assign each target point to its nearest point among all independently transformed source-instance clouds, accepting only distances within max_distance",
        "target_instance_labels_used_for_prediction": False,
        "target_instance_labels_used_for_pair_eligibility": False,
        "target_instance_labels_used_for_scoring": True,
        "official_metric_port": "Rescan Instance Transfer mIoU with per-target ambiguity remapping",
        "bootstrap_unit": "scene; all target scans from sampled scene are kept together",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "interval": "paired scene-cluster percentile 95% confidence interval",
        "arms": {},
        "paired_deltas": {},
    }
    rng = np.random.default_rng(args.seed)
    for arm in arms:
        summary["arms"][arm] = {
            "mean_instance_transfer_miou_over_scans": float(np.nanmean([
                float(row[f"{arm}_instance_transfer_miou"]) for row in scan_rows
            ])),
            "mean_target_point_coverage": float(np.mean([
                float(row[f"{arm}_point_coverage"]) for row in scan_rows
            ])),
        }
    for left, right in (("spatial", "fitness"), ("color", "spatial"), ("color", "fitness")):
        deltas = np.asarray([
            float(row[f"{left}_instance_transfer_miou"]) - float(row[f"{right}_instance_transfer_miou"])
            for row in scan_rows
        ])
        boot = np.empty(args.bootstrap_samples, dtype=np.float64)
        for sample_index in range(args.bootstrap_samples):
            sampled = rng.integers(0, len(scenes), size=len(scenes))
            selected = [
                row
                for scene_index in sampled
                for row in by_scene[scenes[scene_index]]
            ]
            boot[sample_index] = np.nanmean([
                float(row[f"{left}_instance_transfer_miou"]) - float(row[f"{right}_instance_transfer_miou"])
                for row in selected
            ])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        summary["paired_deltas"][f"{left}_minus_{right}"] = {
            "mean_scan_weighted_delta": float(np.nanmean(deltas)),
            "scene_macro_delta": float(np.mean([
                np.nanmean([
                    float(row[f"{left}_instance_transfer_miou"]) - float(row[f"{right}_instance_transfer_miou"])
                    for row in by_scene[scene]
                ]) for scene in scenes
            ])),
            "scene_cluster_bootstrap_95pct": [float(lo), float(hi)],
        }
    per_scene = {}
    for scene in scenes:
        subset = by_scene[scene]
        per_scene[scene] = {
            "target_scans": len(subset),
            **{
                f"{arm}_instance_transfer_miou": float(np.nanmean([
                    float(row[f"{arm}_instance_transfer_miou"]) for row in subset
                ]))
                for arm in arms
            },
        }
    summary["per_scene"] = per_scene
    (args.output_dir / "full_scene_transfer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Rescan full-scene instance transfer from independently selected poses",
        "",
        f"Evaluated {len(groups)} source-object/target-scan proposals over {len(scenes)} held-out scenes and {len(scan_rows)} target scans. The candidate set includes {target_absent_pairs} source-object/target-scan pairs whose raw ID is absent from that target, removing the previous target-presence filter. For each arm, each target point receives the ID of its nearest point across all transformed source-object clouds, or unknown if farther than {args.max_distance:.2f} m. Predictions use target geometry only; target labels are used only by the official Rescan Instance Transfer mIoU scorer and ambiguity remapping.",
        "",
        f"Scene-cluster paired bootstrap uses {args.bootstrap_samples:,} draws (seed {args.seed}); full target scans in a sampled scene remain together.",
        "",
        "| Arm | Mean Instance Transfer mIoU | Mean point coverage |",
        "|---|---:|---:|",
    ]
    for arm in arms:
        values = summary["arms"][arm]
        lines.append(f"| {arm} | {values['mean_instance_transfer_miou_over_scans']:.4f} | {values['mean_target_point_coverage']:.3f} |")
    lines.extend(["", "## Paired scene-cluster 95% intervals", "", "| Comparison | Mean scan-weighted delta | Scene-macro delta | 95% CI |", "|---|---:|---:|---:|"])
    for comparison, values in summary["paired_deltas"].items():
        lo, hi = values["scene_cluster_bootstrap_95pct"]
        lines.append(f"| {comparison.replace('_minus_', ' − ')} | {values['mean_scan_weighted_delta']:+.4f} | {values['scene_macro_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines.extend(["", "## Per-scene scores", "", "| Scene | Scans | Fitness | Spatial-only | Spatial+RGB |", "|---|---:|---:|---:|---:|"])
    for scene, values in per_scene.items():
        lines.append(f"| {scene} | {values['target_scans']} | {values['fitness_instance_transfer_miou']:.4f} | {values['spatial_instance_transfer_miou']:.4f} | {values['color_instance_transfer_miou']:.4f} |")
    lines.extend([
        "",
        "## Limits",
        "",
        "Source instance masks and per-object proposal transforms are available. The target is unsegmented during prediction and target IDs do not select eligible queries. This remains an FPFH geometry proposal system with a frozen spatial/RGB ranker, not the primary DINO method; the held-out scenes are independent Rescan scenes, not official 3RScan validation.",
    ])
    (args.output_dir / "full_scene_transfer_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
