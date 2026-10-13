#!/usr/bin/env python3
"""Score point-level instance transfer from selected saved FPFH transforms."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from rerank_rescan_fpfh_by_color import read_scene_ply


ARMS = ("fitness", "spatial", "color")
METRICS = ("instance_iou", "instance_precision", "instance_recall")


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


def transform(row: dict[str, str]) -> np.ndarray:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_032_026)
    args = parser.parse_args()
    if args.max_distance <= 0 or args.bootstrap_samples < 1:
        raise ValueError("max distance and bootstrap sample count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)
    if not groups or any(len(rows) != 5 for rows in groups.values()):
        raise ValueError("expected a non-empty candidate set with exactly five proposals per pair")

    partial_path = args.output_dir / "instance_transfer_objects.partial.csv"
    object_rows: list[dict[str, str]] = []
    if partial_path.exists():
        with partial_path.open(newline="", encoding="utf-8") as stream:
            object_rows = list(csv.DictReader(stream))
        print(f"resuming from {len(object_rows)} completed pairs", flush=True)
    completed = {
        (row["scene"], row["target_scan"], row["source_instance_id"])
        for row in object_rows
    }

    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cached_scene = ""
    target_tree: cKDTree | None = None
    target_ids: np.ndarray | None = None
    target_classes: np.ndarray | None = None
    current_target: tuple[str, str] | None = None
    sorted_groups = sorted(groups)

    for index, key in enumerate(sorted_groups):
        if key in completed:
            continue
        scene, target_scan, raw_id = key
        if current_target != (scene, target_scan):
            annotation_dir = args.data_root / scene / "gt_segmentation"
            if cached_scene != scene:
                source_points, _, source_ids, _ = read_scene_ply(annotation_dir / f"{scene}_0.ply")
                source_cache = {
                    str(instance_id): source_points[source_ids == int(instance_id)]
                    for instance_id in {
                        group_key[2] for group_key in groups if group_key[0] == scene
                    }
                }
                cached_scene = scene
            target_points, _, target_ids, target_classes = read_scene_ply(annotation_dir / f"{target_scan}.ply")
            target_tree = cKDTree(target_points)
            current_target = (scene, target_scan)
            print(f"loaded {target_scan}", flush=True)

        source_object = source_cache[raw_id]
        target_mask = target_ids == int(raw_id)
        if not np.any(target_mask):
            raise ValueError(f"candidate pair {key} is not target-present; expected eligible pairs only")
        source_class = int(groups[key][0]["source_class_id"])
        result: dict[str, object] = {
            "scene": scene,
            "target_scan": target_scan,
            "source_instance_id": raw_id,
            "source_class_id": source_class,
            "source_points": len(source_object),
            "target_instance_points": int(np.count_nonzero(target_mask)),
        }
        for arm in ARMS:
            candidate = rank_group(groups[key], arm)
            matrix = transform(candidate)
            transformed = source_object @ matrix[:3, :3].T + matrix[:3, 3]
            distances, nearest = target_tree.query(transformed, k=1, workers=1)
            predicted_target_indices = np.unique(nearest[distances <= args.max_distance])
            predicted_mask = np.zeros(len(target_ids), dtype=bool)
            predicted_mask[predicted_target_indices] = True
            true_positive = int(np.count_nonzero(predicted_mask & target_mask))
            false_positive = int(np.count_nonzero(predicted_mask & ~target_mask))
            false_negative = int(np.count_nonzero(target_mask & ~predicted_mask))
            denom_iou = true_positive + false_positive + false_negative
            denom_precision = true_positive + false_positive
            denom_recall = true_positive + false_negative
            result[f"{arm}_predicted_target_points"] = int(np.count_nonzero(predicted_mask))
            result[f"{arm}_instance_iou"] = true_positive / denom_iou if denom_iou else 0.0
            result[f"{arm}_instance_precision"] = true_positive / denom_precision if denom_precision else 0.0
            result[f"{arm}_instance_recall"] = true_positive / denom_recall if denom_recall else 0.0
        object_rows.append(result)

        next_target = sorted_groups[index + 1][:2] if index + 1 < len(sorted_groups) else None
        if next_target != (scene, target_scan):
            write_csv(partial_path, object_rows)
            print(f"checkpointed {target_scan}; completed_pairs={len(object_rows)}", flush=True)

    if len(object_rows) != len(groups):
        raise ValueError(f"incomplete transfer rows: expected {len(groups)}, got {len(object_rows)}")
    write_csv(args.output_dir / "instance_transfer_objects.csv", object_rows)
    partial_path.unlink(missing_ok=True)

    scenes = sorted({row["scene"] for row in object_rows})
    by_scene = {scene: [row for row in object_rows if row["scene"] == scene] for scene in scenes}
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "scenes": len(scenes),
        "target_scans": len({(row["scene"], row["target_scan"]) for row in object_rows}),
        "object_target_pairs": len(object_rows),
        "max_distance_m": args.max_distance,
        "prediction": "for each query, transform source-instance points and transfer its ID to unique nearest target points within the fixed distance",
        "pair_eligibility": "source raw instance ID must be present in target annotations; metrics are conditional on annotated target presence",
        "target_instance_labels_used_for_prediction": False,
        "target_instance_labels_used_for_pair_eligibility": True,
        "target_instance_labels_used_for_scoring": True,
        "bootstrap_unit": "scene, paired ranking arms kept together",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "interval": "percentile 95% confidence interval",
        "arms": {},
    }
    for arm in ARMS:
        summary["arms"][arm] = {
            metric: float(np.mean([float(row[f"{arm}_{metric}"]) for row in object_rows]))
            for metric in METRICS
        }
    summary["per_scene"] = {}
    scene_rows: list[dict[str, object]] = []
    for scene in scenes:
        subset = by_scene[scene]
        scene_result: dict[str, object] = {"scene": scene, "object_target_pairs": len(subset)}
        for arm in ARMS:
            scene_result[f"{arm}_instance_iou"] = float(np.mean([
                float(row[f"{arm}_instance_iou"]) for row in subset
            ]))
        scene_result["color_minus_spatial_iou"] = (
            scene_result["color_instance_iou"] - scene_result["spatial_instance_iou"]
        )
        summary["per_scene"][scene] = scene_result
        scene_rows.append(scene_result)
    write_csv(args.output_dir / "instance_transfer_per_scene.csv", scene_rows)
    positive_scene_count = sum(row["color_minus_spatial_iou"] > 0 for row in scene_rows)
    summary["color_vs_spatial_iou_positive_scenes"] = positive_scene_count
    loo_macro_deltas = [
        float(np.mean([
            row["color_minus_spatial_iou"] for j, row in enumerate(scene_rows) if j != omitted
        ]))
        for omitted in range(len(scene_rows))
    ]
    summary["color_vs_spatial_iou_leave_one_scene_out_macro_delta_range"] = [
        min(loo_macro_deltas), max(loo_macro_deltas)
    ]
    summary["paired_deltas"] = {}
    for metric in METRICS:
        summary["paired_deltas"][metric] = {}
        for left, right in (("spatial", "fitness"), ("color", "spatial"), ("color", "fitness")):
            weighted_delta = float(np.mean([
                float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                for row in object_rows
            ]))
            boot = np.empty(args.bootstrap_samples, dtype=float)
            for sample_index in range(args.bootstrap_samples):
                sampled = rng.integers(0, len(scenes), size=len(scenes))
                numerator = sum(
                    sum(
                        float(row[f"{left}_{metric}"]) - float(row[f"{right}_{metric}"])
                        for row in by_scene[scenes[scene_index]]
                    )
                    for scene_index in sampled
                )
                denominator = sum(len(by_scene[scenes[scene_index]]) for scene_index in sampled)
                boot[sample_index] = numerator / denominator
            lo, hi = np.quantile(boot, [0.025, 0.975])
            summary["paired_deltas"][metric][f"{left}_minus_{right}"] = {
                "object_weighted_delta": weighted_delta,
                "scene_cluster_bootstrap_95pct": [float(lo), float(hi)],
            }

    (args.output_dir / "instance_transfer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Rescan point-level transfer from selected FPFH transforms",
        "",
        f"Conditional evaluation on {len(object_rows)} target-present source-instance queries in {len(scenes)} scenes / {summary['target_scans']} target scans. For each arm's top-ranked transform, all source-instance points query the unsegmented target cloud; each unique nearest target point within {args.max_distance:.2f} m receives that source ID. Target labels are used only for pair eligibility and scoring. Scores are object-target weighted; paired scene-cluster bootstrap uses {args.bootstrap_samples:,} draws (seed {args.seed}).",
        "",
        "This is a per-query instance-mask transfer diagnostic. Queries are scored independently, so it is not a globally conflict-resolved full-scene segmentation or official Rescan Instance Transfer mIoU.",
        "",
        "| Arm | Mean instance IoU | Mean precision | Mean recall |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = summary["arms"][arm]
        lines.append(f"| {arm} | {values['instance_iou']:.3f} | {values['instance_precision']:.3f} | {values['instance_recall']:.3f} |")
    lines.extend(["", "## Paired scene-cluster 95% intervals for mean deltas", "", "| Metric | Comparison | Delta | 95% CI |", "|---|---|---:|---:|"])
    for metric in METRICS:
        for comparison, values in summary["paired_deltas"][metric].items():
            lo, hi = values["scene_cluster_bootstrap_95pct"]
            lines.append(f"| {metric} | {comparison.replace('_minus_', ' − ')} | {values['object_weighted_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines.extend([
        "",
        f"Color minus spatial-only IoU is positive in {positive_scene_count}/{len(scenes)} scenes; leave-one-scene-out scene-macro delta ranges from {min(loo_macro_deltas):+.3f} to {max(loo_macro_deltas):+.3f}.",
        "",
        "## Per-scene mean instance IoU",
        "",
        "| Scene | Pairs | Fitness | Spatial-only | Spatial+RGB | RGB minus spatial |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in scene_rows:
        lines.append(
            f"| {row['scene']} | {row['object_target_pairs']} | {row['fitness_instance_iou']:.3f} | "
            f"{row['spatial_instance_iou']:.3f} | {row['color_instance_iou']:.3f} | "
            f"{row['color_minus_spatial_iou']:+.3f} |"
        )
    (args.output_dir / "instance_transfer_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
