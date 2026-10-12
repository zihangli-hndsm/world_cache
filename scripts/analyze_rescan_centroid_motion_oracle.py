#!/usr/bin/env python3
"""Measure the instance-transfer ceiling of oracle per-object translation.

For each future Rescan scan, stable target instance labels are deliberately
used to associate each target object with its source object and to estimate a
centroid translation. This is an annotation oracle, not a deployable method.
It tests whether object motion compensation could plausibly improve the
first-scan XYZ nearest-neighbor baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_rescan_xyz_transfer import (
    UNKNOWN_ID,
    instance_transfer_miou,
    read_annotated_ply,
    rescan_eval_instance_ids,
)


def score_scan(
    target_path: Path,
    source_points: np.ndarray,
    source_raw_ids: np.ndarray,
    target_points: np.ndarray,
    target_raw_ids: np.ndarray,
    max_distance: float,
) -> dict[str, float | int | str]:
    source_ids = rescan_eval_instance_ids(source_raw_ids)
    target_ids = rescan_eval_instance_ids(target_raw_ids)

    static_dist, static_nearest = cKDTree(source_points).query(target_points, k=1, workers=-1)
    static_prediction = np.where(static_dist <= max_distance, source_ids[static_nearest], UNKNOWN_ID)
    static_miou, _ = instance_transfer_miou(
        target_ids, static_prediction, target_path.with_suffix(".txt")
    )

    source_labels = np.unique(source_raw_ids)
    target_labels = np.unique(target_raw_ids)
    valid_source_labels = set(int(v) for v in source_labels if 0 <= v < 1024)
    valid_target_labels = set(int(v) for v in target_labels if 0 <= v < 1024)
    shared = sorted(valid_source_labels & valid_target_labels)
    moved_points: list[np.ndarray] = []
    moved_ids: list[np.ndarray] = []
    gt_common_points = 0
    for instance_id in shared:
        src_mask = source_raw_ids == instance_id
        tgt_mask = target_raw_ids == instance_id
        src_obj = source_points[src_mask]
        tgt_obj = target_points[tgt_mask]
        if len(src_obj) == 0 or len(tgt_obj) == 0:
            continue
        translation = tgt_obj.mean(axis=0) - src_obj.mean(axis=0)
        moved_points.append(src_obj + translation)
        moved_ids.append(np.full(len(src_obj), int(rescan_eval_instance_ids(np.array([instance_id]))[0]), dtype=np.int32))
        gt_common_points += len(tgt_obj)

    if moved_points:
        template = np.concatenate(moved_points, axis=0)
        template_ids = np.concatenate(moved_ids)
        oracle_dist, oracle_nearest = cKDTree(template).query(target_points, k=1, workers=-1)
        oracle_prediction = np.where(oracle_dist <= max_distance, template_ids[oracle_nearest], UNKNOWN_ID)
        oracle_miou, oracle_instances = instance_transfer_miou(
            target_ids, oracle_prediction, target_path.with_suffix(".txt")
        )
        oracle_coverage = float(np.mean(oracle_dist <= max_distance))
    else:
        oracle_miou, oracle_instances, oracle_coverage = float("nan"), 0, 0.0
    return {
        "target_scan": target_path.stem,
        "shared_source_target_instances": len(shared),
        "target_points_in_shared_instances": gt_common_points,
        "target_point_fraction_from_shared_instances": float(gt_common_points / len(target_points)),
        "static_nn_5cm_coverage": float(np.mean(static_dist <= max_distance)),
        "static_nn_instance_transfer_miou": static_miou,
        "centroid_oracle_5cm_coverage": oracle_coverage,
        "centroid_oracle_instance_transfer_miou": oracle_miou,
        "centroid_oracle_scored_instances": oracle_instances,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = [
        line.strip()
        for line in (args.data_root / "scene_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, float | int | str]] = []
    per_scene: dict[str, dict[str, float | int]] = {}
    for scene in scenes:
        ann = args.data_root / scene / "gt_segmentation"
        source_path = ann / f"{scene}_0.ply"
        source_points, source_raw_ids, _ = read_annotated_ply(source_path)
        targets = sorted(ann.glob(f"{scene}_[1-9]*.ply"))
        scene_rows = []
        for target_path in targets:
            target_points, target_raw_ids, _ = read_annotated_ply(target_path)
            row = score_scan(
                target_path,
                source_points,
                source_raw_ids,
                target_points,
                target_raw_ids,
                args.max_distance,
            )
            row["scene"] = scene
            rows.append(row)
            scene_rows.append(row)
            print(
                f"{target_path.stem}: static={row['static_nn_instance_transfer_miou']:.4f} "
                f"centroid-oracle={row['centroid_oracle_instance_transfer_miou']:.4f} "
                f"shared_ids={row['shared_source_target_instances']}",
                flush=True,
            )
        per_scene[scene] = {
            "target_scans": len(scene_rows),
            "static_nn_miou": float(np.nanmean([r["static_nn_instance_transfer_miou"] for r in scene_rows])),
            "centroid_oracle_miou": float(np.nanmean([r["centroid_oracle_instance_transfer_miou"] for r in scene_rows])),
            "mean_common_instance_count": float(np.mean([r["shared_source_target_instances"] for r in scene_rows])),
        }

    with (args.output_dir / "oracle_per_scan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    static_mean = float(np.nanmean([r["static_nn_instance_transfer_miou"] for r in rows]))
    oracle_mean = float(np.nanmean([r["centroid_oracle_instance_transfer_miou"] for r in rows]))
    summary = {
        "scenes": len(scenes),
        "future_scans": len(rows),
        "centroid_translation_oracle": True,
        "uses_target_instance_labels_for_object_association_and_translation": True,
        "uses_target_labels_for_pointwise_prediction": False,
        "max_distance_m": args.max_distance,
        "static_nn_mean_instance_transfer_miou": static_mean,
        "centroid_oracle_mean_instance_transfer_miou": oracle_mean,
        "centroid_oracle_gain_pp": 100.0 * (oracle_mean - static_mean),
        "per_scene": per_scene,
        "interpretation": "An annotation upper-bound diagnostic. The per-object transform is estimated from target ground-truth instance membership and centroids, so it is not a method result.",
    }
    (args.output_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rescan per-object translation oracle",
        "",
        "This diagnostic uses target ground-truth instance IDs to associate each target object with the same source object, then translates the source points by the difference in object centroids. Target labels are not used for pointwise label prediction, but they do provide oracle object identity and pose. The result is an upper-bound diagnostic, not a deployable method.",
        "",
        f"- Future scans: {len(rows)} across {len(scenes)} scenes",
        f"- Static XYZ NN Instance Transfer mIoU: {static_mean:.4f}",
        f"- Centroid-translation oracle Instance Transfer mIoU: {oracle_mean:.4f}",
        f"- Oracle gain: {100 * (oracle_mean - static_mean):+.2f} pp",
        "",
        "| Scene | Static XYZ NN mIoU | Centroid oracle mIoU | Shared IDs / target |",
        "|---|---:|---:|---:|",
    ]
    for scene, values in per_scene.items():
        lines.append(
            f"| {scene} | {values['static_nn_miou']:.4f} | {values['centroid_oracle_miou']:.4f} | "
            f"{values['mean_common_instance_count']:.1f} |"
        )
    lines.extend([
        "",
        "This oracle tests whether per-object motion compensation could improve the static label-transfer floor. It does not show that the feature matcher can recover these transforms. The full benchmark remains held out from method fitting; any future adaptation should use scene-level folds and preserve untouched final scenes.",
        "",
    ])
    (args.output_dir / "oracle_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
