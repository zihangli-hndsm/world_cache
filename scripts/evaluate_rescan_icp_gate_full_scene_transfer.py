#!/usr/bin/env python3
"""Evaluate corrected ICP/static-gate transforms with full-scene Rescan mIoU."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_rescan_full_scene_pose_transfer import get_transform, rank_group, write_csv
from evaluate_rescan_xyz_transfer import (
    MAX_INSTANCE_ID,
    UNKNOWN_ID,
    instance_transfer_miou,
    read_annotated_ply,
    rescan_eval_instance_ids,
)


def scene_prediction(
    target_points: np.ndarray,
    pair_keys: list[tuple[str, str, str]],
    source_cache: dict[str, np.ndarray],
    transforms: dict[tuple[str, str, str], np.ndarray],
    max_distance: float,
) -> tuple[np.ndarray, float, int]:
    moved_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for key in pair_keys:
        source_id = int(key[2])
        points = source_cache.get(str(source_id))
        if points is None or len(points) == 0:
            continue
        matrix = transforms[key]
        moved_parts.append(points @ matrix[:3, :3].T + matrix[:3, 3])
        label_parts.append(np.full(len(points), source_id, dtype=np.int32))
    if not moved_parts:
        return np.full(len(target_points), UNKNOWN_ID, dtype=np.int32), 0.0, 0

    all_points = np.concatenate(moved_parts, axis=0)
    all_labels = np.concatenate(label_parts)
    distances, nearest = cKDTree(all_points).query(target_points, k=1, workers=1)
    accepted = distances <= max_distance
    prediction = np.full(len(target_points), UNKNOWN_ID, dtype=np.int32)
    labels = all_labels[nearest[accepted]]
    valid = (labels >= 0) & (labels < MAX_INSTANCE_ID)
    prediction[np.flatnonzero(accepted)[valid]] = labels[valid] + 1
    return prediction, float(np.mean(accepted)), int(np.unique(prediction[prediction <= MAX_INSTANCE_ID]).size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True,
                        help="Five original FPFH proposals per object/target query.")
    parser.add_argument("--gated-csv", type=Path, required=True,
                        help="One saved ICP/static-fallback pose per object/target query.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_042_026)
    args = parser.parse_args()
    if args.max_distance <= 0 or args.bootstrap_samples < 1:
        raise ValueError("max distance and bootstrap sample count must be positive")

    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    candidate_rows = read_rows(args.candidate_csv)
    gated_rows = read_rows(args.gated_csv)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["scene"], row["target_scan"], str(row["source_instance_id"]))].append(row)
    if not groups or any(len(rows) != 5 for rows in groups.values()):
        raise ValueError("expected exactly five original proposals for each query")
    gated = {
        (row["scene"], row["target_scan"], str(row["source_instance_id"])): row
        for row in gated_rows
    }
    if set(gated) != set(groups):
        raise ValueError("gated and original candidate query sets differ")
    absent = sum(
        rows[0]["target_instance_present"].lower() == "false"
        for rows in groups.values()
    )
    scans = sorted({key[:2] for key in groups})
    per_scan: list[dict[str, object]] = []
    source_cache: dict[str, np.ndarray] = {}
    cached_scene = ""
    for scene, target_scan in scans:
        annotation_dir = args.data_root / scene / "gt_segmentation"
        if scene != cached_scene:
            source_points, source_ids, _ = read_annotated_ply(annotation_dir / f"{scene}_0.ply")
            source_cache = {
                str(instance_id): source_points[source_ids == instance_id]
                for instance_id in np.unique(source_ids)
                if 0 <= int(instance_id) < 1024
            }
            cached_scene = scene
        target_points, target_ids, _ = read_annotated_ply(annotation_dir / f"{target_scan}.ply")
        gt = rescan_eval_instance_ids(target_ids)
        pair_keys = sorted(key for key in groups if key[:2] == (scene, target_scan))
        color_transforms = {
            key: get_transform(rank_group(groups[key], "color")) for key in pair_keys
        }
        gated_transforms = {key: get_transform(gated[key]) for key in pair_keys}
        identity_transforms = {key: np.eye(4, dtype=np.float64) for key in pair_keys}
        permutation_file = annotation_dir / f"{target_scan}.txt"
        row: dict[str, object] = {
            "scene": scene,
            "target_scan": target_scan,
            "source_objects_with_candidates": len(pair_keys),
            "target_points": len(target_points),
            "target_instances_scored": int(np.unique(gt[(gt > 0) & (gt <= MAX_INSTANCE_ID)]).size),
            "queries_using_icp": sum(gated[key]["used_icp_pose"].lower() == "true" for key in pair_keys),
        }
        for arm, matrices in (("identity", identity_transforms), ("color", color_transforms),
                              ("gated_icp", gated_transforms)):
            pred, coverage, n_instances = scene_prediction(
                target_points, pair_keys, source_cache, matrices, args.max_distance
            )
            miou, scored = instance_transfer_miou(gt, pred, permutation_file)
            row[f"{arm}_instance_transfer_miou"] = miou
            row[f"{arm}_point_coverage"] = coverage
            row[f"{arm}_predicted_instances"] = n_instances
            row[f"{arm}_instances_scored"] = scored
        per_scan.append(row)
        print(f"completed {target_scan}; scans={len(per_scan)}/{len(scans)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "full_scene_transfer_per_scan.csv", per_scan)
    scenes = sorted({str(row["scene"]) for row in per_scan})
    by_scene = {scene: [row for row in per_scan if row["scene"] == scene] for scene in scenes}
    arms = ("identity", "color", "gated_icp")
    summary: dict[str, object] = {
        "scenes": len(scenes), "target_scans": len(per_scan),
        "source_object_target_pairs": len(groups), "target_absent_source_object_pairs": absent,
        "target_absent_pairs_included": absent > 0, "max_distance_m": args.max_distance,
        "prediction": "global nearest point among all independently transformed source-object clouds; reject beyond max_distance",
        "target_instance_labels_used_for_prediction": False,
        "target_instance_labels_used_for_pair_eligibility": False,
        "target_instance_labels_used_for_scoring": True,
        "official_metric_port": "Rescan Instance Transfer mIoU with per-target ambiguity remapping",
        "bootstrap_unit": "scene; all target scans from sampled scene are kept together",
        "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
        "arms": {}, "paired_deltas": {}, "per_scene": {},
    }
    for arm in arms:
        summary["arms"][arm] = {
            "mean_instance_transfer_miou_over_scans": float(np.nanmean([
                float(row[f"{arm}_instance_transfer_miou"]) for row in per_scan
            ])),
            "mean_target_point_coverage": float(np.mean([
                float(row[f"{arm}_point_coverage"]) for row in per_scan
            ])),
        }
    rng = np.random.default_rng(args.seed)
    for left, right in (("gated_icp", "identity"), ("gated_icp", "color")):
        deltas = np.asarray([
            float(row[f"{left}_instance_transfer_miou"]) - float(row[f"{right}_instance_transfer_miou"])
            for row in per_scan
        ])
        draws = np.empty(args.bootstrap_samples)
        for index in range(args.bootstrap_samples):
            sampled = rng.integers(0, len(scenes), size=len(scenes))
            selected = [row for si in sampled for row in by_scene[scenes[si]]]
            draws[index] = np.nanmean([
                float(row[f"{left}_instance_transfer_miou"]) - float(row[f"{right}_instance_transfer_miou"])
                for row in selected
            ])
        lo, hi = np.quantile(draws, [0.025, 0.975])
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
    for scene in scenes:
        subset = by_scene[scene]
        summary["per_scene"][scene] = {
            "target_scans": len(subset),
            **{f"{arm}_instance_transfer_miou": float(np.nanmean([
                float(row[f"{arm}_instance_transfer_miou"]) for row in subset
            ])) for arm in arms},
        }
    (args.output_dir / "full_scene_transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Full-scene Rescan transfer with corrected ICP/static gate", "",
        f"Evaluated {len(groups)} source-object/target-scan queries (including {absent} target-absent objects) across {len(scenes)} scenes and {len(per_scan)} target scans. The frozen gate is ICP fitness >=0.8 and RMSE <=0.05 m, with identity fallback. Target labels are used only for official Instance Transfer mIoU scoring and ambiguity remapping.",
        "", "| Arm | Mean Instance Transfer mIoU | Mean point coverage |",
        "|---|---:|---:|",
    ]
    for arm in arms:
        values = summary["arms"][arm]
        lines.append(f"| {arm} | {values['mean_instance_transfer_miou_over_scans']:.4f} | {values['mean_target_point_coverage']:.3f} |")
    lines.extend(["", "## Paired scene-cluster 95% intervals", "", "| Comparison | Mean scan-weighted delta | Scene-macro delta | 95% CI |", "|---|---:|---:|---:|"])
    for comparison, values in summary["paired_deltas"].items():
        lo, hi = values["scene_cluster_bootstrap_95pct"]
        lines.append(f"| {comparison.replace('_minus_', ' − ')} | {values['mean_scan_weighted_delta']:+.4f} | {values['scene_macro_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines.extend(["", "## Per-scene results", "", "| Scene | Scans | Identity | Raw color pose | Gated ICP |", "|---|---:|---:|---:|---:|"])
    for scene, values in summary["per_scene"].items():
        lines.append(f"| {scene} | {values['target_scans']} | {values['identity_instance_transfer_miou']:.4f} | {values['color_instance_transfer_miou']:.4f} | {values['gated_icp_instance_transfer_miou']:.4f} |")
    (args.output_dir / "full_scene_transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
