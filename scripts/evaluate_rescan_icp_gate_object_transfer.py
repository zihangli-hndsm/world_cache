#!/usr/bin/env python3
"""Evaluate frozen ICP/static fallback on per-object transfer queries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from rerank_rescan_fpfh_by_color import read_scene_ply


METRICS = ("instance_iou", "instance_precision", "instance_recall")
STRATA = ("<0.10m", "0.10-0.30m", ">=0.30m")


def transform(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [float(row[f"transform_{r}{c}"]) for r in range(4) for c in range(4)],
        dtype=np.float64,
    ).reshape(4, 4)


def transfer_iou(source_points: np.ndarray, target_ids: np.ndarray, target_tree: cKDTree,
                 instance_id: int, matrix: np.ndarray, max_distance: float) -> dict[str, float]:
    moved = source_points @ matrix[:3, :3].T + matrix[:3, 3]
    distances, nearest = target_tree.query(moved, k=1, workers=1)
    predicted = np.unique(nearest[distances <= max_distance])
    target_mask = target_ids == instance_id
    tp = int(np.count_nonzero(target_mask[predicted]))
    predicted_count = len(predicted)
    target_count = int(np.count_nonzero(target_mask))
    fp = predicted_count - tp
    fn = target_count - tp
    union = tp + fp + fn
    return {
        "instance_iou": tp / union if union else 0.0,
        "instance_precision": tp / predicted_count if predicted_count else 0.0,
        "instance_recall": tp / target_count if target_count else 0.0,
    }


def scene_bootstrap(values: np.ndarray, scenes: list[str], samples: int,
                    rng: np.random.Generator) -> list[float]:
    unique = sorted(set(scenes))
    index = {scene: np.flatnonzero(np.asarray(scenes) == scene) for scene in unique}
    boot = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        draw = rng.integers(0, len(unique), size=len(unique))
        picked = np.concatenate([index[unique[i]] for i in draw])
        boot[sample] = float(np.mean(values[picked]))
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gated-csv", type=Path, required=True)
    parser.add_argument("--selected-transfer-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_032_026)
    args = parser.parse_args()
    if args.max_distance <= 0 or args.bootstrap_samples < 1:
        raise ValueError("max distance and bootstrap count must be positive")

    with args.gated_csv.open(newline="", encoding="utf-8") as stream:
        gate_rows = list(csv.DictReader(stream))
    gate_metrics_path = args.gated_csv.with_name(args.gated_csv.stem + "_gate.csv")
    with gate_metrics_path.open(newline="", encoding="utf-8") as stream:
        gate_metric_rows = list(csv.DictReader(stream))
    with args.selected_transfer_csv.open(newline="", encoding="utf-8") as stream:
        selected_rows = list(csv.DictReader(stream))
    gate = {
        (row["scene"], row["target_scan"], row["source_instance_id"]): row
        for row in gate_rows
    }
    gate_metrics = {
        (row["scene"], row["target_scan"], row["source_instance_id"]): row
        for row in gate_metric_rows
    }
    selected = {
        (row["scene"], row["target_scan"], row["source_instance_id"]): row
        for row in selected_rows
    }
    if not selected or not set(selected).issubset(gate):
        raise ValueError("eligible selected-transfer queries are not a subset of gated poses")

    records: list[dict[str, object]] = []
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    current_target: tuple[str, str] | None = None
    target_ids: np.ndarray | None = None
    target_tree: cKDTree | None = None
    for key in sorted(selected):
        scene, target_scan, raw_id = key
        instance_id = int(raw_id)
        if scene not in source_cache:
            points, _, ids, _ = read_scene_ply(
                args.data_root / scene / "gt_segmentation" / f"{scene}_0.ply"
            )
            source_cache[scene] = (points, ids)
        if current_target != key[:2]:
            target_points, _, target_ids, _ = read_scene_ply(
                args.data_root / scene / "gt_segmentation" / f"{target_scan}.ply"
            )
            target_tree = cKDTree(target_points)
            current_target = key[:2]

        source_points_all, source_ids = source_cache[scene]
        source_points = source_points_all[source_ids == instance_id]
        target_mask = target_ids == instance_id
        if len(source_points) == 0 or not np.any(target_mask):
            raise ValueError(f"query {key} violates target-present eligibility")

        identity_result = transfer_iou(
            source_points, target_ids, target_tree, instance_id, np.eye(4), args.max_distance
        )
        gate_row = gate[key]
        gate_metric = gate_metrics[key]
        gated_matrix = transform(gate_row)
        gated_result = transfer_iou(
            source_points, target_ids, target_tree, instance_id, gated_matrix, args.max_distance
        )
        displacement = float(np.linalg.norm(
            source_points.mean(axis=0) - target_points[target_mask].mean(axis=0)
        ))
        if displacement < 0.10:
            stratum = STRATA[0]
        elif displacement < 0.30:
            stratum = STRATA[1]
        else:
            stratum = STRATA[2]
        row: dict[str, object] = {
            "scene": scene,
            "target_scan": target_scan,
            "source_instance_id": raw_id,
            "observed_centroid_displacement_m": displacement,
            "displacement_stratum": stratum,
            "used_icp_pose": gate_row["used_icp_pose"].lower() == "true",
            "icp_fitness": float(gate_metric["icp_fitness"]),
            "icp_inlier_rmse_m": float(gate_metric["icp_inlier_rmse_m"]),
            "identity_instance_iou": identity_result["instance_iou"],
            "color_instance_iou": float(selected[key]["color_instance_iou"]),
            "gated_icp_instance_iou": gated_result["instance_iou"],
        }
        for metric in METRICS:
            row[f"identity_{metric}"] = identity_result[metric]
            row[f"gated_icp_{metric}"] = gated_result[metric]
            row[f"color_{metric}"] = float(selected[key][f"color_{metric}"])
        records.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gated_object_transfer.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    rng = np.random.default_rng(args.seed)
    scenes = [str(row["scene"]) for row in records]
    summary: dict[str, object] = {
        "queries": len(records),
        "scenes": len(set(scenes)),
        "target_scans": len({(row["scene"], row["target_scan"]) for row in records}),
        "eligible_queries_using_icp": sum(bool(row["used_icp_pose"]) for row in records),
        "eligible_query_icp_fraction": float(np.mean([bool(row["used_icp_pose"]) for row in records])),
        "max_distance_m": args.max_distance,
        "eligibility": "source raw instance ID annotated in target; conditional on eligible queries",
        "prediction": "identity or frozen scene_a fitness/RMSE-gated, per-object ICP pose; target point cloud remains unsegmented",
        "target_labels_used_for_prediction": False,
        "target_labels_used_for_eligibility_stratification_and_scoring": True,
        "displacement_definition": "source-target annotation centroid displacement; post-hoc proxy that also reflects partial visibility and reconstruction differences",
        "bootstrap_unit": "scene, query-weighted within each resampled scene set",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "overall": {},
        "strata": {},
    }
    for metric in METRICS:
        identity_values = np.asarray([float(row[f"identity_{metric}"]) for row in records])
        gated_values = np.asarray([float(row[f"gated_icp_{metric}"]) for row in records])
        color_values = np.asarray([float(row[f"color_{metric}"]) for row in records])
        summary["overall"][metric] = {
            "identity": float(identity_values.mean()),
            "unrefined_color": float(color_values.mean()),
            "gated_icp": float(gated_values.mean()),
            "gated_minus_identity": float(np.mean(gated_values - identity_values)),
            "gated_minus_identity_scene_cluster_bootstrap_95pct": scene_bootstrap(
                gated_values - identity_values, scenes, args.bootstrap_samples, rng
            ),
            "gated_minus_color": float(np.mean(gated_values - color_values)),
            "gated_minus_color_scene_cluster_bootstrap_95pct": scene_bootstrap(
                gated_values - color_values, scenes, args.bootstrap_samples, rng
            ),
        }
    for stratum in STRATA:
        subset = [row for row in records if row["displacement_stratum"] == stratum]
        if not subset:
            continue
        subset_scenes = [str(row["scene"]) for row in subset]
        row_summary: dict[str, object] = {
            "queries": len(subset),
            "scenes": len(set(subset_scenes)),
            "mean_observed_displacement_m": float(np.mean([
                float(row["observed_centroid_displacement_m"]) for row in subset
            ])),
            "icp_accepted": sum(bool(row["used_icp_pose"]) for row in subset),
        }
        for metric in METRICS:
            identity_values = np.asarray([float(row[f"identity_{metric}"]) for row in subset])
            gated_values = np.asarray([float(row[f"gated_icp_{metric}"]) for row in subset])
            color_values = np.asarray([float(row[f"color_{metric}"]) for row in subset])
            row_summary[metric] = {
                "identity": float(identity_values.mean()),
                "unrefined_color": float(color_values.mean()),
                "gated_icp": float(gated_values.mean()),
                "gated_minus_identity": float(np.mean(gated_values - identity_values)),
                "gated_minus_identity_scene_cluster_bootstrap_95pct": scene_bootstrap(
                    gated_values - identity_values, subset_scenes, args.bootstrap_samples, rng
                ) if len(set(subset_scenes)) > 1 else None,
            }
        summary["strata"][stratum] = row_summary

    (args.output_dir / "gated_object_transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Per-query transfer under a frozen ICP/static fallback",
        "",
        f"Evaluated {len(records)} target-present queries across {summary['scenes']} Rescan scenes. The gate uses scene_a-frozen ICP fitness >=0.8 and RMSE <=0.05 m; failures use identity. Target masks are used only for eligibility, the post-hoc displacement stratum, and scoring.",
        "",
        "| Observed centroid shift | Queries | ICP accepted | Identity IoU | Raw color-pose IoU | Gated ICP IoU | Gated − identity (95% scene CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stratum, row in summary["strata"].items():
        ci = row["instance_iou"]["gated_minus_identity_scene_cluster_bootstrap_95pct"]
        interval = "not estimable" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(
            f"| {stratum} | {row['queries']} | {row['icp_accepted']} | {row['instance_iou']['identity']:.3f} | {row['instance_iou']['unrefined_color']:.3f} | {row['instance_iou']['gated_icp']:.3f} | {row['instance_iou']['gated_minus_identity']:+.3f} {interval} |"
        )
    overall = summary["overall"]["instance_iou"]
    overall_ci = overall["gated_minus_identity_scene_cluster_bootstrap_95pct"]
    lines.extend([
        "",
        f"Overall, identity IoU is {overall['identity']:.3f}, raw color-pose IoU {overall['unrefined_color']:.3f}, and gated ICP IoU {overall['gated_icp']:.3f}; gated-minus-identity is {overall['gated_minus_identity']:+.3f} (scene-cluster 95% CI [{overall_ci[0]:+.3f}, {overall_ci[1]:+.3f}]).",
        "",
        "These are independent per-query instance masks, not globally conflict-resolved full-scene Instance Transfer mIoU. The measured centroid shift is not ground-truth rigid motion.",
    ])
    (args.output_dir / "gated_object_transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
