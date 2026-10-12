#!/usr/bin/env python3
"""Score every frozen FPFH proposal for point-level Rescan instance transfer."""

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


def rank_key(row: dict[str, str], arm: str) -> tuple[float | int, ...]:
    if arm == "fitness":
        return (-float(row["fitness"]), int(row["repeat"]))
    if arm == "spatial":
        return (-float(row["spatial_coverage"]), -float(row["fitness"]), int(row["repeat"]))
    if arm == "color":
        return (
            -float(row["joint_spatial_rgb_inlier_fraction"]),
            -float(row["spatial_coverage"]),
            float(row["median_rgb_distance_on_spatial_inliers"]),
            -float(row["fitness"]),
            int(row["repeat"]),
        )
    raise ValueError(arm)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def scene_bootstrap(
    values: np.ndarray,
    scenes: list[str],
    scene_order: list[str],
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    index = {scene: np.flatnonzero(np.asarray(scenes) == scene) for scene in scene_order}
    boot = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        draw = rng.integers(0, len(scene_order), size=len(scene_order))
        selected = np.concatenate([index[scene_order[k]] for k in draw])
        boot[sample] = float(np.mean(values[selected]))
    return [float(value) for value in np.quantile(boot, [0.025, 0.975])]


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
        raise ValueError("max distance and bootstrap count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)
    if not groups or any(len(rows) != 5 for rows in groups.values()):
        raise ValueError("expected five frozen pose proposals per eligible query")

    partial = args.output_dir / "all_proposal_transfer.partial.csv"
    score_rows: list[dict[str, str]] = []
    if partial.exists():
        with partial.open(newline="", encoding="utf-8") as stream:
            score_rows = list(csv.DictReader(stream))
        counts: dict[tuple[str, str, str], int] = defaultdict(int)
        for row in score_rows:
            counts[(row["scene"], row["target_scan"], row["source_instance_id"])] += 1
        if any(count != 5 for count in counts.values()):
            raise ValueError("partial checkpoint contains incomplete candidate groups")
        completed = set(counts)
        print(f"resuming {len(completed)} complete queries", flush=True)
    else:
        completed = set()

    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_tree: cKDTree | None = None
    target_ids: np.ndarray | None = None
    current_target: tuple[str, str] | None = None
    keys = sorted(groups)
    for index, key in enumerate(keys):
        if key in completed:
            continue
        scene, target_scan, instance_id = key
        if current_target != key[:2]:
            annotation_dir = args.data_root / scene / "gt_segmentation"
            if scene not in source_cache:
                points, _, ids, _ = read_scene_ply(annotation_dir / f"{scene}_0.ply")
                source_cache[scene] = (points, ids)
            target_points, _, target_ids, _ = read_scene_ply(annotation_dir / f"{target_scan}.ply")
            target_tree = cKDTree(target_points)
            current_target = key[:2]
            print(f"loaded {target_scan}", flush=True)

        source_points_all, source_ids_all = source_cache[scene]
        source_points = source_points_all[source_ids_all == int(instance_id)]
        target_mask = target_ids == int(instance_id)
        target_count = int(np.count_nonzero(target_mask))
        if len(source_points) == 0 or target_count == 0:
            raise ValueError(f"query {key} violates the target-present eligibility rule")

        query_rows: list[dict[str, str]] = []
        for candidate in groups[key]:
            matrix = np.asarray(
                [float(candidate[f"transform_{r}{c}"]) for r in range(4) for c in range(4)],
                dtype=np.float64,
            ).reshape(4, 4)
            transformed = source_points @ matrix[:3, :3].T + matrix[:3, 3]
            distances, nearest = target_tree.query(transformed, k=1, workers=1)
            predicted = np.unique(nearest[distances <= args.max_distance])
            predicted_count = len(predicted)
            true_positive = int(np.count_nonzero(target_mask[predicted]))
            false_positive = predicted_count - true_positive
            false_negative = target_count - true_positive
            union = true_positive + false_positive + false_negative
            row = {
                **candidate,
                "source_points": str(len(source_points)),
                "target_instance_points": str(target_count),
                "predicted_target_points": str(predicted_count),
                "instance_iou": repr(true_positive / union if union else 0.0),
                "instance_precision": repr(true_positive / predicted_count if predicted_count else 0.0),
                "instance_recall": repr(true_positive / target_count if target_count else 0.0),
            }
            query_rows.append(row)
        if len(query_rows) != 5:
            raise ValueError(f"candidate group {key} has {len(query_rows)} proposals")
        score_rows.extend(query_rows)

        next_key = keys[index + 1] if index + 1 < len(keys) else None
        if next_key is None or next_key[:2] != key[:2]:
            write_csv(partial, score_rows)
            print(f"checkpointed {target_scan}; completed_queries={len(score_rows) // 5}", flush=True)

    if len(score_rows) != len(groups) * 5:
        raise ValueError(f"incomplete proposal scores: expected {len(groups) * 5}, got {len(score_rows)}")
    output_csv = args.output_dir / "all_proposal_transfer.csv"
    write_csv(output_csv, score_rows)
    partial.unlink(missing_ok=True)

    scored_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        scored_groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)
    query_keys = sorted(scored_groups)
    scene_ids = [key[0] for key in query_keys]
    scene_order = sorted(set(scene_ids))
    chosen = {
        arm: [min(scored_groups[key], key=lambda row: rank_key(row, arm)) for key in query_keys]
        for arm in ARMS
    }
    per_query: list[dict[str, object]] = []
    for i, key in enumerate(query_keys):
        rows = scored_groups[key]
        output: dict[str, object] = {"scene": key[0], "target_scan": key[1], "source_instance_id": key[2]}
        for arm in ARMS:
            row = chosen[arm][i]
            output[f"{arm}_repeat"] = int(row["repeat"])
            for metric in METRICS:
                output[f"{arm}_{metric}"] = float(row[metric])
        for metric in METRICS:
            output[f"uniform_{metric}"] = float(np.mean([float(row[metric]) for row in rows]))
        per_query.append(output)
    write_csv(args.output_dir / "all_proposal_transfer_selected.csv", per_query)

    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "scenes": len(scene_order),
        "target_scans": len({key[:2] for key in query_keys}),
        "object_target_queries": len(query_keys),
        "pose_proposals_per_query": 5,
        "max_distance_m": args.max_distance,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "eligibility": "target annotations contain the source raw instance ID; conditional on target-present queries",
        "prediction": "transform all source-instance points; unique nearest target points within max distance receive the source ID",
        "target_labels_used_for_ranking": False,
        "target_labels_used_for_eligibility": True,
        "target_labels_used_for_scoring": True,
        "uniform_proposal_baseline": "per-query mean across the five fixed candidate pose outcomes; expected value of uniform candidate selection",
        "arms": {},
        "comparisons": {},
    }
    arrays = {
        arm: {
            metric: np.asarray([float(row[f"{arm}_{metric}"]) for row in per_query])
            for metric in METRICS
        }
        for arm in ARMS
    }
    uniform = {
        metric: np.asarray([float(row[f"uniform_{metric}"]) for row in per_query])
        for metric in METRICS
    }
    for arm in ARMS:
        summary["arms"][arm] = {
            metric: float(np.mean(arrays[arm][metric])) for metric in METRICS
        }
    for metric in METRICS:
        summary["comparisons"][metric] = {}
        pairs = [("spatial", "fitness"), ("color", "spatial"), ("color", "fitness"), ("color", "uniform")]
        for left, right in pairs:
            delta = arrays[left][metric] - (uniform[metric] if right == "uniform" else arrays[right][metric])
            summary["comparisons"][metric][f"{left}_minus_{right}"] = {
                "object_weighted_delta": float(np.mean(delta)),
                "scene_cluster_bootstrap_95pct": scene_bootstrap(
                    delta, scene_ids, scene_order, args.bootstrap_samples, rng
                ),
            }
    (args.output_dir / "all_proposal_transfer_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Rescan all-proposal point-level instance transfer",
        "",
        f"Evaluated all five frozen FPFH transforms for {len(query_keys)} target-present source-instance queries across {len(scene_order)} scenes and {summary['target_scans']} target scans. Predictions transfer the source ID to unique nearest target points within {args.max_distance:.2f} m; target labels are used only for eligibility and scoring.",
        "",
        "| Selection rule | Mean instance IoU | Precision | Recall |",
        "|---|---:|---:|---:|",
    ]
    for arm, label in (("fitness", "FPFH fitness"), ("spatial", "Spatial support"), ("color", "Spatial + RGB")):
        score = summary["arms"][arm]
        lines.append(f"| {label} | {score['instance_iou']:.3f} | {score['instance_precision']:.3f} | {score['instance_recall']:.3f} |")
    lines.append(f"| Uniform proposal expectation | {float(uniform['instance_iou'].mean()):.3f} | {float(uniform['instance_precision'].mean()):.3f} | {float(uniform['instance_recall'].mean()):.3f} |")
    lines.extend(["", "## Paired scene-cluster 95% intervals for mean differences", "", "| Metric | Comparison | Delta | 95% CI |", "|---|---|---:|---:|"])
    for metric in METRICS:
        for comparison, value in summary["comparisons"][metric].items():
            lo, hi = value["scene_cluster_bootstrap_95pct"]
            lines.append(f"| {metric} | {comparison.replace('_minus_', ' − ')} | {value['object_weighted_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines.extend([
        "",
        "The uniform-proposal row is the per-query mean outcome over the five saved proposals; it is the expected score if proposal choice is uniform. This is a per-query transfer diagnostic, not globally conflict-resolved full-scene mIoU. Results are conditional on the annotation-defined target-present query set.",
    ])
    (args.output_dir / "all_proposal_transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
