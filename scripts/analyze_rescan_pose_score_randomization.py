#!/usr/bin/env python3
"""Test whether frozen Rescan proposal scores select better poses than chance."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def ranked(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    if arm == "fitness":
        key = lambda row: (-float(row["fitness"]), int(row["repeat"]))
    elif arm == "spatial":
        key = lambda row: (-float(row["spatial_coverage"]), -float(row["fitness"]), int(row["repeat"]))
    elif arm == "color":
        key = lambda row: (
            -float(row["joint_spatial_rgb_inlier_fraction"]),
            -float(row["spatial_coverage"]),
            float(row["median_rgb_distance_on_spatial_inliers"]),
            -float(row["fitness"]),
            int(row["repeat"]),
        )
    else:
        raise ValueError(arm)
    return min(rows, key=key)


def cluster_interval(
    per_group_delta: np.ndarray,
    scenes: list[str],
    scene_order: list[str],
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    indices = {scene: np.flatnonzero(np.asarray(scenes) == scene) for scene in scene_order}
    boot = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        draw = rng.integers(0, len(scene_order), size=len(scene_order))
        selected = np.concatenate([indices[scene_order[index]] for index in draw])
        boot[sample] = float(np.mean(per_group_delta[selected]))
    return [float(value) for value in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_032_026)
    args = parser.parse_args()
    if args.permutations < 1 or args.bootstrap_samples < 1:
        raise ValueError("permutation and bootstrap counts must be positive")

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)
    if not groups or any(len(group) != 5 for group in groups.values()):
        raise ValueError("expected five FPFH proposals for every eligible object-target query")

    keys = sorted(groups)
    scenes = [key[0] for key in keys]
    scene_order = sorted(set(scenes))
    selected = {arm: [ranked(groups[key], arm) for key in keys] for arm in ("fitness", "spatial", "color")}
    metrics = ("exact_instance_hit", "same_class_hit")
    actual = {
        arm: {
            metric: np.asarray([row[metric].lower() == "true" for row in selected[arm]], dtype=np.float64)
            for metric in metrics
        }
        for arm in selected
    }
    candidate_hits = {
        metric: np.asarray([
            [row[metric].lower() == "true" for row in groups[key]] for key in keys
        ], dtype=np.float64)
        for metric in metrics
    }
    null_expected = {metric: candidate_hits[metric].mean(axis=1) for metric in metrics}

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, 5, size=(args.permutations, len(keys)))
    summary: dict[str, object] = {
        "groups": len(keys),
        "scenes": len(scene_order),
        "proposals_per_group": 5,
        "permutations": args.permutations,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "eligibility": "source raw instance ID annotated in the target; conditional on eligible queries",
        "scoring": "centroid localization flags saved with each pose proposal, not full 6-DoF pose accuracy",
        "prediction_labels_used": False,
        "null": "uniformly select one of each query's five proposals after breaking the association between frozen ranking scores and proposal pose",
        "arms": {},
        "score_randomization": {},
    }
    for metric in metrics:
        null = candidate_hits[metric][np.arange(len(keys))[None, :], draws].mean(axis=1)
        color_value = float(actual["color"][metric].mean())
        null_value = float(null_expected[metric].mean())
        summary["score_randomization"][metric] = {
            "color_top1": color_value,
            "uniform_proposal_expectation": null_value,
            "color_minus_uniform": color_value - null_value,
            "randomization_null_95pct": [float(v) for v in np.quantile(null, [0.025, 0.975])],
            "upper_tail_p": float((1 + np.count_nonzero(null >= color_value)) / (args.permutations + 1)),
            "cluster_bootstrap_95pct_for_color_minus_uniform": cluster_interval(
                actual["color"][metric] - null_expected[metric], scenes, scene_order,
                args.bootstrap_samples, rng,
            ),
        }
        summary["arms"][metric] = {
            arm: float(actual[arm][metric].mean()) for arm in ("fitness", "spatial", "color")
        }
        for arm in ("fitness", "spatial"):
            summary["arms"][metric][f"color_minus_{arm}"] = float(
                np.mean(actual["color"][metric] - actual[arm][metric])
            )
            summary["arms"][metric][f"color_minus_{arm}_cluster_bootstrap_95pct"] = cluster_interval(
                actual["color"][metric] - actual[arm][metric], scenes, scene_order,
                args.bootstrap_samples, rng,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pose_score_randomization.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Rescan proposal-ranker score randomization",
        "",
        f"On {len(keys)} target-present object/scan queries in {len(scene_order)} scenes, each frozen FPFH proposal set contains five poses. We compare top-1 centroid-localization hits under fitness, spatial, and spatial+RGB ranking. As a conditional null, we break the association between the frozen score and pose by selecting one of the five proposals uniformly per query. The target labels define eligibility and score only; they do not rank poses.",
        "",
        "| Metric | Fitness | Spatial | Spatial+RGB | RGB − spatial (95% scene CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, label in (("exact_instance_hit", "Exact-instance centroid hit@1"), ("same_class_hit", "Same-class centroid hit@1")):
        row = summary["arms"][metric]
        ci = row["color_minus_spatial_cluster_bootstrap_95pct"]
        lines.append(
            f"| {label} | {row['fitness']:.3f} | {row['spatial']:.3f} | {row['color']:.3f} | {row['color_minus_spatial']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    lines.extend(["", "## Score-to-pose randomization", ""])
    for metric, label in (("exact_instance_hit", "Exact-instance"), ("same_class_hit", "Same-class")):
        row = summary["score_randomization"][metric]
        ci = row["cluster_bootstrap_95pct_for_color_minus_uniform"]
        null_ci = row["randomization_null_95pct"]
        lines.append(
            f"- {label}: RGB top-1={row['color_top1']:.3f}; uniform-proposal expectation={row['uniform_proposal_expectation']:.3f}; difference={row['color_minus_uniform']:+.3f}, scene-bootstrap 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]. The null's 95% range is [{null_ci[0]:.3f}, {null_ci[1]:.3f}] and its one-sided randomization tail area is {row['upper_tail_p']:.5f}."
        )
    lines.extend([
        "",
        "The localization flag is a centroid threshold diagnostic, not full 6-DoF pose accuracy. Results are conditional on an annotated target instance and do not establish an improvement over static XYZ transfer. Thresholds were frozen on scene_a before B–M evaluation.",
    ])
    (args.output_dir / "pose_score_randomization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
