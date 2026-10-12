#!/usr/bin/env python3
"""Scene-cluster bootstrap intervals for Rescan instance-centroid recall."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "exact_instance_recall_at_1",
    "exact_instance_recall_at_5",
    "same_class_recall_at_1",
    "same_class_recall_at_5",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7_102_026)
    parser.add_argument("--centroid-threshold", type=float, default=0.20)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if args.centroid_threshold <= 0:
        raise ValueError("--centroid-threshold must be positive")

    with args.input_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no object rows in {args.input_csv}")
    scenes = sorted({row["scene"] for row in rows})
    grouped = {
        scene: {
            metric: np.asarray(
                [row[metric].lower() == "true" for row in rows if row["scene"] == scene],
                dtype=np.float64,
            )
            for metric in METRICS
        }
        for scene in scenes
    }
    rng = np.random.default_rng(args.seed)
    result: dict[str, object] = {
        "input_csv": str(args.input_csv),
        "scenes": len(scenes),
        "object_target_pairs": len(rows),
        "bootstrap_unit": "scene, with all object-target pairs in a sampled scene",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "centroid_distance_threshold_m": args.centroid_threshold,
        "pair_eligibility": "source raw instance ID must be present in target annotations; estimates are conditional on annotated target presence",
        "interval": "percentile 95% confidence interval",
        "metrics": {},
    }
    for metric in METRICS:
        scene_means = [grouped[scene][metric].mean() for scene in scenes]
        pair_weighted = float(np.mean([row[metric].lower() == "true" for row in rows]))
        bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
        for sample_index in range(args.bootstrap_samples):
            sampled_scenes = rng.integers(0, len(scenes), size=len(scenes))
            positives = sum(grouped[scenes[index]][metric].sum() for index in sampled_scenes)
            denominator = sum(len(grouped[scenes[index]][metric]) for index in sampled_scenes)
            bootstrap[sample_index] = positives / denominator
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        result["metrics"][metric] = {
            "object_target_weighted_mean": pair_weighted,
            "scene_macro_mean": float(np.mean(scene_means)),
            "scene_cluster_bootstrap_95pct": [float(lower), float(upper)],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        lines = [
            "# Rescan FPFH instance-centroid-recall uncertainty",
            "",
            f"Scene-cluster percentile bootstrap with {args.bootstrap_samples:,} draws over {len(scenes)} scenes and {len(rows)} object-target pairs; seed {args.seed}.",
            "Pairs are conditional on the source raw ID being present in target annotations.",
            f"Positive hits are based on transformed source-centroid distance to an annotated target-instance centroid ({args.centroid_threshold:.2f} m in this run); this is not full 6-DoF pose accuracy.",
            "",
            "| Metric | Object-target weighted mean | Scene macro mean | Scene-cluster 95% CI |",
            "|---|---:|---:|---:|",
        ]
        for metric in METRICS:
            values = result["metrics"][metric]
            lo, hi = values["scene_cluster_bootstrap_95pct"]
            lines.append(
                f"| {metric} | {values['object_target_weighted_mean']:.3f} | "
                f"{values['scene_macro_mean']:.3f} | [{lo:.3f}, {hi:.3f}] |"
            )
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
