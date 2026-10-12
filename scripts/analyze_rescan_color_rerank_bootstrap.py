#!/usr/bin/env python3
"""Scene-cluster paired bootstrap for the Rescan color reranker."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIRS = {
    "exact_recall_at_1": ("color_exact_recall_at_1", "fitness_exact_recall_at_1"),
    "exact_recall_at_5": ("color_exact_recall_at_5", "fitness_exact_recall_at_5"),
    "same_class_recall_at_1": ("color_same_class_recall_at_1", "fitness_same_class_recall_at_1"),
    "same_class_recall_at_5": ("color_same_class_recall_at_5", "fitness_same_class_recall_at_5"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_012_026)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    with args.input_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no object rows in {args.input_csv}")
    scenes = sorted({row["scene"] for row in rows})
    grouped = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    rng = np.random.default_rng(args.seed)
    result: dict[str, object] = {
        "input_csv": str(args.input_csv),
        "scenes": len(scenes),
        "object_target_pairs": len(rows),
        "bootstrap_unit": "scene, paired color and fitness rows retained together",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "interval": "percentile 95% confidence interval",
        "scoring_scope": "eligible pairs require the source raw ID to be present in target annotations; input recall flags measure centroid localization within 0.20 m, not full 6-DoF pose accuracy",
        "metrics": {},
    }
    for metric, (color_col, fitness_col) in PAIRS.items():
        scene_values: dict[str, tuple[float, float, int]] = {}
        for scene in scenes:
            scene_rows = grouped[scene]
            color_hits = np.asarray([row[color_col].lower() == "true" for row in scene_rows], dtype=float)
            fitness_hits = np.asarray([row[fitness_col].lower() == "true" for row in scene_rows], dtype=float)
            scene_values[scene] = (float(color_hits.sum()), float(fitness_hits.sum()), len(scene_rows))

        color_mean = sum(scene_values[s][0] for s in scenes) / len(rows)
        fitness_mean = sum(scene_values[s][1] for s in scenes) / len(rows)
        scene_macro_delta = float(np.mean([
            (scene_values[s][0] - scene_values[s][1]) / scene_values[s][2] for s in scenes
        ]))
        bootstrap_delta = np.empty(args.bootstrap_samples, dtype=float)
        for i in range(args.bootstrap_samples):
            sampled = rng.integers(0, len(scenes), size=len(scenes))
            numerator = sum(
                scene_values[scenes[j]][0] - scene_values[scenes[j]][1] for j in sampled
            )
            denominator = sum(scene_values[scenes[j]][2] for j in sampled)
            bootstrap_delta[i] = numerator / denominator
        lower, upper = np.quantile(bootstrap_delta, [0.025, 0.975])
        result["metrics"][metric] = {
            "fitness_object_weighted_mean": float(fitness_mean),
            "color_object_weighted_mean": float(color_mean),
            "paired_object_weighted_delta": float(color_mean - fitness_mean),
            "scene_macro_paired_delta": scene_macro_delta,
            "scene_cluster_bootstrap_95pct_delta": [float(lower), float(upper)],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        lines = [
            "# Rescan color reranker paired uncertainty",
            "",
            f"Paired scene-cluster percentile bootstrap with {args.bootstrap_samples:,} draws over {len(scenes)} scenes and {len(rows)} object-target pairs; seed {args.seed}.",
            "Pairs are conditional on the source raw ID being present in target annotations; recall flags measure centroid localization within 0.20 m, not full 6-DoF pose accuracy.",
            "These deltas compare the combined spatial+RGB rank against FPFH fitness and do not isolate the RGB contribution; the spatial-only control and color-vs-spatial intervals are in `rerank_ablation_summary.md`.",
            "",
            "| Metric | Fitness | Color | Paired delta | Scene-macro delta | Scene-cluster 95% CI for delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for metric, values in result["metrics"].items():
            lo, hi = values["scene_cluster_bootstrap_95pct_delta"]
            lines.append(
                f"| {metric} | {values['fitness_object_weighted_mean']:.3f} | "
                f"{values['color_object_weighted_mean']:.3f} | "
                f"{values['paired_object_weighted_delta']:+.3f} | "
                f"{values['scene_macro_paired_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
            )
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
