#!/usr/bin/env python3
"""Compare fitness, spatial-only, and spatial+RGB ranks on saved proposals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "exact_recall_at_1": "exact_instance_hit",
    "exact_recall_at_5": "exact_instance_hit",
    "same_class_recall_at_1": "same_class_hit",
    "same_class_recall_at_5": "same_class_hit",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--color-object-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_022_026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    with args.color_object_csv.open(newline="", encoding="utf-8") as stream:
        color_objects = list(csv.DictReader(stream))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)
    if not groups:
        raise ValueError("candidate CSV is empty")
    if any(len(rows) != 5 for rows in groups.values()):
        raise ValueError("expected exactly five proposals per object-target pair")

    object_rows: list[dict[str, object]] = []
    for key in sorted(groups):
        rows = groups[key]
        fitness = sorted(rows, key=lambda r: (-float(r["fitness"]), int(r["repeat"])))
        spatial = sorted(
            rows,
            key=lambda r: (
                -float(r["spatial_coverage"]),
                -float(r["fitness"]),
                int(r["repeat"]),
            ),
        )
        color = sorted(
            rows,
            key=lambda r: (
                -float(r["joint_spatial_rgb_inlier_fraction"]),
                -float(r["spatial_coverage"]),
                float(r["median_rgb_distance_on_spatial_inliers"]),
                -float(r["fitness"]),
                int(r["repeat"]),
            ),
        )
        result: dict[str, object] = {
            "scene": key[0],
            "target_scan": key[1],
            "source_instance_id": key[2],
            "candidate_count": len(rows),
        }
        for arm, ranked in (("fitness", fitness), ("spatial", spatial), ("color", color)):
            for k in (1, 5):
                result[f"{arm}_exact_recall_at_{k}"] = any(
                    row["exact_instance_hit"].lower() == "true" for row in ranked[:k]
                )
                result[f"{arm}_same_class_recall_at_{k}"] = any(
                    row["same_class_hit"].lower() == "true" for row in ranked[:k]
                )
        object_rows.append(result)

    color_lookup = {
        (r["scene"], r["target_scan"], str(r["source_instance_id"])): r
        for r in color_objects
    }
    if set(color_lookup) != set(groups):
        raise ValueError("color object CSV and candidate CSV have different pair keys")
    for row in object_rows:
        key = (row["scene"], row["target_scan"], row["source_instance_id"])
        saved = color_lookup[key]
        for metric, hit_col in METRICS.items():
            k = metric.rsplit("_", 1)[-1]
            prefix = "exact" if hit_col == "exact_instance_hit" else "same_class"
            saved_col = f"color_{prefix}_recall_at_{k}"
            if row[f"color_{metric}"] != (saved[saved_col].lower() == "true"):
                raise ValueError(f"recomputed color ranking disagrees with saved output for {key}")

    csv_path = args.output_dir / "rerank_ablation_objects.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(object_rows[0]))
        writer.writeheader()
        writer.writerows(object_rows)

    scenes = sorted({r["scene"] for r in object_rows})
    by_scene = {scene: [r for r in object_rows if r["scene"] == scene] for scene in scenes}
    rng = np.random.default_rng(args.seed)
    arms = ("fitness", "spatial", "color")
    metrics = (
        "exact_recall_at_1", "exact_recall_at_5",
        "same_class_recall_at_1", "same_class_recall_at_5",
    )
    summary: dict[str, object] = {
        "scenes": len(scenes),
        "object_target_pairs": len(object_rows),
        "candidate_count": len(candidates),
        "bootstrap_unit": "scene; paired rankings are kept together within each sampled scene",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "interval": "percentile 95% confidence interval",
        "scoring_scope": "pairs are conditional on the source raw ID being present in target annotations; input exact/same-class flags use a 0.20 m centroid-localization threshold, not full 6-DoF pose accuracy",
        "metrics": {},
    }
    for metric in metrics:
        values = {
            arm: float(np.mean([bool(r[f"{arm}_{metric}"]) for r in object_rows]))
            for arm in arms
        }
        pairs: dict[str, list[float]] = {}
        for left, right in (("color", "fitness"), ("spatial", "fitness"), ("color", "spatial")):
            scene_delta = {
                scene: float(np.mean([
                    bool(r[f"{left}_{metric}"]) - bool(r[f"{right}_{metric}"])
                    for r in by_scene[scene]
                ]))
                for scene in scenes
            }
            macro_delta = float(np.mean(list(scene_delta.values())))
            weighted_delta = float(np.mean([
                bool(r[f"{left}_{metric}"]) - bool(r[f"{right}_{metric}"])
                for r in object_rows
            ]))
            boot = np.empty(args.bootstrap_samples, dtype=float)
            for i in range(args.bootstrap_samples):
                sampled = rng.integers(0, len(scenes), size=len(scenes))
                numerator = sum(
                    sum(
                        bool(r[f"{left}_{metric}"]) - bool(r[f"{right}_{metric}"])
                        for r in by_scene[scenes[j]]
                    )
                    for j in sampled
                )
                denominator = sum(len(by_scene[scenes[j]]) for j in sampled)
                boot[i] = numerator / denominator
            lo, hi = np.quantile(boot, [0.025, 0.975])
            pairs[f"{left}_minus_{right}"] = [weighted_delta, macro_delta, float(lo), float(hi)]
        summary["metrics"][metric] = {"object_weighted_recall": values, "paired_deltas": pairs}

    json_path = args.output_dir / "rerank_ablation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rescan FPFH reranking ablation",
        "",
        f"Same {len(object_rows)} object-target pairs / five proposals each / {len(scenes)} held-out scenes. Pairs are conditional on the source raw ID being present in target annotations. `spatial` ranks by the fraction of source points within 0.12 m of the unsegmented target, with FPFH fitness as tie-break; `color` adds the frozen RGB agreement criterion (0.15 L2). Hits use the input 0.20 m centroid-localization flags, not full 6-DoF pose accuracy. Scene-cluster paired percentile bootstrap: {args.bootstrap_samples:,} draws, seed {args.seed}.",
        "",
        "| Metric | Fitness | Spatial-only | Spatial+RGB | Spatial vs fitness delta (95% CI) | RGB vs fitness delta (95% CI) | RGB vs spatial delta (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        m = summary["metrics"][metric]
        sf = m["paired_deltas"]["spatial_minus_fitness"]
        cf = m["paired_deltas"]["color_minus_fitness"]
        cs = m["paired_deltas"]["color_minus_spatial"]
        v = m["object_weighted_recall"]
        lines.append(
            f"| {metric} | {v['fitness']:.3f} | {v['spatial']:.3f} | {v['color']:.3f} | "
            f"{sf[0]:+.3f} [{sf[2]:+.3f}, {sf[3]:+.3f}] | "
            f"{cf[0]:+.3f} [{cf[2]:+.3f}, {cf[3]:+.3f}] | "
            f"{cs[0]:+.3f} [{cs[2]:+.3f}, {cs[3]:+.3f}] |"
        )
    (args.output_dir / "rerank_ablation_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
