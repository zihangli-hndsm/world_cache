#!/usr/bin/env python3
"""Object-identity-cluster bootstrap for the bounded HILO view-transfer pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/reports/20261009_hilo_pose_projection/hash8_identity_pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/20261009_hilo_pose_projection/identity_cluster_bootstrap.json"))
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    rows = source.get("query_outcomes")
    if not rows:
        raise ValueError("input lacks per-query outcomes; rerun the frozen scoring script")
    ids = sorted({row["object_id"] for row in rows})
    captures = sorted({row["capture"] for row in rows})
    if len(ids) < 2 or len(captures) != 3:
        raise ValueError(f"expected multiple object IDs and three query captures; got {len(ids)} IDs, {captures}")
    scenes = sorted({row["scene"] for row in rows})
    by_id: dict[str, list[dict]] = {oid: [] for oid in ids}
    for row in rows:
        by_id[row["object_id"]].append(row)

    metrics = ["dino_top1", "rgb_top1", "top1_difference", "same_arc_dino", "cross_arc_dino",
               "cross_minus_same_dino", "dino_mutual_precision", "dino_mutual_coverage",
               "rgb_mutual_precision", "rgb_mutual_coverage"]
    per_id = {name: [] for name in metrics}
    for oid in ids:
        part = by_id[oid]
        dino = np.asarray([r["dino_top1_correct"] for r in part], dtype=float)
        rgb = np.asarray([r["rgb_histogram_top1_correct"] for r in part], dtype=float)
        same = np.asarray([r["dino_top1_correct"] for r in part if r["capture"] == "arc0_image12"], dtype=float)
        cross = np.asarray([r["dino_top1_correct"] for r in part if r["capture"] in {"arc4_image7", "arc4_image12"}], dtype=float)
        dino_acc = float(dino.mean())
        rgb_acc = float(rgb.mean())
        same_acc = float(same.mean()) if same.size else np.nan
        cross_acc = float(cross.mean()) if cross.size else np.nan
        dacc = [r for r in part if r["dino_mutual_accepted"]]
        racc = [r for r in part if r["rgb_histogram_mutual_accepted"]]
        per_id["dino_top1"].append(dino_acc)
        per_id["rgb_top1"].append(rgb_acc)
        per_id["top1_difference"].append(dino_acc - rgb_acc)
        per_id["same_arc_dino"].append(same_acc)
        per_id["cross_arc_dino"].append(cross_acc)
        per_id["cross_minus_same_dino"].append(cross_acc - same_acc)
        per_id["dino_mutual_precision"].append(float(np.mean([r["dino_mutual_correct"] for r in dacc])) if dacc else np.nan)
        per_id["dino_mutual_coverage"].append(float(np.mean([r["dino_mutual_accepted"] for r in part])))
        per_id["rgb_mutual_precision"].append(float(np.mean([r["rgb_histogram_mutual_correct"] for r in racc])) if racc else np.nan)
        per_id["rgb_mutual_coverage"].append(float(np.mean([r["rgb_histogram_mutual_accepted"] for r in part])))

    # A separate configuration-cluster bootstrap captures variability across
    # the sampled HILO scene configurations. It does not imply site-level
    # independence; all scenes share the same controlled capture setup.
    by_scene: dict[str, list[dict]] = {scene: [] for scene in scenes}
    for row in rows:
        by_scene[row["scene"]].append(row)
    scene_values = {name: [] for name in metrics}
    for scene in scenes:
        part = by_scene[scene]
        dino = np.asarray([r["dino_top1_correct"] for r in part], dtype=float)
        rgb = np.asarray([r["rgb_histogram_top1_correct"] for r in part], dtype=float)
        same = np.asarray([r["dino_top1_correct"] for r in part if r["capture"] == "arc0_image12"], dtype=float)
        cross = np.asarray([r["dino_top1_correct"] for r in part if r["capture"] in {"arc4_image7", "arc4_image12"}], dtype=float)
        dacc = [r for r in part if r["dino_mutual_accepted"]]
        racc = [r for r in part if r["rgb_histogram_mutual_accepted"]]
        scene_values["dino_top1"].append(float(dino.mean()))
        scene_values["rgb_top1"].append(float(rgb.mean()))
        scene_values["top1_difference"].append(float((dino - rgb).mean()))
        scene_values["same_arc_dino"].append(float(same.mean()) if same.size else np.nan)
        scene_values["cross_arc_dino"].append(float(cross.mean()) if cross.size else np.nan)
        scene_values["cross_minus_same_dino"].append(float(cross.mean() - same.mean()) if cross.size and same.size else np.nan)
        scene_values["dino_mutual_precision"].append(float(np.mean([r["dino_mutual_correct"] for r in dacc])) if dacc else np.nan)
        scene_values["dino_mutual_coverage"].append(float(np.mean([r["dino_mutual_accepted"] for r in part])))
        scene_values["rgb_mutual_precision"].append(float(np.mean([r["rgb_histogram_mutual_correct"] for r in racc])) if racc else np.nan)
        scene_values["rgb_mutual_coverage"].append(float(np.mean([r["rgb_histogram_mutual_accepted"] for r in part])))

    rng = np.random.default_rng(args.seed)
    draw_ids = rng.integers(0, len(ids), size=(args.draws, len(ids)))
    results = {}
    scene_rng = np.random.default_rng(args.seed + 1)
    scene_draws = scene_rng.integers(0, len(scenes), size=(args.draws, len(scenes)))
    for name in metrics:
        values = np.asarray(per_id[name], dtype=float)
        valid = np.isfinite(values)
        observed = float(values[valid].mean())
        draws = np.empty(args.draws, dtype=float)
        # A draw resamples the unique object identities, then averages their
        # within-identity outcomes. IDs without accepted matches are omitted
        # only for precision, as precision is undefined for them.
        for i, chosen in enumerate(draw_ids):
            sample = values[chosen]
            sample = sample[np.isfinite(sample)]
            draws[i] = sample.mean() if sample.size else np.nan
        draws = draws[np.isfinite(draws)]
        results[name] = {
            "identity_macro_estimate": observed,
            "identity_cluster_bootstrap_95pct_percentile_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            "identities_in_estimate": int(valid.sum()),
            "unique_identities": len(ids),
            "configuration_macro_estimate": float(np.nanmean(scene_values[name])),
            "configuration_cluster_bootstrap_95pct_percentile_interval": np.quantile(
                np.asarray([np.nanmean(np.asarray(scene_values[name])[chosen]) for chosen in scene_draws]),
                [0.025, 0.975],
            ).tolist(),
            "configurations_in_estimate": len(scenes),
        }

    summary = {
        "dataset": "HILO",
        "input": str(args.input),
        "protocol": "separate nonparametric cluster bootstraps over unique persistent object model IDs and over sampled scene configurations; each cluster contributes its mean across observed views",
        "draws": args.draws,
        "seed": args.seed,
        "objects": len(ids),
        "configurations": len(scenes),
        "query_rows": len(rows),
        "captures": captures,
        "results": results,
        "limitations": [
            "identity-cluster intervals quantify object-identity sampling conditional on the sampled configurations, camera calibrations, and fixed reference/query views",
            "configuration-cluster intervals quantify only the sampled HILO capture configurations; all share one controlled robot/turntable capture setup and do not establish independent site generalization",
            "masks use rendered mesh poses and depth agreement; this is not room-rescan or independent RGB-D site validation",
            "same-arc versus cross-arc also changes camera calibration and is descriptive, not a causal viewpoint effect",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
