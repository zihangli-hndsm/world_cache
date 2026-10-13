#!/usr/bin/env python3
"""Nested relation bootstrap that repeats the source-LOPO selector fit.

The outer bootstrap resamples eligible held-out relations. For every sampled
test relation, an inner bootstrap resamples the other source relations with
replacement, selects the mode/margin/support configuration on those training
relations, and scores it on the untouched test relation. Thus both relation
sampling and selector-fit variation enter the reported interval.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_3rscan_multimotion_rescue_mechanisms import (
    MARGINS_M,
    MIN_SUPPORTS,
    MODE_COUNTS,
    predict,
    read_pair,
)
from analyze_pose_locked_multimotion_extended_source import environment_groups


SEEDS = (17, 29, 41)


def choose_config_index(scores: np.ndarray, configs: list[tuple[int, float, int]]) -> int:
    best = int(scores.max())
    return min(
        (i for i, score in enumerate(scores) if int(score) == best),
        key=lambda i: (configs[i][0], -configs[i][2], -configs[i][1]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261005)
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or any(seed not in SEEDS for seed in args.seeds):
        raise ValueError("positive bootstrap count and seeds from 17/29/41 are required")

    pair_dirs: dict[str, Path] = {}
    for root in args.input_dir:
        for path in sorted((root / "descriptor_candidates").glob("*.npz")):
            pair = path.stem
            if pair == "target":
                continue
            if pair in pair_dirs:
                raise ValueError(f"duplicate relation {pair}")
            pair_dirs[pair] = root
    if len(pair_dirs) != 18:
        raise ValueError(f"expected 18 source relations, found {len(pair_dirs)}")
    meta = environment_groups(args.input_dir, args.metadata)
    if len({meta[pair]["environment_id"] for pair in pair_dirs}) != 18:
        raise ValueError("source relations are not environment-disjoint")

    configs = list(itertools.product(MODE_COUNTS, MARGINS_M, MIN_SUPPORTS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    fold_records: list[dict[str, object]] = []
    bootstrap_records: list[dict[str, object]] = []

    for seed in args.seeds:
        names = sorted(pair_dirs)
        data = {
            pair: read_pair(pair_dirs[pair], pair, args.annotation_root, seed=seed)
            for pair in names
        }
        # Rows are relations; columns are fixed selector configurations.
        method_hits = np.zeros((len(names), len(configs)), dtype=np.int64)
        baseline_hits = np.zeros(len(names), dtype=np.int64)
        query_counts = np.zeros(len(names), dtype=np.int64)
        for j, pair in enumerate(names):
            item = data[pair]
            moved = np.asarray(item["moved"], dtype=bool)
            query_counts[j] = int(moved.sum())
            baseline_correct = np.asarray(item["baseline_correct"], dtype=bool)
            baseline_hits[j] = int(baseline_correct.sum())
            for k, config in enumerate(configs):
                correct = predict(item, *config)[0]
                method_hits[j, k] = int(correct.sum())
        eligible = np.flatnonzero(query_counts > 0)
        if len(eligible) != 14 or int(query_counts.sum()) != 1685:
            raise ValueError(
                f"expected 14 eligible relations / 1,685 queries, got "
                f"{len(eligible)} / {int(query_counts.sum())}"
            )

        fold_results: list[tuple[int, int, int, int]] = []
        config_frequency = np.zeros(len(configs), dtype=np.int64)
        for test_idx in range(len(names)):
            train_idx = np.asarray([j for j in range(len(names)) if j != test_idx], dtype=np.int64)
            pooled_scores = method_hits[train_idx].sum(axis=0)
            selected = choose_config_index(pooled_scores, configs)
            delta = int(method_hits[test_idx, selected] - baseline_hits[test_idx])
            fold_results.append((test_idx, selected, delta, int(query_counts[test_idx])))
            config_frequency[selected] += 1
            fold_records.append({
                "seed": seed, "heldout_relation": names[test_idx],
                "environment_id": meta[names[test_idx]]["environment_id"],
                "queries": int(query_counts[test_idx]),
                "baseline_correct": int(baseline_hits[test_idx]),
                "method_correct": int(method_hits[test_idx, selected]),
                "delta": delta,
                "modes": configs[selected][0], "margin_m": configs[selected][1],
                "min_support": configs[selected][2],
            })

        rng = np.random.default_rng(args.bootstrap_seed + seed)
        outer_test_draws = rng.choice(eligible, size=(args.bootstrap_samples, len(eligible)), replace=True)
        macro_draws = np.empty(args.bootstrap_samples, dtype=np.float64)
        for b in range(args.bootstrap_samples):
            test_deltas: list[float] = []
            for test_idx in outer_test_draws[b]:
                train_idx = np.asarray([j for j in range(len(names)) if j != test_idx], dtype=np.int64)
                # Relation bootstrap weights preserve the 17 distinct source
                # environments eligible to train this fold.
                train_counts = rng.multinomial(len(train_idx), np.full(len(train_idx), 1 / len(train_idx)))
                train_scores = train_counts @ method_hits[train_idx]
                selected = choose_config_index(train_scores, configs)
                n = int(query_counts[test_idx])
                if n <= 0:
                    raise AssertionError("outer held-out draw included an ineligible relation")
                delta = int(method_hits[test_idx, selected] - baseline_hits[test_idx])
                test_deltas.append(delta / n)
            macro_draws[b] = float(np.mean(test_deltas))

        fixed_deltas = np.asarray([
            delta / n for test_idx, selected, delta, n in fold_results if n > 0
        ], dtype=np.float64)
        fixed_macro = float(fixed_deltas.mean())
        lo, hi = np.quantile(macro_draws, [0.025, 0.975])
        summaries.append({
            "seed": seed,
            "source_relations": len(names),
            "eligible_relations": len(eligible),
            "queries": int(query_counts.sum()),
            "configs": len(configs),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed + seed,
            "fixed_lopo_relation_macro_delta": fixed_macro,
            "nested_relation_selector_bootstrap_macro_ci95": [float(lo), float(hi)],
            "nested_bootstrap_probability_macro_delta_le_zero": float(np.mean(macro_draws <= 0)),
            "fixed_lopo_all_eligible_relations_nonnegative": bool(np.all(fixed_deltas >= 0)),
            "fixed_lopo_selected_config_counts": {
                f"m{config[0]}_g{config[1]:.2f}_s{config[2]}": int(config_frequency[i])
                for i, config in enumerate(configs) if config_frequency[i]
            },
            "bootstrap_conditioning": (
                "outer resampling of eligible held-out relations; inner relation bootstrap "
                "within each fold refits the 144-way source selector from the other 17 relations"
            ),
            "limitations": [
                "all observations are availability-selected official-train relations",
                "bootstrap resamples relation/environment IDs as the independent unit",
                "fixed per-relation RANSAC and secondary-mode predictions are reused; only selector fitting is repeated",
                "this is not external validation or a guarantee for new acquisition domains",
            ],
        })
        for b, value in enumerate(macro_draws):
            bootstrap_records.append({"seed": seed, "replicate": b, "relation_macro_delta": float(value)})

    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    pd.DataFrame(fold_records).to_csv(args.output_dir / "per_relation.csv", index=False)
    pd.DataFrame(bootstrap_records).to_csv(
        args.output_dir / "bootstrap_draws.csv.gz", index=False, compression="gzip"
    )
    lines = [
        "# Pose-locked selector-fit nested relation bootstrap", "",
        "The fixed source-LOPO result is recomputed over all 18 environment-disjoint source relations. For uncertainty, an outer bootstrap resamples the 14 eligible test relations; for each held-out draw, an inner relation bootstrap resamples the other 17 training relations and refits the 144-configuration selector before scoring the untouched test relation. The same fixed geometry predictions are reused; the selector fit is repeated.", "",
        "| Seed | Fixed-LOPO relation-macro gain (pp) | Nested bootstrap 95% interval (pp) | P(gain ≤ 0) |", "|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lo, hi = row["nested_relation_selector_bootstrap_macro_ci95"]
        lines.append(
            f"| {row['seed']} | {row['fixed_lopo_relation_macro_delta']*100:+.2f} | "
            f"[{lo*100:+.2f},{hi*100:+.2f}] | "
            f"{row['nested_bootstrap_probability_macro_delta_le_zero']:.4f} |"
        )
    lines.extend([
        "", "These intervals include relation sampling and selector-fit variability conditional on the cached pose/mode predictions. They do not include uncertainty from fitting those geometric hypotheses, availability selection, or domain shift. The resampling unit is the source relation/environment, not a query or object cluster.", "",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
