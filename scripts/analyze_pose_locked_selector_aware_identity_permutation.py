#!/usr/bin/env python3
"""Selection-aware physical-ID permutation control for pose-locked LOPO.

Unlike the conditional identity permutation report, every draw deranges the
secondary-branch IDs in every source relation and then refits the full LOPO
configuration selector. This controls for the possibility that the selector
itself exploits chance ID matches in its training relations.
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
from analyze_pose_locked_object_cluster_sensitivity import choose_config


SEEDS = (17, 29, 41)


def choose_index(scores: np.ndarray, configs: list[tuple[int, float, int]]) -> int:
    best = int(scores.max())
    return min(
        (i for i, score in enumerate(scores) if int(score) == best),
        key=lambda i: (configs[i][0], -configs[i][2], -configs[i][1]),
    )


def make_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 1:
        return np.arange(size, dtype=np.int64)
    identity = np.arange(size, dtype=np.int64)
    perm = rng.permutation(size)
    while np.any(perm == identity):
        perm = rng.permutation(size)
    return perm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20261005)
    args = parser.parse_args()
    if args.permutations < 1 or any(seed not in SEEDS for seed in args.seeds):
        raise ValueError("positive permutations and seeds from 17/29/41 are required")

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
    relation_meta = environment_groups(args.input_dir, args.metadata)
    pairs = sorted(pair_dirs)
    if len({relation_meta[pair]["environment_id"] for pair in pairs}) != 18:
        raise ValueError("source relations are not environment-disjoint")

    configs = list(itertools.product(MODE_COUNTS, MARGINS_M, MIN_SUPPORTS))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    relation_records: list[dict[str, object]] = []
    draw_records: list[dict[str, object]] = []

    for seed in args.seeds:
        rng = np.random.default_rng(args.rng_seed + seed)
        data = {
            pair: read_pair(pair_dirs[pair], pair, args.annotation_root, seed=seed)
            for pair in pairs
        }
        query_counts = np.asarray([
            int(np.asarray(data[pair]["moved"], dtype=bool).sum()) for pair in pairs
        ], dtype=np.int64)
        eligible = np.flatnonzero(query_counts > 0)
        if len(eligible) != 14 or int(query_counts.sum()) != 1685:
            raise ValueError("expected 14 eligible relations and 1,685 linked queries")

        # For each relation/config, hold fixed the contribution of predictions
        # that stayed on the global mode. Store secondary outcomes as a sparse
        # ID-by-ID contingency tensor so a derangement is fast to rescore.
        static_hits: list[np.ndarray] = []
        contingencies: list[np.ndarray] = []
        vocabularies: list[np.ndarray] = []
        baseline_hits = np.zeros(len(pairs), dtype=np.int64)
        for relation_idx, pair in enumerate(pairs):
            item = data[pair]
            moved = np.asarray(item["moved"], dtype=bool)
            expected = np.asarray(item["expected"], dtype=np.int64)
            vocabulary = np.asarray(item["reference_object_ids"], dtype=np.int64)
            id_to_index = {int(obj_id): i for i, obj_id in enumerate(vocabulary)}
            base_correct = np.asarray(item["baseline_correct"], dtype=bool)
            baseline_hits[relation_idx] = int(base_correct.sum())
            static = np.zeros(len(configs), dtype=np.int64)
            cube = np.zeros((len(configs), len(vocabulary), len(vocabulary)), dtype=np.int32)
            for config_idx, config in enumerate(configs):
                correct, mode, _, prediction = predict(item, *config)
                # The actual selected mode and physical IDs must reproduce the
                # fixed-prediction source-LOPO records before null scoring.
                global_mask = moved & (mode == 0)
                static[config_idx] = int((global_mask & (prediction == expected)).sum())
                secondary = moved & (mode > 0) & (prediction > 0)
                if secondary.any():
                    pred_idx = np.asarray([id_to_index[int(x)] for x in prediction[secondary]], dtype=np.int64)
                    truth_idx = np.asarray([id_to_index[int(x)] for x in expected[secondary]], dtype=np.int64)
                    np.add.at(cube[config_idx], (pred_idx, truth_idx), 1)
                if not np.array_equal(correct, moved & (prediction == expected)):
                    raise ValueError(f"{pair}: prediction correctness mismatch for {config}")
            static_hits.append(static)
            contingencies.append(cube)
            vocabularies.append(vocabulary)

        # Reproduce the observed LOPO score and per-fold configuration first.
        observed_weighted_hits = 0
        observed_weighted_delta = 0
        observed_relation_deltas: list[float] = []
        observed_configs: dict[str, int] = {}
        for relation_idx, pair in enumerate(pairs):
            chosen = choose_config(data, pair)
            config_idx = configs.index(chosen)
            method_correct = int(predict(data[pair], *chosen)[0].sum())
            base_correct = int(baseline_hits[relation_idx])
            if method_correct != int(static_hits[relation_idx][config_idx]
                                      + contingencies[relation_idx][config_idx].trace()):
                # Identity mapping is the tensor diagonal.
                raise ValueError(f"{pair}: observed identity score does not match contingency table")
            observed_weighted_hits += method_correct
            observed_weighted_delta += method_correct - base_correct
            if query_counts[relation_idx] > 0:
                observed_relation_deltas.append(
                    (method_correct - base_correct) / int(query_counts[relation_idx])
                )
            key = f"m{chosen[0]}_g{chosen[1]:.2f}_s{chosen[2]}"
            observed_configs[key] = observed_configs.get(key, 0) + 1
        total_queries = int(query_counts.sum())
        observed_weighted = observed_weighted_delta / total_queries
        observed_macro = float(np.mean(observed_relation_deltas))

        # Each draw applies a separate full-vocabulary derangement per
        # relation, then selects a fresh configuration in every LOPO fold.
        null_weighted = np.empty(args.permutations, dtype=np.float64)
        null_macro = np.empty(args.permutations, dtype=np.float64)
        for draw in range(args.permutations):
            permuted_hits = np.empty((len(pairs), len(configs)), dtype=np.int64)
            for relation_idx in range(len(pairs)):
                vocab_size = len(vocabularies[relation_idx])
                permutation = make_derangement(vocab_size, rng)
                identity = np.arange(vocab_size)
                secondary_hits = contingencies[relation_idx][:, identity, permutation].sum(axis=1)
                permuted_hits[relation_idx] = static_hits[relation_idx] + secondary_hits

            relation_delta: list[float] = []
            weighted_delta = 0
            for heldout_idx in eligible:
                training_idx = np.asarray([j for j in range(len(pairs)) if j != heldout_idx], dtype=np.int64)
                selected = choose_index(permuted_hits[training_idx].sum(axis=0), configs)
                delta = int(permuted_hits[heldout_idx, selected] - baseline_hits[heldout_idx])
                weighted_delta += delta
                relation_delta.append(delta / int(query_counts[heldout_idx]))
            null_weighted[draw] = weighted_delta / total_queries
            null_macro[draw] = float(np.mean(relation_delta))
            draw_records.append({
                "seed": seed, "permutation": draw,
                "weighted_delta": float(null_weighted[draw]),
                "relation_macro_delta": float(null_macro[draw]),
            })

        weighted_p = float((1 + np.count_nonzero(null_weighted >= observed_weighted))
                           / (args.permutations + 1))
        macro_p = float((1 + np.count_nonzero(null_macro >= observed_macro))
                        / (args.permutations + 1))
        summaries.append({
            "seed": seed, "source_relations": len(pairs), "eligible_relations": len(eligible),
            "queries": total_queries, "permutations": args.permutations,
            "configs": len(configs), "observed_weighted_delta": observed_weighted,
            "observed_relation_macro_delta": observed_macro,
            "observed_weighted_gain_correct": int(observed_weighted_delta),
            "observed_selected_config_counts": observed_configs,
            "null_weighted_delta_95pct": [float(v) for v in np.quantile(null_weighted, [0.025, 0.975])],
            "null_relation_macro_delta_95pct": [float(v) for v in np.quantile(null_macro, [0.025, 0.975])],
            "null_weighted_delta_mean": float(null_weighted.mean()),
            "null_weighted_delta_p_ge_observed": weighted_p,
            "null_relation_macro_delta_p_ge_observed": macro_p,
            "randomization_unit": "relation; complete reference object-ID vocabulary; same derangement across all queries/configurations with that predicted ID",
            "selection_repeated_under_null": "for each draw, refit the 144-way source-LOPO selector in every held-out relation using the deranged training-relation outcomes",
            "limitations": [
                "source-only, availability-selected official-train relations",
                "cached poses, residuals, and secondary-mode hypotheses are fixed",
                "null tests physical-ID dependence conditional on geometry hypotheses, not external generalization",
                "global-mode outputs and truth labels remain unchanged",
            ],
            "rng_seed": args.rng_seed + seed,
        })
        for relation_idx, pair in enumerate(pairs):
            relation_records.append({
                "seed": seed, "pair": pair,
                "environment_id": relation_meta[pair]["environment_id"],
                "queries": int(query_counts[relation_idx]),
                "baseline_correct": int(baseline_hits[relation_idx]),
                "reference_id_vocabulary": int(len(vocabularies[relation_idx])),
            })

    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    pd.DataFrame(relation_records).to_csv(args.output_dir / "per_relation.csv", index=False)
    pd.DataFrame(draw_records).to_csv(
        args.output_dir / "permutation_draws.csv.gz", index=False, compression="gzip"
    )
    lines = [
        "# Selection-aware pose-locked identity permutation control", "",
        "Each draw deranges the full positive reference object-ID vocabulary independently within every source relation. The same relation-specific mapping is applied to all secondary-branch predictions and all 144 configurations, while global-mode predictions, truth, geometry, and query coverage remain fixed. The complete source-LOPO configuration selector is refit under each draw, so training-relation chance matches can influence model selection just as they do in the observed pipeline.", "",
        "| Seed | Observed weighted gain (pp) | Null weighted 95% interval (pp) | Weighted p | Null relation-macro 95% interval (pp) | Macro p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        wlo, whi = row["null_weighted_delta_95pct"]
        mlo, mhi = row["null_relation_macro_delta_95pct"]
        lines.append(
            f"| {row['seed']} | {row['observed_weighted_delta']*100:+.2f} | "
            f"[{wlo*100:+.2f},{whi*100:+.2f}] | {row['null_weighted_delta_p_ge_observed']:.4f} | "
            f"[{mlo*100:+.2f},{mhi*100:+.2f}] | {row['null_relation_macro_delta_p_ge_observed']:.4f} |"
        )
    lines.extend([
        "", "This selection-aware null is stricter than a post-selection label shuffle: it repeats the source selector for every derangement. It remains conditional on the cached geometric hypotheses and availability-selected train relations; it is not an independent-site test. Tail areas are descriptive randomization diagnostics.", "",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
