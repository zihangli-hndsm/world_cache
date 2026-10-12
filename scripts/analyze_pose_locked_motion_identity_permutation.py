#!/usr/bin/env python3
"""Negative control for pose-locked secondary-motion identity assignments.

The source-LOPO selector, global pose, residuals, mode choices, and assignment
coverage are held fixed. Within each held-out relation, reference object IDs
predicted by the secondary branch are subjected to a random derangement before
scoring. A single mapping is used for every query with a given object ID, so
the control preserves repeated-point/object dependence and ID frequencies.
"""

from __future__ import annotations

import argparse
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
from analyze_pose_locked_object_cluster_sensitivity import choose_config
from analyze_pose_locked_multimotion_extended_source import environment_groups


UNASSIGNED = -1
SEEDS = (17, 29, 41)


def deranged_map(ids: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    unique = np.unique(ids[ids > 0])
    if len(unique) <= 1:
        return {int(value): int(value) for value in unique}
    values = rng.permutation(unique)
    while np.any(values == unique):
        values = rng.permutation(unique)
    return {int(source): int(target) for source, target in zip(unique, values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--rng-seed", type=int, default=20261004)
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
    if len({relation_meta[pair]["environment_id"] for pair in pair_dirs}) != 18:
        raise ValueError("source relations share an environment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    relation_records: list[dict[str, object]] = []
    permutation_records: list[dict[str, object]] = []
    for seed in args.seeds:
        rng = np.random.default_rng(args.rng_seed + seed)
        data = {
            pair: read_pair(root, pair, args.annotation_root, seed=seed)
            for pair, root in sorted(pair_dirs.items())
        }
        records: list[dict[str, object]] = []
        heldout_by_pair: dict[str, dict[str, np.ndarray]] = {}
        total_queries = 0
        base_total = 0
        method_total = 0
        for pair in sorted(data):
            config = choose_config(data, pair)
            method_correct, mode, _, method_prediction = predict(data[pair], *config)
            item = data[pair]
            moved = np.asarray(item["moved"], dtype=bool)
            baseline_prediction = np.asarray(item["baseline_prediction"], dtype=np.int64)
            expected = np.asarray(item["expected"], dtype=np.int64)
            if len(moved) != len(method_prediction) or len(expected) != len(moved):
                raise ValueError(f"{pair}: held-out query arrays are misaligned")
            n = int(moved.sum())
            if n == 0:
                continue
            base_correct = moved & (baseline_prediction == expected)
            if not np.array_equal(base_correct, np.asarray(item["baseline_correct"], dtype=bool)):
                raise ValueError(f"{pair}: baseline score differs from LOPO evaluator")
            observed = moved & (method_prediction == expected)
            if not np.array_equal(observed, method_correct):
                raise ValueError(f"{pair}: method correctness does not match selected IDs")
            secondary = (mode > 0) & (method_prediction > 0)
            heldout_by_pair[pair] = {
                "moved": moved,
                "expected": expected,
                "mode": np.asarray(mode, dtype=np.int64),
                "prediction": np.asarray(method_prediction, dtype=np.int64),
                "baseline_prediction": baseline_prediction,
            }
            record = {
                "seed": seed, "pair": pair,
                "environment_id": relation_meta[pair]["environment_id"],
                "queries": n, "baseline_correct": int(base_correct.sum()),
                "method_correct": int(observed.sum()),
                "observed_delta": int(observed.sum() - base_correct.sum()),
                "secondary_assignments": int(secondary.sum()),
                "reference_id_vocabulary": int(len(item["reference_object_ids"])),
                "modes": config[0], "margin_m": config[1], "min_support": config[2],
            }
            records.append(record)
            relation_records.append(record)
            total_queries += n
            base_total += int(base_correct.sum())
            method_total += int(observed.sum())

        observed_delta = method_total - base_total
        observed_weighted = observed_delta / total_queries
        rel_deltas = np.asarray([
            row["observed_delta"] / row["queries"] for row in records
        ], dtype=np.float64)
        observed_macro = float(rel_deltas.mean())
        null_weighted: list[float] = []
        null_macro: list[float] = []
        null_method_hits: list[int] = []
        for draw in range(args.permutations):
            trial_hits = 0
            trial_deltas: list[float] = []
            for pair in sorted(data):
                pair_data = heldout_by_pair.get(pair)
                if pair_data is None:
                    continue
                moved = pair_data["moved"]
                if not moved.any():
                    continue
                mode = pair_data["mode"]
                prediction = pair_data["prediction"]
                expected = pair_data["expected"]
                baseline_prediction = pair_data["baseline_prediction"]
                secondary = (mode > 0) & (prediction > 0)
                mapped = prediction.copy()
                if secondary.any():
                    ids = np.unique(np.concatenate([
                        np.asarray(item["reference_object_ids"], dtype=np.int64),
                        expected[moved & (expected > 0)], prediction[secondary],
                    ]))
                    mapping = deranged_map(ids, rng)
                    mapped[secondary] = np.asarray(
                        [mapping[int(value)] for value in prediction[secondary]], dtype=np.int64
                    )
                base_hits = int((moved & (baseline_prediction == expected)).sum())
                method_hits = int((moved & (mapped == expected)).sum())
                n = int(moved.sum())
                trial_hits += method_hits
                trial_deltas.append((method_hits - base_hits) / n)
            null_weighted.append((trial_hits - base_total) / total_queries)
            null_macro.append(float(np.mean(trial_deltas)))
            null_method_hits.append(trial_hits)
            permutation_records.append({
                "seed": seed, "permutation": draw,
                "method_correct": trial_hits,
                "weighted_delta": null_weighted[-1],
                "relation_macro_delta": null_macro[-1],
            })

        null_arr = np.asarray(null_weighted, dtype=np.float64)
        null_macro_arr = np.asarray(null_macro, dtype=np.float64)
        summaries.append({
            "seed": seed,
            "source_relations": 18,
            "eligible_relations": len(records),
            "queries": total_queries,
            "baseline_correct": base_total,
            "method_correct": method_total,
            "observed_delta": observed_delta,
            "observed_weighted_delta": observed_weighted,
            "observed_relation_macro_delta": observed_macro,
            "secondary_assigned_queries": int(sum(r["secondary_assignments"] for r in records)),
            "single_id_vocabulary_relations": int(sum(
                r["reference_id_vocabulary"] <= 1 for r in records
            )),
            "permutations": args.permutations,
            "randomization_unit": "relation; one deranged reference-object-ID mapping shared across all queries with an ID",
            "control": "pose, residuals, mode decisions, coverage, and per-mode/object-ID counts held fixed; secondary branch IDs relabeled, global baseline unchanged",
            "null_weighted_delta_95pct": [float(v) for v in np.quantile(null_arr, [0.025, 0.975])],
            "null_relation_macro_delta_95pct": [float(v) for v in np.quantile(null_macro_arr, [0.025, 0.975])],
            "null_weighted_delta_mean": float(null_arr.mean()),
            "null_weighted_delta_p_ge_observed": float((1 + np.sum(null_arr >= observed_weighted)) / (args.permutations + 1)),
            "null_macro_delta_p_ge_observed": float((1 + np.sum(null_macro_arr >= observed_macro)) / (args.permutations + 1)),
            "limitations": [
                "source-only relation-LOPO outcomes on official train relations",
                "object-ID relabeling is a negative control, not an independent-site generalization test",
                "selector is held fixed within each LOPO fold; this control tests the identity assignment conditional on fitted modes",
            ],
            "rng_seed": args.rng_seed + seed,
        })

    per_relation = pd.DataFrame(relation_records)
    per_relation.to_csv(args.output_dir / "per_relation.csv", index=False)
    (pd.DataFrame(permutation_records)
     .to_csv(args.output_dir / "permutation_draws.csv.gz", index=False, compression="gzip"))
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Pose-locked secondary-motion identity permutation control", "",
        "The source-LOPO model, pose, residuals, selected secondary modes, and assignment coverage are fixed. For each held-out relation, a random derangement permutes the annotated reference scan's object-ID vocabulary; the same map applies to every query point with a given predicted ID. This preserves point/object clusters and prediction frequencies while breaking the query-to-physical-instance label mapping. Global-baseline IDs and all expected labels are unchanged. Relations with only one positive reference ID cannot be deranged and retain their original ID mapping.", "",
        "| Seed | Baseline correct | Observed method correct | Observed gain (pp) | Null gain 95% interval (pp) | Permutation p (weighted) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lo, hi = row["null_weighted_delta_95pct"]
        lines.append(
            f"| {row['seed']} | {row['baseline_correct']}/1,685 | "
            f"{row['method_correct']}/1,685 | {row['observed_weighted_delta']*100:+.2f} | "
            f"[{lo*100:+.2f},{hi*100:+.2f}] | {row['null_weighted_delta_p_ge_observed']:.4f} |"
        )
    lines += [
        "", "The permutation interval is a conditional negative-control distribution, not a confidence interval for new environments. It retains the out-of-fold model decisions and all object-level dependence; only the physical-ID mapping for secondary-mode outputs is scrambled. The finite-sample permutation tail area is descriptive because source relations are availability-selected train data.", "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
