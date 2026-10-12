#!/usr/bin/env python3
"""Nested source-group bootstrap for the source-fitted DINO cosine gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, load_split


def grouped_arrays(relations: list[dict], thresholds: np.ndarray):
    by_group: dict[str, list[dict]] = {}
    for row in relations:
        by_group.setdefault(row["environment_id"], []).append(row)
    names = sorted(by_group)
    accepts, correct, queries = [], [], []
    for name in names:
        scores = np.concatenate([r["scores"] for r in by_group[name]])
        labels = np.concatenate([r["correct"] for r in by_group[name]])
        valid = np.concatenate([r["valid"] for r in by_group[name]])
        accept = np.asarray([np.sum(scores[valid] >= t) for t in thresholds], dtype=np.int32)
        hit = np.asarray([np.sum((scores >= t) & labels & valid) for t in thresholds], dtype=np.int32)
        accepts.append(accept)
        correct.append(hit)
        queries.append(len(scores))
    accepts = np.stack(accepts)
    correct = np.stack(correct)
    precision = np.divide(correct, accepts, out=np.full(correct.shape, np.nan, dtype=float), where=accepts > 0)
    coverage = correct / np.asarray(queries, dtype=float)[:, None]
    return names, precision, coverage


def fit_bootstrap_threshold(source_precision: np.ndarray, source_coverage: np.ndarray,
                            sampled_groups: np.ndarray, target: float) -> int | None:
    p = source_precision[sampled_groups]
    c = source_coverage[sampled_groups]
    valid = np.isfinite(p)
    counts = valid.sum(axis=0)
    macro_p = np.divide(np.nansum(p, axis=0), counts,
                        out=np.full(p.shape[1], np.nan), where=counts > 0)
    macro_c = c.mean(axis=0)
    feasible = np.flatnonzero((macro_p >= target) & (counts > 0))
    if not len(feasible):
        return None
    best_cov = np.max(macro_c[feasible])
    tied = feasible[np.isclose(macro_c[feasible], best_cov, rtol=0, atol=1e-14)]
    return int(tied[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-input-root", type=Path, default=ROOT / "outputs/reports/20261009_unseen_oasis_equal_budget")
    parser.add_argument("--source-prediction-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank_train8")
    parser.add_argument("--validation-input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    parser.add_argument("--validation-prediction-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    parser.add_argument("--remind-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank/summary.json")
    parser.add_argument("--target-source-precision", type=float, default=.90)
    parser.add_argument("--exclude-source-uuids", nargs="*", default=[])
    parser.add_argument("--exclude-validation-uuids", nargs="*", default=[])
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/20261009_dino_source_gate_fit_uncertainty")
    args = parser.parse_args()

    source = load_split(args.source_input_root, args.source_prediction_root)
    validation = load_split(args.validation_input_root, args.validation_prediction_root)
    source_before = {r["environment_id"] for r in source}
    validation_before = {r["environment_id"] for r in validation}
    excluded_source = {x.lower() for x in args.exclude_source_uuids}
    excluded_validation = {x.lower() for x in args.exclude_validation_uuids}
    if not excluded_source.issubset(source_before):
        raise ValueError(f"excluded source UUIDs not found: {sorted(excluded_source - source_before)}")
    if not excluded_validation.issubset(validation_before):
        raise ValueError(f"excluded validation UUIDs not found: {sorted(excluded_validation - validation_before)}")
    source = [r for r in source if r["environment_id"] not in excluded_source]
    validation = [r for r in validation if r["environment_id"] not in excluded_validation]
    source_ids = sorted({r["environment_id"] for r in source})
    val_ids = sorted({r["environment_id"] for r in validation})
    if set(source_ids) & set(val_ids):
        raise ValueError("source and validation reference UUIDs overlap")
    source_scores = np.concatenate([r["scores"] for r in source])
    source_valid = np.concatenate([r["valid"] for r in source])
    thresholds = np.unique(source_scores[source_valid])
    src_names, src_p, src_c = grouped_arrays(source, thresholds)
    val_names, val_p, val_c = grouped_arrays(validation, thresholds)
    if src_names != source_ids or val_names != val_ids:
        raise AssertionError("group order changed while building bootstrap arrays")

    remind_summary = json.loads(args.remind_summary.read_text())
    remind_rows = {r["environment_id"]: r["remind_one_to_one"] for r in remind_summary["reference_groups"]}
    if not set(val_names).issubset(remind_rows):
        raise ValueError("REMIND is missing one or more retained DINO validation UUIDs")
    remind_p = np.asarray([float(remind_rows[x]["precision"]) for x in val_names])
    remind_c = np.asarray([float(remind_rows[x]["coverage"]) for x in val_names])

    rng = np.random.default_rng(args.seed)
    n_source, n_val = len(source_ids), len(val_ids)
    draws = args.draws
    fit_indices = np.full(draws, -1, dtype=np.int32)
    for i in range(draws):
        sampled = rng.integers(0, n_source, size=n_source)
        fitted = fit_bootstrap_threshold(src_p, src_c, sampled, args.target_source_precision)
        fit_indices[i] = -1 if fitted is None else fitted
    success = fit_indices >= 0

    original_successes = int(success.sum())
    if not original_successes:
        raise RuntimeError("no source bootstrap draw found a feasible threshold")
    threshold_draws = thresholds[fit_indices[success]]
    result_rows = []
    # Resample validation groups independently, preserving paired DINO/REMIND outcomes.
    for draw_id in np.flatnonzero(success):
        threshold_idx = int(fit_indices[draw_id])
        sampled_val = rng.integers(0, n_val, size=n_val)
        p = val_p[sampled_val, threshold_idx]
        c = val_c[sampled_val, threshold_idx]
        available = np.isfinite(p)
        if not available.any():
            continue
        dino_p = float(np.mean(p[available]))
        remind_paired = float(np.mean(remind_p[sampled_val][available]))
        dino_c = float(np.mean(c))
        remind_cov = float(np.mean(remind_c[sampled_val]))
        result_rows.append((dino_p, remind_paired, dino_c, remind_cov,
                            int(available.sum()), float(thresholds[threshold_idx])))
    rows = np.asarray(result_rows, dtype=float)
    if not len(rows):
        raise RuntimeError("no validation bootstrap draws had accepted groups")

    def interval(x: np.ndarray, adjusted: bool = False) -> list[float]:
        q = [.0125, .9875] if adjusted else [.025, .975]
        return [float(v) for v in np.quantile(x, q)]

    precision_diff = rows[:, 0] - rows[:, 1]
    coverage_diff = rows[:, 2] - rows[:, 3]
    summary = {
        "status": "complete",
        "estimand": "nested source-reference-group and validation-reference-group bootstrap; source threshold refit in each source draw, validation DINO-minus-REMIND paired effects evaluated at that draw's threshold",
        "source_group_count": n_source,
        "validation_group_count": n_val,
        "original_source_group_count": len(source_before),
        "original_validation_group_count": len(validation_before),
        "excluded_source_uuids": sorted(excluded_source),
        "excluded_validation_uuids": sorted(excluded_validation),
        "source_target_macro_precision": args.target_source_precision,
        "source_candidate_threshold_count": int(len(thresholds)),
        "draws_requested": draws,
        "draws_feasible": original_successes,
        "infeasible_source_fit_fraction": float(1 - original_successes / draws),
        "seed": args.seed,
        "fitted_threshold_distribution": {
            "median": float(np.median(threshold_draws)),
            "percentile_95": interval(threshold_draws),
            "percentile_98_75": interval(threshold_draws, adjusted=True),
            "unique_thresholds": int(len(np.unique(threshold_draws))),
        },
        "validation_macro_precision": {"median": float(np.median(rows[:, 0])), "percentile_95": interval(rows[:, 0])},
        "validation_paired_precision_difference": {
            "median": float(np.median(precision_diff)),
            "percentile_95": interval(precision_diff),
            "two_outcome_adjusted_percentile_95": interval(precision_diff, adjusted=True),
            "draws_with_all_sampled_groups_accepting": int(np.sum(rows[:, 4] == n_val)),
            "precision_defined_on_accepting_sampled_groups_only": True,
        },
        "validation_paired_coverage_difference": {
            "median": float(np.median(coverage_diff)),
            "percentile_95": interval(coverage_diff),
            "two_outcome_adjusted_percentile_95": interval(coverage_diff, adjusted=True),
        },
        "validation_dino_correct_coverage": {"median": float(np.median(rows[:, 2])), "percentile_95": interval(rows[:, 2])},
        "limits": [
            f"Only {n_source} source reference groups support threshold fitting; the cluster bootstrap is a sensitivity analysis, not a finite-sample guarantee.",
            "The precision contrast conditions on sampled groups with at least one accepted DINO assignment if a refitted threshold accepts none in a group; the number of such groups is recorded per draw.",
            "Validation is the previously explored 3RScan cohort with privileged masks; this does not establish independent-site generalization.",
            "This uncertainty analysis does not correct potential physical-content overlap between the source and validation references.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Nested source-fit uncertainty for the DINO cosine gate", "",
        f"We resampled the {n_source} source reference UUIDs with replacement, refit the maximum-coverage threshold subject to {args.target_source_precision:.0%} source macro precision, then independently resampled the {n_val} validation UUIDs and evaluated the paired DINO-minus-REMIND effects at that draw's fitted threshold. There were {draws} joint draws; {summary['draws_feasible']} had a feasible source fit. Excluded source UUIDs: {sorted(excluded_source)}. Excluded validation UUIDs: {sorted(excluded_validation)}.", "",
        f"The fitted cutoff median is {summary['fitted_threshold_distribution']['median']:.6f}, with 95% percentile range [{summary['fitted_threshold_distribution']['percentile_95'][0]:.6f}, {summary['fitted_threshold_distribution']['percentile_95'][1]:.6f}]. The nested paired precision difference median is {np.median(precision_diff)*100:+.1f} pp (two-outcome-adjusted 95% range [{summary['validation_paired_precision_difference']['two_outcome_adjusted_percentile_95'][0]*100:+.1f}, {summary['validation_paired_precision_difference']['two_outcome_adjusted_percentile_95'][1]*100:+.1f}]); correct-coverage difference is {np.median(coverage_diff)*100:+.1f} pp (adjusted range [{summary['validation_paired_coverage_difference']['two_outcome_adjusted_percentile_95'][0]*100:+.1f}, {summary['validation_paired_coverage_difference']['two_outcome_adjusted_percentile_95'][1]*100:+.1f}]).", "",
        f"This nested bootstrap propagates source-fit and validation-group resampling, but {n_source} source groups are too few for a strong finite-sample guarantee. Precision averages only over sampled groups with at least one DINO accept; the validation cohort is previously explored and uses privileged masks.", "",
    ]
    (args.output / "report.md").write_text("\n".join(report))
    print(json.dumps({k: summary[k] for k in ("draws_requested", "draws_feasible", "infeasible_source_fit_fraction", "fitted_threshold_distribution", "validation_paired_precision_difference", "validation_paired_coverage_difference")}, indent=2))


if __name__ == "__main__":
    main()
