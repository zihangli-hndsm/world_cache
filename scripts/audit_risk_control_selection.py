"""Audit post-selection validity of the residual risk operating point.

The original source-fit selector scans many residual prefixes and reports a
raw exact-binomial lower bound.  This script keeps that descriptive result,
then applies a Bonferroni correction over every eligible prefix and evaluates
a fixed 2 cm audit point separately, while marking its unadjusted interval as
retrospective rather than formal family-wise evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

try:
    from .evaluate_risk_controlled_calibration import (
        evaluate,
        fit_threshold,
        lower_confidence_bound,
        read,
    )
except ImportError:  # pragma: no cover - supports direct CLI execution
    from evaluate_risk_controlled_calibration import evaluate, fit_threshold, lower_confidence_bound, read


def candidate_prefixes(frame: pd.DataFrame, min_queries: int) -> list[tuple[float, int, int]]:
    residuals = frame.spatial_residual_m.to_numpy(float)
    correct = frame.semantic_correct.to_numpy(bool)
    order = np.argsort(residuals, kind="stable")
    residuals = residuals[order]
    correct = correct[order]
    ends = np.flatnonzero(np.r_[np.diff(residuals) > 0.0, True])
    candidates = []
    for end in ends:
        queries = int(end + 1)
        if queries >= min_queries:
            candidates.append((float(residuals[end]), queries, int(correct[:queries].sum())))
    return candidates


def familywise_scan(
    frame: pd.DataFrame,
    precision_target: float,
    confidence_level: float,
    min_queries: int,
) -> dict[str, float | int | bool | None]:
    candidates = candidate_prefixes(frame, min_queries)
    if not candidates:
        raise ValueError("no eligible residual prefixes")
    alpha = (1.0 - confidence_level) / len(candidates)
    best_adjusted: tuple[float, float, int, float] | None = None
    max_adjusted: tuple[float, float, int, float] | None = None
    for threshold, queries, successes in candidates:
        adjusted = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, queries - successes + 1))
        record = (adjusted, threshold, queries, successes / queries)
        if max_adjusted is None or adjusted > max_adjusted[0]:
            max_adjusted = record
        if adjusted >= precision_target and (best_adjusted is None or queries > best_adjusted[2]):
            best_adjusted = record
    return {
        "candidate_prefixes": len(candidates),
        "bonferroni_alpha": alpha,
        "familywise_feasible": best_adjusted is not None,
        "familywise_threshold": None if best_adjusted is None else best_adjusted[1],
        "familywise_queries": None if best_adjusted is None else best_adjusted[2],
        "familywise_lower_bound": None if best_adjusted is None else best_adjusted[0],
        "max_adjusted_lower_bound": max_adjusted[0],
        "max_adjusted_threshold": max_adjusted[1],
        "max_adjusted_queries": max_adjusted[2],
    }


def row(dataset: str, operating_point: str, frame: pd.DataFrame, threshold: float, bound: float | None) -> dict[str, object]:
    metrics = evaluate(frame, threshold)
    return {
        "dataset": dataset,
        "operating_point": operating_point,
        "threshold": threshold,
        **metrics,
        "semantic_lower_bound": bound,
        "lower_bound_status": "fixed-threshold" if bound is not None else "not reported",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.90)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-queries", type=int, default=100)
    parser.add_argument("--fixed-threshold", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.concat([read(path) for path in args.source], ignore_index=True)
    target = read(args.target)
    descriptive = fit_threshold(source, args.precision_target, args.confidence_level, args.min_queries)
    source_fit_bound = float(descriptive["semantic_lower_bound"])
    fixed_mask = source.spatial_residual_m.to_numpy(float) <= args.fixed_threshold
    fixed_successes = int(source.loc[fixed_mask, "semantic_correct"].sum())
    fixed_trials = int(fixed_mask.sum())
    fixed_bound = lower_confidence_bound(fixed_successes, fixed_trials, args.confidence_level)
    rows = [
        row("source", "source_fit_descriptive", source, float(descriptive["threshold"]), source_fit_bound),
        row("target", "source_fit_descriptive", target, float(descriptive["threshold"]), None),
        row("source", "fixed_2cm_audit", source, args.fixed_threshold, fixed_bound),
        row("target", "fixed_2cm_audit", target, args.fixed_threshold, None),
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    scan = familywise_scan(source, args.precision_target, args.confidence_level, args.min_queries)
    status = {
        "status": "complete",
        "selection_correction": "Bonferroni over all eligible residual prefixes",
        "source_fit_is_descriptive_after_prefix_selection": True,
        "fixed_threshold_is_not_selected_from_target_labels": True,
        "fixed_threshold_is_retrospective_audit": True,
        "precision_target": args.precision_target,
        "confidence_level": args.confidence_level,
        "min_queries": args.min_queries,
        "fixed_threshold": args.fixed_threshold,
        "source_fit_threshold": descriptive["threshold"],
        "source_fit_raw_lower_bound": source_fit_bound,
        "fixed_source_lower_bound": fixed_bound,
        **scan,
        "target_evaluation_is_post_hoc": True,
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
