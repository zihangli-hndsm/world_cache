"""Fit a source-only finite-sample risk-controlled residual threshold."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


PAIR_RE = re.compile(r"(0cac|20c993|4aca|5630|6bde|751a|95be|c670|chairs|pair0[2-8]|target)")


def read(path: Path) -> pd.DataFrame:
    match = PAIR_RE.search(path.parent.name)
    pair = match.group(1) if match else path.parent.name
    frame = pd.read_csv(path).assign(pair=pair)
    if "query_evaluable" not in frame:
        raise ValueError(f"{path} is not an annotation-independent diagnostic")
    return frame[frame.query_evaluable.astype(bool)].reset_index(drop=True)


def lower_confidence_bound(successes: int, trials: int, confidence_level: float) -> float:
    if trials == 0 or successes == 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence_level, successes, trials - successes + 1))


def fit_threshold(
    frame: pd.DataFrame,
    precision_target: float,
    confidence_level: float,
    min_queries: int,
) -> dict[str, float | int]:
    residuals = frame.spatial_residual_m.to_numpy(float)
    correct = frame.semantic_correct.to_numpy(bool)
    best: dict[str, float | int] | None = None
    for threshold in np.unique(residuals):
        selected = residuals <= threshold
        queries = int(selected.sum())
        if queries < min_queries:
            continue
        successes = int(correct[selected].sum())
        precision = successes / queries
        lower = lower_confidence_bound(successes, queries, confidence_level)
        if lower < precision_target:
            continue
        candidate = {
            "threshold": float(threshold),
            "queries": queries,
            "coverage": queries / len(frame),
            "semantic_precision": precision,
            "semantic_lower_bound": lower,
        }
        if best is None or queries > int(best["queries"]):
            best = candidate
    if best is None:
        raise ValueError("no residual threshold satisfies the requested risk bound")
    return best


def mean_iou(actual: np.ndarray, predicted: np.ndarray) -> float:
    labels = np.union1d(actual, predicted)
    values = []
    for label in labels:
        union = np.logical_or(actual == label, predicted == label).sum()
        if union:
            values.append(np.logical_and(actual == label, predicted == label).sum() / union)
    return float(np.mean(values)) if values else float("nan")


def evaluate(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    mask = frame.spatial_residual_m.to_numpy(float) <= threshold
    count = int(mask.sum())
    if count == 0:
        raise ValueError("risk-controlled threshold retained no queries")
    semantic = frame.loc[mask, "semantic_correct"].to_numpy(bool)
    instance = frame.loc[mask, "instance_correct"].to_numpy(bool)
    query_semantic = frame.loc[mask, "query_semantic_id"].to_numpy()
    selected_semantic = frame.loc[mask, "selected_reference_semantic_id"].to_numpy()
    query_instance = frame.loc[mask, "query_instance_id"].to_numpy()
    selected_instance = frame.loc[mask, "selected_reference_instance_id"].to_numpy()
    return {
        "queries": count,
        "coverage": count / len(frame),
        "semantic_precision": float(semantic.mean()),
        "instance_precision": float(instance.mean()),
        "semantic_mIoU": mean_iou(query_semantic, selected_semantic),
        "instance_mIoU": mean_iou(query_instance, selected_instance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.90)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-queries", type=int, default=100)
    args = parser.parse_args()
    if not 0.0 < args.precision_target < 1.0:
        raise ValueError("precision target must lie in (0, 1)")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence level must lie in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.concat([read(path) for path in args.source], ignore_index=True)
    target = read(args.target)
    fit = fit_threshold(source, args.precision_target, args.confidence_level, args.min_queries)
    rows = []
    for dataset, frame in (("source", source), ("target", target)):
        rows.append({"dataset": dataset, **fit, **evaluate(frame, float(fit["threshold"]))})
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "summary.csv", index=False)

    pair_rows = []
    for pair, frame in source.groupby("pair", sort=True):
        pair_rows.append({"dataset": "source", "pair": pair, **evaluate(frame, float(fit["threshold"]))})
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "per_pair.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "selection_population": "all source query_evaluable diagnostics",
        "risk_bound": "one-sided exact binomial lower confidence bound",
        "confidence_level": args.confidence_level,
        "precision_target": args.precision_target,
        "min_queries": args.min_queries,
        **fit,
        "target_evaluation_is_post_hoc": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    main()
