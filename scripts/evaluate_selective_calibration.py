"""Evaluate source-only, leave-one-pair-out selective calibration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCORES = (
    "descriptor_margin",
    "top1_descriptor_score",
    "selected_descriptor_score",
    "low_spatial_residual",
    "consensus_inlier",
)


def score_map(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "descriptor_margin": frame["top1_top2_margin"].to_numpy(float),
        "top1_descriptor_score": frame["top1_descriptor_score"].to_numpy(float),
        "selected_descriptor_score": frame["selected_descriptor_score"].to_numpy(float),
        "low_spatial_residual": -frame["spatial_residual_m"].to_numpy(float),
        "consensus_inlier": frame["consensus_inlier"].astype(float).to_numpy(),
    }


def fit_threshold(
    frame: pd.DataFrame,
    values: np.ndarray,
    precision_target: float,
    min_queries: int,
) -> tuple[float, int, float] | None:
    """Choose the least restrictive score threshold meeting source precision."""
    order = np.argsort(-values, kind="stable")
    ordered_values = values[order]
    correct = frame["semantic_correct"].to_numpy(bool)[order].astype(np.int64)
    cumulative = np.cumsum(correct)
    ends = np.flatnonzero(np.r_[ordered_values[1:] != ordered_values[:-1], True])
    counts = ends + 1
    precision = cumulative[ends] / counts
    valid = (counts >= min_queries) & (precision >= precision_target)
    if not valid.any():
        return None
    # Largest retained group is the least restrictive valid threshold.
    index = int(np.flatnonzero(valid)[-1])
    threshold = float(ordered_values[ends[index]])
    selected = values >= threshold
    return threshold, int(selected.sum()), float(frame.loc[selected, "semantic_correct"].mean())


def evaluate(frame: pd.DataFrame, values: np.ndarray, threshold: float) -> dict[str, float]:
    selected = values >= threshold
    count = int(selected.sum())
    return {
        "queries": count,
        "coverage": float(selected.mean()),
        "semantic_precision": float(frame.loc[selected, "semantic_correct"].mean()) if count else float("nan"),
        "instance_precision": float(frame.loc[selected, "instance_correct"].mean()) if count else float("nan"),
    }


def summarize_cross_fitted(result: pd.DataFrame, source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate pair-held-out predictions without reusing each held pair.

    Each row in ``result`` was produced by fitting its threshold on all source
    pairs except ``held_pair``.  Pooling only those held-out rows gives a
    cross-fitted source-risk estimate; it is not a family-wise guarantee.
    """
    total_queries = sum(len(frame) for frame in source.values())
    held = result[result["held_pair"] != "target_source_only_fit"]
    rows: list[dict[str, object]] = []
    for (precision_target, score), group in held.groupby(
        ["precision_target", "score"], sort=True
    ):
        calibrated = group[group["status"] == "calibrated"].copy()
        row: dict[str, object] = {
            "precision_target": float(precision_target),
            "score": score,
            "source_pairs": len(source),
            "calibrated_pairs": len(calibrated),
            "source_queries": total_queries,
        }
        if calibrated.empty:
            row.update({
                "status": "incomplete",
                "selected_queries": 0,
                "coverage": 0.0,
                "weighted_semantic_precision": float("nan"),
                "weighted_instance_precision": float("nan"),
                "macro_pair_coverage": float("nan"),
                "macro_pair_semantic_precision": float("nan"),
                "pairs_below_precision_target": len(source),
            })
        else:
            selected = calibrated["held_queries"].astype(float)
            selected_queries = int(selected.sum())
            semantic_correct = float(
                (selected * calibrated["held_semantic_precision"].astype(float)).sum()
            )
            instance_correct = float(
                (selected * calibrated["held_instance_precision"].astype(float)).sum()
            )
            row.update({
                "status": "complete" if len(calibrated) == len(source) else "incomplete",
                "selected_queries": selected_queries,
                "coverage": selected_queries / total_queries,
                "weighted_semantic_precision": semantic_correct / selected_queries,
                "weighted_instance_precision": instance_correct / selected_queries,
                "macro_pair_coverage": calibrated["held_coverage"].astype(float).mean(),
                "macro_pair_semantic_precision": calibrated["held_semantic_precision"].astype(float).mean(),
                "pairs_below_precision_target": int(
                    (calibrated["held_semantic_precision"].astype(float) < float(precision_target)).sum()
                    + len(source) - len(calibrated)
                ),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def read(path: Path) -> pd.DataFrame:
    raw_pair = re.sub(r"^\d+_3rscan_consensus_", "", path.parent.name).replace("_vitg_diag", "")
    match = re.search(r"(0cac|20c993|4aca|5630|6bde|751a|95be|c670|chairs|pair0[2-8])", raw_pair)
    pair = match.group(1) if match else ("target" if "target" in raw_pair else raw_pair)
    frame = pd.read_csv(path).assign(pair=pair)
    if "query_evaluable" in frame:
        frame = frame[frame["query_evaluable"].astype(bool)].reset_index(drop=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, nargs="+", default=[0.90, 0.95])
    parser.add_argument("--min-calibration-queries", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = {frame.pair.iloc[0]: frame for path in args.source for frame in [read(path)]}
    target = read(args.target)
    rows: list[dict[str, object]] = []
    for requested in args.precision_target:
        for held_pair, held in sorted(source.items()):
            train = pd.concat([frame for pair, frame in source.items() if pair != held_pair], ignore_index=True)
            for score_name in SCORES:
                threshold_fit = fit_threshold(train, score_map(train)[score_name], requested, args.min_calibration_queries)
                row: dict[str, object] = {
                    "precision_target": requested,
                    "held_pair": held_pair,
                    "score": score_name,
                    "train_pairs": len(source) - 1,
                }
                if threshold_fit is None:
                    row.update({"status": "no_source_threshold", "threshold": np.nan})
                else:
                    threshold, train_count, train_precision = threshold_fit
                    row.update({
                        "status": "calibrated",
                        "threshold": threshold,
                        "train_queries": train_count,
                        "train_coverage": train_count / len(train),
                        "train_semantic_precision": train_precision,
                    })
                    row.update({f"held_{key}": value for key, value in evaluate(held, score_map(held)[score_name], threshold).items()})
                    row.update({f"target_{key}": value for key, value in evaluate(target, score_map(target)[score_name], threshold).items()})
                rows.append(row)

        # Fit once on all source pairs for the source-only target evaluation.
        train = pd.concat(source.values(), ignore_index=True)
        for score_name in SCORES:
            threshold_fit = fit_threshold(train, score_map(train)[score_name], requested, args.min_calibration_queries)
            row = {
                "precision_target": requested,
                "held_pair": "target_source_only_fit",
                "score": score_name,
                "train_pairs": len(source),
            }
            if threshold_fit is None:
                row.update({"status": "no_source_threshold", "threshold": np.nan})
            else:
                threshold, train_count, train_precision = threshold_fit
                row.update({
                    "status": "calibrated",
                    "threshold": threshold,
                    "train_queries": train_count,
                    "train_coverage": train_count / len(train),
                    "train_semantic_precision": train_precision,
                })
                row.update({f"target_{key}": value for key, value in evaluate(target, score_map(target)[score_name], threshold).items()})
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "leave_one_pair_out.csv", index=False)
    summarize_cross_fitted(result, source).to_csv(
        args.output_dir / "cross_fitted_summary.csv", index=False
    )
    print(result[result["held_pair"] == "target_source_only_fit"].to_string(index=False))


if __name__ == "__main__":
    main()
