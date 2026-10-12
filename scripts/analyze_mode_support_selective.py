#!/usr/bin/env python3
"""Test mode-support as a source-only selective signal for multi-motion.

The support statistic is computed from unlabeled consensus hypotheses. Semantic
labels are used only to fit a relation-held-out threshold on source pairs;
physical-instance transfer is reported as an evaluation-only secondary metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def add_support(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for k in range(4):
        column = f"hypothesis_{k}_inlier"
        if column not in frame:
            frame[column] = False
        frame[column] = frame[column].fillna(False).astype(bool)
    support = {}
    for (pair, mode), group in frame.groupby(["pair", "assigned_mode"], sort=False):
        mode = int(mode)
        if mode >= 0:
            support[(str(pair), mode)] = int(group[f"hypothesis_{mode}_inlier"].sum())
    pair_sizes = frame.groupby("pair").size().to_dict()
    frame["mode_support"] = [support.get((str(pair), int(mode)), 0) for pair, mode in zip(frame.pair, frame.assigned_mode)]
    frame["mode_support_fraction"] = [value / pair_sizes[pair] for value, pair in zip(frame.mode_support, frame.pair)]
    frame["accepted"] = frame.consensus_inlier.astype(bool)
    frame["source_evaluable"] = frame.query_evaluable.astype(bool)
    frame["physical_evaluable"] = frame.source_evaluable & frame.official_rigid_reference_instance_id.ge(0)
    return frame


def fit_threshold(train: pd.DataFrame, target_precision: float, min_queries: int) -> dict[str, float | int] | None:
    train = train[train.accepted & train.source_evaluable]
    if train.empty:
        return None
    values = np.sort(train.mode_support_fraction.unique())
    candidates = []
    for threshold in values:
        selected = train.mode_support_fraction >= threshold
        count = int(selected.sum())
        if count < min_queries:
            continue
        precision = float(train.loc[selected, "semantic_correct"].mean())
        if precision >= target_precision:
            candidates.append((count, float(threshold), precision))
    if not candidates:
        return None
    count, threshold, precision = max(candidates, key=lambda row: (row[0], -row[1]))
    return {"threshold": threshold, "train_queries": count, "train_semantic_precision": precision}


def evaluate(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    pool = frame[frame.source_evaluable]
    selected = pool.accepted & (pool.mode_support_fraction >= threshold)
    count = int(selected.sum())
    physical_pool = pool.physical_evaluable
    physical_correct = selected & pool.official_rigid_instance_correct.astype(bool) & physical_pool
    physical_assigned = selected & physical_pool
    return {
        "selected_queries": count,
        "coverage": count / len(pool) if len(pool) else 0.0,
        "semantic_precision": float(pool.loc[selected, "semantic_correct"].mean()) if count else float("nan"),
        "physical_queries": int(physical_pool.sum()),
        "physical_correct_transfers": int(physical_correct.sum()),
        "physical_transfer_coverage": float(physical_correct.sum() / physical_pool.sum()) if physical_pool.any() else float("nan"),
        "physical_assigned_precision": float(physical_correct.sum() / physical_assigned.sum()) if physical_assigned.any() else float("nan"),
    }


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # Each diagnostic export contains several arms.  The hypothesis columns
    # and assigned mode are defined for the multi-motion arm; mixing arms would
    # inflate support counts and make the statistic unavailable at inference.
    if "arm" in frame.columns:
        frame = frame[frame.arm.eq("multi_motion")].copy()
    return add_support(frame)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-target", type=float, nargs="+", default=[0.90, 0.95])
    parser.add_argument("--min-calibration-queries", type=int, default=100)
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.semantic_target):
        raise ValueError("semantic targets must lie in (0,1)")
    frame = read(args.diagnostics)
    # The query-level diagnostic export predates the summary export and has no
    # role column.  The protocol freezes the literal ``target`` pair as the
    # held-out relation; every other pair is a source relation.
    source_pairs = sorted(str(pair) for pair in frame.loc[frame.pair.ne("target"), "pair"].unique())
    target = frame[frame.pair.eq("target")].copy()
    source = {pair: frame[frame.pair.eq(pair)].copy() for pair in source_pairs}
    rows = []
    for precision_target in args.semantic_target:
        for held_pair in source_pairs:
            train = pd.concat([source[pair] for pair in source_pairs if pair != held_pair], ignore_index=True)
            fit = fit_threshold(train, precision_target, args.min_calibration_queries)
            row = {"semantic_target": precision_target, "held_pair": held_pair, "status": "calibrated" if fit else "no_threshold"}
            if fit:
                row.update(fit)
                row.update({f"held_{key}": value for key, value in evaluate(source[held_pair], fit["threshold"]).items()})
            rows.append(row)
        train = pd.concat(source.values(), ignore_index=True)
        fit = fit_threshold(train, precision_target, args.min_calibration_queries)
        row = {"semantic_target": precision_target, "held_pair": "target_source_only_fit", "status": "calibrated" if fit else "no_threshold"}
        if fit:
            row.update(fit)
            row.update({f"target_{key}": value for key, value in evaluate(target, fit["threshold"]).items()})
        rows.append(row)
    result = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "leave_one_relation_out.csv", index=False)
    calibrated = result[(result.status == "calibrated") & (result.held_pair != "target_source_only_fit")]
    summaries = []
    for target_precision, group in calibrated.groupby("semantic_target"):
        summaries.append({
            "semantic_target": target_precision,
            "calibrated_relations": int(len(group)),
            "held_relation_mean_coverage": float(group.held_coverage.mean()),
            "held_relation_mean_semantic_precision": float(group.held_semantic_precision.mean()),
            "held_relation_mean_physical_transfer_coverage": float(group.held_physical_transfer_coverage.dropna().mean()),
            "held_relation_mean_physical_assigned_precision": float(group.held_physical_assigned_precision.dropna().mean()),
            "relations_below_semantic_target": int((group.held_semantic_precision < target_precision).sum()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "cross_fitted_summary.csv", index=False)
    target_rows = result[result.held_pair == "target_source_only_fit"]
    target_rows.to_csv(args.output_dir / "target_source_fit.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "signal": "selected-mode consensus support divided by all depth-valid diagnostic queries in relation",
        "arms": "multi_motion only",
        "labels_used_for_threshold_fit": "source semantic_correct only",
        "physical_instance_used_for_threshold_fit": False,
        "source_relations": len(source_pairs),
        "target_present": bool(not target.empty),
        "semantic_targets": args.semantic_target,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(target_rows.to_string(index=False))


if __name__ == "__main__":
    main()
