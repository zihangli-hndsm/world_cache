#!/usr/bin/env python3
"""Score 2-of-3 seed-voted object predictions from source-LOPO runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def majority_vote(values: np.ndarray, quorum: int = 2) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=np.int64)
    for i, row in enumerate(values):
        valid = row[row > 0]
        if not len(valid):
            continue
        labels, counts = np.unique(valid, return_counts=True)
        winner = int(np.argmax(counts))
        if int(counts[winner]) >= quorum:
            out[i] = int(labels[winner])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-root", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    p.add_argument("--motion-oracle-queries", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap-samples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261002)
    args = p.parse_args()
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("exactly three distinct RANSAC seeds are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oracle = pd.read_csv(args.motion_oracle_queries)
    oracle = oracle[(oracle.split == "source") & (oracle.group == "rigid")]
    expected = oracle[["pair", "query_index", "reference_object_id"]].copy()
    key = ["pair", "query_index"]
    predictions: list[pd.DataFrame] = []
    for seed in args.seeds:
        path = args.seed_root / f"seed_{seed}" / "query_outcomes.csv"
        frame = pd.read_csv(path)
        frame = frame[frame.role == "source"].copy()
        if frame.duplicated(key).any():
            raise ValueError(f"seed {seed}: duplicate source query predictions")
        needed = {"method_prediction_object_id", "baseline_prediction_object_id", "moved_rigid"}
        if not needed <= set(frame.columns):
            raise ValueError(f"seed {seed}: missing prediction columns {needed - set(frame.columns)}")
        frame = frame[key + ["method_prediction_object_id", "baseline_prediction_object_id"]]
        frame = frame.rename(columns={
            "method_prediction_object_id": f"method_seed_{seed}",
            "baseline_prediction_object_id": f"baseline_seed_{seed}",
        })
        predictions.append(frame)

    merged = predictions[0]
    for frame in predictions[1:]:
        merged = merged.merge(frame, on=key, how="outer", validate="one_to_one")
    merged = merged.merge(expected, on=key, how="inner", validate="one_to_one")
    if len(merged) != len(expected) or merged[list(key)].duplicated().any():
        raise ValueError("seed predictions do not align one-to-one with official rigid queries")
    pred_cols = [f"method_seed_{s}" for s in args.seeds]
    base_cols = [f"baseline_seed_{s}" for s in args.seeds]
    method_preds = merged[pred_cols].to_numpy(dtype=np.int64)
    baseline_preds = merged[base_cols].to_numpy(dtype=np.int64)
    method_vote = majority_vote(method_preds)
    baseline_vote = majority_vote(baseline_preds)
    truth = merged.reference_object_id.to_numpy(dtype=np.int64)
    merged["method_vote_prediction"] = method_vote
    merged["baseline_vote_prediction"] = baseline_vote
    merged["method_seed_agreement"] = (method_preds > 0).sum(axis=1)
    merged["method_quorum_agreement"] = np.asarray([
        max((np.count_nonzero(row == label) for label in np.unique(row) if label > 0), default=0)
        for row in method_preds
    ], dtype=np.int64)
    merged["method_vote_assigned"] = method_vote > 0
    merged["baseline_vote_assigned"] = baseline_vote > 0
    merged["method_vote_correct"] = method_vote == truth
    merged["baseline_vote_correct"] = baseline_vote == truth
    merged["method_vote_correct"] &= merged.method_vote_assigned
    merged["baseline_vote_correct"] &= merged.baseline_vote_assigned

    relation_rows = []
    for pair, rows in merged.groupby("pair", sort=True):
        method_preds_pair = rows[pred_cols].to_numpy(dtype=np.int64)
        baseline_preds_pair = rows[base_cols].to_numpy(dtype=np.int64)
        truth_pair = rows.reference_object_id.to_numpy(dtype=np.int64)
        for arm, values in (("multi_motion_vote", method_preds_pair), ("global_ransac_vote", baseline_preds_pair)):
            voted = majority_vote(values)
            assigned = voted > 0
            correct = assigned & (voted == truth_pair)
            relation_rows.append({
                "pair": pair,
                "arm": arm,
                "queries": len(rows),
                "assigned": int(assigned.sum()),
                "correct": int(correct.sum()),
                "coverage": float(correct.mean()),
                "assignment_coverage": float(assigned.mean()),
                "precision_when_assigned": float(correct.sum() / assigned.sum()) if assigned.any() else np.nan,
            })
    relation = pd.DataFrame(relation_rows)
    wide = relation.pivot(index="pair", columns="arm", values=["queries", "correct", "coverage", "assignment_coverage"])
    eligible = wide["queries"]["multi_motion_vote"] > 0
    pairs = wide.index[eligible]
    delta = (
        wide.loc[pairs, ("correct", "multi_motion_vote")].to_numpy(dtype=np.float64)
        - wide.loc[pairs, ("correct", "global_ransac_vote")].to_numpy(dtype=np.float64)
    ) / wide.loc[pairs, ("queries", "multi_motion_vote")].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
    boot = delta[draws].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975])

    vote_summary = {}
    for name, pred in (("method", method_preds), ("baseline", baseline_preds)):
        voted = majority_vote(pred)
        assigned = voted > 0
        correct = assigned & (voted == truth)
        vote_summary[name] = {
            "assigned": int(assigned.sum()),
            "correct": int(correct.sum()),
            "fixed_query_transfer_rate": float(correct.mean()),
            "assignment_coverage": float(assigned.mean()),
            "precision_when_assigned": float(correct.sum() / assigned.sum()) if assigned.any() else None,
        }
    agreement_hist = merged.method_quorum_agreement.value_counts().sort_index().to_dict()
    summary = {
        "source_queries": len(merged),
        "source_relations_with_rigid_queries": len(pairs),
        "seeds": args.seeds,
        "vote_rule": "predict the reference object ID only when at least 2 of 3 seeds agree; otherwise abstain",
        "method_vote": vote_summary["method"],
        "baseline_vote": vote_summary["baseline"],
        "paired_relation_macro_transfer_rate_delta": float(delta.mean()),
        "paired_relation_bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "relations_nonnegative": int((delta >= 0).sum()),
        "method_quorum_agreement_histogram": {str(k): int(v) for k, v in agreement_hist.items()},
        "provenance": "per-seed predictions use source-LOPO selected motion settings; reference object IDs are used only for final scoring",
    }
    merged.to_csv(args.output_dir / "query_vote_diagnostics.csv", index=False)
    relation.to_csv(args.output_dir / "per_relation.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Multi-motion seed-vote diagnostic",
        "",
        f"For each source-LOPO query, we combine three seed-specific RANSAC predictions (seeds {', '.join(map(str, args.seeds))}). The method and its global-RANSAC baseline use the same 2-of-3 object-ID vote; queries without a two-seed agreement abstain. Each seed's secondary-mode settings were selected on the other 15 source relations. The 1,510-query denominator and official object-ID mapping are used only for scoring.",
        "",
        "| Arm | Correct fixed-query transfers | Fixed-query rate | Assignment coverage | Precision when assigned |",
        "|---|---:|---:|---:|---:|",
        f"| Global RANSAC vote | {vote_summary['baseline']['correct']}/{len(merged)} | {vote_summary['baseline']['fixed_query_transfer_rate']:.1%} | {vote_summary['baseline']['assignment_coverage']:.1%} | {vote_summary['baseline']['precision_when_assigned']:.1%} |",
        f"| Multi-motion vote | {vote_summary['method']['correct']}/{len(merged)} | {vote_summary['method']['fixed_query_transfer_rate']:.1%} | {vote_summary['method']['assignment_coverage']:.1%} | {vote_summary['method']['precision_when_assigned']:.1%} |",
        "",
        f"Paired relation-macro fixed-query-rate difference is {delta.mean()*100:+.2f} pp (95% relation-bootstrap CI [{ci[0]*100:+.2f},{ci[1]*100:+.2f}] pp); nonnegative relations: {int((delta >= 0).sum())}/{len(delta)}. Method seed agreement counts (max matching predictions per query): {agreement_hist}.",
        "",
        "This tests whether seed agreement can stabilize a label-free test-time prediction. It remains an internal cross-fit on official 3RScan train relations, multiplies inference cost by three, and uses relation-specific source-LOPO settings. It is not independent-site validation; the target relation was not included in this source-only vote report.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
