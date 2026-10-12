#!/usr/bin/env python3
"""Aggregate source-only candidate-size sweep for tracked instances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_vitl_vitg_candidate_audit import PAIR_TO_G_AUDIT, as_bool


KS = (1, 5, 10, 20, 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relation_rows: dict[int, list[dict[str, object]]] = {k: [] for k in KS}
    summary_paths = []
    for pair in PAIR_TO_G_AUDIT:
        if "target" in pair:
            continue
        pair_dir = args.audit_root / pair
        diag_path = pair_dir / "query_diagnostics.csv"
        summary_path = pair_dir / "summary.csv"
        status_path = pair_dir / "status.json"
        if not (diag_path.is_file() and summary_path.is_file() and status_path.is_file()):
            raise FileNotFoundError(f"incomplete K-sweep output for {pair}")
        diag = pd.read_csv(diag_path)
        summ = pd.read_csv(summary_path)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            raise ValueError(f"incomplete status for {pair}")
        if set(diag.top_k.unique()) != set(KS) or set(summ.top_k.unique()) != set(KS):
            raise ValueError(f"{pair}: expected K values {KS}")
        summary_paths.append(str(summary_path))
        baseline_keys = None
        baseline_links = None
        for k in KS:
            d = diag[diag.top_k == k].sort_values(["query_index", "frame_id"]).reset_index(drop=True)
            keys = d[["query_index", "frame_id", "query_object_id", "mapped_reference_object_id"]]
            if baseline_keys is None:
                baseline_keys = keys
                baseline_links = as_bool(d.physical_instance_evaluable)
            elif not baseline_keys.equals(keys) or not np.array_equal(baseline_links, as_bool(d.physical_instance_evaluable)):
                raise ValueError(f"{pair}: query or instance-link denominator differs across K")

            sem_eval = as_bool(d.query_evaluable)
            linked = as_bool(d.physical_instance_evaluable)
            sem_hit = as_bool(d.candidate_semantic_hit) & sem_eval
            phys_hit = as_bool(d.physical_instance_candidate_hit) & linked
            correct_phys = as_bool(d.consensus_inlier) & as_bool(d.physical_instance_correct) & linked
            correct_sem = as_bool(d.consensus_inlier) & as_bool(d.semantic_correct) & sem_eval
            s = summ[summ.top_k == k].iloc[0]
            strict_pose = bool(float(s.rotation_error_deg) < 5.0 and float(s.translation_error_m) < 0.25)
            relation_rows[k].append({
                "pair": pair,
                "top_k": k,
                "semantic_queries": int(sem_eval.sum()),
                "linked_queries": int(linked.sum()),
                "semantic_hits": int(sem_hit.sum()),
                "physical_hits": int(phys_hit.sum()),
                "physical_correct_inliers": int(correct_phys.sum()),
                "semantic_correct_inliers": int(correct_sem.sum()),
                "evaluable_inliers": int((as_bool(d.consensus_inlier) & sem_eval).sum()),
                "physical_recall": float(phys_hit.sum() / linked.sum()) if linked.sum() else 0.0,
                "physical_correct_inlier_coverage": float(correct_phys.sum() / linked.sum()) if linked.sum() else 0.0,
                "physical_recovery": float(correct_phys.sum() / phys_hit.sum()) if phys_hit.sum() else 0.0,
                "semantic_recall": float(sem_hit.sum() / sem_eval.sum()) if sem_eval.sum() else 0.0,
                "strict_pose": int(strict_pose),
                "inlier_coverage": float(s.inlier_coverage),
                "inlier_semantic_accuracy": float(s.inlier_semantic_accuracy),
            })

    rng = np.random.default_rng(args.seed)
    baseline = {k: relation_rows[k] for k in KS}
    aggregate_rows = []
    for k in KS:
        rows = relation_rows[k]
        n = len(rows)
        linked = sum(int(r["linked_queries"]) for r in rows)
        sem_n = sum(int(r["semantic_queries"]) for r in rows)
        phits = sum(int(r["physical_hits"]) for r in rows)
        recovered = sum(int(r["physical_correct_inliers"]) for r in rows)
        semhits = sum(int(r["semantic_hits"]) for r in rows)
        semantic_correct = sum(int(r["semantic_correct_inliers"]) for r in rows)
        evaluable_inliers = sum(int(r["evaluable_inliers"]) for r in rows)
        aggregate_rows.append({
            "top_k": k,
            "relations": n,
            "semantic_queries": sem_n,
            "linked_instance_queries": linked,
            "physical_instance_recall": phits / linked if linked else 0.0,
            "semantic_recall": semhits / sem_n if sem_n else 0.0,
            "physical_correct_inlier_coverage": recovered / linked if linked else 0.0,
            "physical_recovery_given_candidate": recovered / phits if phits else 0.0,
            "strict_pose_successes": sum(int(r["strict_pose"]) for r in rows),
            "mean_inlier_coverage": float(np.mean([float(r["inlier_coverage"]) for r in rows])),
            "mean_retained_semantic_accuracy": float(np.mean([float(r["inlier_semantic_accuracy"]) for r in rows])),
            "pooled_retained_semantic_accuracy": semantic_correct / evaluable_inliers if evaluable_inliers else 0.0,
        })

    baseline_k_rows = {k: {r["pair"]: r for r in relation_rows[k]} for k in KS}
    for row in aggregate_rows:
        k = int(row["top_k"])
        values = {}
        if k != 20:
            b = baseline_k_rows[20]
            cur = baseline_k_rows[k]
            paired = sorted(cur)
            draws = rng.integers(0, len(paired), size=(args.bootstrap_samples, len(paired)))
            for metric, count_key, denom_key in (
                ("physical_recall", "physical_hits", "linked_queries"),
                ("physical_correct_inlier_coverage", "physical_correct_inliers", "linked_queries"),
                ("physical_recovery", "physical_correct_inliers", "physical_hits"),
                ("semantic_recall", "semantic_hits", "semantic_queries"),
            ):
                nums = np.asarray([
                    (int(cur[p][count_key]) / max(int(cur[p][denom_key]), 1)) -
                     (int(b[p][count_key]) / max(int(b[p][denom_key]), 1))
                    for p in paired
                ], dtype=np.float64)
                values[metric] = nums[draws].mean(axis=1)
            row["vs_k20_delta_physical_recall_relation_macro"] = float(np.mean([
                float(cur[p]["physical_recall"]) - float(b[p]["physical_recall"]) for p in paired
            ]))
            for metric, draws_arr in values.items():
                lo, hi = np.quantile(draws_arr, [0.025, 0.975])
                row[f"vs_k20_delta_{metric}_relation_macro"] = float(np.mean(draws_arr))
                row[f"vs_k20_delta_{metric}_pair_bootstrap_95pct"] = [float(lo), float(hi)]
        else:
            for metric in ("physical_recall", "physical_correct_inlier_coverage", "physical_recovery", "semantic_recall"):
                row[f"vs_k20_delta_{metric}_pair_bootstrap_95pct"] = [0.0, 0.0]

    pd.DataFrame(aggregate_rows).to_csv(args.output_dir / "k_sweep_summary.csv", index=False)
    pd.DataFrame([r for k in KS for r in relation_rows[k]]).to_csv(args.output_dir / "per_relation.csv", index=False)
    report = [
        "# Source-only physical-instance candidate-size sweep",
        "",
        "ViT-G/14, blocks (2,20,40); same 13 source query pools; K in {1,5,10,20,50}; 1,500 RANSAC iterations per K; physical links and labels used only post hoc. Relation-level bootstrap resamples the 13 relation pairs.",
        "",
        "| K | Semantic Recall@K | Physical-instance Recall@K | Correct physical inlier coverage | Recovery given physical candidate | Strict pose | Mean coverage / retained semantic accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_rows:
        report.append(
            f"| {r['top_k']} | {r['semantic_recall']:.1%} | {r['physical_instance_recall']:.1%} | "
            f"{r['physical_correct_inlier_coverage']:.1%} | {r['physical_recovery_given_candidate']:.1%} | "
            f"{r['strict_pose_successes']}/13 | {r['mean_inlier_coverage']:.1%} / "
            f"{r['mean_retained_semantic_accuracy']:.1%} relation-macro ({r['pooled_retained_semantic_accuracy']:.1%} pooled) |"
        )
    report += [
        "",
        "K=20 remains the pre-existing operating point; the K sweep is descriptive and does not select a new value. At K=50 versus K=20, the relation-macro paired changes and 95% bootstrap intervals are: physical-instance Recall@K +{:.1f} pp [{:.1f},{:.1f}], correct physical-inlier coverage +{:.1f} pp [{:.1f},{:.1f}], conditional recovery +{:.1f} pp [{:.1f},{:.1f}], and semantic Recall@K +{:.1f} pp [{:.1f},{:.1f}]. Correct physical-inlier coverage requires both a top-K linked-instance hit and unchanged-consensus selection of that instance as an inlier. The source-only relation pool does not estimate independent-site generalization.".format(
            aggregate_rows[-1]["vs_k20_delta_physical_recall_relation_macro"] * 100,
            *[x * 100 for x in aggregate_rows[-1]["vs_k20_delta_physical_recall_pair_bootstrap_95pct"]],
            aggregate_rows[-1]["vs_k20_delta_physical_correct_inlier_coverage_relation_macro"] * 100,
            *[x * 100 for x in aggregate_rows[-1]["vs_k20_delta_physical_correct_inlier_coverage_pair_bootstrap_95pct"]],
            aggregate_rows[-1]["vs_k20_delta_physical_recovery_relation_macro"] * 100,
            *[x * 100 for x in aggregate_rows[-1]["vs_k20_delta_physical_recovery_pair_bootstrap_95pct"]],
            aggregate_rows[-1]["vs_k20_delta_semantic_recall_relation_macro"] * 100,
            *[x * 100 for x in aggregate_rows[-1]["vs_k20_delta_semantic_recall_pair_bootstrap_95pct"]],
        ),
        "",
        "Bootstrap paired deltas versus K=20 are in `k_sweep_summary.csv`; query-level observations and relation-level values are in the accompanying CSV files.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
