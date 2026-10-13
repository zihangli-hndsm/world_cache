#!/usr/bin/env python3
"""Joint source-relation / validation-environment bootstrap for score transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.025, 0.5, 0.975])
    return {"q025": float(q[0]), "median": float(q[1]), "q975": float(q[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_3rscan_score_threshold_cluster_bootstrap"))
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--exclude-environment", action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted((root / "outputs/runs").glob("20260922_3rscan_consensus_*_vitg_diag_clean/query_diagnostics.csv"))
    paths = [p for p in paths if p.parent.name != "20260922_3rscan_consensus_target_vitg_diag_clean"]
    if len(paths) != 16:
        raise ValueError(f"expected 16 source relations, found {len(paths)}")
    source_scores = []
    for path in paths:
        d = pd.read_csv(path)
        d = d[d.geometry_weight.eq(0.0) & d.top_k.eq(20) & d.query_evaluable.astype(bool)]
        if d.empty or d.top1_descriptor_score.isna().any():
            raise ValueError(f"invalid source calibration rows in {path}")
        source_scores.append(d.top1_descriptor_score.to_numpy(dtype=float))

    pred_path = root / "outputs/reports/20261006_3rscan_score_selectivity_lopo/crossfit_predictions.csv.gz"
    val = pd.read_csv(pred_path)
    val = val[
        val.calibration_protocol.eq("source_train_fixed")
        & val.signal.eq("DINO top-1 similarity")
        & val.train_acceptance_target.eq(0.25)
        & val.physical_eval.astype(bool)
    ].copy()
    if val.environment_id.nunique() != 47 or len(val) != 98221:
        raise ValueError(f"unexpected validation population: {len(val)} rows / {val.environment_id.nunique()} environments")
    if val.physical_correct.isna().any() or val.top1_descriptor_score.isna().any():
        raise ValueError("missing validation outcomes or scores")

    absent = set(args.exclude_environment) - set(val.environment_id.unique())
    if absent:
        raise ValueError(f"excluded environments not present: {sorted(absent)}")
    if args.exclude_environment:
        val = val[~val.environment_id.isin(args.exclude_environment)].copy()
    env_names = np.array(sorted(val.environment_id.unique()))
    env_arrays = {}
    for name, group in val.groupby("environment_id", sort=True):
        env_arrays[name] = (
            group.top1_descriptor_score.to_numpy(dtype=float),
            group.physical_correct.to_numpy(dtype=bool),
        )

    fixed_threshold = float(np.quantile(np.concatenate(source_scores), 0.75))
    fixed_accept = val.top1_descriptor_score.to_numpy(dtype=float) >= fixed_threshold
    fixed_coverage = float(fixed_accept.mean())
    fixed_precision = float(val.loc[fixed_accept, "physical_correct"].astype(bool).mean())
    base_by_env = np.array([
        env_arrays[name][1].mean() for name in env_names
    ])
    fixed_delta_by_env = []
    fixed_cov_by_env = []
    for name in env_names:
        scores, correct = env_arrays[name]
        keep = scores >= fixed_threshold
        fixed_cov_by_env.append(float(keep.mean()))
        if keep.any():
            fixed_delta_by_env.append(float(correct[keep].mean() - correct.mean()))

    rng = np.random.default_rng(20261006)
    draws = np.empty((args.replicates, 4), dtype=float)
    n_rel = len(source_scores)
    for b in range(args.replicates):
        sampled_relations = rng.integers(0, n_rel, size=n_rel)
        threshold = float(np.quantile(np.concatenate([source_scores[i] for i in sampled_relations]), 0.75))
        sampled_envs = rng.integers(0, len(env_names), size=len(env_names))
        deltas = []
        coverages = []
        accepted_precisions = []
        for eidx in sampled_envs:
            scores, correct = env_arrays[env_names[eidx]]
            keep = scores >= threshold
            coverages.append(float(keep.mean()))
            if keep.any():
                accepted_precisions.append(float(correct[keep].mean()))
                deltas.append(float(correct[keep].mean() - correct.mean()))
        draws[b] = (
            threshold,
            float(np.mean(coverages)),
            float(np.mean(accepted_precisions)),
            float(np.mean(deltas)),
        )

    draw_frame = pd.DataFrame(draws, columns=["threshold", "environment_macro_coverage", "environment_macro_accepted_precision", "environment_macro_precision_delta"])
    draw_frame.to_csv(output / "joint_bootstrap_draws.csv.gz", index=False, compression="gzip")
    summary = {
        "source_relations": len(source_scores),
        "source_queries": int(sum(map(len, source_scores))),
        "validation_environments": int(len(env_names)),
        "excluded_validation_environments": sorted(args.exclude_environment),
        "validation_physical_queries": int(len(val)),
        "bootstrap_replicates": int(args.replicates),
        "bootstrap_unit": f"resample 16 source relations with replacement to refit the pooled query-score 75th percentile, and independently resample the {len(env_names)} included validation environments with replacement",
        "fixed_fit": {
            "threshold": fixed_threshold,
            "pooled_validation_coverage": fixed_coverage,
            "pooled_validation_accepted_precision": fixed_precision,
            "environment_macro_coverage": float(np.mean(fixed_cov_by_env)),
            "environment_macro_baseline_precision": float(np.mean(base_by_env)),
            "environment_macro_accepted_precision": float(np.mean([env_arrays[n][1][env_arrays[n][0] >= fixed_threshold].mean() for n in env_names if (env_arrays[n][0] >= fixed_threshold).any()])),
            "environment_macro_precision_delta": float(np.mean(fixed_delta_by_env)),
        },
        "joint_bootstrap_95_percentile_intervals": {
            col: quantiles(draw_frame[col].to_numpy())
            for col in draw_frame.columns
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    s = summary["joint_bootstrap_95_percentile_intervals"]
    report = f"""# Joint cluster bootstrap for source-fitted DINO score abstention

The source-fitted threshold was recomputed in each bootstrap draw by resampling the 16 source relations with replacement, pooling their query scores, and taking the 75th percentile. Independently, the {len(env_names)} included validation environments were resampled with replacement. Metrics use {len(val):,} physically evaluable validation queries. Excluded environments: {', '.join(args.exclude_environment) if args.exclude_environment else 'none'}. This propagates threshold-fit and validation-environment sampling variability; it does not correct post-hoc exploration, annotation-conditioned query selection, or dataset-specific dependence.

| Quantity | Fixed point estimate | Joint bootstrap 95% percentile interval |
|---|---:|---:|
| Source threshold | {fixed_threshold:.4f} | [{s['threshold']['q025']:.4f}, {s['threshold']['q975']:.4f}] |
| Environment-macro physical coverage | {summary['fixed_fit']['environment_macro_coverage']:.1%} | [{s['environment_macro_coverage']['q025']:.1%}, {s['environment_macro_coverage']['q975']:.1%}] |
| Environment-macro accepted precision | {summary['fixed_fit']['environment_macro_accepted_precision']:.1%} | [{s['environment_macro_accepted_precision']['q025']:.1%}, {s['environment_macro_accepted_precision']['q975']:.1%}] |
| Environment-macro precision gain | {summary['fixed_fit']['environment_macro_precision_delta']*100:+.1f} pp | [{s['environment_macro_precision_delta']['q025']*100:+.1f}, {s['environment_macro_precision_delta']['q975']*100:+.1f}] pp |

The earlier environment-only interval conditions on the point estimate threshold. This joint interval is the primary uncertainty summary for the fixed source-fit gate. Bootstrap draws and exact summary are saved alongside this report.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
