#!/usr/bin/env python3
"""Evaluate label-free score-margin abstention with leave-one-site-out Rescan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


SITES = ["scene_a", *[f"scene_{name}" for name in "defghijklm"]]
METHODS = ("dino_synthetic_views", "rgb_color")
SIGNALS = ("top1_top2_margin", "top1_similarity")
TRAIN_ACCEPTANCE_TARGETS = (0.50, 0.25, 0.10, 0.05)
BOOTSTRAP_N = 20_000
SEED = 20261007


def auc_score(score: np.ndarray, label: np.ndarray) -> float | None:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    positive = label.sum()
    negative = len(label) - positive
    if positive == 0 or negative == 0:
        return None
    ranks = rankdata(score, method="average")
    return float((ranks[label].sum() - positive * (positive + 1) / 2) / (positive * negative))


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--query-metrics",
        type=Path,
        default=Path("outputs/reports/20261006_rescan_dinog_scoreconfidence_corrected/query_metrics.csv"),
    )
    parser.add_argument(
        "--pair-metrics",
        type=Path,
        default=Path("outputs/reports/20261006_rescan_dinog_scoreconfidence_corrected/pair_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/20261006_rescan_score_margin_selectivity_allqueries"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    query_path = args.query_metrics if args.query_metrics.is_absolute() else root / args.query_metrics
    pair_path = args.pair_metrics if args.pair_metrics.is_absolute() else root / args.pair_metrics
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    queries = pd.read_csv(query_path)
    queries = queries[queries.scene.isin(SITES) & queries.method.isin(METHODS)].copy()
    if queries.scene.nunique() != 11 or queries.target_scan.nunique() != 25:
        raise ValueError("expected 11 sites and 25 target scans")
    queries["is_target_present"] = queries.target_present_in_full_source.astype(bool)
    queries["physical_top1"] = queries.is_target_present & queries.physical_rank_top20.eq(1)
    queries["semantic_top1"] = queries.semantic_rank_top20.eq(1)

    # Independent consistency audit: the rescored pair aggregates must match
    # both previously saved query diagnostics and scene-A pair results exactly.
    old_pairs = pd.concat(
        [
            pd.read_csv(root / "outputs/reports/20261006_rescan_dinog_querydiagnostics_bm/pair_metrics.csv"),
            pd.read_csv(root / "outputs/reports/20261006_rescan_scene_a_external/dino_g_querydiagnostics/pair_metrics.csv"),
        ],
        ignore_index=True,
    )
    new_pairs = pd.read_csv(pair_path)
    cols = ["scene", "source_scan", "target_scan", "method", "k", "target_present_queries", "physical_correct", "semantic_correct_present"]
    old = old_pairs[old_pairs.k.eq(20) & old_pairs.scene.isin(SITES)][cols].sort_values(cols[:5]).reset_index(drop=True)
    new = new_pairs[new_pairs.k.eq(20) & new_pairs.scene.isin(SITES)][cols].sort_values(cols[:5]).reset_index(drop=True)
    keys = cols[:5]
    same_keys = old[keys].astype(str).equals(new[keys].astype(str))
    same_counts = all(
        np.array_equal(old[name].to_numpy(dtype=np.int64), new[name].to_numpy(dtype=np.int64))
        for name in cols[5:]
    ) if len(old) == len(new) else False
    if not same_keys or not same_counts:
        merged = old.merge(new, on=cols[:5], suffixes=("_old", "_new"), how="outer", indicator=True)
        raise ValueError(f"score rerun pair counts differ: {merged[merged._merge != 'both'].head()}")

    # The all-query population is the deployable estimate. The target-present
    # population is retained only as an oracle-conditioned diagnostic.
    populations = {
        "all_queries": queries,
        "target_present_conditioned": queries[queries.is_target_present].copy(),
    }
    fold_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for population, pop_data in populations.items():
        for method in METHODS:
            data = pop_data[pop_data.method == method]
            for signal in SIGNALS:
                for train_acceptance in TRAIN_ACCEPTANCE_TARGETS:
                  for site in SITES:
                    train = data[data.scene != site]
                    test = data[data.scene == site].copy()
                    threshold = float(train[signal].quantile(1.0 - train_acceptance))
                    test["population"] = population
                    test["signal"] = signal
                    test["train_acceptance_target"] = train_acceptance
                    test["threshold"] = threshold
                    test["accepted"] = test[signal] >= threshold
                    test["method_fold"] = method
                    predictions.append(test[["scene", "target_scan", "source_scan", "target_instance_id", "target_sample_index", "population", "signal", "train_acceptance_target", "method_fold", "threshold", "accepted", "is_target_present", "physical_top1", "semantic_top1", signal]])
                    accepted = test.accepted
                    fold_rows.append(
                        {
                            "population": population,
                            "method": method,
                            "signal": signal,
                            "train_acceptance_target": train_acceptance,
                            "held_out_site": site,
                            "threshold_from_other_sites": threshold,
                            "queries": int(len(test)),
                            "target_present_fraction": float(test.is_target_present.mean()),
                            "accepted_queries": int(accepted.sum()),
                            "coverage": float(accepted.mean()),
                            "physical_top1_precision_all": float(test.physical_top1.mean()),
                            "physical_top1_precision_accepted": float(test.loc[accepted, "physical_top1"].mean()) if accepted.any() else None,
                            "physical_top1_recall_retained": float((accepted & test.physical_top1).mean()),
                            "semantic_top1_precision_accepted": float(test.loc[accepted, "semantic_top1"].mean()) if accepted.any() else None,
                            "score_auc_physical_top1": auc_score(test[signal].to_numpy(), test.physical_top1.to_numpy()),
                            "score_auc_semantic_top1": auc_score(test[signal].to_numpy(), test.semantic_top1.to_numpy()),
                        }
                    )
    folds = pd.DataFrame(fold_rows)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(output / "lopo_selectivity_predictions.csv", index=False)
    folds.to_csv(output / "lopo_site_metrics.csv", index=False)

    summaries = []
    for index, ((population, method, signal, train_acceptance), group) in enumerate(folds.groupby(["population", "method", "signal", "train_acceptance_target"], sort=True)):
        coverage = group.coverage.to_numpy()
        precision = group.physical_top1_precision_accepted.to_numpy(dtype=float)
        baseline = group.physical_top1_precision_all.to_numpy()
        valid_precision = np.isfinite(precision)
        delta = precision[valid_precision] - baseline[valid_precision]
        if valid_precision.any():
            p_mean, p_low, p_high = bootstrap_mean(precision[valid_precision], SEED + 4 * index)
            d_mean, d_low, d_high = bootstrap_mean(delta, SEED + 4 * index + 1)
        else:
            p_mean = p_low = p_high = d_mean = d_low = d_high = None
        aucs = group.score_auc_physical_top1.dropna().to_numpy()
        auc_mean, auc_low, auc_high = bootstrap_mean(aucs, SEED + 4 * index + 2)
        summaries.append(
            {
                "method": method,
                "population": population,
                "signal": signal,
                "train_acceptance_target": float(train_acceptance),
                "mean_heldout_coverage": float(coverage.mean()),
                "mean_heldout_physical_top1_precision_all": float(baseline.mean()),
                "mean_heldout_physical_top1_precision_accepted": p_mean,
                "accepted_precision_site_bootstrap_ci95": [p_low, p_high],
                "sites_with_accepted_queries": int(valid_precision.sum()),
                "mean_precision_delta_pp": float(d_mean * 100) if d_mean is not None else None,
                "precision_delta_ci95_pp": [d_low * 100, d_high * 100] if d_low is not None else [None, None],
                "sites_with_positive_precision_delta": int((delta > 0).sum()),
                "sites_with_defined_precision_delta": int(len(delta)),
                "site_macro_auc_physical_top1": auc_mean,
                "site_macro_auc_ci95": [auc_low, auc_high],
                "auc_defined_sites": int(len(aucs)),
            }
        )

    summary = {
        "status": "complete",
        "data": "Rescan rendered point-cloud retrieval; 11 sites (D-M plus A), 25 target scans",
        "ranking_changed": False,
        "pair_level_count_reproduction": {"matched_pairs_k20": int(len(old)), "exact": True},
        "queries_per_method": int(queries[queries.method == METHODS[0]].shape[0]),
        "target_present_fraction": float(queries[queries.method == METHODS[0]].is_target_present.mean()),
        "target_present_queries_per_method": int(queries[queries.method == METHODS[0]].is_target_present.sum()),
        "threshold_protocol": "within each outer held-out site and evaluation population, score thresholds are training-score quantiles corresponding to 50%, 25%, 10%, and 5% intended acceptance; score-only calibration uses no labels. all_queries includes target-absent objects; target_present_conditioned is an oracle-conditioned diagnostic",
        "outcome": "top-1 physical-instance hit; target-absent queries are always incorrect",
        "uncertainty": "site-cluster bootstrap over 11 leave-one-site-out held-out site metrics; intervals are descriptive with few sites",
        "bootstrap": {"samples": BOOTSTRAP_N, "seed": SEED},
        "summaries": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Label-free top-score abstention on Rescan",
        "",
        "The ranking is unchanged. The primary `all_queries` population includes target-absent objects, so those queries count as incorrect physical matches. For each outer held-out site, score thresholds come only from the other ten sites, at score quantiles targeting 50%, 25%, 10%, and 5% training acceptance; no labels select the threshold. The query population still uses valid annotated target instances/classes and the dataset's object-balanced point sampling. `target_present_conditioned` is an oracle-conditioned diagnostic, not the deployment estimate.",
        "",
        "Outcome is physical-instance hit at rank 1. Coverage is the accepted fraction of the held-out query population. Precision intervals bootstrap sites and should be read as descriptive because there are only 11 outer folds; sites with no accepted queries have undefined precision and are omitted from that mean/interval.",
        "",
        "| Population | Train target | Method | Score | Held-out coverage | Physical top-1 precision, all | Physical top-1 precision, accepted (95% site-bootstrap CI) | Precision change (pp, 95% CI) | Sites improved | Site-macro AUC (95% CI) |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        p_ci = row["accepted_precision_site_bootstrap_ci95"]
        d_ci = row["precision_delta_ci95_pp"]
        a_ci = row["site_macro_auc_ci95"]
        p_text = "n/a" if row["mean_heldout_physical_top1_precision_accepted"] is None else (
            f"{100*row['mean_heldout_physical_top1_precision_accepted']:.1f}% "
            f"[{100*p_ci[0]:.1f},{100*p_ci[1]:.1f}]"
        )
        d_text = "n/a" if row["mean_precision_delta_pp"] is None else (
            f"{row['mean_precision_delta_pp']:+.1f} [{d_ci[0]:+.1f},{d_ci[1]:+.1f}]"
        )
        lines.append(
            f"| {row['population']} | {100*row['train_acceptance_target']:.0f}% | {row['method']} | {row['signal']} | {100*row['mean_heldout_coverage']:.1f}% | "
            f"{100*row['mean_heldout_physical_top1_precision_all']:.1f}% | "
            f"{p_text} | {d_text} | "
            f"{row['sites_with_positive_precision_delta']}/{row['sites_with_defined_precision_delta']} | "
            f"{row['site_macro_auc_physical_top1']:.3f} [{a_ci[0]:.3f},{a_ci[1]:.3f}] |"
        )
    lines.extend(
        [
            "",
            "This post-hoc confidence analysis is not a new independent benchmark. Similarity margins measure patch-level score separation, not instance-level uncertainty; outer-site results and the RGB control are reported to reveal transfer limits.",
            "",
            "See `lopo_site_metrics.csv`, `lopo_selectivity_predictions.csv`, and `summary.json` for per-site evidence.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
