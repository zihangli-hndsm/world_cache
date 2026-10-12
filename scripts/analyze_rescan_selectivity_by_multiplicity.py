#!/usr/bin/env python3
"""Stratify held-out score-only abstention by same-class source multiplicity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261008
BOOTSTRAP_N = 20_000
BINS = ((1, 1, "1"), (2, 3, "2--3"), (4, 5, "4--5"), (6, np.inf, "6+"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/reports/20261006_rescan_score_margin_selectivity_allqueries/lopo_selectivity_predictions.csv"),
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("outputs/reports/20261006_rescan_dinog_scoreconfidence_corrected/query_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/20261006_rescan_selectivity_by_multiplicity"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    predictions = pd.read_csv(rooted(args.predictions))
    predictions = predictions[
        predictions.population.eq("all_queries")
        & predictions.method_fold.eq("dino_synthetic_views")
        & predictions.signal.eq("top1_similarity")
        & predictions.train_acceptance_target.eq(0.25)
        & predictions.is_target_present
    ].copy()
    keys = ["scene", "target_scan", "source_scan", "target_instance_id", "target_sample_index"]
    query_metrics = pd.read_csv(rooted(args.queries))
    query_metrics = query_metrics[
        query_metrics.method.eq("dino_synthetic_views")
    ][keys + ["source_class_instance_count_full"]]
    predictions = predictions.merge(query_metrics, on=keys, validate="many_to_one")
    if predictions.physical_top1.isna().any() or predictions.accepted.isna().any():
        raise ValueError("missing outcome or acceptance decision")
    predictions["multiplicity_bin"] = pd.cut(
        predictions.source_class_instance_count_full,
        bins=[0, 1, 3, 5, np.inf],
        labels=[label for _, _, label in BINS],
    )

    output = rooted(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    site_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED)
    for low, high, label in BINS:
        group = predictions[predictions.multiplicity_bin.eq(label)]
        for site, part in group.groupby("scene", sort=True):
            accepted = part[part.accepted]
            site_rows.append(
                {
                    "multiplicity_bin": label,
                    "scene": site,
                    "queries": len(part),
                    "accepted_queries": len(accepted),
                    "coverage": float(part.accepted.mean()),
                    "baseline_precision": float(part.physical_top1.mean()),
                    "accepted_precision": float(accepted.physical_top1.mean()) if len(accepted) else np.nan,
                    "precision_delta": float(accepted.physical_top1.mean() - part.physical_top1.mean()) if len(accepted) else np.nan,
                }
            )
        site = pd.DataFrame([row for row in site_rows if row["multiplicity_bin"] == label])
        defined = site.dropna(subset=["precision_delta"])
        deltas = defined.precision_delta.to_numpy(dtype=float)
        if len(deltas):
            draws = deltas[rng.integers(0, len(deltas), size=(BOOTSTRAP_N, len(deltas)))].mean(axis=1)
            ci = np.quantile(draws, [0.025, 0.975])
            delta_mean = float(deltas.mean())
        else:
            ci = [np.nan, np.nan]
            delta_mean = np.nan
        summary_rows.append(
            {
                "multiplicity_bin": label,
                "target_present_queries": len(group),
                "sites_with_queries": int(group.scene.nunique()),
                "sites_with_accepted_queries": len(defined),
                "mean_site_coverage_defined": float(site.coverage.mean()),
                "mean_site_baseline_precision": float(site.baseline_precision.mean()),
                "mean_site_accepted_precision_defined": float(defined.accepted_precision.mean()) if len(defined) else np.nan,
                "mean_site_precision_delta_defined": delta_mean,
                "precision_delta_ci_low": float(ci[0]),
                "precision_delta_ci_high": float(ci[1]),
                "sites_improved": int((deltas > 0).sum()),
                "site_bootstrap_seed": SEED,
                "site_bootstrap_draws": BOOTSTRAP_N,
            }
        )

    site_df = pd.DataFrame(site_rows)
    summary = pd.DataFrame(summary_rows)
    site_df.to_csv(output / "site_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    lines = [
        "# Score-only abstention stratified by source multiplicity",
        "",
        "This post-hoc diagnostic uses the all-query 25%-target score thresholds already fit from the other ten sites in each outer leave-one-site-out fold. It then conditions the held-out evaluation on target-present queries and stratifies by the annotation-derived number of same-class source instances. Thresholds and acceptance do not use these labels or strata. This is not a deployable unlabeled stratifier or an independent validation set.",
        "",
        "Within each site and stratum, precision change is accepted physical top-1 precision minus the unfiltered precision. Reported intervals resample the available site-fold deltas; folds with no accepted query in that stratum have undefined accepted precision and are excluded from the delta summary.",
        "",
        "| Same-class source instances | Target-present queries | Sites with accepted queries | Mean site coverage | Baseline precision | Accepted precision | Delta, pp (95% site bootstrap CI) | Sites improved |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['multiplicity_bin']} | {row['target_present_queries']:,} | {row['sites_with_accepted_queries']}/{row['sites_with_queries']} | "
            f"{row['mean_site_coverage_defined']:.1%} | {row['mean_site_baseline_precision']:.1%} | "
            f"{row['mean_site_accepted_precision_defined']:.1%} | "
            f"{100*row['mean_site_precision_delta_defined']:+.1f} [{100*row['precision_delta_ci_low']:+.1f},{100*row['precision_delta_ci_high']:+.1f}] | "
            f"{row['sites_improved']}/{row['sites_with_accepted_queries']} |"
        )
    lines.extend(
        [
            "",
            "The bins are defined from saved cross-scan annotations and therefore serve only to test whether the score association disappears in high-ambiguity strata. Site counts are small, and the bootstrap intervals are descriptive.",
            "",
            "Per-site values: `site_metrics.csv`; summary: `summary.csv`.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
