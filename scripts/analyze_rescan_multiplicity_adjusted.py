#!/usr/bin/env python3
"""Site/class-adjusted descriptive check of Rescan identity-gap multiplicity.

This is an exploratory fixed-effects linear-probability analysis of the
query-level semantic-only error indicator. It is not a causal model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SITES = ["scene_a", *[f"scene_{name}" for name in "defghijklm"]]
BINS = ["1", "2-3", "4-5", "6+"]


def load_queries(root: Path, include_render_pilots: bool = False) -> pd.DataFrame:
    paths = [
        root / "outputs/reports/20261006_rescan_dinog_querydiagnostics_bm/query_metrics.csv",
        root / "outputs/reports/20261006_rescan_scene_a_external/dino_g_querydiagnostics/query_metrics.csv",
    ]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    sites = ["scene_a", *[f"scene_{name}" for name in ("bcdefghijklm" if include_render_pilots else "defghijklm")]]
    frame = frame[
        frame["scene"].isin(sites)
        & frame["target_present_in_full_source"].astype(bool)
    ].copy()
    edges = [0, 1, 3, 5, np.inf]
    frame["multiplicity_bin"] = pd.cut(
        frame["source_class_instance_count_full"],
        bins=edges,
        labels=BINS,
        right=True,
    ).astype(str)
    expected_sites = 13 if include_render_pilots else 11
    if frame["scene"].nunique() != expected_sites or set(frame["multiplicity_bin"]) != set(BINS):
        raise ValueError(f"Expected all {expected_sites} sites and four multiplicity bins")
    # The one-instance stratum has a structural zero for semantic-only errors.
    # Exclude it from the adjusted contrast among nontrivial multiplicities.
    return frame[frame["multiplicity_bin"] != "1"].reset_index(drop=True)


def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    sites = sorted(frame["scene"].unique())
    classes = sorted(frame["target_class_id"].unique())
    columns = [np.ones(len(frame), dtype=np.float64)]
    names = ["intercept"]
    for value in sites[1:]:
        columns.append((frame["scene"].to_numpy() == value).astype(float))
        names.append(f"site[{value}]")
    for value in classes[1:]:
        columns.append((frame["target_class_id"].to_numpy() == value).astype(float))
        names.append(f"class[{value}]")
    for value in ("4-5", "6+"):
        columns.append((frame["multiplicity_bin"].to_numpy() == value).astype(float))
        names.append(f"multiplicity[{value}]")
    matrix = np.column_stack(columns)
    site_index = pd.Categorical(frame["scene"], categories=sites).codes
    return matrix, names, site_index


def fit_with_leave_one_site_out(
    frame: pd.DataFrame,
    outcome: str,
) -> tuple[np.ndarray, list[np.ndarray], list[str], list[str], list[str]]:
    matrix, names, _ = design_matrix(frame)
    y = frame[outcome].astype(float).to_numpy()
    if int(np.linalg.matrix_rank(matrix)) != matrix.shape[1]:
        raise ValueError("Full-sample design is rank deficient")
    beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
    loo: list[np.ndarray] = []
    estimable: list[str] = []
    deficient: list[str] = []
    for site in sorted(frame["scene"].unique()):
        subset = frame[frame["scene"] != site].reset_index(drop=True)
        x, loo_names, _ = design_matrix(subset)
        if int(np.linalg.matrix_rank(x)) != x.shape[1]:
            deficient.append(site)
            continue
        b = np.linalg.lstsq(x, subset[outcome].astype(float).to_numpy(), rcond=None)[0]
        loo.append(np.array([b[loo_names.index("multiplicity[4-5]")],
                             b[loo_names.index("multiplicity[6+]")]]))
        estimable.append(site)
    return beta, loo, estimable, deficient, names


def bootstrap_rank_support(frame: pd.DataFrame, n_draws: int = 40_000, seed: int = 20261006) -> int:
    """Count cluster resamples retaining all multiplicity bins and full rank."""
    matrix, _, site_index = design_matrix(frame)
    n_sites = int(site_index.max()) + 1
    x_by_site = np.stack(
        [matrix[site_index == site].T @ matrix[site_index == site] for site in range(n_sites)]
    )
    bins = frame["multiplicity_bin"].to_numpy()
    seen: dict[tuple[int, ...], bool] = {}
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(n_draws):
        counts = np.bincount(rng.integers(0, n_sites, size=n_sites), minlength=n_sites)
        active = tuple(np.flatnonzero(counts))
        if active not in seen:
            active_mask = np.isin(site_index, active)
            has_all_bins = set(bins[active_mask]) == set(BINS[1:])
            xtx = x_by_site[list(active)].sum(axis=0)
            seen[active] = has_all_bins and np.linalg.matrix_rank(xtx) == matrix.shape[1]
        valid += int(seen[active])
    return valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/20261006_rescan_multiplicity_adjusted"),
    )
    parser.add_argument("--include-render-pilots", action="store_true", help="include scenes B/C for a labeled all-site sensitivity")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    frame = load_queries(root, include_render_pilots=args.include_render_pilots)
    support_frame = frame[frame.method == "dino_synthetic_views"]
    class_bin = support_frame.groupby(["target_class_id", "multiplicity_bin"], observed=True).size().unstack(fill_value=0)
    for bucket in BINS:
        if bucket not in class_bin:
            class_bin[bucket] = 0
    class_bin = class_bin[BINS]
    common_classes = class_bin.index[(class_bin["2-3"] > 0) & (class_bin["6+"] > 0)]
    common_class_query_rows = int(class_bin.loc[common_classes, ["2-3", "6+"]].to_numpy().sum()) if len(common_classes) else 0
    contrast_query_rows = int(class_bin[["2-3", "6+"]].to_numpy().sum())
    site_class_bin = support_frame.groupby(["scene", "target_class_id", "multiplicity_bin"], observed=True).size().unstack(fill_value=0)
    for bucket in BINS:
        if bucket not in site_class_bin:
            site_class_bin[bucket] = 0
    site_class_bin = site_class_bin[BINS]
    site_class_overlap_cells = int(((site_class_bin["2-3"] > 0) & (site_class_bin["6+"] > 0)).sum())
    site_class_multi_bin_cells = int(((site_class_bin > 0).sum(axis=1) > 1).sum())
    class_bin.rename_axis(columns="multiplicity_bin").reset_index().to_csv(output / "class_multiplicity_support.csv", index=False)
    site_class_bin.rename_axis(index=["scene", "target_class_id"], columns="multiplicity_bin").reset_index().to_csv(output / "site_class_multiplicity_support.csv", index=False)
    rows = []
    failures = {}
    bootstrap_support = None
    for method in ["dino_synthetic_views", "rgb_color"]:
        subset = frame[frame["method"] == method].reset_index(drop=True)
        if method == "dino_synthetic_views":
            bootstrap_support = bootstrap_rank_support(subset)
        beta, loo, estimable, deficient, names = fit_with_leave_one_site_out(
            subset,
            "semantic_only_hit_top20",
        )
        failures[method] = {"estimable_sites": estimable, "rank_deficient_sites": deficient}
        for term, label in [(names.index("multiplicity[4-5]"), "4-5 vs 2-3"),
                            (names.index("multiplicity[6+]"), "6+ vs 2-3")]:
            idx = 0 if label.startswith("4-5") else 1
            values = np.array([row[idx] for row in loo], dtype=float) * 100
            rows.append(
                {
                    "method": method,
                    "contrast": label,
                    "adjusted_risk_difference_pp": float(beta[term] * 100),
                    "leave_one_site_out_min_pp": float(values.min()) if len(values) else None,
                    "leave_one_site_out_max_pp": float(values.max()) if len(values) else None,
                    "queries": int(len(subset)),
                    "site_clusters": int(subset["scene"].nunique()),
                    "classes": int(subset["target_class_id"].nunique()),
                    "leave_one_site_out_estimable": len(values),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output / "adjusted_contrasts.csv", index=False)
    summary = {
        "estimand": "query-level semantic-only-hit risk difference among multiplicity >=2",
        "reference_bin": "2-3 source instances of query class",
        "adjustment": ["site fixed effects", "query semantic-class fixed effects"],
        "sensitivity": "leave-one-site-out refit; ranges are not confidence intervals",
        "causal": False,
        "queries_per_arm": int(frame["method"].value_counts().min()),
        "site_clusters": sorted(frame["scene"].unique()),
        "class_ids": sorted(int(value) for value in frame["target_class_id"].unique()),
        "support_overlap": {
            "class_ids_with_both_2_3_and_6_plus": [int(value) for value in common_classes],
            "class_count_with_both_bins": int(len(common_classes)),
            "class_count_total": int(class_bin.shape[0]),
            "queries_in_common_classes_across_2_3_and_6_plus": common_class_query_rows,
            "queries_in_2_3_and_6_plus_contrast_total": contrast_query_rows,
            "common_class_query_share_of_contrast": common_class_query_rows / contrast_query_rows if contrast_query_rows else None,
            "site_class_cells_with_both_2_3_and_6_plus": site_class_overlap_cells,
            "site_class_cells_with_multiple_nontrivial_bins": site_class_multi_bin_cells,
        },
        "leave_one_site_out_rank_deficiency": failures,
        "site_bootstrap_full_rank_support": {
            "valid_resamples": bootstrap_support,
            "total_resamples": 40000,
            "seed": 20261006,
            "required_valid_for_reported_interval": 2000,
        },
        "contrasts": result.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Rescan identity-gap multiplicity: site/class-adjusted sensitivity check",
        "",
        "Among target-present queries with at least two source instances of the query class, we fit a query-level linear-probability model for a semantic hit without the physical instance. The model includes site and semantic-class fixed effects; the reference is multiplicity 2--3. We report leave-one-site-out ranges as a sensitivity check, not confidence intervals.",
        "",
        f"This remains post-hoc descriptive analysis, not a causal estimate. Object size, target-scan composition, annotation density, and other site-level differences are not controlled. The one-instance bin is omitted because semantic-only error is structurally impossible there. Sites included: {', '.join(sorted(frame.scene.unique()))}. Render pilots B/C are {'included' if args.include_render_pilots else 'excluded'}. Only {bootstrap_support}/40,000 site-cluster bootstrap draws retained full-rank site/class/multiplicity overlap, so no bootstrap interval is reported.",
        f"Support audit: only {len(common_classes)} of {class_bin.shape[0]} semantic classes appear in both the 2--3 and 6+ multiplicity bins; those classes account for {common_class_query_rows:,}/{contrast_query_rows:,} ({common_class_query_rows / contrast_query_rows:.1%}) queries in the two-bin contrast. No site--class cells contain observations from multiple nontrivial bins ({site_class_multi_bin_cells} overlapping cells). The fixed-effects coefficient therefore relies on additive site/class assumptions and does not estimate a within-site/class matched contrast.",
        "",
        "| Method | Adjusted contrast | Risk difference (pp) | Leave-one-site-out range (pp) | Queries | Sites estimable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['contrast']} | {row['adjusted_risk_difference_pp']:.1f} | "
            f"[{row['leave_one_site_out_min_pp']:.1f},{row['leave_one_site_out_max_pp']:.1f}] | {row['queries']:,} | "
            f"{row['leave_one_site_out_estimable']}/{row['site_clusters']} |"
        )
    lines.extend(
        [
            "",
            "The full fixed-effects specification and support diagnostics are in `summary.json`; machine-readable contrasts are in `adjusted_contrasts.csv`.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
