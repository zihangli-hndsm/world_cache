#!/usr/bin/env python3
"""Adjust Rescan multiplicity diagnostics for annotated object support/size proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rerank_rescan_fpfh_by_color import read_scene_ply


SITES = ["scene_a", *[f"scene_{name}" for name in "defghijklm"]]
BINS = ["1", "2-3", "4-5", "6+"]
BOOTSTRAP_N = 20_000
QUERY_FILES = [
    Path("outputs/reports/20261006_rescan_dinog_querydiagnostics_bm/query_metrics.csv"),
    Path("outputs/reports/20261006_rescan_scene_a_external/dino_g_querydiagnostics/query_metrics.csv"),
]


def scan_object_support(path: Path, scene: str, scan: str) -> pd.DataFrame:
    points, _, ids, classes = read_scene_ply(path)
    valid = (ids > 0) & (classes > 0)
    ids = ids[valid]
    classes = classes[valid]
    points = points[valid]
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    sorted_classes = classes[order]
    sorted_points = points[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1]
    counts = np.diff(np.r_[starts, len(sorted_ids)])
    min_xyz = np.minimum.reduceat(sorted_points, starts, axis=0)
    max_xyz = np.maximum.reduceat(sorted_points, starts, axis=0)
    classes_by_id = sorted_classes[starts]
    if any(np.unique(sorted_classes[start : start + count]).size != 1 for start, count in zip(starts, counts)):
        raise ValueError(f"an instance has multiple semantic classes: {path}")
    return pd.DataFrame(
        {
            "scene": scene,
            "scan": scan,
            "instance_id": sorted_ids[starts].astype(int),
            "class_id": classes_by_id.astype(int),
            "annotated_vertex_count": counts.astype(int),
            "bbox_diagonal_m": np.linalg.norm(max_xyz - min_xyz, axis=1),
        }
    )


def load_support(data_root: Path, scans: set[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for scene, scan in sorted(scans):
        path = data_root / scene / "gt_segmentation" / f"{scan}.ply"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(scan_object_support(path, scene, scan))
    if not rows:
        raise ValueError("no source/target scans selected")
    return pd.concat(rows, ignore_index=True)


def make_x(frame: pd.DataFrame, covariates: list[str]) -> tuple[np.ndarray, list[str]]:
    sites = sorted(frame.scene.unique())
    classes = sorted(frame.target_class_id.unique())
    cols = [np.ones(len(frame))]
    names = ["intercept"]
    for value in sites[1:]:
        cols.append((frame.scene.to_numpy() == value).astype(float))
        names.append(f"site[{value}]")
    for value in classes[1:]:
        cols.append((frame.target_class_id.to_numpy() == value).astype(float))
        names.append(f"class[{value}]")
    for value in ("4-5", "6+"):
        cols.append((frame.multiplicity_bin.to_numpy() == value).astype(float))
        names.append(f"multiplicity[{value}]")
    for value in covariates:
        cols.append(frame[value].to_numpy(dtype=float))
        names.append(value)
    return np.column_stack(cols), names


def fit_and_loo(frame: pd.DataFrame, covariates: list[str]) -> tuple[np.ndarray, list[np.ndarray], int]:
    x, names = make_x(frame, covariates)
    y = frame.semantic_only_hit_top20.astype(float).to_numpy()
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise ValueError(f"rank deficient full design with covariates {covariates}: {rank}/{x.shape[1]}")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    loo = []
    for site in sorted(frame.scene.unique()):
        subset = frame[frame.scene != site].reset_index(drop=True)
        x_loo, names_loo = make_x(subset, covariates)
        if int(np.linalg.matrix_rank(x_loo)) != x_loo.shape[1]:
            continue
        b = np.linalg.lstsq(x_loo, subset.semantic_only_hit_top20.astype(float).to_numpy(), rcond=None)[0]
        loo.append(
            [b[names_loo.index("multiplicity[4-5]")], b[names_loo.index("multiplicity[6+]")]]
        )
    return beta, loo, rank


def site_bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=(Path.home() / 'data/worldcache_datasets/rescan_subset'))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_rescan_multiplicity_object_support"))
    args = parser.parse_args()
    root = args.root.resolve()
    data_root = args.data_root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    queries = pd.concat([pd.read_csv(root / path) for path in QUERY_FILES], ignore_index=True)
    queries = queries[
        queries.scene.isin(SITES)
        & queries.target_present_in_full_source.astype(bool)
        & queries.method.isin(["dino_synthetic_views", "rgb_color"])
    ].copy()
    queries["multiplicity_bin"] = pd.cut(
        queries.source_class_instance_count_full,
        bins=[0, 1, 3, 5, np.inf],
        labels=BINS,
    ).astype(str)
    queries = queries[queries.multiplicity_bin != "1"].reset_index(drop=True)

    scans = set(zip(queries.scene, queries.source_scan)) | set(zip(queries.scene, queries.target_scan))
    support = load_support(data_root, scans)
    support["log_vertex_count"] = np.log1p(support.annotated_vertex_count)
    support["log_bbox_diagonal"] = np.log1p(support.bbox_diagonal_m.clip(lower=0))

    source = support.rename(
        columns={
            "scan": "source_scan",
            "instance_id": "target_instance_id",
            "class_id": "source_instance_class_id",
            "annotated_vertex_count": "source_vertex_count",
            "bbox_diagonal_m": "source_bbox_diagonal_m",
            "log_vertex_count": "log_source_vertex_count",
            "log_bbox_diagonal": "log_source_bbox_diagonal",
        }
    )[["scene", "source_scan", "target_instance_id", "source_instance_class_id", "source_vertex_count", "source_bbox_diagonal_m", "log_source_vertex_count", "log_source_bbox_diagonal"]]
    target = support.rename(
        columns={
            "scan": "target_scan",
            "instance_id": "target_instance_id",
            "class_id": "target_instance_class_id",
            "annotated_vertex_count": "target_vertex_count",
            "bbox_diagonal_m": "target_bbox_diagonal_m",
            "log_vertex_count": "log_target_vertex_count",
            "log_bbox_diagonal": "log_target_bbox_diagonal",
        }
    )[["scene", "target_scan", "target_instance_id", "target_instance_class_id", "target_vertex_count", "target_bbox_diagonal_m", "log_target_vertex_count", "log_target_bbox_diagonal"]]
    queries = queries.merge(source, on=["scene", "source_scan", "target_instance_id"], validate="many_to_one")
    queries = queries.merge(target, on=["scene", "target_scan", "target_instance_id"], validate="many_to_one")
    if not (queries.target_class_id == queries.target_instance_class_id).all():
        raise ValueError("query class disagrees with target PLY instance class")
    queries["source_target_class_mismatch"] = (
        queries.target_class_id != queries.source_instance_class_id
    ).astype(float)
    mismatch_query_rows = int(queries["source_target_class_mismatch"].sum())
    mismatch_object_pairs = int(
        queries.loc[
            queries["source_target_class_mismatch"].eq(1.0),
            ["scene", "source_scan", "target_scan", "target_instance_id"],
        ].drop_duplicates().shape[0]
    )
    if queries[["source_vertex_count", "target_vertex_count"]].isna().any().any():
        raise ValueError("missing source/target object support")

    # Center/scale continuous controls once so coefficients stay numerically stable.
    covariate_groups = {
        "base": [],
        "target_support": ["log_target_vertex_count"],
        "target_extent": ["log_target_bbox_diagonal"],
        "source_support": ["log_source_vertex_count"],
        "source_extent": ["log_source_bbox_diagonal"],
        "all_support_proxies": [
            "log_target_vertex_count",
            "log_target_bbox_diagonal",
            "log_source_vertex_count",
            "log_source_bbox_diagonal",
        ],
        "all_support_plus_label_consistency": [
            "log_target_vertex_count",
            "log_target_bbox_diagonal",
            "log_source_vertex_count",
            "log_source_bbox_diagonal",
            "source_target_class_mismatch",
        ],
    }
    continuous_controls = {
        name
        for names in covariate_groups.values()
        for name in names
        if name != "source_target_class_mismatch"
    }
    for name in sorted(continuous_controls):
        mean = queries[name].mean()
        sd = queries[name].std()
        queries[name] = (queries[name] - mean) / (sd if sd > 0 else 1.0)

    result_rows = []
    rank_meta = {}
    for method in ("dino_synthetic_views", "rgb_color"):
        method_rows = []
        data = queries[queries.method == method].reset_index(drop=True)
        for model_name, covariates in covariate_groups.items():
            beta, loo, rank = fit_and_loo(data, covariates)
            _, names = make_x(data, covariates)
            contrast_index = names.index("multiplicity[6+]")
            loo_values = np.asarray(loo, dtype=float)[:, 1] * 100
            row = {
                "method": method,
                "model": model_name,
                "six_plus_vs_two_three_risk_difference_pp": float(beta[contrast_index] * 100),
                "leave_one_site_out_min_pp": float(loo_values.min()),
                "leave_one_site_out_max_pp": float(loo_values.max()),
                "loo_estimable_sites": int(len(loo)),
                "queries": int(len(data)),
                "sites": int(data.scene.nunique()),
                "design_rank": rank,
                "design_columns": int(len(names)),
            }
            result_rows.append(row)
            method_rows.append(row)
        consistent = data[data.source_target_class_mismatch == 0].reset_index(drop=True)
        for model_name, covariates in (
            ("label_consistent_base", []),
            (
                "label_consistent_all_support_proxies",
                [
                    "log_target_vertex_count",
                    "log_target_bbox_diagonal",
                    "log_source_vertex_count",
                    "log_source_bbox_diagonal",
                ],
            ),
        ):
            beta, loo, rank = fit_and_loo(consistent, covariates)
            _, names = make_x(consistent, covariates)
            contrast_index = names.index("multiplicity[6+]")
            loo_values = np.asarray(loo, dtype=float)[:, 1] * 100
            row = {
                "method": method,
                "model": model_name,
                "six_plus_vs_two_three_risk_difference_pp": float(beta[contrast_index] * 100),
                "leave_one_site_out_min_pp": float(loo_values.min()),
                "leave_one_site_out_max_pp": float(loo_values.max()),
                "loo_estimable_sites": int(len(loo)),
                "queries": int(len(consistent)),
                "sites": int(consistent.scene.nunique()),
                "design_rank": rank,
                "design_columns": int(len(names)),
            }
            result_rows.append(row)
            method_rows.append(row)
        rank_meta[method] = method_rows

    result = pd.DataFrame(result_rows)
    result.to_csv(out / "adjusted_size_controls.csv", index=False)
    mismatch_rows = queries[queries.source_target_class_mismatch > 0].groupby(
        [
            "scene",
            "source_scan",
            "target_scan",
            "target_instance_id",
            "target_class_id",
            "source_instance_class_id",
            "source_class_instance_count_full",
            "method",
        ],
        as_index=False,
    ).agg(
        queries=("target_sample_index", "size"),
        semantic_hits=("semantic_hit_top20", "sum"),
        physical_hits=("physical_hit_top20", "sum"),
        semantic_only_hits=("semantic_only_hit_top20", "sum"),
    )
    mismatch_rows.to_csv(out / "class_mismatch_objects.csv", index=False)
    consistent = queries[queries.source_target_class_mismatch == 0].copy()
    consistency_site = consistent.groupby(
        ["scene", "method", "multiplicity_bin"], observed=True
    ).agg(
        queries=("semantic_hit_top20", "size"),
        semantic_recall=("semantic_hit_top20", "mean"),
        physical_recall=("physical_hit_top20", "mean"),
    ).reset_index()
    consistency_site["gap"] = consistency_site.semantic_recall - consistency_site.physical_recall
    gap_rows = []
    for index, ((method, bucket), group) in enumerate(
        consistency_site.groupby(["method", "multiplicity_bin"], observed=True, sort=True)
    ):
        estimate, low, high = site_bootstrap_mean(group.gap.to_numpy() * 100, seed=20261006 + index)
        gap_rows.append(
            {
                "method": method,
                "multiplicity_bin": bucket,
                "queries": int(group.queries.sum()),
                "sites": int(group.scene.nunique()),
                "semantic_minus_physical_gap_pp": estimate,
                "ci_low_pp": low,
                "ci_high_pp": high,
            }
        )
    consistent_gaps = pd.DataFrame(gap_rows)
    consistent_gaps.to_csv(out / "class_consistency_site_gaps.csv", index=False)
    unique_queries = queries.drop_duplicates(
        ["scene", "source_scan", "target_scan", "target_instance_id", "target_sample_index"]
    )
    focal_queries = unique_queries[unique_queries.multiplicity_bin.isin(["2-3", "6+"])]
    focal_class_bins = focal_queries.groupby("target_class_id").multiplicity_bin.nunique()
    common_classes = sorted(focal_class_bins[focal_class_bins == 2].index.astype(int).tolist())
    common_class_queries = int(focal_queries.target_class_id.isin(common_classes).sum())
    site_class_bins = unique_queries.groupby(["scene", "target_class_id"]).multiplicity_bin.nunique()
    overlapping_site_class_cells = int((site_class_bins > 1).sum())
    support_audit = {
        "focal_bins": ["2-3", "6+"],
        "semantic_classes_in_both_focal_bins": common_classes,
        "semantic_class_count_in_both_focal_bins": len(common_classes),
        "semantic_class_count_total_in_focal_bins": int(focal_queries.target_class_id.nunique()),
        "common_class_query_count": common_class_queries,
        "all_focal_query_count": int(len(focal_queries)),
        "common_class_query_fraction": common_class_queries / len(focal_queries),
        "site_class_cells_with_multiple_nontrivial_bins": overlapping_site_class_cells,
        "unique_query_rows": int(len(unique_queries)),
    }
    summary = {
        "status": "complete",
        "outcome": "semantic hit without tracked physical instance at top-20",
        "analysis_unit": "sampled target query; target-point sampling is object-balanced and repeated observations share scan/object clusters",
        "multiplicity": "number of distinct source-PLY instance IDs of the query class",
        "controls": "site and semantic-class fixed effects; annotated source/target instance vertex counts and axis-aligned bounding-box diagonals are scan-support proxies, not true object dimensions",
        "causal": False,
        "sites": sorted(queries.scene.unique()),
        "query_rows_per_method": int(queries.groupby("method").size().min()),
        "source_target_scans_loaded": int(len(scans)),
        "object_support_rows": int(len(support)),
        "source_target_class_mismatch_query_rows": mismatch_query_rows,
        "source_target_class_mismatch_object_pairs": mismatch_object_pairs,
        "common_support_audit": support_audit,
        "class_consistency_bootstrap": {
            "unit": "site clusters with at least one consistent-label query in the multiplicity stratum",
            "samples": BOOTSTRAP_N,
            "seed_base": 20261006,
            "rows": gap_rows,
        },
        "models": rank_meta,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Rescan multiplicity and observed object support",
        "",
        "Outcome is a query-level semantic hit without the tracked physical instance in top-20. The model includes site and semantic-class fixed effects; the multiplicity contrast is 6+ versus 2--3 same-class source instances. Added covariates use the annotated target/source instance vertex count and axis-aligned bounding-box diagonal from the corresponding PLY. Vertex count measures reconstruction support/visibility as well as size; bbox diagonal is an incomplete extent estimate under partial scans. Neither is a ground-truth physical size measure.",
        "",
        "Estimates are query-weighted linear-probability coefficients. Leave-one-site-out ranges are sensitivity ranges, not confidence intervals. Queries repeat within objects and scans; with 11 sites and sparse class-by-site multiplicity overlap, these results are exploratory and non-causal.",
        "",
        f"Common-support audit: only {len(common_classes)}/{focal_queries.target_class_id.nunique()} semantic classes occur in both the 2--3 and 6+ bins, covering {common_class_queries:,}/{len(focal_queries):,} ({100 * support_audit['common_class_query_fraction']:.1f}%) of focal-bin queries. No site--class cell contains queries from multiple nontrivial multiplicity bins ({overlapping_site_class_cells} overlapping cells). The fixed-effects coefficients therefore rely on additive assumptions rather than within-cell comparisons and cannot rule out site--class-specific confounding.",
        "",
        "| Method | Model / query filter | Adjusted difference (pp) | Leave-one-site-out range (pp) | Queries |",
        "|---|---|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['method']} | {row['model']} | {row['six_plus_vs_two_three_risk_difference_pp']:.1f} | "
            f"[{row['leave_one_site_out_min_pp']:.1f},{row['leave_one_site_out_max_pp']:.1f}] | {row['queries']:,} |"
        )
    lines.extend(
        [
            "",
            "After removing those source/target label-disagreement object groups, the site-macro semantic-minus-physical gaps are:",
            "",
            "| Method | Multiplicity | Queries | Sites | Gap (pp, 95% site-bootstrap CI) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in gap_rows:
        lines.append(
            f"| {row['method']} | {row['multiplicity_bin']} | {row['queries']:,} | {row['sites']} | "
            f"{row['semantic_minus_physical_gap_pp']:.1f} [{row['ci_low_pp']:.1f},{row['ci_high_pp']:.1f}] |"
        )
    lines.extend(
        [
            "",
            f"Loaded {len(scans)} source/target scans and {len(support):,} scan-instance rows. {summary['source_target_class_mismatch_query_rows']:,} method/query rows ({summary['source_target_class_mismatch_object_pairs']:,} unique scene/source/target/instance groups) have a source-versus-target class-label mismatch; these groups are excluded in the `label_consistent_*` sensitivity rows. Their query-level hit counts are in `class_mismatch_objects.csv`. Machine-readable model and support metadata are in `adjusted_size_controls.csv`, `class_consistency_site_gaps.csv`, and `summary.json`.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines))
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
