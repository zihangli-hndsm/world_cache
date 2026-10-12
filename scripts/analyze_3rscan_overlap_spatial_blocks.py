#!/usr/bin/env python3
"""Measure spatial support and block-bootstrap overlap metrics for fixed scan pairs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from audit_3rscan_cross_split_site_overlap import read_annotated_vertices


FIELD_COLUMNS = {"NYU40": 8, "Eigen13": 9, "RIO27": 10}
PAIR_DEFAULTS = (
    ("6bde6055-9162-246f-8f44-e3efc21c9e14", "b1f23308-d255-2761-94da-981d962c6bf8", "paper-source to evaluated-validation candidate"),
    ("d7d40d5c-7a5d-2b36-9433-ee98b893988d", "4fbad31e-465b-2a5d-84b7-c0ddea978db4", "strong shared-target cluster candidate"),
    ("6bde6055-9162-246f-8f44-e3efc21c9e14", "6bde6057-9162-246f-8d07-dfba45622e09", "same-environment positive control"),
    ("b1f23308-d255-2761-94da-981d962c6bf8", "77361fd4-d054-2a22-88c4-b5b404f904ca", "same-environment positive control for evaluated validation reference"),
    ("4fbad31e-465b-2a5d-84b7-c0ddea978db4", "4fbad32f-465b-2a5d-8408-146ab1d72808", "same-environment positive control"),
)


def agreement_stats(contingency: np.ndarray) -> tuple[float | None, float | None, float | None, int]:
    count = int(contingency.sum())
    if count == 0:
        return None, None, None, 0
    observed = float(np.trace(contingency) / count)
    row = contingency.sum(axis=1).astype(np.float64)
    col = contingency.sum(axis=0).astype(np.float64)
    marginal = float(np.dot(row, col) / (count * count))
    return observed, marginal, observed - marginal, count


def block_metrics(
    aligned: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    target_idx: np.ndarray,
    source_match: np.ndarray,
    reverse_coverage: float,
    block_size: float,
    bootstrap_reps: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    block_coords = np.floor(aligned / block_size).astype(np.int64)
    _, block_index = np.unique(block_coords, axis=0, return_inverse=True)
    block_count = int(block_index.max()) + 1 if len(block_index) else 0
    source_per_block = np.bincount(block_index, minlength=block_count).astype(np.int64)
    matched_per_block = np.bincount(block_index[source_match], minlength=block_count).astype(np.int64)
    top_order = np.argsort(matched_per_block)[::-1]
    matched_total = int(matched_per_block.sum())
    result: dict[str, object] = {
        "block_size_m": block_size,
        "occupied_source_blocks": block_count,
        "blocks_with_matches": int(np.count_nonzero(matched_per_block)),
        "source_coverage": float(source_match.mean()) if len(source_match) else 0.0,
        "target_coverage": reverse_coverage,
        "matched_vertices": matched_total,
        "largest_block_match_share": float(matched_per_block[top_order[0]] / matched_total) if matched_total else 0.0,
        "top_5_block_match_share": float(matched_per_block[top_order[:5]].sum() / matched_total) if matched_total else 0.0,
        "fields": {},
    }
    if block_count == 0:
        return result

    draw_weights = np.zeros((bootstrap_reps, block_count), dtype=np.int32)
    for draw in range(bootstrap_reps):
        draw_weights[draw] = np.bincount(
            rng.integers(0, block_count, size=block_count), minlength=block_count
        )
    bootstrap_source_n = draw_weights @ source_per_block
    bootstrap_match_n = draw_weights @ matched_per_block
    source_boot = np.divide(
        bootstrap_match_n,
        bootstrap_source_n,
        out=np.zeros(bootstrap_reps, dtype=np.float64),
        where=bootstrap_source_n > 0,
    )
    coverage_ci = np.quantile(source_boot, [0.025, 0.975]).tolist()
    result["source_coverage_spatial_block_ci95"] = [float(x) for x in coverage_ci]

    aligned_indices = np.flatnonzero(source_match)
    metrics_by_field: dict[str, object] = {}
    for field, column in FIELD_COLUMNS.items():
        source_values = source[:, column].astype(np.int64)
        target_values = target[target_idx, column].astype(np.int64)
        valid = source_match & (source_values != 0) & (target_values != 0)
        valid_blocks = block_index[valid]
        a = source_values[valid]
        b = target_values[valid]
        max_label = int(max(a.max(initial=0), b.max(initial=0)))
        contingency_by_block = np.zeros((block_count, max_label + 1, max_label + 1), dtype=np.int32)
        if len(a):
            np.add.at(contingency_by_block, (valid_blocks, a, b), 1)
        observed, marginal, lift, n_valid = agreement_stats(contingency_by_block.sum(axis=0))

        diagonal = np.diagonal(contingency_by_block, axis1=1, axis2=2).sum(axis=1).astype(np.int64)
        row_by_block = contingency_by_block.sum(axis=2).astype(np.int64)
        col_by_block = contingency_by_block.sum(axis=1).astype(np.int64)
        boot_n = draw_weights @ np.bincount(valid_blocks, minlength=block_count).astype(np.int64)
        boot_exact = draw_weights @ diagonal
        boot_rows = draw_weights @ row_by_block
        boot_cols = draw_weights @ col_by_block
        boot_agreement = np.divide(boot_exact, boot_n, out=np.zeros(bootstrap_reps, dtype=np.float64), where=boot_n > 0)
        boot_marginal = np.divide(
            np.einsum("bi,bi->b", boot_rows, boot_cols),
            boot_n.astype(np.float64) ** 2,
            out=np.zeros(bootstrap_reps, dtype=np.float64),
            where=boot_n > 0,
        )
        boot_lift = boot_agreement - boot_marginal
        metrics_by_field[field] = {
            "nonzero_match_count": n_valid,
            "exact_agreement": observed,
            "matched_subset_marginal_reference": marginal,
            "lift_over_marginal": lift,
            "exact_agreement_spatial_block_ci95": [float(x) for x in np.quantile(boot_agreement, [0.025, 0.975])],
            "lift_spatial_block_ci95": [float(x) for x in np.quantile(boot_lift, [0.025, 0.975])],
        }
    result["fields"] = metrics_by_field

    extra_fields: dict[str, object] = {}
    point_pairs = target[target_idx]
    exact_masks = {
        "RGB_all_channels": (np.ones(len(source_match), dtype=bool), np.all(source[:, 3:6] == point_pairs[:, 3:6], axis=1)),
        "objectId_nonzero": ((source[:, 6] != 0) & (point_pairs[:, 6] != 0), source[:, 6] == point_pairs[:, 6]),
        "globalId_nonzero": ((source[:, 7] != 0) & (point_pairs[:, 7] != 0), source[:, 7] == point_pairs[:, 7]),
    }
    for name, (valid_all, exact_all) in exact_masks.items():
        valid = source_match & valid_all
        valid_blocks = block_index[valid]
        denom_by_block = np.bincount(valid_blocks, minlength=block_count).astype(np.int64)
        exact_by_block = np.bincount(valid_blocks, weights=exact_all[valid].astype(np.int64), minlength=block_count).astype(np.int64)
        denom = int(denom_by_block.sum())
        observed_rate = float(exact_by_block.sum() / denom) if denom else None
        boot_denom = draw_weights @ denom_by_block
        boot_exact = draw_weights @ exact_by_block
        boot_rate = np.divide(boot_exact, boot_denom, out=np.zeros(bootstrap_reps, dtype=np.float64), where=boot_denom > 0)
        extra_fields[name] = {
            "matched_point_count": denom,
            "exact_agreement": observed_rate,
            "exact_agreement_spatial_block_ci95": [float(x) for x in np.quantile(boot_rate, [0.025, 0.975])],
        }
    result["exact_field_rates"] = extra_fields
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--controls-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distance-m", type=float, default=0.02)
    parser.add_argument("--block-sizes-m", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    if args.distance_m <= 0 or args.bootstrap_reps <= 0 or any(x <= 0 for x in args.block_sizes_m):
        parser.error("distance, block sizes, and bootstrap reps must be positive")

    rows: dict[tuple[str, str], dict[str, str]] = {}
    for path in (args.pairs_csv, args.controls_csv):
        with path.open(newline="", encoding="utf-8") as stream:
            rows.update({(row["source_id"], row["target_id"]): row for row in csv.DictReader(stream)})

    rng = np.random.default_rng(args.seed)
    pair_outputs = []
    for source_id, target_id, label in PAIR_DEFAULTS:
        row = rows[(source_id, target_id)]
        transform = np.asarray(json.loads(row["transform"]), dtype=np.float64)
        source = read_annotated_vertices(args.dataset_root / source_id / "labels.instances.annotated.v2.ply")
        target = read_annotated_vertices(args.dataset_root / target_id / "labels.instances.annotated.v2.ply")
        aligned = source[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        distance, target_idx = cKDTree(target[:, :3]).query(aligned, k=1)
        source_match = distance < args.distance_m
        reverse_distance = cKDTree(aligned).query(target[:, :3], k=1)[0]
        reverse_coverage = float(np.mean(reverse_distance < args.distance_m))
        size_outputs = [
            block_metrics(
                aligned, source, target, target_idx, source_match, reverse_coverage,
                block_size, args.bootstrap_reps, rng,
            )
            for block_size in args.block_sizes_m
        ]
        pair_outputs.append({
            "source_id": source_id,
            "target_id": target_id,
            "pair_role": label,
            "symmetric_fitness": float(row["icp_symmetric_fitness"]),
            "symmetric_rmse_m": float(row["icp_symmetric_rmse_m"]),
            "distance_threshold_m": args.distance_m,
            "spatial_summaries": size_outputs,
        })

    summary = {
        "status": "complete_descriptive_spatial_block_bootstrap",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "block_sizes_m": args.block_sizes_m,
        "pairs": pair_outputs,
        "scope_note": (
            "Spatial blocks are resampled within each fixed pair to summarize spatial sensitivity of vertex overlap and label agreement. "
            "Registration transforms and pairs were selected using the full clouds; these intervals do not correct registration-selection bias, "
            "and neighboring blocks may remain spatially dependent. Treat them as descriptive, not confirmatory site-identity tests."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Spatial support and block-bootstrap sensitivity for scan overlap", "",
        f"Nearest-neighbor distance threshold: {args.distance_m:g} m.",
        f"Block sizes: {', '.join(f'{x:g} m' for x in args.block_sizes_m)}; {args.bootstrap_reps} block-bootstrap draws per size and pair.",
        "",
    ]
    for pair in pair_outputs:
        lines += [
            f"## {pair['pair_role']}: `{pair['source_id']}` → `{pair['target_id']}`", "",
            f"Geometry screen: fitness {pair['symmetric_fitness']:.3f}, RMSE {pair['symmetric_rmse_m']:.3f} m.", "",
            "| Block size | Source blocks | Blocks with matches | Source/target coverage (source block CI) | Largest / top-5 block share | NYU40 agreement [95% block CI] | NYU40 lift [95% block CI] | Eigen13 agreement [95% block CI] | Eigen13 lift [95% block CI] | RIO27 agreement [95% block CI] | RIO27 lift [95% block CI] |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in pair["spatial_summaries"]:
            fields = item["fields"]
            def pct(value: float | None) -> str:
                return "n/a" if value is None else f"{value:.1%}"
            def interval(field: str, key: str) -> str:
                value = fields[field][key]
                ci_key = "lift_spatial_block_ci95" if key == "lift_over_marginal" else key + "_spatial_block_ci95"
                ci = fields[field][ci_key]
                return f"{pct(value)} [{pct(ci[0])},{pct(ci[1])}]"
            lines.append(
                f"| {item['block_size_m']:g} m | {item['occupied_source_blocks']} | {item['blocks_with_matches']} | "
                f"{item['source_coverage']:.1%}/{item['target_coverage']:.1%} "
                f"[{item['source_coverage_spatial_block_ci95'][0]:.1%},{item['source_coverage_spatial_block_ci95'][1]:.1%}] | "
                f"{item['largest_block_match_share']:.1%}/{item['top_5_block_match_share']:.1%} | "
                f"{interval('NYU40','exact_agreement')} | {interval('NYU40','lift_over_marginal')} | "
                f"{interval('Eigen13','exact_agreement')} | {interval('Eigen13','lift_over_marginal')} | "
                f"{interval('RIO27','exact_agreement')} | {interval('RIO27','lift_over_marginal')} |"
            )
        lines += [
            "", "Exact color/identity-field agreement (matched points with both identity labels nonzero):", "",
            "| Block size | RGB all-channel exact [95% block CI] | objectId exact [95% block CI] | globalId exact [95% block CI] |",
            "|---:|---:|---:|---:|",
        ]
        for item in pair["spatial_summaries"]:
            extra = item["exact_field_rates"]
            def exact_interval(name: str) -> str:
                metrics = extra[name]
                ci = metrics["exact_agreement_spatial_block_ci95"]
                return f"{metrics['exact_agreement']:.1%} [{ci[0]:.1%},{ci[1]:.1%}]"
            lines.append(
                f"| {item['block_size_m']:g} m | {exact_interval('RGB_all_channels')} | "
                f"{exact_interval('objectId_nonzero')} | {exact_interval('globalId_nonzero')} |"
            )
        lines.append("")
    lines += [summary["scope_note"], ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"pairs": len(pair_outputs), "block_sizes_m": args.block_sizes_m, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
