#!/usr/bin/env python3
"""Compare annotation-field agreement in cross-split candidates and controls.

This descriptive comparison is restricted to pairs passing the geometric
screen. It is not a test of physical-site identity or a recall estimate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CLASS_FIELDS = ("NYU40", "Eigen13", "RIO27")
ALL_FIELDS = ("objectId", "globalId", *CLASS_FIELDS)


def metric_rows(summary: dict[str, object]) -> list[dict[str, float | str]]:
    rows = []
    for item in summary["candidates"]:
        metrics = item["field_agreement"]
        row: dict[str, float | str] = {
            "source_id": item["source_id"],
            "target_id": item["target_id"],
            "min_side_coverage": min(item["source_fraction_with_match"], item["target_fraction_with_match"]),
        }
        for name in ALL_FIELDS:
            field = metrics[name]
            row[f"{name}_agreement"] = field["nonzero_both_exact_agreement"]
            row[f"{name}_reference"] = field["nonzero_both_marginal_reference"]
        rows.append(row)
    return rows


def describe(rows: list[dict[str, float | str]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows if row[key] is not None], dtype=float)
    if not len(values):
        return {}
    return {
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def is_semantically_coherent(row: dict[str, float | str], coverage: float, agreement: float, lift: float) -> bool:
    if float(row["min_side_coverage"]) < coverage:
        return False
    return all(
        float(row[f"{name}_agreement"]) >= agreement
        and float(row[f"{name}_agreement"]) - float(row[f"{name}_reference"]) >= lift
        for name in CLASS_FIELDS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-side-coverage", type=float, default=0.10)
    parser.add_argument("--class-agreement", type=float, default=0.70)
    parser.add_argument("--class-lift", type=float, default=0.30)
    args = parser.parse_args()
    cross_summary = json.loads(args.cross_summary.read_text(encoding="utf-8"))
    control_summary = json.loads(args.control_summary.read_text(encoding="utf-8"))
    cross_rows, control_rows = metric_rows(cross_summary), metric_rows(control_summary)

    descriptions = {}
    for label, rows in (("cross_split", cross_rows), ("same_environment_controls", control_rows)):
        descriptions[label] = {
            key: describe(rows, key)
            for key in ("min_side_coverage", *(f"{name}_agreement" for name in ALL_FIELDS))
        }
    cross_hits = [
        row for row in cross_rows
        if is_semantically_coherent(row, args.min_side_coverage, args.class_agreement, args.class_lift)
    ]
    control_hits = [
        row for row in control_rows
        if is_semantically_coherent(row, args.min_side_coverage, args.class_agreement, args.class_lift)
    ]
    summary = {
        "status": "interim" if cross_summary["status"] != "complete" else "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cross_split_pair_count": cross_summary["pair_count"],
        "cross_split_geometric_candidate_count": len(cross_rows),
        "control_pair_count": control_summary["pair_count"],
        "control_geometric_candidate_count": len(control_rows),
        "distribution_summaries": descriptions,
        "descriptive_semantic_coherence_rule": {
            "min_side_coverage": args.min_side_coverage,
            "minimum_nonzero_exact_agreement_each_taxonomy": args.class_agreement,
            "minimum_lift_over_matched_subset_marginal_reference": args.class_lift,
        },
        "cross_split_semantically_coherent_candidates": cross_hits,
        "same_environment_controls_meeting_rule": len(control_hits),
        "same_environment_control_eligible_count": len(control_rows),
        "scope_note": (
            "This post hoc rule summarizes pairs that already passed a geometry cutoff. "
            "The control and cross-split candidate sets are selected samples with different "
            "coverage distributions; counts are descriptive and do not estimate physical-site "
            "identity or the full screen's overlap recall."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Annotation agreement: geometric candidates ({summary['status']})",
        "",
        f"Cross-split CSV: {summary['cross_split_pair_count']:,} pairs; {len(cross_rows)} geometric candidates.",
        f"Same-environment controls: {len(control_rows)}/{summary['control_pair_count']} pass the same geometric cutoff.",
        "",
        "## Distribution summaries among geometric candidates",
        "",
        "| Set | Min-side coverage median (p10–p90) | objectId median | globalId median | NYU40 median | Eigen13 median | RIO27 median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, display in (("cross_split", "Cross-split"), ("same_environment_controls", "Same-environment controls")):
        d = descriptions[label]
        cov = d["min_side_coverage"]
        fields = [d[f"{name}_agreement"]["median"] for name in ALL_FIELDS]
        lines.append(
            f"| {display} | {cov['median']:.1%} ({cov['p10']:.1%}–{cov['p90']:.1%}) | "
            + " | ".join(f"{value:.1%}" for value in fields)
            + " |"
        )
    lines += [
        "",
        "## Exploratory semantically coherent candidates",
        "",
        f"Rule: min-side coverage >= {args.min_side_coverage:.0%}; each nonzero NYU40/Eigen13/RIO27 field has exact agreement >= {args.class_agreement:.0%} and lift >= {args.class_lift:.0%} over matched-subset marginals.",
        f"Cross-split: {len(cross_hits)} / {len(cross_rows)} candidates. Same-environment controls: {len(control_hits)} / {len(control_rows)}.",
        "",
        "| Train/reference scan | Validation/query scan | Coverage | NYU40 | Eigen13 | RIO27 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in cross_hits:
        lines.append(
            f"| `{row['source_id']}` | `{row['target_id']}` | {float(row['min_side_coverage']):.1%} | "
            + " | ".join(f"{float(row[f'{name}_agreement']):.1%}" for name in CLASS_FIELDS)
            + " |"
        )
    lines += ["", summary["scope_note"], ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "cross_split_geometric_candidates": len(cross_rows),
        "control_geometric_candidates": len(control_rows),
        "cross_split_semantically_coherent": len(cross_hits),
        "controls_meeting_rule": len(control_hits),
    }, indent=2))


if __name__ == "__main__":
    main()
