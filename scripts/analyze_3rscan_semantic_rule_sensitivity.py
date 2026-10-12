#!/usr/bin/env python3
"""Describe how post-hoc semantic-coherence counts vary with their cutoffs.

Counts are computed only among geometry-screened candidates and matched
same-environment controls. They are descriptive, selected-sample diagnostics;
they are not estimates of site identity or screen recall.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path


FIELDS = ("NYU40", "Eigen13", "RIO27")
COVERAGE_THRESHOLDS = (0.05, 0.075, 0.10, 0.125)
AGREEMENT_THRESHOLDS = (0.60, 0.65, 0.70)
LIFT_THRESHOLDS = (0.15, 0.20, 0.25, 0.30)


def read_candidates(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in summary["candidates"]:
        metrics = item["field_agreement"]
        row: dict[str, object] = {
            "pair": f"{item['source_id']}->{item['target_id']}",
            "min_side_coverage": min(
                float(item["source_fraction_with_match"]),
                float(item["target_fraction_with_match"]),
            ),
        }
        for field in FIELDS:
            values = metrics[field]
            agreement = values.get("nonzero_both_exact_agreement")
            marginal = values.get("nonzero_both_marginal_reference")
            row[f"{field}_agreement"] = None if agreement is None else float(agreement)
            row[f"{field}_lift"] = (
                None if agreement is None or marginal is None else float(agreement) - float(marginal)
            )
        rows.append(row)
    return summary, rows


def qualifies(row: dict[str, object], coverage: float, agreement: float, lift: float) -> bool:
    if float(row["min_side_coverage"]) < coverage:
        return False
    return all(
        row[f"{field}_agreement"] is not None
        and row[f"{field}_lift"] is not None
        and float(row[f"{field}_agreement"]) >= agreement
        and float(row[f"{field}_lift"]) >= lift
        for field in FIELDS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cross_summary, cross_rows = read_candidates(args.cross_summary)
    control_summary, control_rows = read_candidates(args.control_summary)
    rows = []
    for coverage, agreement, lift in itertools.product(
        COVERAGE_THRESHOLDS, AGREEMENT_THRESHOLDS, LIFT_THRESHOLDS
    ):
        cross_hits = [r for r in cross_rows if qualifies(r, coverage, agreement, lift)]
        control_hits = [r for r in control_rows if qualifies(r, coverage, agreement, lift)]
        rows.append(
            {
                "min_side_coverage": coverage,
                "min_class_agreement_each": agreement,
                "min_lift_each": lift,
                "cross_split_hits": len(cross_hits),
                "cross_split_pairs": [r["pair"] for r in cross_hits],
                "control_hits": len(control_hits),
                "control_candidate_count": len(control_rows),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "interim" if cross_summary["status"] != "complete" else "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cross_split_pair_count": cross_summary["pair_count"],
        "cross_split_geometric_candidate_count": len(cross_rows),
        "control_pair_count": control_summary["pair_count"],
        "control_geometric_candidate_count": len(control_rows),
        "fields_required_each": list(FIELDS),
        "threshold_grid": rows,
        "scope_note": (
            "Post-hoc descriptive counts among geometry-selected candidates and controls. "
            "Cutoff variation is not a confirmatory test and does not establish physical-site identity."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "min_side_coverage",
                "min_class_agreement_each",
                "min_lift_each",
                "cross_split_hits",
                "cross_split_pairs",
                "control_hits",
                "control_candidate_count",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "cross_split_pairs": ";".join(row["cross_split_pairs"])})

    lines = [
        f"# Semantic-coherence cutoff sensitivity ({payload['status']})",
        "",
        f"Input: {len(cross_rows)} cross-split and {len(control_rows)} same-environment geometry candidates.",
        "Counts are descriptive and post hoc; no grid cell is a confirmatory site-identity test.",
        "",
        "| Min coverage | Min agreement | Min lift | Cross-split | Controls |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['min_side_coverage']:.3f} | {row['min_class_agreement_each']:.2f} | "
            f"{row['min_lift_each']:.2f} | {row['cross_split_hits']} | "
            f"{row['control_hits']}/{row['control_candidate_count']} |"
        )
    lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "grid_cells": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
