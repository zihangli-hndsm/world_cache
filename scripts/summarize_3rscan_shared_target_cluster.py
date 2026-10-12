#!/usr/bin/env python3
"""Summarize pairwise source comparisons for a post-hoc shared-target cluster."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

CLASS_FIELDS = ("NYU40", "Eigen13", "RIO27")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    comparisons = [json.loads(path.read_text(encoding="utf-8")) for path in args.comparison_json]
    if not comparisons:
        parser.error("provide at least one comparison")
    target_ids = {item["shared_validation_target"] for item in comparisons}
    if len(target_ids) != 1:
        parser.error(f"comparisons must share one validation target, found {sorted(target_ids)}")

    pair_rows = []
    directions = []
    for item in comparisons:
        pair_row = {
            "source_a": item["source_a"],
            "source_b": item["source_b"],
            "directions": [],
        }
        for key, label in (("source_a_to_source_b", "A to B"), ("source_b_to_source_a", "B to A")):
            metrics = item[key]
            fields = metrics["annotation_field_agreement"]
            direction = {
                "label": label,
                "source_coverage": metrics["source_coverage"],
                "matched_point_count": metrics["matched_point_count"],
                "rgb_exact_agreement": metrics["rgb_exact_agreement"],
                "class_metrics": {
                    name: {
                        "nonzero_exact_agreement": fields[name]["exact_agreement_both_nonzero"],
                        "nonzero_marginal_reference": fields[name]["marginal_reference_both_nonzero"],
                        "nonzero_lift": fields[name]["lift_over_marginal_both_nonzero"],
                    }
                    for name in CLASS_FIELDS
                },
            }
            pair_row["directions"].append(direction)
            directions.append(direction)
        pair_rows.append(pair_row)

    coverage = [item["source_coverage"] for item in directions]
    lifts = [item["class_metrics"][field]["nonzero_lift"] for item in directions for field in CLASS_FIELDS]
    agreements = [item["class_metrics"][field]["nonzero_exact_agreement"] for item in directions for field in CLASS_FIELDS]
    marginals = [item["class_metrics"][field]["nonzero_marginal_reference"] for item in directions for field in CLASS_FIELDS]
    summary = {
        "status": "exploratory_post_hoc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shared_validation_target": next(iter(target_ids)),
        "pairwise_comparison_count": len(pair_rows),
        "directed_comparison_count": len(directions),
        "train_source_ids": sorted({source for pair in pair_rows for source in (pair["source_a"], pair["source_b"])}),
        "directed_coverage_range": [float(min(coverage)), float(max(coverage))],
        "nonzero_class_exact_agreement_range": [float(min(agreements)), float(max(agreements))],
        "nonzero_class_marginal_reference_range": [float(min(marginals)), float(max(marginals))],
        "nonzero_class_lift_range": [float(min(lifts)), float(max(lifts))],
        "pairwise_comparisons": pair_rows,
        "scope_note": (
            "These source pairs were selected post hoc because they independently passed a geometric "
            "screen against the same validation reference. Matched-point outcomes are descriptive; "
            "spatial autocorrelation and selection are not accounted for, and this analysis does not "
            "prove that the scans depict the same physical site."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    lines = [
        "# Post-hoc shared-target train-source cluster",
        "",
        f"Shared validation target: `{summary['shared_validation_target']}`.",
        f"The {len(pair_rows)} pairwise comparisons cover {len(summary['train_source_ids'])} train references.",
        "All source transforms were estimated independently against the shared validation target.",
        "",
        "| Train source A | Train source B | A→B / B→A coverage | NYU40 exact / marginal | Eigen13 exact / marginal | RIO27 exact / marginal |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pair in pair_rows:
        a, b = pair["directions"]
        classes = []
        for field in CLASS_FIELDS:
            x, y = a["class_metrics"][field], b["class_metrics"][field]
            classes.append(
                f"{pct(x['nonzero_exact_agreement'])}/{pct(x['nonzero_marginal_reference'])}; "
                f"{pct(y['nonzero_exact_agreement'])}/{pct(y['nonzero_marginal_reference'])}"
            )
        lines.append(
            f"| `{pair['source_a']}` | `{pair['source_b']}` | "
            f"{a['source_coverage']:.1%}/{b['source_coverage']:.1%} | "
            + " | ".join(classes)
            + " |"
        )
    lines += [
        "",
        f"Across directed class-field comparisons, exact agreement is {pct(min(agreements))}–{pct(max(agreements))}; matched-subset marginal reference is {pct(min(marginals))}–{pct(max(marginals))}; the lift ranges from {pct(min(lifts))} to {pct(max(lifts))}.",
        "",
        summary["scope_note"],
        "",
        "The marginal reference independently shuffles labels within the matched subsets. It is descriptive and is not a calibrated significance test.",
        "",
    ]
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "pairwise_comparison_count": len(pair_rows),
        "train_source_count": len(summary["train_source_ids"]),
        "directed_coverage_range": summary["directed_coverage_range"],
        "nonzero_class_exact_agreement_range": summary["nonzero_class_exact_agreement_range"],
        "nonzero_class_marginal_reference_range": summary["nonzero_class_marginal_reference_range"],
        "nonzero_class_lift_range": summary["nonzero_class_lift_range"],
    }, indent=2))


if __name__ == "__main__":
    main()
