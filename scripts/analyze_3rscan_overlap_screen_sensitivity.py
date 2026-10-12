#!/usr/bin/env python3
"""Summarize cross-split screening yield against same-environment controls.

This is a threshold-sensitivity diagnostic for the FPFH/FGR/ICP screen. A
control recall below 100% means that absence of candidates cannot establish
absence of cross-split overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
MAX_RMSE_M = 0.08


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def qualifies(row: dict[str, str], threshold: float) -> bool:
    return (
        float(row["icp_symmetric_fitness"]) >= threshold
        and float(row["icp_symmetric_rmse_m"]) <= MAX_RMSE_M
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-split-csv", type=Path, required=True)
    parser.add_argument("--controls-csv", type=Path, required=True)
    parser.add_argument("--expected-pairs", type=int, default=18_095)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_pairs <= 0:
        parser.error("--expected-pairs must be positive")

    cross = read_rows(args.cross_split_csv)
    controls = read_rows(args.controls_csv)
    table = []
    for threshold in THRESHOLDS:
        cross_count = sum(qualifies(row, threshold) for row in cross)
        control_hits = sum(qualifies(row, threshold) for row in controls)
        table.append(
            {
                "symmetric_fitness_min": threshold,
                "symmetric_rmse_m_max": MAX_RMSE_M,
                "cross_split_candidate_count": cross_count,
                "control_hits": control_hits,
                "control_count": len(controls),
                "control_recall": control_hits / len(controls) if controls else None,
            }
        )

    complete = len(cross) == args.expected_pairs
    status = "complete" if complete else "interim"
    summary = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cross_split_pair_count": len(cross),
        "expected_cross_split_pair_count": args.expected_pairs,
        "cross_split_pair_fraction": min(1.0, len(cross) / args.expected_pairs),
        "same_environment_control_count": len(controls),
        "symmetric_rmse_m_max": MAX_RMSE_M,
        "threshold_sensitivity": table,
        "interpretation": (
            "Control recall below 100% means this screen can miss same-environment "
            "overlap at that threshold; a zero candidate count would not establish "
            "absence of cross-split overlap."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# Cross-split overlap screen sensitivity ({status})",
        "",
        f"Evaluated {len(cross):,}/{args.expected_pairs:,} cross-split pairs "
        f"({summary['cross_split_pair_fraction']:.1%}) and {len(controls)} "
        "same-environment positive controls.",
        "",
        f"Candidates require symmetric ICP fitness above the listed threshold "
        f"and symmetric RMSE <= {MAX_RMSE_M:.2f} m.",
        "",
        "| Fitness threshold | Cross-split candidates | Positive controls recalled | Recall |",
        "|---:|---:|---:|---:|",
    ]
    for row in table:
        recall = "n/a" if row["control_recall"] is None else f"{row['control_recall']:.1%}"
        lines.append(
            f"| {row['symmetric_fitness_min']:.2f} | "
            f"{row['cross_split_candidate_count']} | "
            f"{row['control_hits']}/{row['control_count']} | {recall} |"
        )
    lines += ["", "## Interpretation", "", summary["interpretation"], ""]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
