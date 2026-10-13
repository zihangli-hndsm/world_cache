#!/usr/bin/env python3
"""Recompute the paired validation contrast after excluding overlap candidates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-table", type=Path, required=True)
    parser.add_argument("--exclude-environment", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10022026)
    args = parser.parse_args()
    with args.environment_table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    excluded = set(args.exclude_environment)
    selected = [row for row in rows if row["environment_id"] not in excluded]
    missing = excluded - {row["environment_id"] for row in rows}
    if missing:
        raise SystemExit(f"requested exclusions are absent from table: {sorted(missing)}")
    if not selected:
        raise SystemExit("no environments remain after exclusion")

    vitg = np.asarray([float(row["ViT-G_semantic_inlier_coverage_env"]) for row in selected])
    fcgf = np.asarray([float(row["FCGF_semantic_inlier_coverage_env"]) for row in selected])
    delta = vitg - fcgf
    vitg_pose = np.asarray([float(row["ViT-G_strict_pose_success_env"]) for row in selected])
    fcgf_pose = np.asarray([float(row["FCGF_strict_pose_success_env"]) for row in selected])
    pose_delta = vitg_pose - fcgf_pose
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
    draws = delta[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    pose_draws = pose_delta[indices].mean(axis=1)
    pose_low, pose_high = np.quantile(pose_draws, [0.025, 0.975])
    # Exhaustively quantify the strongest possible influence of one or two
    # environment exclusions on the environment-macro coverage contrast.
    # The worst-case rows are descriptive stress tests, not confidence bounds.
    all_vitg = np.asarray([float(row["ViT-G_semantic_inlier_coverage_env"]) for row in rows])
    all_fcgf = np.asarray([float(row["FCGF_semantic_inlier_coverage_env"]) for row in rows])
    all_delta = all_vitg - all_fcgf
    removal_rows = []
    removal_summary = {}
    for count in (1, 2):
        values = []
        for removed in itertools.combinations(range(len(rows)), count):
            keep = np.ones(len(rows), dtype=bool)
            keep[list(removed)] = False
            gap = float(all_delta[keep].mean())
            values.append((gap, removed))
        values.sort(key=lambda item: item[0])
        worst_gap, worst_removed = values[0]
        removal_summary[str(count)] = {
            "combinations": len(values),
            "minimum_gap": worst_gap,
            "minimum_gap_excluded_environments": [rows[i]["environment_id"] for i in worst_removed],
            "q05_gap": float(np.quantile([x[0] for x in values], 0.05)),
            "median_gap": float(np.median([x[0] for x in values])),
            "positive_combinations": sum(gap > 0 for gap, _ in values),
            "ties": sum(gap == 0 for gap, _ in values),
            "negative_combinations": sum(gap < 0 for gap, _ in values),
        }
        for gap, removed in values:
            keep_indices = [i for i in range(len(rows)) if i not in removed]
            removal_rows.append({
                "removed_count": count,
                "excluded_environment_ids": ";".join(rows[i]["environment_id"] for i in removed),
                "remaining_environments": len(keep_indices),
                "ViT-G_macro_coverage": float(all_vitg[keep_indices].mean()),
                "FCGF_macro_coverage": float(all_fcgf[keep_indices].mean()),
                "coverage_gap": gap,
            })
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.environment_table),
        "posthoc_exclusions_from_geometry_overlap_screen": sorted(excluded),
        "environments": len(selected),
        "relations": sum(int(row["relations"]) for row in selected),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "ViT-G_environment_macro_coverage": float(vitg.mean()),
        "FCGF_environment_macro_coverage": float(fcgf.mean()),
        "ViT-G_minus_FCGF_coverage_difference": float(delta.mean()),
        "difference_ci95": [float(low), float(high)],
        "ViT-G_strict_pose_successes": sum(int(row["ViT-G_strict_pose_successes"]) for row in selected),
        "FCGF_strict_pose_successes": sum(int(row["FCGF_strict_pose_successes"]) for row in selected),
        "ViT-G_environment_macro_strict_pose_success": float(vitg_pose.mean()),
        "FCGF_environment_macro_strict_pose_success": float(fcgf_pose.mean()),
        "strict_pose_environment_macro_difference": float(pose_delta.mean()),
        "strict_pose_environment_macro_difference_ci95": [float(pose_low), float(pose_high)],
        "environment_win_tie_loss": {
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
        },
        "exhaustive_environment_removal_stress_test": removal_summary,
        "interpretation": "Post-hoc sensitivity analysis only; excluding a flagged environment does not establish physical-site independence for the remaining split.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "environment_removal_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(removal_rows[0]))
        writer.writeheader()
        writer.writerows(removal_rows)
    removal_lines = []
    for count, label in ((1, "one"), (2, "two")):
        item = removal_summary[str(count)]
        removal_lines.append(
            f"- Removing any {label} environment{'s' if count == 2 else ''} ({item['combinations']:,} combinations): "
            f"minimum gap {item['minimum_gap'] * 100:+.2f} pp "
            f"after excluding {', '.join(f'`{env}`' for env in item['minimum_gap_excluded_environments'])}; "
            f"5th percentile {item['q05_gap'] * 100:+.2f} pp; "
            f"all {item['positive_combinations']:,} combinations remain positive."
        )
    report = f"""# Validation sensitivity excluding geometric overlap candidates

This post-hoc analysis excludes {', '.join(f'`{item}`' for item in sorted(excluded))} from the frozen official validation environment table after a cross-split geometry/annotation overlap screen.

- Environments / relations: {summary['environments']} / {summary['relations']}
- ViT-G macro correct-semantic inlier coverage: {summary['ViT-G_environment_macro_coverage']:.1%}
- FCGF macro correct-semantic inlier coverage: {summary['FCGF_environment_macro_coverage']:.1%}
- Paired difference: {summary['ViT-G_minus_FCGF_coverage_difference'] * 100:+.1f} percentage points (95% environment-bootstrap CI [{low * 100:+.1f}, {high * 100:+.1f}])
- Environment wins / ties / losses: {summary['environment_win_tie_loss']['wins']} / {summary['environment_win_tie_loss']['ties']} / {summary['environment_win_tie_loss']['losses']}
- Strict pose successes: ViT-G {summary['ViT-G_strict_pose_successes']}/{summary['relations']}, FCGF {summary['FCGF_strict_pose_successes']}/{summary['relations']}
- Environment-macro strict pose success: ViT-G {summary['ViT-G_environment_macro_strict_pose_success']:.1%}, FCGF {summary['FCGF_environment_macro_strict_pose_success']:.1%}; paired gap {summary['strict_pose_environment_macro_difference'] * 100:+.1f} points (95% environment-bootstrap CI [{pose_low * 100:+.1f}, {pose_high * 100:+.1f}])

This was selected after observing both the validation metrics and the geometry screen. It is an exploratory robustness check and does not establish site independence for the remaining environments.

## Exhaustive environment-removal stress test

Across the full 47-environment table, we enumerated every one- and two-environment removal and recomputed the same unweighted environment-macro coverage difference. This asks whether a small number of environments can determine the sign or size of the observed contrast; it is not a bootstrap confidence interval and uses no new inference.

{chr(10).join(removal_lines)}

The minimum is selected from the observed validation outcomes, so report it as a descriptive stress-test bound rather than an inferential guarantee. Per-combination values are in `environment_removal_sensitivity.csv`.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
