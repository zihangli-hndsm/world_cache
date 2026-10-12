"""Paired query-level rescue/loss audit for added motion hypotheses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_3rscan_multimotion_lopo import assign_mode_prefix


def paired_intervention(
    method_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    mode_count: int,
    threshold: float = 0.10,
) -> dict[str, int | float | bool]:
    """Measure correct-transfer rescues and losses on exactly matched queries."""
    method = method_frame.sort_values("query_index").reset_index(drop=True)
    baseline = baseline_frame.sort_values("query_index").reset_index(drop=True)
    query_set_match = np.array_equal(method.query_index.to_numpy(), baseline.query_index.to_numpy())
    if not query_set_match:
        raise ValueError("paired intervention requires an identical query population")
    method_output = assign_mode_prefix(method, mode_count, threshold)
    baseline_output = assign_mode_prefix(baseline, 1, threshold)
    method_moved = method_output.lopo_moved_rigid_query.to_numpy(dtype=bool)
    baseline_moved = baseline_output.lopo_moved_rigid_query.to_numpy(dtype=bool)
    if not np.array_equal(method_moved, baseline_moved):
        raise ValueError("moved-rigid scoring denominator differs between paired arms")
    method_correct = method_output.lopo_correct_moved_rigid_transfer.to_numpy(dtype=bool)
    baseline_correct = baseline_output.lopo_correct_moved_rigid_transfer.to_numpy(dtype=bool)
    moved = method_moved
    denominator = int(moved.sum())
    correct = int((method_correct & moved).sum())
    baseline_count = int((baseline_correct & moved).sum())
    rescued = int((method_correct & ~baseline_correct & moved).sum())
    lost = int((baseline_correct & ~method_correct & moved).sum())
    return {
        "query_set_match": query_set_match,
        "queries": int(len(method)),
        "moved_rigid_queries": denominator,
        "correct_transfers": correct,
        "baseline_correct_transfers": baseline_count,
        "coverage": correct / denominator if denominator else float("nan"),
        "baseline_coverage": baseline_count / denominator if denominator else float("nan"),
        "rescued_vs_baseline": rescued,
        "lost_vs_baseline": lost,
        "net_gain": rescued - lost,
    }


def run_probe(output_dir: Path, threshold: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(output_dir / "query_diagnostics.csv")
    pairs = sorted(diagnostics.pair.unique())
    baseline_by_pair = {
        pair: diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == "global_ransac")]
        for pair in pairs
    }
    method_specs = {
        "multi_motion": [1, 2, 3, 4],
        "adaptive_k": [1],
        "combined": [1, 2, 3, 4],
    }
    rows: list[dict[str, object]] = []
    for pair in pairs:
        for arm, mode_counts in method_specs.items():
            method_frame = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == arm)]
            for mode_count in mode_counts:
                rows.append({
                    "pair": pair,
                    "role": "target" if pair == "target" else "source",
                    "arm": arm,
                    "mode_count": mode_count,
                    **paired_intervention(method_frame, baseline_by_pair[pair], mode_count, threshold),
                })
    by_pair = pd.DataFrame(rows)
    by_pair.to_csv(output_dir / "mode_intervention_by_pair.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for (role, arm, mode_count), frame in by_pair.groupby(["role", "arm", "mode_count"], sort=False):
        denominator = int(frame.moved_rigid_queries.sum())
        correct = int(frame.correct_transfers.sum())
        base_correct = int(frame.baseline_correct_transfers.sum())
        comparable = frame.moved_rigid_queries > 0
        nonnegative = int((
            frame.loc[comparable, "coverage"].to_numpy()
            >= frame.loc[comparable, "baseline_coverage"].to_numpy()
        ).sum())
        aggregate_rows.append({
            "role": role,
            "arm": arm,
            "mode_count": mode_count,
            "query_set_match_all_pairs": bool(frame.query_set_match.all()),
            "pairs": int(len(frame)),
            "eligible_pairs": int(comparable.sum()),
            "moved_rigid_queries": denominator,
            "correct_transfers": correct,
            "baseline_correct_transfers": base_correct,
            "coverage": correct / denominator if denominator else float("nan"),
            "baseline_coverage": base_correct / denominator if denominator else float("nan"),
            "coverage_delta_pp": (correct - base_correct) * 100.0 / denominator if denominator else float("nan"),
            "rescued_vs_baseline": int(frame.rescued_vs_baseline.sum()),
            "lost_vs_baseline": int(frame.lost_vs_baseline.sum()),
            "net_gain": int(frame.net_gain.sum()),
            "nonnegative_pairs": nonnegative,
            "not_comparable_pairs": int(len(frame) - comparable.sum()),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "mode_intervention_summary.csv", index=False)
    report_path = output_dir / "multimotion_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# 3RScan multi-motion pilot\n"
    marker = "## Same-query mode intervention"
    report = report.split(marker)[0].rstrip()
    lines = [
        "## Same-query mode intervention",
        "",
        "This paired intervention holds each relation's full depth-valid query population fixed and changes only the allowed motion hypotheses. `rescued` counts mapped moved-rigid queries correct under the intervention but incorrect under global RANSAC; `lost` counts the reverse.",
        "",
        "| Role | Arm | Modes retained | Coverage | Change | Rescued | Lost | Net |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.role} | {row.arm} | {row.mode_count} | {100 * row.coverage:.2f}% | "
            f"{row.coverage_delta_pp:+.2f} pp | {row.rescued_vs_baseline} | "
            f"{row.lost_vs_baseline} | {row.net_gain:+d} |"
        )
    report_path.write_text(report + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return by_pair, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
    args = parser.parse_args()
    _, aggregate = run_probe(args.output_dir, args.threshold)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
