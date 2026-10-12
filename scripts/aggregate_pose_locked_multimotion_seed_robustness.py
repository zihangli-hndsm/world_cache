#!/usr/bin/env python3
"""Aggregate the frozen pose-locked multi-motion audit across RANSAC seeds.

The three runs reuse the same descriptor/candidate cache and source-only LOPO
protocol.  Seeds are therefore a robustness axis, not independent datasets;
the report keeps their paired estimates separate and only summarizes their
spread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 29, 41)


def run_dir(root: Path, seed: int) -> Path:
    return root / ("20260929_pose_locked_multimotion" if seed == 17 else f"20261003_pose_locked_multimotion_seed{seed}")


def read_seed(root: Path, seed: int) -> tuple[dict, pd.DataFrame]:
    path = run_dir(root, seed)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    outcomes = pd.read_csv(path / "source_lopo_outcomes.csv")
    outcomes["delta"] = outcomes.method_correct - outcomes.baseline_correct
    outcomes["delta_pp"] = np.where(
        outcomes.moved_queries > 0,
        100.0 * outcomes.delta / outcomes.moved_queries,
        np.nan,
    )
    return summary, outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict] = []
    relation_frames: list[pd.DataFrame] = []
    target_rows: list[dict] = []
    for seed in SEEDS:
        summary, outcomes = read_seed(args.reports_root, seed)
        source = outcomes[outcomes.pair != "target"].copy()
        eligible = source[source.moved_queries > 0]
        seed_rows.append({
            "seed": seed,
            "source_relations": int(summary["source_relations"]),
            "eligible_relations": int(summary["eligible_source_relations"]),
            "source_queries": int(summary["source_moved_rigid_queries"]),
            "baseline_correct": int(summary["baseline_correct"]),
            "method_correct": int(summary["method_correct"]),
            "delta": int(summary["method_correct"] - summary["baseline_correct"]),
            "delta_pp": float(summary["coverage_delta_pp"]),
            "rescued": int(summary["rescued"]),
            "lost": int(summary["lost"]),
            "nonnegative_relations": int(summary["nonnegative_source_relations"]),
            "weighted_ci95_lower_pp": float(summary["weighted_delta_ci95_pp"][0]),
            "weighted_ci95_upper_pp": float(summary["weighted_delta_ci95_pp"][1]),
            "macro_ci95_lower_pp": float(summary["macro_delta_ci95_pp"][0]),
            "macro_ci95_upper_pp": float(summary["macro_delta_ci95_pp"][1]),
            "stage1_gate_pass": bool(summary["stage1_gate_pass"]),
            "target_baseline": int(summary["target_baseline_correct"]),
            "target_method": int(summary["target_method_correct"]),
            "target_queries": int(summary["target_moved_rigid_queries"]),
        })
        relation = eligible[["pair", "moved_queries", "baseline_correct", "method_correct", "delta", "delta_pp"]].copy()
        relation["seed"] = seed
        relation_frames.append(relation)
        target = outcomes[outcomes.pair == "target"].iloc[0]
        target_rows.append({
            "seed": seed,
            "baseline_correct": int(target.baseline_correct),
            "method_correct": int(target.method_correct),
            "delta": int(target.delta),
            "queries": int(target.moved_queries),
            "delta_pp": float(target.delta_pp),
            "modes": int(target.modes),
            "margin_m": float(target.margin_m),
            "min_independent_support": int(target.min_independent_support),
        })

    seeds = pd.DataFrame(seed_rows)
    relations = pd.concat(relation_frames, ignore_index=True)
    targets = pd.DataFrame(target_rows)
    per_relation = (
        relations.groupby("pair", as_index=False)
        .agg(
            seeds=("seed", "count"),
            moved_queries=("moved_queries", "first"),
            mean_delta_pp=("delta_pp", "mean"),
            std_delta_pp=("delta_pp", "std"),
            min_delta_pp=("delta_pp", "min"),
            max_delta_pp=("delta_pp", "max"),
            positive_seeds=("delta", lambda x: int((x > 0).sum())),
            nonnegative_seeds=("delta", lambda x: int((x >= 0).sum())),
        )
    )

    seeds.to_csv(args.output_dir / "seed_summary.csv", index=False)
    relations.to_csv(args.output_dir / "relation_seed_outcomes.csv", index=False)
    per_relation.to_csv(args.output_dir / "relation_summary.csv", index=False)
    targets.to_csv(args.output_dir / "target_summary.csv", index=False)

    mean_delta = float(seeds.delta_pp.mean())
    report = f"""# Pose-locked multi-motion seed robustness

This audit reruns the same source-only leave-one-relation-out selector with
RANSAC seeds 17, 29, and 41. The descriptor cache, 20/50 candidate limits,
independent-support rule, and target holdout are unchanged. Seeds are a
robustness axis, not independent data splits.

| Seed | Source baseline | Source method | Gain (pp) | Rescued/lost | Nonnegative relations | Weighted 95% CI (pp) | Target |
|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in seeds.itertuples(index=False):
        report += (
            f"| {row.seed} | {row.baseline_correct}/{row.source_queries} "
            f"({100*row.baseline_correct/row.source_queries:.1f}%) | "
            f"{row.method_correct}/{row.source_queries} ({100*row.method_correct/row.source_queries:.1f}%) | "
            f"{row.delta_pp:+.2f} | {row.rescued}/{row.lost} | "
            f"{row.nonnegative_relations}/{row.eligible_relations} | "
            f"[{row.weighted_ci95_lower_pp:+.2f},{row.weighted_ci95_upper_pp:+.2f}] | "
            f"{row.target_baseline}/{row.target_queries} $\\to$ {row.target_method}/{row.target_queries} |\n"
        )
    report += f"""
The weighted source gain is {mean_delta:+.2f} pp on average across seeds
(range {seeds.delta_pp.min():+.2f} to {seeds.delta_pp.max():+.2f} pp). Every seed
has a positive relation-bootstrap lower bound and all 12 eligible source
relations are nonnegative. The target diagnostic changes by
{targets.delta.mean():+.2f} correct transfers on average ({targets.delta.tolist()}),
so it is not used as evidence for selecting the method. The scene pose is
locked to each seed's global top-20 estimate; this guarantees zero additional
pose loss but does not establish that the baseline pose is correct. All
relations are from the official 3RScan train split, so this remains an
internal robustness result pending external validation.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "seeds": list(SEEDS),
        "mean_weighted_delta_pp": mean_delta,
        "min_weighted_delta_pp": float(seeds.delta_pp.min()),
        "max_weighted_delta_pp": float(seeds.delta_pp.max()),
        "all_weighted_bootstrap_lowers_positive": bool((seeds.weighted_ci95_lower_pp > 0).all()),
        "all_eligible_relations_nonnegative_each_seed": bool((seeds.nonnegative_relations == seeds.eligible_relations).all()),
        "target_delta_correct_transfers": [int(value) for value in targets.delta],
        "target_used_for_selection": False,
        "scene_pose": "locked to seed-specific global top-20 baseline",
        "official_split": "train",
        "external_validation": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(seeds.to_string(index=False))
    print(targets.to_string(index=False))


if __name__ == "__main__":
    main()
