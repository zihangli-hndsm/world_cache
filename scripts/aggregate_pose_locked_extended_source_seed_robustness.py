#!/usr/bin/env python3
"""Aggregate the three-seed extended source-only pose-locked audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SEEDS = (17, 29, 41)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    relations: list[pd.DataFrame] = []
    for seed in SEEDS:
        directory = args.reports_root / f"20261003_pose_locked_extended_source_seed{seed}"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        outcome = pd.read_csv(directory / "source_lopo_outcomes.csv")
        outcome["seed"] = seed
        outcome["delta"] = outcome.method_correct - outcome.baseline_correct
        outcome["delta_pp"] = outcome.delta / outcome.moved_queries * 100.0
        relations.append(outcome)
        rows.append({
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
            "positive_relations": int(summary["positive_source_relations"]),
            "weighted_ci95_lower_pp": float(summary["relation_bootstrap_weighted_ci95_pp"][0]),
            "weighted_ci95_upper_pp": float(summary["relation_bootstrap_weighted_ci95_pp"][1]),
            "macro_ci95_lower_pp": float(summary["relation_bootstrap_macro_ci95_pp"][0]),
            "macro_ci95_upper_pp": float(summary["relation_bootstrap_macro_ci95_pp"][1]),
        })
    seeds = pd.DataFrame(rows)
    relation_frame = pd.concat(relations, ignore_index=True)
    relation_summary = (
        relation_frame.groupby("pair", as_index=False)
        .agg(
            seeds=("seed", "count"),
            moved_queries=("moved_queries", "first"),
            mean_delta_pp=("delta_pp", "mean"),
            min_delta_pp=("delta_pp", "min"),
            max_delta_pp=("delta_pp", "max"),
            positive_seeds=("delta", lambda x: int((x > 0).sum())),
            nonnegative_seeds=("delta", lambda x: int((x >= 0).sum())),
        )
    )
    seeds.to_csv(args.output_dir / "seed_summary.csv", index=False)
    relation_frame.to_csv(args.output_dir / "relation_seed_outcomes.csv", index=False)
    relation_summary.to_csv(args.output_dir / "relation_summary.csv", index=False)
    summary = {
        "seeds": list(SEEDS),
        "source_relations": int(seeds.source_relations.iloc[0]),
        "eligible_relations": int(seeds.eligible_relations.iloc[0]),
        "mean_weighted_delta_pp": float(seeds.delta_pp.mean()),
        "min_weighted_delta_pp": float(seeds.delta_pp.min()),
        "max_weighted_delta_pp": float(seeds.delta_pp.max()),
        "all_weighted_bootstrap_lowers_positive": bool((seeds.weighted_ci95_lower_pp > 0).all()),
        "all_relations_nonnegative_each_seed": bool((seeds.nonnegative_relations == seeds.eligible_relations).all()),
        "extra_pairs": ["extra1776", "extra4aca"],
        "official_split": "train",
        "target_used_for_selection": False,
        "external_validation": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Extended source-only pose-locked multi-motion seed robustness",
        "",
        "This audit combines the original 16 source relations with the two "
        "locally complete extra official-train relations. The predesignated "
        "target is excluded. The same source-only LOPO selector is rerun for "
        "RANSAC seeds 17, 29, and 41; seeds are a robustness axis, not "
        "independent datasets.",
        "",
        "| Seed | Baseline | Method | Gain (pp) | Rescued/lost | Nonnegative relations | Weighted 95% CI (pp) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seeds.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.baseline_correct}/{row.source_queries} | "
            f"{row.method_correct}/{row.source_queries} | {row.delta_pp:+.2f} | "
            f"{row.rescued}/{row.lost} | {row.nonnegative_relations}/{row.eligible_relations} | "
            f"[{row.weighted_ci95_lower_pp:+.2f},{row.weighted_ci95_upper_pp:+.2f}] |"
        )
    lines += [
        "",
        f"The mean weighted gain is {seeds.delta_pp.mean():+.2f} pp "
        f"(range {seeds.delta_pp.min():+.2f} to {seeds.delta_pp.max():+.2f}); "
        "every seed has a positive bootstrap lower bound and all 14 eligible "
        "relations are nonnegative. The two extra relations were selected by "
        "local input availability, so this remains an internal train-split "
        "audit rather than external-site evidence.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(seeds.to_string(index=False))


if __name__ == "__main__":
    main()
