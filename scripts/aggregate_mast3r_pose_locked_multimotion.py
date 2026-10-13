#!/usr/bin/env python3
"""Aggregate fixed-candidate MASt3R multi-motion LOPO seed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    relation_frames = []
    for folder in args.input_dir:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        relations = pd.read_csv(folder / "source_lopo_outcomes.csv")
        if len(relations) != int(summary["source_relations"]):
            raise ValueError(f"relation count does not match summary in {folder}")
        summaries.append(summary)
        relations["seed"] = int(summary["seed"])
        relation_frames.append(relations)
    if len({summary["seed"] for summary in summaries}) != len(summaries):
        raise ValueError("duplicate RANSAC seeds")
    query_counts = {summary["source_moved_rigid_queries"] for summary in summaries}
    relation_sets = {tuple(sorted(frame.pair)) for frame in relation_frames}
    if len(query_counts) != 1 or len(relation_sets) != 1:
        raise ValueError("seed runs do not share the same relation/query population")

    per_seed = pd.DataFrame([{
        "seed": item["seed"],
        "source_relations": item["source_relations"],
        "moved_rigid_queries": item["source_moved_rigid_queries"],
        "top20_candidate_recall": item["source_candidate_top20_recall"],
        "baseline_correct": item["source_baseline_correct"],
        "method_correct": item["source_method_correct"],
        "gain_pp": item["source_gain_pp"],
        "rescued": item["source_rescued"],
        "lost": item["source_lost"],
        "nonnegative_relations": item["source_nonnegative_relations"],
        "weighted_ci95_lower_pp": item["source_weighted_relation_bootstrap_ci95_pp"][0],
        "weighted_ci95_upper_pp": item["source_weighted_relation_bootstrap_ci95_pp"][1],
        "macro_ci95_lower_pp": item["source_macro_relation_bootstrap_ci95_pp"][0],
        "macro_ci95_upper_pp": item["source_macro_relation_bootstrap_ci95_pp"][1],
    } for item in sorted(summaries, key=lambda row: row["seed"])])
    relation_frame = pd.concat(relation_frames, ignore_index=True).sort_values(["pair", "seed"])
    by_relation = relation_frame.pivot(index="pair", columns="seed", values="method_correct")
    baseline = relation_frame.pivot(index="pair", columns="seed", values="baseline_correct")
    if not by_relation.columns.equals(baseline.columns):
        raise ValueError("baseline/method seed columns mismatch")
    nonnegative_every_seed = int(((by_relation - baseline) >= 0).all(axis=1).sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.output_dir / "per_seed.csv", index=False)
    relation_frame.to_csv(args.output_dir / "per_relation_seed.csv", index=False)
    gains = per_seed.gain_pp.to_numpy(dtype=np.float64)
    summary = {
        "seeds": per_seed.seed.astype(int).tolist(),
        "source_relations": int(per_seed.source_relations.iloc[0]),
        "source_moved_rigid_queries": int(per_seed.moved_rigid_queries.iloc[0]),
        "top20_candidate_recall": float(per_seed.top20_candidate_recall.iloc[0]),
        "gain_pp_by_seed": {str(int(row.seed)): float(row.gain_pp) for row in per_seed.itertuples(index=False)},
        "mean_gain_pp": float(gains.mean()),
        "sample_sd_gain_pp": float(gains.std(ddof=1)) if len(gains) > 1 else 0.0,
        "min_gain_pp": float(gains.min()),
        "max_gain_pp": float(gains.max()),
        "relation_bootstrap_weighted_ci95_by_seed": {
            str(int(row.seed)): [float(row.weighted_ci95_lower_pp), float(row.weighted_ci95_upper_pp)]
            for row in per_seed.itertuples(index=False)
        },
        "nonnegative_relations_all_seeds": nonnegative_every_seed,
        "relations_nonnegative_per_seed": {
            str(int(row.seed)): int(row.nonnegative_relations)
            for row in per_seed.itertuples(index=False)
        },
        "independent_replications": False,
        "target_relation_evaluated": False,
        "scope": "14 official-train source relations; fixed MASt3R top-20 candidate sets; source-only leave-one-relation-out selector",
        "warning": "Seed runs share data and candidate sets; relation-bootstrap intervals include source relation variability but not independent-site validation.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MASt3R candidate multi-motion seed robustness", "",
        f"The frozen MASt3R top-20 candidate sets were re-evaluated over {summary['source_relations']} official-train source relations and {summary['source_moved_rigid_queries']} linked moved-rigid queries. Candidate Recall@20 is {100*summary['top20_candidate_recall']:.1f}%. The source-only leave-one-relation-out secondary-motion switch improves over the same global RANSAC candidate baseline by {summary['mean_gain_pp']:+.2f} pp on average (seed range {summary['min_gain_pp']:+.2f} to {summary['max_gain_pp']:+.2f} pp); {nonnegative_every_seed}/{summary['source_relations']} relations are nonnegative at every seed.",
        "", "| Seed | Baseline correct | Method correct | Gain (pp) | Relation-bootstrap 95% CI | Relations nonnegative |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed.itertuples(index=False):
        lines.append(
            f"| {int(row.seed)} | {int(row.baseline_correct)} | {int(row.method_correct)} | {row.gain_pp:+.2f} | [{row.weighted_ci95_lower_pp:+.2f},{row.weighted_ci95_upper_pp:+.2f}] | {int(row.nonnegative_relations)}/{int(row.source_relations)} |"
        )
    lines += [
        "", "The three runs use the same relation/query and candidate pools, so they are algorithmic seed-sensitivity checks rather than independent replications. All relations are official train data; the predesignated held-out target is not present in these caches. Physical site independence is unverified.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
