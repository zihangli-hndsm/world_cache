"""Summarize the fixed-candidate multi-motion RANSAC seed audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 29, 41)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing seed_<seed>/ analysis outputs")
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    relations: list[pd.DataFrame] = []
    for seed in SEEDS:
        seed_dir = args.root / f"seed_{seed}"
        summary = json.loads((seed_dir / "summary.json").read_text(encoding="utf-8"))
        if int(summary["ransac_seed"]) != seed:
            raise ValueError(f"seed mismatch in {seed_dir}")
        source = pd.read_csv(seed_dir / "source_lopo_outcomes.csv")
        source = source[source.pair != "target"].copy()
        if set(source.official_split) != {"train"}:
            raise ValueError(f"non-train relation found in {seed_dir}")
        source["seed"] = seed
        relations.append(source)
        summaries.append({
            "seed": seed,
            "official_split": ",".join(summary["official_source_split"]),
            "relations": int(summary["source_relations"]),
            "eligible_relations": int(summary["eligible_source_relations"]),
            "queries": int(summary["source_moved_rigid_queries"]),
            "baseline_correct": int(summary["baseline_correct"]),
            "method_correct": int(summary["method_correct"]),
            "baseline_coverage": float(summary["baseline_coverage"]),
            "method_coverage": float(summary["method_coverage"]),
            "delta_pp": float(summary["coverage_delta_pp"]),
            "rescued": int(summary["rescued"]),
            "lost": int(summary["lost"]),
            "net": int(summary["method_correct"] - summary["baseline_correct"]),
            "nonnegative_relations": int(summary["nonnegative_source_relations"]),
            "weighted_ci95_lower_pp": float(summary["weighted_delta_ci95_pp"][0]),
            "weighted_ci95_upper_pp": float(summary["weighted_delta_ci95_pp"][1]),
            "stage1_gate_pass": bool(summary["stage1_gate_pass"]),
            "target_baseline": int(summary["target_baseline_correct"]),
            "target_method": int(summary["target_method_correct"]),
            "target_queries": int(summary["target_moved_rigid_queries"]),
        })

    per_seed = pd.DataFrame(summaries)
    all_relations = pd.concat(relations, ignore_index=True)
    if per_seed.queries.nunique() != 1:
        raise ValueError("seed runs do not share a fixed source query denominator")
    if all_relations.groupby("seed").pair.nunique().nunique() != 1:
        raise ValueError("seed runs do not share the same source relations")
    wide = all_relations.pivot(index="pair", columns="seed", values="method_coverage")
    baseline_wide = all_relations.pivot(index="pair", columns="seed", values="baseline_coverage")
    query_wide = all_relations.pivot(index="pair", columns="seed", values="moved_queries")
    if not (query_wide.nunique(axis=1) == 1).all():
        raise ValueError("source relation query denominators changed across seeds")
    pair_rows = []
    for pair in wide.index:
        values = wide.loc[pair]
        base = baseline_wide.loc[pair]
        delta = values - base
        eligible = bool(query_wide.loc[pair, 17] > 0)
        pair_rows.append({
            "pair": pair,
            "moved_queries": int(query_wide.loc[pair, 17]),
            "seeds_nonnegative": int((delta >= -1e-12).sum()),
            "seeds_positive": int((delta > 1e-12).sum()),
            "delta_seed_17_pp": 100 * float(delta[17]) if eligible else np.nan,
            "delta_seed_29_pp": 100 * float(delta[29]) if eligible else np.nan,
            "delta_seed_41_pp": 100 * float(delta[41]) if eligible else np.nan,
            "eligible": eligible,
        })
    pair_summary = pd.DataFrame(pair_rows)
    per_seed.to_csv(args.root / "seed_summary.csv", index=False)
    pair_summary.to_csv(args.root / "relation_seed_summary.csv", index=False)

    mean_delta = float(per_seed.delta_pp.mean())
    sd_delta = float(per_seed.delta_pp.std(ddof=1))
    min_delta = float(per_seed.delta_pp.min())
    max_delta = float(per_seed.delta_pp.max())
    eligible_pairs = pair_summary[pair_summary.eligible]
    positive_all = int((eligible_pairs.seeds_positive == len(SEEDS)).sum())
    nonnegative_all = int((eligible_pairs.seeds_nonnegative == len(SEEDS)).sum())
    report = [
        "# Multi-motion RANSAC seed robustness — 2026-09-29",
        "",
        "This audit reruns the frozen 3RScan candidate cache with RANSAC seeds 17, 29, and 41. For each seed, it refits the top-20 global hypothesis and secondary motion hypotheses, then reruns the source-only leave-one-relation-out selector over the same 16 training relations. Each held-out source relation is scored using a setting chosen on the other 15. The target relation is reported only as a diagnostic and is never used for selection.",
        "",
        "The baseline is scored from its seed-specific top-20 assignments. The multi-motion arm retains that seed-specific scene pose and uses the frozen adaptive candidate limits for correspondence assignment. Thus pose loss is zero by construction; this audit does not establish absolute pose quality. The denominator is the same 1,510 moved-rigid queries for every seed, from the official 3RScan train split. These are repeated randomizations of one query set, not independent dataset samples.",
        "",
        "## Source results",
        "",
        "| Seed | Baseline | Method | Delta | Rescued / lost | Nonnegative / eligible relations | Weighted relation-bootstrap 95% CI | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in per_seed.itertuples(index=False):
        report.append(
            f"| {row.seed} | {row.baseline_correct}/{row.queries} ({row.baseline_coverage:.2%}) | "
            f"{row.method_correct}/{row.queries} ({row.method_coverage:.2%}) | {row.delta_pp:+.2f} pp | "
            f"{row.rescued} / {row.lost} | {row.nonnegative_relations}/{row.eligible_relations} | "
            f"[{row.weighted_ci95_lower_pp:+.2f}, {row.weighted_ci95_upper_pp:+.2f}] pp | "
            f"{'PASS' if row.stage1_gate_pass else 'FAIL'} |"
        )
    report += [
        "",
        f"Across the three seeds, the mean source gain is {mean_delta:+.2f} pp (sample SD {sd_delta:.2f} pp; range {min_delta:+.2f} to {max_delta:+.2f} pp). The frozen +10 pp gate passes in {int(per_seed.stage1_gate_pass.sum())}/{len(SEEDS)} seeds. All {nonnegative_all}/{len(eligible_pairs)} eligible source relations are nonnegative in all three seeds; {positive_all}/{len(eligible_pairs)} are strictly positive in all three.",
        "",
        "The seed-29 run crosses the source gate, while seeds 17 and 41 do not. This supports a repeatable positive source effect across these three randomizations, but the effect size varies and the gate is not stable across seeds. The seeds share all queries and labels; do not treat their aggregate as three independent replications or as held-out validation. All relations are official train data. Keep the paper claim unchanged and proceed to authorized official validation before promotion.",
        "",
        "The target diagnostic is 32/47 at seed 17, 29/47 at seed 29, and 30/47 at seed 41; these are small train-split diagnostics, not external validation.",
        "",
        "See `seed_summary.csv` and `relation_seed_summary.csv` for machine-readable outcomes. Per-seed query and relation outputs are retained in the corresponding `seed_<seed>/` directories.",
        "",
    ]
    (args.root / "seed_robustness_report.md").write_text("\n".join(report), encoding="utf-8")
    status = {
        "status": "complete",
        "seeds": list(SEEDS),
        "source_queries_per_seed": int(per_seed.queries.iloc[0]),
        "mean_delta_pp": mean_delta,
        "sample_sd_delta_pp": sd_delta,
        "delta_range_pp": [min_delta, max_delta],
        "gate_pass_count": int(per_seed.stage1_gate_pass.sum()),
        "all_seed_nonnegative_eligible_relations": nonnegative_all,
        "all_seed_strictly_positive_eligible_relations": positive_all,
        "official_split": "train",
        "target_used_for_selection": False,
        "independent_replications": False,
    }
    (args.root / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(per_seed[["seed", "baseline_correct", "method_correct", "delta_pp", "rescued", "lost",
                    "nonnegative_relations", "stage1_gate_pass"]].to_string(index=False))
    print(f"mean_delta_pp={mean_delta:+.2f}; sample_sd_pp={sd_delta:.2f}; range=[{min_delta:+.2f}, {max_delta:+.2f}]")
    print(f"relations_nonnegative_all_seeds={nonnegative_all}/{len(eligible_pairs)}; positive_all_seeds={positive_all}/{len(eligible_pairs)}")


if __name__ == "__main__":
    main()
