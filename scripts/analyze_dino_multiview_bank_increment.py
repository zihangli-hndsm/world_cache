#!/usr/bin/env python3
"""Paired reference-group analysis of increasing DINO exemplar-bank size.

The view-count curve already reports each bank against the REMIND-style
tracker. This analysis instead compares every larger bank with the one-view
bank under the same Hungarian matcher and borrowed per-relation acceptance
budgets. It is an exploratory reanalysis of the existing sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/reports/20261009_dino_multiview_source_bank/view_budget_curve.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/20261009_dino_multiview_bank_increment"
BANK_SIZES = (1, 2, 4, 8, 12, 20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--subset-bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        input_display = input_path.relative_to(ROOT).as_posix()
    except ValueError:
        input_display = str(input_path)
    try:
        output_display = output_dir.relative_to(ROOT).as_posix()
    except ValueError:
        output_display = str(output_dir)

    data = json.loads(input_path.read_text())
    if data.get("status") != "complete":
        raise ValueError("view-count sweep is incomplete")
    rows = data["relation_repeat_rows"]
    envs = sorted({str(row["environment_id"]) for row in rows})
    expected_groups = int(data["cohort"]["reference_groups"])
    if len(envs) != expected_groups:
        raise ValueError(f"expected {expected_groups} reference groups, found {len(envs)}")

    # First pool relation counts within each reference group and repeat, then
    # compute group-macro precision/coverage after averaging subset repeats.
    grouped: dict[tuple[int, int, str], dict[str, int]] = {}
    for row in rows:
        key = (int(row["bank_size"]), int(row["repeat"]), str(row["environment_id"]))
        cell = grouped.setdefault(key, {"query_masks": 0, "accepted": 0, "correct": 0})
        for field in cell:
            cell[field] += int(row[field])

    group_rates: dict[int, dict[str, np.ndarray]] = {}
    relation_ids = sorted({str(row["run_id"]) for row in rows})
    relation_group = {
        run_id: next(str(row["environment_id"]) for row in rows if str(row["run_id"]) == run_id)
        for run_id in relation_ids
    }
    relation_index = {run_id: i for i, run_id in enumerate(relation_ids)}
    group_relations = {
        env: [relation_index[run_id] for run_id in relation_ids if relation_group[run_id] == env]
        for env in envs
    }
    subset_group_rates: dict[int, dict[str, np.ndarray]] = {}
    subset_rng = np.random.default_rng(args.seed + 1)
    for size in BANK_SIZES:
        repeats = sorted({rep for bank, rep, _ in grouped if bank == size})
        expected_repeats = 1 if size == 20 else 20
        if len(repeats) != expected_repeats:
            raise ValueError(f"bank size {size}: expected {expected_repeats} repeats, got {repeats}")
        precision = np.empty((len(repeats), len(envs)), dtype=np.float64)
        coverage = np.empty_like(precision)
        for ri, repeat in enumerate(repeats):
            for ei, env in enumerate(envs):
                cell = grouped[(size, repeat, env)]
                if cell["accepted"] <= 0 or cell["query_masks"] <= 0:
                    raise ValueError(f"empty denominator for bank={size}, repeat={repeat}, group={env}")
                precision[ri, ei] = cell["correct"] / cell["accepted"]
                coverage[ri, ei] = cell["correct"] / cell["query_masks"]
        group_rates[size] = {
            "precision": precision.mean(axis=0),
            "coverage": coverage.mean(axis=0),
        }

        # Resample the 20 independently drawn reference-view subsets within
        # each relation, then pool relations into their reference group. The
        # 20-view bank is deterministic; resampling its single run is inert.
        subset_correct = np.empty((args.subset_bootstrap_draws, len(relation_ids)), dtype=np.float64)
        relation_accepts = np.empty(len(relation_ids), dtype=np.float64)
        relation_queries = np.empty(len(relation_ids), dtype=np.float64)
        for run_id, ri in relation_index.items():
            run_rows = sorted(
                [row for row in rows if row["bank_size"] == size and str(row["run_id"]) == run_id],
                key=lambda row: int(row["repeat"]),
            )
            correct_values = np.asarray([int(row["correct"]) for row in run_rows], dtype=np.float64)
            accepts = np.asarray([int(row["accepted"]) for row in run_rows], dtype=np.int64)
            queries = np.asarray([int(row["query_masks"]) for row in run_rows], dtype=np.int64)
            if len(correct_values) != expected_repeats or np.unique(accepts).size != 1 or np.unique(queries).size != 1:
                raise ValueError(f"inconsistent repeat rows for bank={size}, relation={run_id}")
            sample = subset_rng.integers(
                0, len(correct_values), size=(args.subset_bootstrap_draws, len(correct_values))
            )
            subset_correct[:, ri] = correct_values[sample].mean(axis=1)
            relation_accepts[ri] = accepts[0]
            relation_queries[ri] = queries[0]
        sampled_precision = np.empty((args.subset_bootstrap_draws, len(envs)), dtype=np.float64)
        sampled_coverage = np.empty_like(sampled_precision)
        for ei, env in enumerate(envs):
            ridx = group_relations[env]
            sampled_precision[:, ei] = subset_correct[:, ridx].sum(axis=1) / relation_accepts[ridx].sum()
            sampled_coverage[:, ei] = subset_correct[:, ridx].sum(axis=1) / relation_queries[ridx].sum()
        subset_group_rates[size] = {"precision": sampled_precision, "coverage": sampled_coverage}

    # The one-view arm is deterministic because its only source frame is fixed
    # by query pairing. Broadcast it across subset-bootstrap draws.
    for outcome in ("precision", "coverage"):
        subset_group_rates[1][outcome] = np.broadcast_to(
            group_rates[1][outcome], (args.subset_bootstrap_draws, len(envs))
        )

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(envs), size=(args.draws, len(envs)))
    nested_rng = np.random.default_rng(args.seed + 2)
    contrasts = []
    for size in BANK_SIZES[1:]:
        for outcome in ("precision", "coverage"):
            delta = group_rates[size][outcome] - group_rates[1][outcome]
            bootstrap = delta[indices].mean(axis=1)
            subset_choice = nested_rng.integers(0, args.subset_bootstrap_draws, size=args.draws)
            nested_delta = (
                subset_group_rates[size][outcome] - subset_group_rates[1][outcome]
            )
            nested_bootstrap = nested_delta[subset_choice[:, None], indices].mean(axis=1)
            contrasts.append({
                "bank_views": size,
                "outcome": outcome,
                "estimate": float(delta.mean()),
                "percentile_95": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
                "bonferroni_10_effect_family_95": [
                    float(x) for x in np.quantile(bootstrap, [0.0025, 0.9975])
                ],
                "subset_and_group_bootstrap_95": [
                    float(x) for x in np.quantile(nested_bootstrap, [0.025, 0.975])
                ],
                "subset_and_group_bonferroni_10_effect_family_95": [
                    float(x) for x in np.quantile(nested_bootstrap, [0.0025, 0.9975])
                ],
                "groups_positive": int((delta > 0).sum()),
                "groups_negative": int((delta < 0).sum()),
                "groups_tied": int((delta == 0).sum()),
            })

    summary = {
        "status": "complete",
        "scope": f"exploratory reanalysis of the already explored {data['cohort']['relations']}-relation 3RScan sweep",
        "estimand": "group-macro paired difference from one-view Hungarian exemplar bank; each bank uses the exact per-relation REMIND accepted-link count ranked by assigned cosine",
        "association_rule": "per-query-frame rectangular Hungarian assignment for every bank size",
        "view_subset_repeats_averaged_within_group": 20,
        "bootstrap": {
            "draws": args.draws,
            "seed": args.seed,
            "unit": f"{len(envs)} released reference groups",
            "subset_bootstrap_draws": args.subset_bootstrap_draws,
            "nested_protocol": "for each bank size, resample the 20 observed subset-run outcomes with replacement independently within each relation, pool sampled correct counts within reference groups, then resample reference groups; the one-view bank is deterministic",
            "multiple_comparison_family": "five larger bank sizes by two outcomes; Bonferroni percentile intervals",
        },
        "limits": [
            "group-only intervals are conditional on the 20 observed view subsets",
            "nested intervals use a nonparametric bootstrap over those 20 observed subsets within each relation; they reflect subset sampling variability only to the extent represented by those runs",
            "view subsets are Monte Carlo samples, not independent physical sites",
            "the acceptance budget is borrowed post hoc from the REMIND-style tracker",
            "annotation-derived masks and a previously explored benchmark are used",
            "this does not establish independent-site generalization or novelty over existing multi-view memory systems",
        ],
        "contrasts_vs_one_view": contrasts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# DINO exemplar-bank increment over one view",
        "",
        f"Exploratory paired reanalysis of the existing {data['cohort']['relations']}-relation 3RScan sweep over {len(envs)} reference groups. Every bank size uses the same per-frame Hungarian matcher and the exact per-relation REMIND acceptance count ranked by assigned-pair cosine. Group rates average the 20 sampled subsets within each reference group before paired bootstrap resampling. A nested sensitivity additionally resamples the 20 subset-run outcomes with replacement independently within each relation, then resamples reference groups (10,000 inner and 100,000 outer draws).",
        "",
        "| Views | Outcome | Difference vs. one view (pp) | Group-only Bonferroni interval | Nested subset + group Bonferroni interval | Groups positive / negative / tied |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['bank_views']} | {row['outcome']} | {100*row['estimate']:+.2f} | "
            f"[{100*row['bonferroni_10_effect_family_95'][0]:+.2f}, {100*row['bonferroni_10_effect_family_95'][1]:+.2f}] | "
            f"[{100*row['subset_and_group_bonferroni_10_effect_family_95'][0]:+.2f}, {100*row['subset_and_group_bonferroni_10_effect_family_95'][1]:+.2f}] | "
            f"{row['groups_positive']} / {row['groups_negative']} / {row['groups_tied']} |"
        )
    lines.extend([
        "",
        f"Group-only intervals condition on the observed view-subset draws; nested intervals resample their 20 run outcomes within each relation and therefore reflect subset sampling variability only to the extent represented by those runs. Both methods resample the {len(envs)} released reference groups, which do not quantify physical-site independence. This cohort was explored, and the acceptance count is borrowed post hoc. Treat all effects as descriptive. This analysis does not establish a new memory method or independent-site generalization.",
        "",
        f"Reproduce with `python scripts/analyze_dino_multiview_bank_increment.py --input {input_display} --output-dir {output_display} --draws {args.draws} --subset-bootstrap-draws {args.subset_bootstrap_draws} --seed {args.seed}`.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
