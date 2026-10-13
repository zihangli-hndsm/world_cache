#!/usr/bin/env python3
"""Measure query-level transfer under fixed MASt3R RANSAC seed agreement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 29, 41)
KEY = ["query_index", "frame_id"]


def relation_bootstrap(values: np.ndarray, rng: np.random.Generator, n: int) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def ratio_bootstrap(nums: np.ndarray, dens: np.ndarray, rng: np.random.Generator,
                    n: int) -> list[float]:
    ix = rng.integers(0, len(nums), size=(n, len(nums)))
    num, den = nums[ix].sum(axis=1), dens[ix].sum(axis=1)
    vals = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
    return [float(v) for v in np.nanquantile(vals, [0.025, 0.975])]


def read_relation(root: Path, relation: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for seed in SEEDS:
        path = root / relation / f"seed_{seed}" / "query_diagnostics.csv"
        frame = pd.read_csv(path)
        required = set(KEY + ["candidate_count", "physical_instance_candidate_hit",
                              "physical_instance_evaluable", "physical_instance_correct",
                              "consensus_inlier", "selected_reference_object_id",
                              "physical_correct_transfer"])
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if frame.duplicated(KEY).any():
            raise ValueError(f"{path}: duplicate query/frame rows")
        cols = KEY + ["candidate_count", "physical_instance_candidate_hit",
                      "physical_instance_evaluable", "physical_instance_correct",
                      "consensus_inlier", "selected_reference_object_id",
                      "physical_correct_transfer"]
        frame = frame[cols].rename(columns={
            c: f"{c}_s{seed}" for c in cols if c not in KEY
        })
        merged = frame if merged is None else merged.merge(
            frame, on=KEY, how="outer", validate="one_to_one", indicator=f"merge_{seed}"
        )
        if f"merge_{seed}" in merged:
            if not (merged[f"merge_{seed}"] == "both").all():
                raise ValueError(f"{relation}: seed {seed} query population mismatch")
            merged = merged.drop(columns=f"merge_{seed}")
    assert merged is not None
    for field in ("candidate_count", "physical_instance_candidate_hit",
                  "physical_instance_evaluable"):
        columns = [f"{field}_s{s}" for s in SEEDS]
        if not (merged[columns].nunique(axis=1) == 1).all():
            raise ValueError(f"{relation}: {field} labels differ across seeds")
    # The saved outcome must match the physical correctness label whenever a
    # consensus inlier selected a reference object.
    for seed in SEEDS:
        inlier = merged[f"consensus_inlier_s{seed}"].astype(bool)
        selected = merged[f"selected_reference_object_id_s{seed}"].fillna(-1).astype(int)
        outcome = merged[f"physical_correct_transfer_s{seed}"].astype(bool)
        expected = inlier & merged[f"physical_instance_correct_s{seed}"].astype(bool)
        if not (outcome == expected).all():
            raise ValueError(f"{relation}: seed {seed} saved transfer outcome is inconsistent")
        merged[f"vote_s{seed}"] = np.where(inlier & (selected > 0), selected, -1)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    relation_dirs = sorted(p.name for p in args.seed_root.iterdir()
                           if p.is_dir() and (p / "seed_17/query_diagnostics.csv").is_file())
    if len(relation_dirs) != 14:
        raise ValueError(f"expected the 14 matched relations, found {len(relation_dirs)}")
    relations = {name: read_relation(args.seed_root, name) for name in relation_dirs}
    if sum(int(df.physical_instance_evaluable_s17.astype(bool).sum())
           for df in relations.values()) != 12_188:
        raise ValueError("expected 12,188 physically evaluable query rows")

    rows: list[dict[str, object]] = []
    per_relation: dict[str, dict[str, dict[str, int]]] = {}
    for relation, df in relations.items():
        eligible = df.physical_instance_evaluable_s17.astype(bool).to_numpy()
        votes = df[[f"vote_s{s}" for s in SEEDS]].to_numpy(dtype=int)
        methods: dict[str, np.ndarray] = {}
        single_assigned: dict[str, np.ndarray] = {}
        for seed in SEEDS:
            single_assigned[f"seed_{seed}"] = (
                df[f"consensus_inlier_s{seed}"].astype(bool).to_numpy() & eligible
            )
            methods[f"seed_{seed}"] = (
                df[f"physical_correct_transfer_s{seed}"].astype(bool).to_numpy() & eligible
            )
        for quorum in (2, 3):
            prediction = np.full(len(df), -1, dtype=int)
            agreement = np.zeros(len(df), dtype=int)
            for i, row_votes in enumerate(votes):
                valid = row_votes[row_votes > 0]
                if len(valid):
                    ids, counts = np.unique(valid, return_counts=True)
                    best = int(np.argmax(counts))
                    prediction[i], agreement[i] = int(ids[best]), int(counts[best])
            assigned = (agreement >= quorum) & eligible
            correct = np.zeros(len(df), dtype=bool)
            # Check against stored transfer outcomes from each seed supporting the winning vote.
            for i in np.flatnonzero(assigned):
                supporters = [s for j, s in enumerate(SEEDS) if votes[i, j] == prediction[i]]
                expected = [bool(df.iloc[i][f"physical_correct_transfer_s{s}"]) for s in supporters]
                if len(set(expected)) != 1:
                    raise ValueError(f"{relation}: supporting seed transfer labels disagree")
                correct[i] = expected[0]
            methods[f"quorum_{quorum}"] = correct
            row = {
                "relation": relation, "operating_point": f"{quorum}_of_3",
                "eligible_queries": int(eligible.sum()), "assigned": int(assigned.sum()),
                "correct": int(correct.sum()),
                "coverage": float(assigned.sum() / eligible.sum()),
                "precision": float(correct.sum() / assigned.sum()) if assigned.sum() else np.nan,
                "correct_transfer_rate": float(correct.sum() / eligible.sum()),
            }
            rows.append(row)
            per_relation.setdefault(relation, {})[f"q{quorum}"] = {
                "n": int(eligible.sum()), "assigned": int(assigned.sum()),
                "correct": int(correct.sum()),
            }
        for seed in SEEDS:
            correct = methods[f"seed_{seed}"]
            assigned = single_assigned[f"seed_{seed}"]
            n = int(eligible.sum())
            hits = int(correct.sum())
            assigned_n = int(assigned.sum())
            rows.append({
                "relation": relation, "operating_point": f"seed_{seed}",
                "eligible_queries": n, "assigned": assigned_n, "correct": hits,
                "coverage": assigned_n / n,
                "precision": hits / assigned_n if assigned_n else np.nan,
                "correct_transfer_rate": hits / n,
            })
            per_relation[relation][f"seed{seed}"] = {
                "n": n, "assigned": assigned_n, "correct": hits
            }

    per_df = pd.DataFrame(rows)
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "relations": len(relations), "eligible_queries": 12_188,
        "bootstrap_unit": "relation; each relation contributes equal weight to macro estimates",
        "bootstrap_samples": args.bootstrap_samples, "rng_seed": args.seed,
        "limitations": [
            "fixed candidates and fixed train relations; this is a seed sensitivity diagnostic, not an independent replication",
            "seed agreement costs up to three RANSAC runs and is not a calibrated confidence guarantee",
            "no official validation split or unseen-site generalization is represented",
        ],
        "operating_points": {},
    }
    piv = per_df.pivot(index="relation", columns="operating_point")
    for op in ("seed_17", "seed_29", "seed_41", "2_of_3", "3_of_3"):
        n = per_df[per_df.operating_point == op].eligible_queries.to_numpy(dtype=float)
        a = per_df[per_df.operating_point == op].assigned.to_numpy(dtype=float)
        c = per_df[per_df.operating_point == op].correct.to_numpy(dtype=float)
        summary["operating_points"][op] = {
            "assigned": int(a.sum()), "correct": int(c.sum()),
            "pooled_coverage": float(a.sum() / n.sum()),
            "pooled_precision": float(c.sum() / a.sum()) if a.sum() else None,
            "pooled_correct_transfer_rate": float(c.sum() / n.sum()),
            "relation_macro_coverage": float(np.mean(a / n)),
            "relation_macro_coverage_ci95": relation_bootstrap(a / n, rng, args.bootstrap_samples),
            "relation_macro_correct_transfer_rate": float(np.mean(c / n)),
            "relation_macro_correct_transfer_ci95": relation_bootstrap(c / n, rng, args.bootstrap_samples),
            "pooled_precision_relation_bootstrap_ci95": ratio_bootstrap(c, a, rng, args.bootstrap_samples),
        }
    for q in ("2_of_3", "3_of_3"):
        diff = (piv["correct_transfer_rate"][q] - piv["correct_transfer_rate"]["seed_17"]).to_numpy()
        summary["operating_points"][q]["vs_seed_17_relation_macro_transfer_delta"] = float(diff.mean())
        summary["operating_points"][q]["vs_seed_17_delta_ci95"] = relation_bootstrap(
            diff, rng, args.bootstrap_samples
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_df.to_csv(args.output_dir / "per_relation.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Dense MASt3R seed agreement diagnostic", "",
        "Three RANSAC runs reuse the same fixed dense candidate correspondences. A query is assigned when at least two or all three consensus-inlier runs select the same positive reference object ID. The selected ID is scored against the linked physical instance. Single-seed rows assign each seed's own consensus inliers.", "",
        "| Operating point | Correct / eligible | Assignment coverage | Precision when assigned | Correct transfer / eligible |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = [("seed_17", "Seed 17"), ("seed_29", "Seed 29"), ("seed_41", "Seed 41"),
              ("2_of_3", "Agreement 2/3"), ("3_of_3", "Agreement 3/3")]
    for op, label in labels:
        x = summary["operating_points"][op]
        ci = x["pooled_precision_relation_bootstrap_ci95"]
        precision = "n/a" if x["pooled_precision"] is None else f"{x['pooled_precision']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]"
        lines.append(f"| {label} | {x['correct']:,}/12,188 | {x['pooled_coverage']:.1%} | {precision} | {x['pooled_correct_transfer_rate']:.1%} |")
    lines += ["", "Intervals resample the 14 relations. For precision, the ratio of summed correct and assigned counts is recalculated in each relation bootstrap draw.", ""]
    for q, label in (("2_of_3", "2/3"), ("3_of_3", "3/3")):
        x = summary["operating_points"][q]
        ci = x["vs_seed_17_delta_ci95"]
        lines.append(f"At {label} agreement, relation-macro correct-transfer rate differs from seed 17 by {x['vs_seed_17_relation_macro_transfer_delta']*100:+.2f} pp (95% relation-bootstrap CI [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp).")
    lines += ["", "This is a post-hoc train-set sensitivity analysis on 14 availability-selected relations. Agreement improves selective precision at the cost of coverage, but the result is not an independent replication or a calibrated risk guarantee. Three seeds multiply RANSAC inference cost; feature extraction and candidate sets are shared.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
