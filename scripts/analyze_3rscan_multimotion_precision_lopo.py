#!/usr/bin/env python3
"""Select pose-locked motion rules under a source-only precision floor.

Each outer source relation is held out. On the other 15 relations, choose the
configuration with the most correct moved-rigid transfers subject to pooled
physical-instance assignment precision >= the fixed 90% floor. The held-out
relation is scored only after selection. A second comparator selects without
the precision floor. The designated historical target is selected once from
all 16 sources and reported as a source-split diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_3rscan_multimotion_rescue_mechanisms import (
    MARGINS_M,
    MIN_SUPPORTS,
    MODE_COUNTS,
    predict,
    read_pair,
)


PRECISION_FLOOR = 0.90
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 10_062_027


def predict_counts(
    data: dict[str, Any], config: tuple[int, float, int]
) -> tuple[int, int, int, int]:
    correct, _, _, prediction = predict(data, *config)
    moved = np.asarray(data["moved"], dtype=bool)
    assigned = moved & (prediction >= 0)
    true_positive = int(correct.sum())
    assigned_count = int(assigned.sum())
    return true_positive, assigned_count, int(moved.sum()), int(data["baseline_correct"].sum())


def config_metrics(
    data_by_pair: dict[str, dict[str, Any]],
    pairs: list[str],
    config: tuple[int, float, int],
) -> dict[str, float | int]:
    counts = [predict_counts(data_by_pair[pair], config) for pair in pairs]
    correct = sum(row[0] for row in counts)
    assigned = sum(row[1] for row in counts)
    moved = sum(row[2] for row in counts)
    return {
        "correct": correct,
        "assigned": assigned,
        "moved": moved,
        "precision": correct / assigned if assigned else float("nan"),
        "coverage": correct / moved if moved else float("nan"),
    }


def choose(
    data_by_pair: dict[str, dict[str, Any]],
    train_pairs: list[str],
    precision_floor: float | None,
) -> tuple[tuple[int, float, int], dict[str, float | int], bool]:
    scored: list[tuple[tuple[int, float, int], dict[str, float | int]]] = []
    for config in (
        (mode_count, margin, support)
        for mode_count in MODE_COUNTS
        for margin in MARGINS_M
        for support in MIN_SUPPORTS
    ):
        scored.append((config, config_metrics(data_by_pair, train_pairs, config)))
    feasible = [
        item for item in scored
        if precision_floor is None
        or (item[1]["assigned"] > 0 and item[1]["precision"] >= precision_floor)
    ]
    has_feasible = bool(feasible)
    if not feasible:
        # If the requested floor is infeasible, prefer the safest configuration
        # first, then recover as much coverage as possible at that precision.
        best_precision = max(float(item[1]["precision"]) for item in scored)
        feasible = [item for item in scored if float(item[1]["precision"]) == best_precision]
    best_correct = max(int(metrics["correct"]) for _, metrics in feasible)
    tied = [(config, metrics) for config, metrics in feasible if int(metrics["correct"]) == best_correct]
    config, metrics = min(tied, key=lambda item: (item[0][0], -item[0][2], -item[0][1]))
    return config, metrics, has_feasible


def bootstrap_ci(values: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def analyze(data_by_pair: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    source_pairs = sorted(pair for pair in data_by_pair if pair != "target")
    if len(source_pairs) != 16 or "target" not in data_by_pair:
        raise ValueError("expected the audited 16-source plus one-target relation set")

    selections: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for heldout in source_pairs:
        train_pairs = [pair for pair in source_pairs if pair != heldout]
        unconstrained, train_unconstrained, _ = choose(data_by_pair, train_pairs, None)
        guarded, train_guarded, feasible = choose(
            data_by_pair, train_pairs, PRECISION_FLOOR
        )
        baseline_counts = [
            predict_counts(data_by_pair[pair], (1, 0.0, 0))[0:2]
            for pair in [heldout]
        ][0]
        data = data_by_pair[heldout]
        moved = int(np.asarray(data["moved"], dtype=bool).sum())
        row: dict[str, Any] = {
            "pair": heldout,
            "training_relations": len(train_pairs),
            "precision_floor": PRECISION_FLOOR,
            "guard_feasible": feasible,
            "guard_modes": guarded[0],
            "guard_margin_m": guarded[1],
            "guard_min_independent_support": guarded[2],
            "guard_training_correct": train_guarded["correct"],
            "guard_training_assigned": train_guarded["assigned"],
            "guard_training_precision": train_guarded["precision"],
            "guard_training_coverage": train_guarded["coverage"],
            "unconstrained_modes": unconstrained[0],
            "unconstrained_margin_m": unconstrained[1],
            "unconstrained_min_independent_support": unconstrained[2],
            "unconstrained_training_correct": train_unconstrained["correct"],
            "unconstrained_training_assigned": train_unconstrained["assigned"],
            "unconstrained_training_precision": train_unconstrained["precision"],
            "unconstrained_training_coverage": train_unconstrained["coverage"],
            "moved_queries": moved,
        }
        for name, config in (("guard", guarded), ("unconstrained", unconstrained)):
            correct, _, _, prediction = predict(data, *config)
            moved_mask = np.asarray(data["moved"], dtype=bool)
            assigned = moved_mask & (prediction >= 0)
            tp = int(correct.sum())
            assigned_count = int(assigned.sum())
            row[f"{name}_correct"] = tp
            row[f"{name}_assigned"] = assigned_count
            row[f"{name}_precision"] = tp / assigned_count if assigned_count else np.nan
            row[f"{name}_coverage"] = tp / moved if moved else np.nan
            row[f"{name}_rescued"] = int((correct & ~data["baseline_correct"]).sum())
            row[f"{name}_lost"] = int((data["baseline_correct"] & ~correct).sum())
        base_tp, base_assigned = baseline_counts
        row["baseline_correct"] = base_tp
        row["baseline_assigned"] = base_assigned
        row["baseline_precision"] = base_tp / base_assigned if base_assigned else np.nan
        row["baseline_coverage"] = base_tp / moved if moved else np.nan
        selections.append(row)

        for name, config in (("guard", guarded), ("unconstrained", unconstrained)):
            correct, selected_mode, gap, prediction = predict(data, *config)
            moved_mask = np.asarray(data["moved"], dtype=bool)
            assigned = moved_mask & (prediction >= 0)
            query_rows.extend({
                "pair": heldout,
                "selector": name,
                "query_index": int(index),
                "moved_rigid": bool(moved_mask[index]),
                "assigned": bool(assigned[index]),
                "correct": bool(correct[index]),
                "prediction_reference_object_id": int(prediction[index]),
                "selected_mode": int(selected_mode[index]),
                "residual_gap_m": float(gap[index]),
            } for index in np.flatnonzero(moved_mask))
        outcomes.append(row)

    # Select one final config using all 16 source relations, then score the
    # historical target once without using its labels to choose settings.
    target_guard, target_train_guard, target_feasible = choose(
        data_by_pair, source_pairs, PRECISION_FLOOR
    )
    target_unconstrained, target_train_unconstrained, _ = choose(
        data_by_pair, source_pairs, None
    )
    target = data_by_pair["target"]
    target_row: dict[str, Any] = {"pair": "target", "moved_queries": int(np.asarray(target["moved"]).sum())}
    for name, config, train_metrics in (
        ("guard", target_guard, target_train_guard),
        ("unconstrained", target_unconstrained, target_train_unconstrained),
    ):
        correct, selected_mode, gap, prediction = predict(target, *config)
        moved_mask = np.asarray(target["moved"], dtype=bool)
        assigned = moved_mask & (prediction >= 0)
        target_row.update({
            f"{name}_modes": config[0],
            f"{name}_margin_m": config[1],
            f"{name}_min_independent_support": config[2],
            f"{name}_training_precision": train_metrics["precision"],
            f"{name}_training_coverage": train_metrics["coverage"],
            f"{name}_correct": int(correct.sum()),
            f"{name}_assigned": int(assigned.sum()),
            f"{name}_precision": float(correct.sum() / assigned.sum()) if assigned.any() else np.nan,
            f"{name}_coverage": float(correct.sum() / moved_mask.sum()) if moved_mask.any() else np.nan,
        })
        query_rows.extend({
            "pair": "target",
            "selector": name,
            "query_index": int(index),
            "moved_rigid": bool(moved_mask[index]),
            "assigned": bool(assigned[index]),
            "correct": bool(correct[index]),
            "prediction_reference_object_id": int(prediction[index]),
            "selected_mode": int(selected_mode[index]),
            "residual_gap_m": float(gap[index]),
        } for index in np.flatnonzero(moved_mask))

    source_frame = pd.DataFrame(selections)
    eligible = source_frame[source_frame.moved_queries > 0].copy()
    summary: dict[str, Any] = {
        "status": "complete",
        "source_relations": len(source_pairs),
        "source_eligible_relations": int(len(eligible)),
        "source_moved_queries": int(source_frame.moved_queries.sum()),
        "precision_floor": PRECISION_FLOOR,
        "selector": "maximize correct transfer count under pooled source-training assignment precision floor; tie-break fewer modes, higher support, larger residual margin",
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "target_used_for_selection": False,
        "target_train_precision_floor_feasible": bool(target_feasible),
        "target_diagnostic": target_row,
    }
    for name in ("baseline", "unconstrained", "guard"):
        correct_col = f"{name}_correct"
        assigned_col = f"{name}_assigned"
        total_correct = int(source_frame[correct_col].sum())
        total_assigned = int(source_frame[assigned_col].sum())
        total_moved = int(source_frame.moved_queries.sum())
        cov_deltas: list[float] = []
        precision_deltas: list[float] = []
        paired = eligible.dropna(subset=[f"{name}_coverage", "baseline_coverage"])
        cov_deltas = (
            paired[f"{name}_coverage"] - paired.baseline_coverage
        ).to_numpy(dtype=np.float64)
        pair_prec = eligible.dropna(subset=[f"{name}_precision", "baseline_precision"])
        precision_deltas = (
            pair_prec[f"{name}_precision"] - pair_prec.baseline_precision
        ).to_numpy(dtype=np.float64)
        summary[name] = {
            "correct": total_correct,
            "assigned": total_assigned,
            "moved_queries": total_moved,
            "pooled_coverage": total_correct / total_moved if total_moved else np.nan,
            "pooled_precision": total_correct / total_assigned if total_assigned else np.nan,
            "eligible_relations": int(len(eligible)),
            "precision_floor_feasible_folds": int(source_frame.guard_feasible.sum()) if name == "guard" else None,
        }
        if name != "baseline":
            summary[name]["coverage_delta_vs_baseline_pp"] = 100.0 * (
                summary[name]["pooled_coverage"] - summary["baseline"]["pooled_coverage"]
            )
            summary[name]["precision_delta_vs_baseline_pp"] = 100.0 * (
                summary[name]["pooled_precision"] - summary["baseline"]["pooled_precision"]
            )
            summary[name]["relation_macro_coverage_delta_pp_ci95"] = [
                100.0 * value for value in bootstrap_ci(cov_deltas, seed=BOOTSTRAP_SEED)
            ]
            summary[name]["relation_macro_precision_delta_pp_ci95"] = [
                100.0 * value for value in bootstrap_ci(precision_deltas, seed=BOOTSTRAP_SEED + 1)
            ]

    output_dir.mkdir(parents=True, exist_ok=True)
    source_frame.to_csv(output_dir / "source_lopo.csv", index=False)
    pd.DataFrame([target_row]).to_csv(output_dir / "target_once.csv", index=False)
    pd.DataFrame(query_rows).to_csv(output_dir / "query_outcomes.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = f"""# Source-only precision-constrained pose-locked motion selection

The 16 official-train source relations are evaluated with outer leave-one-relation-out (LOPO): each held-out relation is excluded from selector fitting. Candidate, motion, and assignment hypotheses are fixed before selector scoring. The precision guard is 90% pooled physical-instance precision on the other 15 relations. Among feasible settings, the selector maximizes correct moved-rigid transfers; deterministic ties prefer fewer modes, higher independent support, then a larger residual margin. If no configuration meets the floor, it maximizes training precision first and records that the floor was infeasible. The historical target is selected once from all 16 sources and is shown as a train-split diagnostic only.

## Outer-LOPO source results

- Baseline: {summary['baseline']['correct']}/{summary['baseline']['moved_queries']} correct ({summary['baseline']['pooled_coverage']:.1%} coverage), {summary['baseline']['pooled_precision']:.1%} pooled assignment precision.
- Unconstrained coverage selector: {summary['unconstrained']['correct']}/{summary['unconstrained']['moved_queries']} correct ({summary['unconstrained']['pooled_coverage']:.1%}), {summary['unconstrained']['pooled_precision']:.1%} pooled precision; coverage change {summary['unconstrained']['coverage_delta_vs_baseline_pp']:+.1f} pp, precision change {summary['unconstrained']['precision_delta_vs_baseline_pp']:+.1f} pp.
- 90%-floor selector: {summary['guard']['correct']}/{summary['guard']['moved_queries']} correct ({summary['guard']['pooled_coverage']:.1%}), {summary['guard']['pooled_precision']:.1%} pooled precision; coverage change {summary['guard']['coverage_delta_vs_baseline_pp']:+.1f} pp, precision change {summary['guard']['precision_delta_vs_baseline_pp']:+.1f} pp. The floor was feasible in {summary['guard']['precision_floor_feasible_folds']}/16 LOPO folds.

Relation-macro paired coverage and precision intervals are in `summary.json`. The bootstrap resamples eligible source relations, not query patches. All relations come from the official train split, so these are internal cross-fit diagnostics rather than independent validation.

The source-trained final configurations and their historical target outcomes are in `target_once.csv`. This target is also official-train data and is not external evidence. Query-level outer-fold decisions are in `query_outcomes.csv`.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--method-arm", choices=("multi_motion", "combined"), default="multi_motion")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_frame = pd.read_csv(args.input_dir / "summary.csv")
    pairs = sorted(summary_frame.loc[summary_frame.role == "source", "pair"].unique())
    if len(pairs) != 16 or sorted(summary_frame.loc[summary_frame.role == "target", "pair"].unique()) != ["target"]:
        raise ValueError("input run must contain all 16 source relations and the historical target")
    data_by_pair = {
        pair: read_pair(args.input_dir, pair, args.annotation_root, seed=args.seed, method_arm=args.method_arm)
        for pair in pairs + ["target"]
    }
    summary = analyze(data_by_pair, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
