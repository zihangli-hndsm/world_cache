#!/usr/bin/env python3
"""Outer-LOPO audit of source-fitted descriptor-confidence abstention.

Candidate and motion hypotheses are fixed and label-free. For each held-out
relation, the mode/margin/support setting and top-1/top-20 descriptor-gap
threshold are selected from the other 15 source relations to maximize correct
physical-instance transfers subject to a pooled assignment-precision floor.
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
)


PRECISION_FLOOR = 0.90
THRESHOLD_M = 0.10
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 10_062_027
CONFIGS = tuple(
    (mode_count, margin, support)
    for mode_count in MODE_COUNTS
    for margin in MARGINS_M
    for support in MIN_SUPPORTS
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def bootstrap_ci(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def build_pair(data: dict[str, Any]) -> dict[str, Any]:
    moved = np.asarray(data["moved"], dtype=bool)
    confidence = data["base"].descriptor_gap_top1_top20.to_numpy(dtype=np.float64)
    correct_by_config: list[np.ndarray] = []
    assigned_by_config: list[np.ndarray] = []
    for config in CONFIGS:
        correct, _, _, prediction = predict(data, *config)
        correct_by_config.append(np.asarray(correct, dtype=bool)[moved])
        assigned_by_config.append((prediction[moved] >= 0))
    return {
        "query_index": np.flatnonzero(moved),
        "confidence": confidence[moved],
        "correct": correct_by_config,
        "assigned": assigned_by_config,
        "moved": int(moved.sum()),
    }


def read_saved_pair(
    diagnostics: pd.DataFrame, pair: str, method_arm: str
) -> dict[str, Any]:
    base = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == "global_ransac")]
    motion = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == method_arm)]
    base = base.sort_values("query_index").reset_index(drop=True)
    motion = motion.sort_values("query_index").reset_index(drop=True)
    if base.empty or motion.empty:
        raise ValueError(f"missing saved query diagnostics for {pair}")
    if (not np.array_equal(base.query_index.to_numpy(), motion.query_index.to_numpy())
            or not np.array_equal(base.frame_id.to_numpy(), motion.frame_id.to_numpy())):
        raise ValueError(f"saved query rows do not align for {pair}")
    for column in ("query_evaluable", "official_rigid_reference_instance_id",
                   "descriptor_gap_top1_top20"):
        if not np.array_equal(base[column].to_numpy(), motion[column].to_numpy(), equal_nan=True):
            raise ValueError(f"saved global/motion query field differs for {pair}: {column}")

    base_residual = motion.hypothesis_0_residual_m.to_numpy(dtype=np.float64)
    base_object = motion.hypothesis_0_selected_reference_instance_id.fillna(-1).to_numpy(dtype=np.int64)
    if not np.allclose(base.spatial_residual_m.to_numpy(dtype=np.float64), base_residual,
                       rtol=0.0, atol=1e-6, equal_nan=True):
        raise ValueError(f"saved top-20 baseline residual differs from mode 0 for {pair}")
    if not np.array_equal(base.selected_reference_instance_id.to_numpy(dtype=np.int64), base_object):
        raise ValueError(f"saved top-20 baseline identity differs from mode 0 for {pair}")

    secondary_residuals: list[np.ndarray] = []
    secondary_objects: list[np.ndarray] = []
    secondary_support: list[int] = []
    previous_unexplained = np.isfinite(base_residual) & (base_residual > THRESHOLD_M)
    for mode in range(1, 4):
        residual = motion[f"hypothesis_{mode}_residual_m"].to_numpy(dtype=np.float64)
        object_id = motion[f"hypothesis_{mode}_selected_reference_instance_id"].fillna(-1).to_numpy(dtype=np.int64)
        inlier = np.isfinite(residual) & (residual <= THRESHOLD_M)
        secondary_support.append(int((inlier & previous_unexplained).sum()))
        secondary_residuals.append(residual)
        secondary_objects.append(object_id)
        previous_unexplained &= ~inlier

    expected = base.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
    moved = base.query_evaluable.astype(bool).to_numpy() & (expected > 0)
    return {
        "moved": moved,
        "expected": expected,
        "base_residual": base_residual,
        "base_object": base_object,
        "secondary_residuals": secondary_residuals,
        "secondary_objects": secondary_objects,
        "secondary_independent_support": secondary_support,
        "base": base,
    }


def choose(
    by_pair: dict[str, dict[str, Any]], train_pairs: list[str]
) -> tuple[int, float, dict[str, Any], bool]:
    candidates: list[tuple[int, float, int, int, float, int, int, float]] = []
    # tuple fields: correct, precision, assigned, config_index, threshold,
    # mode_count, support, margin. Only thresholds present among assigned
    # training queries need consideration; tied scores enter as one group.
    for config_index, config in enumerate(CONFIGS):
        correct_parts = [by_pair[p]["correct"][config_index] for p in train_pairs]
        assigned_parts = [by_pair[p]["assigned"][config_index] for p in train_pairs]
        score_parts = [by_pair[p]["confidence"] for p in train_pairs]
        correct = np.concatenate(correct_parts) if correct_parts else np.empty(0, bool)
        assigned = np.concatenate(assigned_parts) if assigned_parts else np.empty(0, bool)
        scores = np.concatenate(score_parts) if score_parts else np.empty(0, np.float64)
        keep = assigned & np.isfinite(scores)
        if not keep.any():
            continue
        selected_scores = scores[keep]
        selected_correct = correct[keep]
        order = np.argsort(-selected_scores, kind="stable")
        sorted_scores = selected_scores[order]
        sorted_correct = selected_correct[order].astype(np.int64)
        group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(sorted_scores) - 1]
        cumulative_correct = np.cumsum(sorted_correct)[group_ends]
        cumulative_assigned = group_ends + 1
        precision = cumulative_correct / cumulative_assigned
        for idx, threshold in enumerate(sorted_scores[group_ends]):
            candidates.append((
                int(cumulative_correct[idx]),
                float(precision[idx]),
                int(cumulative_assigned[idx]),
                config_index,
                float(threshold),
                config[0],
                config[2],
                config[1],
            ))

    feasible = [c for c in candidates if c[1] >= PRECISION_FLOOR]
    has_feasible = bool(feasible)
    pool = feasible if feasible else candidates
    if not pool:
        raise RuntimeError("no nonempty assigned configuration in selector training data")
    if has_feasible:
        # Maximize correct transfers; then prefer a simpler rule, more support,
        # a larger geometric margin, and the least restrictive score threshold.
        best_correct = max(c[0] for c in pool)
        tied = [c for c in pool if c[0] == best_correct]
        selected = min(tied, key=lambda c: (c[5], -c[6], -c[7], c[4]))
    else:
        best_precision = max(c[1] for c in pool)
        safest = [c for c in pool if c[1] == best_precision]
        best_correct = max(c[0] for c in safest)
        selected = min(
            (c for c in safest if c[0] == best_correct),
            key=lambda c: (c[5], -c[6], -c[7], c[4]),
        )
    correct, precision, assigned, config_index, threshold, *_ = selected
    return config_index, threshold, {
        "correct": correct,
        "assigned": assigned,
        "precision": precision,
        "coverage": correct / sum(by_pair[p]["moved"] for p in train_pairs),
    }, has_feasible


def score_pair(
    pair_data: dict[str, Any], config_index: int, threshold: float
) -> dict[str, Any]:
    assigned = pair_data["assigned"][config_index] & (pair_data["confidence"] >= threshold)
    correct = pair_data["correct"][config_index] & assigned
    tp = int(correct.sum())
    count = int(assigned.sum())
    return {
        "correct": tp,
        "assigned": count,
        "precision": tp / count if count else np.nan,
        "coverage": tp / pair_data["moved"] if pair_data["moved"] else np.nan,
        "moved": pair_data["moved"],
        "query_correct": correct,
        "query_assigned": assigned,
    }


def run(by_pair: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    source_pairs = sorted(pair for pair in by_pair if pair != "target")
    if len(source_pairs) != 16 or "target" not in by_pair:
        raise ValueError("expected the audited 16-source plus one-target relation set")

    fold_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for heldout in source_pairs:
        train_pairs = [pair for pair in source_pairs if pair != heldout]
        config_index, threshold, training, feasible = choose(by_pair, train_pairs)
        config = CONFIGS[config_index]
        result = score_pair(by_pair[heldout], config_index, threshold)
        fold_rows.append({
            "pair": heldout,
            "moved_queries": result["moved"],
            "precision_floor_feasible": feasible,
            "modes": config[0],
            "residual_margin_m": config[1],
            "min_independent_support": config[2],
            "descriptor_gap_threshold": threshold,
            "training_precision": training["precision"],
            "training_correct": training["correct"],
            "training_assigned": training["assigned"],
            "heldout_correct": result["correct"],
            "heldout_assigned": result["assigned"],
            "heldout_precision": result["precision"],
            "heldout_coverage": result["coverage"],
        })
        for local_index, query_index in enumerate(by_pair[heldout]["query_index"]):
            query_rows.append({
                "pair": heldout,
                "query_index": int(query_index),
                "descriptor_gap_top1_top20": float(by_pair[heldout]["confidence"][local_index]),
                "assigned": bool(result["query_assigned"][local_index]),
                "correct": bool(result["query_correct"][local_index]),
            })

    folds = pd.DataFrame(fold_rows)
    eligible = folds[folds.moved_queries > 0]
    total_moved = int(folds.moved_queries.sum())
    total_correct = int(folds.heldout_correct.sum())
    total_assigned = int(folds.heldout_assigned.sum())
    baseline = (1, 0.15, 128)
    baseline_index = CONFIGS.index(baseline)
    base_fold = [score_pair(by_pair[p], baseline_index, -np.inf) for p in source_pairs]
    base_correct = sum(x["correct"] for x in base_fold)
    base_assigned = sum(x["assigned"] for x in base_fold)
    base_by_pair = {
        p: score_pair(by_pair[p], baseline_index, -np.inf) for p in source_pairs
    }
    method_macro = eligible.heldout_coverage.to_numpy(dtype=np.float64)
    baseline_macro = np.asarray([
        base_by_pair[row.pair]["correct"] / row.moved_queries
        for row in eligible.itertuples(index=False)
    ])
    delta_ci = bootstrap_ci(method_macro - baseline_macro)
    method_precision = eligible.heldout_precision.to_numpy(dtype=np.float64)
    baseline_precision = np.asarray([
        base_by_pair[row.pair]["precision"]
        for row in eligible.itertuples(index=False)
    ])
    precision_valid = np.isfinite(method_precision) & np.isfinite(baseline_precision)
    precision_delta_ci = bootstrap_ci(
        method_precision[precision_valid] - baseline_precision[precision_valid],
        seed=BOOTSTRAP_SEED + 1,
    )
    summary = {
        "status": "complete",
        "source_relations": 16,
        "eligible_relations": int(len(eligible)),
        "moved_queries": total_moved,
        "precision_floor": PRECISION_FLOOR,
        "selector": "jointly maximize correct physical-instance transfers over 144 geometric settings and a source-fitted descriptor-gap threshold, subject to pooled training precision",
        "score": "query top-1 to top-20 descriptor score gap; larger is treated as higher confidence",
        "target_used_for_selection": False,
        "feasible_training_folds": int(folds.precision_floor_feasible.sum()),
        "baseline": {
            "correct": base_correct,
            "assigned": base_assigned,
            "pooled_coverage": base_correct / total_moved,
            "pooled_precision": base_correct / base_assigned,
        },
        "confidence_guard": {
            "correct": total_correct,
            "assigned": total_assigned,
            "pooled_coverage": total_correct / total_moved,
            "pooled_precision": total_correct / total_assigned,
            "coverage_delta_pp": 100 * (total_correct - base_correct) / total_moved,
            "precision_delta_pp": 100 * (total_correct / total_assigned - base_correct / base_assigned),
            "relation_macro_coverage_delta_ci95_pp": [100 * x for x in delta_ci],
            "relation_macro_precision_delta_ci95_pp": [100 * x for x in precision_delta_ci],
            "precision_macro_relations": int(precision_valid.sum()),
            "eligible_relations_improved": int((method_macro > baseline_macro).sum()),
            "eligible_relations_tied": int((method_macro == baseline_macro).sum()),
            "eligible_relations_worse": int((method_macro < baseline_macro).sum()),
        },
        "folds": fold_rows,
    }

    target_config, target_threshold, target_train, target_feasible = choose(by_pair, source_pairs)
    target = score_pair(by_pair["target"], target_config, target_threshold)
    summary["target_train_diagnostic"] = {
        "moved_queries": target["moved"],
        "precision_floor_feasible": target_feasible,
        "modes": CONFIGS[target_config][0],
        "residual_margin_m": CONFIGS[target_config][1],
        "min_independent_support": CONFIGS[target_config][2],
        "descriptor_gap_threshold": target_threshold,
        "training_precision": target_train["precision"],
        "correct": target["correct"],
        "assigned": target["assigned"],
        "precision": target["precision"],
        "coverage": target["coverage"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "source_lopo.csv", index=False)
    pd.DataFrame(query_rows).to_csv(output_dir / "query_outcomes.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    report = f"""# Descriptor-confidence precision guard: outer source LOPO

For each of the 16 source relations, the outer held-out relation is excluded while jointly selecting a geometric mode/margin/support setting and a minimum descriptor-gap threshold on the other 15 relations. The confidence score is the top-1 minus top-20 descriptor-score gap. The selector maximizes correct moved-rigid physical-instance transfers subject to 90% pooled training assignment precision. All geometric hypotheses and per-query descriptor scores are fixed before this label-based selection. The historical target is shown only as a source-trained diagnostic.

## Results

- Baseline: {base_correct}/{total_moved} correct ({base_correct/total_moved:.1%} coverage), {base_correct/base_assigned:.1%} pooled precision.
- Confidence-guard LOPO: {total_correct}/{total_moved} correct ({total_correct/total_moved:.1%} coverage), {total_correct/total_assigned:.1%} pooled precision; change {summary['confidence_guard']['coverage_delta_pp']:+.1f} coverage pp and {summary['confidence_guard']['precision_delta_pp']:+.1f} precision pp. The training floor was feasible in {summary['feasible_training_folds']}/16 folds.

The paired relation-macro coverage-difference interval is {summary['confidence_guard']['relation_macro_coverage_delta_ci95_pp'][0]:+.1f} to {summary['confidence_guard']['relation_macro_coverage_delta_ci95_pp'][1]:+.1f} points. The paired relation-macro precision-difference interval is {summary['confidence_guard']['relation_macro_precision_delta_ci95_pp'][0]:+.1f} to {summary['confidence_guard']['relation_macro_precision_delta_ci95_pp'][1]:+.1f} points over relations with defined precision; both bootstrap relations rather than queries. This is an internal, post hoc cross-fit on the official train split and is conditional on cached candidate and motion hypotheses; it is not independent validation.

The one-time target diagnostic and fold-level selections are in `summary.json` and `source_lopo.csv`; query decisions are in `query_outcomes.csv`.
"""
    (output_dir / "report.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--method-arm", choices=("multi_motion", "combined"), default="multi_motion")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_frame = pd.read_csv(args.input_dir / "summary.csv")
    diagnostics = pd.read_csv(args.input_dir / "query_diagnostics.csv")
    source_pairs = sorted(summary_frame.loc[summary_frame.role == "source", "pair"].unique())
    if len(source_pairs) != 16:
        raise ValueError("input run must contain the original 16 official-train source relations")
    raw = {
        pair: read_saved_pair(diagnostics, pair, args.method_arm)
        for pair in source_pairs + ["target"]
    }
    summary = run({pair: build_pair(value) for pair, value in raw.items()}, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
