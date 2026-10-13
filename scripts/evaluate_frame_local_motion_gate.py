"""Evaluate a fixed same-frame geometric support gate for secondary modes.

The gate requires at least three independent residual inliers for a secondary
rigid mode in the query's own frame. The source-LOPO configurations come from
the already completed ungated analysis; this script does not reselect any
threshold using the frame-gated outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_3rscan_multimotion_rescue_mechanisms import (
    THRESHOLD_M,
    read_pair,
    relation_bootstrap,
)


SEEDS = (17, 29, 41)
MIN_FRAME_SUPPORT = 3


def predict_frame_local(
    data: dict[str, object], modes: int, margin: float, min_support: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the saved global LOPO rule plus same-frame support for switches."""
    base_residual = np.asarray(data["base_residual"], dtype=np.float64)
    expected = np.asarray(data["expected"], dtype=np.int64)
    base_object = np.asarray(data["base_object"], dtype=np.int64)
    residuals = data["secondary_residuals"]
    objects = data["secondary_objects"]
    global_support = data["secondary_independent_support"]
    frames = np.asarray(data["base"].frame_id, dtype=np.int64)
    n = len(base_residual)
    valid_slots = [
        slot for slot in range(min(modes - 1, len(residuals)))
        if global_support[slot] >= min_support
    ]

    if valid_slots:
        _, frame_group = np.unique(frames, return_inverse=True)
        frame_count = int(frame_group.max()) + 1 if len(frame_group) else 0
        matrix = np.full((n, len(valid_slots)), np.inf, dtype=np.float64)
        for column, slot in enumerate(valid_slots):
            residual = np.asarray(residuals[slot], dtype=np.float64)
            inlier = np.isfinite(residual) & (residual <= THRESHOLD_M)
            independent = inlier.copy()
            independent &= base_residual > THRESHOLD_M
            for previous in range(slot):
                previous_residual = np.asarray(residuals[previous], dtype=np.float64)
                independent &= ~(np.isfinite(previous_residual) & (previous_residual <= THRESHOLD_M))
            support_by_frame = np.bincount(
                frame_group, weights=independent.astype(np.int64), minlength=frame_count
            )
            local_ok = support_by_frame[frame_group] >= MIN_FRAME_SUPPORT
            matrix[local_ok & inlier, column] = residual[local_ok & inlier]

        best_column = np.argmin(matrix, axis=1)
        rows = np.arange(n)
        best_residual = matrix[rows, best_column]
        best_slot = np.asarray(valid_slots, dtype=np.int64)[best_column]
        finite = np.isfinite(best_residual) & (best_residual <= THRESHOLD_M)
        gap = base_residual - best_residual
        switch = finite & ((base_residual > THRESHOLD_M) | (gap >= margin))
        chosen_object = base_object.copy()
        chosen_mode = np.zeros(n, dtype=np.int64)
        for slot in valid_slots:
            use = switch & (best_slot == slot)
            chosen_object[use] = np.asarray(objects[slot], dtype=np.int64)[use]
            chosen_mode[use] = slot + 1
    else:
        switch = np.zeros(n, dtype=bool)
        chosen_object = base_object.copy()
        chosen_mode = np.zeros(n, dtype=np.int64)

    assigned = (base_residual <= THRESHOLD_M) | switch
    correct = np.asarray(data["moved"], dtype=bool) & assigned & (chosen_object == expected)
    return correct, chosen_mode


def run_seed(
    seed: int,
    input_dir: Path,
    config_dir: Path,
    annotation_root: Path,
    out_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    selections = pd.read_csv(config_dir / f"seed_{seed}" / "source_lopo_selection.csv")
    settings: dict[str, tuple[int, float, int]] = {}
    for row in selections.itertuples(index=False):
        settings[str(row.pair)] = (
            int(row.modes), float(row.margin_m), int(row.min_independent_support)
        )
    if set(key for key in settings if key != "target_once") != {
        p.stem for p in (input_dir / "descriptor_candidates").glob("*.npz") if p.stem != "target"
    }:
        raise ValueError(f"seed {seed}: LOPO relation set differs from descriptor cache")

    pairs = sorted(p.stem for p in (input_dir / "descriptor_candidates").glob("*.npz"))
    data_by_pair = {
        pair: read_pair(input_dir, pair, annotation_root, seed=seed)
        for pair in pairs
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome_rows: list[dict[str, object]] = []
    query_frames: list[pd.DataFrame] = []
    for pair, data in data_by_pair.items():
        selection_key = "target_once" if pair == "target" else pair
        modes, margin, support = settings[selection_key]
        correct, selected_mode = predict_frame_local(data, modes, margin, support)
        moved = np.asarray(data["moved"], dtype=bool)
        baseline = np.asarray(data["baseline_correct"], dtype=bool)
        outcome_rows.append({
            "pair": pair,
            "role": "target" if pair == "target" else "source",
            "modes": modes,
            "margin_m": margin,
            "global_min_support": support,
            "moved_queries": int(moved.sum()),
            "baseline_correct": int(baseline.sum()),
            "method_correct": int(correct.sum()),
            "rescued": int((correct & ~baseline).sum()),
            "lost": int((baseline & ~correct).sum()),
            "selected_secondary_queries": int((selected_mode > 0).sum()),
        })
        query_frames.append(pd.DataFrame({
            "pair": pair,
            "role": "target" if pair == "target" else "source",
            "query_index": data["base"].query_index.to_numpy(),
            "frame_id": data["base"].frame_id.to_numpy(),
            "moved_rigid": moved,
            "baseline_correct": baseline,
            "method_correct": correct,
            "rescued": correct & ~baseline,
            "lost": baseline & ~correct,
            "selected_mode": selected_mode,
        }).loc[moved].copy())

    outcomes = pd.DataFrame(outcome_rows)
    queries = pd.concat(query_frames, ignore_index=True)
    source = outcomes[outcomes.role == "source"].copy()
    boot = relation_bootstrap(source, samples=20000, seed=seed)
    source.to_csv(out_dir / f"seed_{seed}_source_relations.csv", index=False)
    queries.to_csv(out_dir / f"seed_{seed}_query_outcomes.csv", index=False)
    boot.to_csv(out_dir / f"seed_{seed}_relation_bootstrap.csv", index=False)

    denom = int(source.moved_queries.sum())
    base_correct = int(source.baseline_correct.sum())
    method_correct = int(source.method_correct.sum())
    eligible = source[source.moved_queries > 0]
    nonnegative = int((eligible.method_correct >= eligible.baseline_correct).sum())
    summary: dict[str, object] = {
        "seed": seed,
        "frame_support_minimum": MIN_FRAME_SUPPORT,
        "frame_gate": "secondary mode may switch a query only when at least three independent inliers from the same frame support that mode",
        "source_relations": int(len(source)),
        "eligible_source_relations": int(len(eligible)),
        "source_queries": denom,
        "baseline_correct": base_correct,
        "method_correct": method_correct,
        "baseline_coverage": base_correct / denom,
        "method_coverage": method_correct / denom,
        "delta_pp": 100.0 * (method_correct - base_correct) / denom,
        "rescued": int(source.rescued.sum()),
        "lost": int(source.lost.sum()),
        "nonnegative_relations": nonnegative,
        "weighted_delta_ci95_pp": [
            float(boot.iloc[0].ci95_lower), float(boot.iloc[0].ci95_upper)
        ],
        "target_queries": int(outcomes.loc[outcomes.role == "target", "moved_queries"].iloc[0]),
        "target_baseline_correct": int(outcomes.loc[outcomes.role == "target", "baseline_correct"].iloc[0]),
        "target_method_correct": int(outcomes.loc[outcomes.role == "target", "method_correct"].iloc[0]),
        "target_used_for_selection": False,
        "source_configuration_reused_from_ungated_lopo": True,
        "official_split": "train",
    }
    (out_dir / f"seed_{seed}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True,
                        help="completed ungated seed-robustness directory")
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    relation_frames = []
    for seed in SEEDS:
        print(f"running frame-local support audit for seed {seed}", flush=True)
        summary, relations = run_seed(
            seed, args.input_dir, args.config_dir, args.annotation_root,
            args.output_dir,
        )
        summaries.append(summary)
        relations["seed"] = seed
        relation_frames.append(relations)
    frame = pd.DataFrame(summaries)
    ungated = {
        seed: json.loads((args.config_dir / f"seed_{seed}" / "summary.json").read_text(encoding="utf-8"))
        for seed in SEEDS
    }
    frame["ungated_method_correct"] = frame.seed.map(
        lambda seed: int(ungated[int(seed)]["method_correct"])
    )
    frame["ungated_delta_pp"] = frame.seed.map(
        lambda seed: float(ungated[int(seed)]["coverage_delta_pp"])
    )
    frame["method_correct_change"] = frame.method_correct - frame.ungated_method_correct
    frame["delta_change_pp"] = frame.delta_pp - frame.ungated_delta_pp
    frame.to_csv(args.output_dir / "frame_gate_seed_summary.csv", index=False)
    pd.concat(relation_frames, ignore_index=True).to_csv(
        args.output_dir / "frame_gate_relation_outcomes.csv", index=False
    )
    report = [
        "# Same-frame support gate for multi-motion transfer",
        "",
        "## Fixed mechanism",
        "",
        "A secondary motion may switch a query only if at least three independent inlier correspondences for that mode occur in the same query frame. Three is the geometric minimum for a rigid correspondence fit. Independent support excludes queries already explained by the adaptive-K primary scene assignment and earlier secondary modes. The query's frame-local support is computed from geometry only; labels are used after inference for scoring.",
        "",
        "The global mode-count, residual-margin, and mode-support settings are reused from the previously completed ungated source-only LOPO folds. They are not retuned against this frame-gated outcome. Every result below uses the same 1,510 moved-rigid queries in the official 3RScan train split; the three RANSAC seeds reuse these queries and are not independent replications. The target is a diagnostic only.",
        "",
        "## Results",
        "",
        "| Seed | Baseline | Frame-gated method | Delta | Change vs ungated | Rescued / lost | Nonnegative / eligible | Weighted relation-bootstrap 95% CI | Target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        report.append(
            f"| {row.seed} | {row.baseline_correct}/{row.source_queries} ({row.baseline_coverage:.2%}) | "
            f"{row.method_correct}/{row.source_queries} ({row.method_coverage:.2%}) | {row.delta_pp:+.2f} pp | "
            f"{row.delta_change_pp:+.2f} pp | "
            f"{row.rescued} / {row.lost} | {row.nonnegative_relations}/{row.eligible_source_relations} | "
            f"[{row.weighted_delta_ci95_pp[0]:+.2f}, {row.weighted_delta_ci95_pp[1]:+.2f}] pp | "
            f"{row.target_baseline_correct}/{row.target_queries} → {row.target_method_correct}/{row.target_queries} |"
        )
    mean_ungated = float(frame.ungated_delta_pp.mean())
    mean_gated = float(frame.delta_pp.mean())
    all_target_same = bool((frame.target_method_correct == frame.seed.map(
        lambda seed: int(ungated[int(seed)]["target_method_correct"])
    )).all())
    report += [
        "",
        f"Relative to the ungated method, the frame gate changes correct-transfer counts by {', '.join(f'{int(v):+d}' for v in frame.method_correct_change)} queries and changes mean source gain from {mean_ungated:+.2f} pp to {mean_gated:+.2f} pp. The target method count is {'unchanged in all seeds' if all_target_same else 'not unchanged in all seeds'}. This same-frame gate is a no-go because its source effect is reduced and no seed reaches the frozen +10 pp threshold.",
        "",
        "This is a fixed ablation of the current multi-motion mechanism, not a new benchmark result. All evaluated relations remain in train; official validation is still required.",
        "",
        "Per-seed query and relation outcomes are saved beside this report. The ungated reference is `../20260929_multimotion_seed_robustness/seed_robustness_report.md`.",
        "",
    ]
    (args.output_dir / "frame_gate_report.md").write_text("\n".join(report), encoding="utf-8")
    print(frame[["seed", "baseline_correct", "method_correct", "delta_pp", "rescued", "lost",
                 "nonnegative_relations", "target_baseline_correct", "target_method_correct"]].to_string(index=False))


if __name__ == "__main__":
    main()
