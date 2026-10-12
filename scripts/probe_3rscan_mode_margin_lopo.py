"""Source-only LOPO probe of conservative mode-switch margins."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def gated_mode_counts(
    frame: pd.DataFrame,
    mode_count: int,
    switch_margin_m: float,
    inlier_threshold_m: float = 0.10,
) -> dict[str, int | float | np.ndarray]:
    """Use a secondary motion only when it beats the scene mode by a margin."""
    residual_columns = sorted(
        (column for column in frame if column.startswith("hypothesis_") and column.endswith("_residual_m")),
        key=lambda column: int(column.split("_")[1]),
    )
    object_columns = [
        f"hypothesis_{int(column.split('_')[1])}_selected_reference_instance_id"
        for column in residual_columns
    ]
    present = [
        i for i, column in enumerate(residual_columns)
        if frame[column].notna().any() and frame[object_columns[i]].notna().any()
    ]
    if not present:
        raise ValueError("no per-hypothesis assignment columns found")
    take = present[: max(1, min(mode_count, len(present)))]
    residuals = frame[[residual_columns[i] for i in take]].to_numpy(dtype=np.float64)
    object_ids = frame[[object_columns[i] for i in take]].to_numpy(dtype=np.float64)
    scene_residual = residuals[:, 0]
    scene_valid = np.isfinite(scene_residual) & (scene_residual <= inlier_threshold_m)
    selected_object = object_ids[:, 0].copy()
    selected_mode = np.zeros(len(frame), dtype=np.int64)
    assigned = scene_valid.copy()
    if len(take) > 1:
        motion_residual = residuals[:, 1:]
        motion_valid = np.isfinite(motion_residual) & (motion_residual <= inlier_threshold_m)
        masked = np.where(motion_valid, motion_residual, np.inf)
        best_slot = np.argmin(masked, axis=1)
        row = np.arange(len(frame))
        best_motion_residual = masked[row, best_slot]
        best_motion_valid = np.isfinite(best_motion_residual)
        switch = best_motion_valid & (
            ~scene_valid | ((scene_residual - best_motion_residual) >= switch_margin_m)
        )
        chosen_motion_column = best_slot + 1
        selected_object[switch] = object_ids[row[switch], chosen_motion_column[switch]]
        selected_mode[switch] = np.asarray(take, dtype=np.int64)[chosen_motion_column[switch]]
        assigned |= switch
    query_evaluable = frame.query_evaluable.astype(bool).to_numpy()
    expected = frame.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
    moved = query_evaluable & (expected > 0)
    correct = moved & assigned & (selected_object.astype(np.int64) == expected)
    denominator = int(moved.sum())
    return {
        "moved_rigid_queries": denominator,
        "correct_transfers": int(correct.sum()),
        "coverage": float(correct.sum() / denominator) if denominator else float("nan"),
        "assigned_mode": selected_mode,
        "correct_mask": correct,
        "moved_mask": moved,
        "effective_mode_count": len(take),
    }


def run_lopo(
    output_dir: Path,
    mode_counts: tuple[int, ...] = (1, 2, 3, 4),
    margins_m: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10),
    threshold_m: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(output_dir / "query_diagnostics.csv")
    summary = pd.read_csv(output_dir / "summary.csv")
    source_pairs = sorted(summary.loc[summary.role == "source", "pair"].unique().tolist())
    if len(source_pairs) != 16:
        raise ValueError("mode-margin LOPO requires the complete 16-relation source set")
    specs = {"multi_motion": mode_counts, "combined": mode_counts}
    per_pair_cache: dict[tuple[str, str, int, float], dict[str, int | float | np.ndarray]] = {}
    baseline_cache: dict[str, dict[str, int | float | np.ndarray]] = {}
    for pair in source_pairs + ["target"]:
        base = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == "global_ransac")].sort_values("query_index")
        baseline_cache[pair] = gated_mode_counts(base, 1, 0.0, threshold_m)
        for arm, counts in specs.items():
            frame = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == arm)].sort_values("query_index")
            for modes in counts:
                for margin in margins_m:
                    per_pair_cache[(pair, arm, modes, margin)] = gated_mode_counts(frame, modes, margin, threshold_m)

    selection_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for arm, counts in specs.items():
        for heldout in source_pairs:
            train = [pair for pair in source_pairs if pair != heldout]
            scores = {
                (modes, margin): sum(
                    int(per_pair_cache[(pair, arm, modes, margin)]["correct_transfers"])
                    for pair in train
                )
                for modes in counts
                for margin in margins_m
            }
            best_score = max(scores.values())
            chosen = min(
                (config for config, value in scores.items() if value == best_score),
                key=lambda config: (config[0], -config[1]),
            )
            metric = per_pair_cache[(heldout, arm, *chosen)]
            baseline = baseline_cache[heldout]
            selection_rows.append({
                "arm": arm,
                "role": "source_lopo",
                "heldout_pair": heldout,
                "training_relations": len(train),
                "selected_mode_count": chosen[0],
                "selected_switch_margin_m": chosen[1],
                "training_correct_transfers": best_score,
                "heldout_correct_transfers": metric["correct_transfers"],
                "heldout_moved_rigid_queries": metric["moved_rigid_queries"],
                "heldout_coverage": metric["coverage"],
                "baseline_correct_transfers": baseline["correct_transfers"],
                "baseline_coverage": baseline["coverage"],
            })
            outcome_rows.append({
                "pair": heldout,
                "role": "source",
                "arm": arm,
                "mode_count": chosen[0],
                "switch_margin_m": chosen[1],
                **{k: v for k, v in metric.items() if not isinstance(v, np.ndarray)},
                "baseline_correct_transfers": baseline["correct_transfers"],
                "baseline_coverage": baseline["coverage"],
            })

        scores = {
            (modes, margin): sum(
                int(per_pair_cache[(pair, arm, modes, margin)]["correct_transfers"])
                for pair in source_pairs
            )
            for modes in counts
            for margin in margins_m
        }
        best_score = max(scores.values())
        chosen = min((cfg for cfg, value in scores.items() if value == best_score), key=lambda cfg: (cfg[0], -cfg[1]))
        target = per_pair_cache[("target", arm, *chosen)]
        selection_rows.append({
            "arm": arm,
            "role": "target_once",
            "heldout_pair": "target",
            "training_relations": len(source_pairs),
            "selected_mode_count": chosen[0],
            "selected_switch_margin_m": chosen[1],
            "training_correct_transfers": best_score,
            "heldout_correct_transfers": target["correct_transfers"],
            "heldout_moved_rigid_queries": target["moved_rigid_queries"],
            "heldout_coverage": target["coverage"],
            "baseline_correct_transfers": np.nan,
            "baseline_coverage": np.nan,
        })
        outcome_rows.append({
            "pair": "target",
            "role": "target_once",
            "arm": arm,
            "mode_count": chosen[0],
            "switch_margin_m": chosen[1],
            **{k: v for k, v in target.items() if not isinstance(v, np.ndarray)},
            "baseline_correct_transfers": np.nan,
            "baseline_coverage": np.nan,
        })

    selections = pd.DataFrame(selection_rows)
    outcomes = pd.DataFrame(outcome_rows)
    selections.to_csv(output_dir / "mode_margin_lopo_selection.csv", index=False)
    outcomes.to_csv(output_dir / "mode_margin_lopo_outcomes.csv", index=False)
    aggregate_rows = []
    for arm in specs:
        source = outcomes[(outcomes.role == "source") & (outcomes.arm == arm)]
        denominator = int(source.moved_rigid_queries.sum())
        correct = int(source.correct_transfers.sum())
        base_correct = int(source.baseline_correct_transfers.sum())
        comparable = source.moved_rigid_queries > 0
        nonnegative = int((
            source.loc[comparable, "coverage"].to_numpy()
            >= source.loc[comparable, "baseline_coverage"].to_numpy()
        ).sum())
        target = outcomes[(outcomes.role == "target_once") & (outcomes.arm == arm)].iloc[0]
        method_pose = summary[(summary.arm == arm) & (summary.role == "source")]
        baseline_pose = summary[(summary.arm == "global_ransac") & (summary.role == "source")]
        pose_loss = int(baseline_pose.strict_scene_pose_success.sum() - method_pose.strict_scene_pose_success.sum())
        coverage = correct / denominator if denominator else float("nan")
        base_coverage = base_correct / denominator if denominator else float("nan")
        aggregate_rows.append({
            "arm": arm,
            "source_lopo_correct_transfers": correct,
            "source_moved_rigid_queries": denominator,
            "source_lopo_coverage": coverage,
            "baseline_correct_transfers": base_correct,
            "baseline_coverage": base_coverage,
            "coverage_delta_pp": 100.0 * (coverage - base_coverage),
            "source_eligible_pairs": int(comparable.sum()),
            "source_nonnegative_pairs": nonnegative,
            "source_not_comparable_pairs": int(len(source) - comparable.sum()),
            "source_scene_pose_successes": int(method_pose.strict_scene_pose_success.sum()),
            "scene_pose_success_loss": pose_loss,
            "target_mode_count": int(target.mode_count),
            "target_switch_margin_m": float(target.switch_margin_m),
            "target_correct_transfers": int(target.correct_transfers),
            "target_moved_rigid_queries": int(target.moved_rigid_queries),
            "target_coverage": float(target.coverage),
            "stage1_gate_pass": bool(coverage - base_coverage >= 0.10 and nonnegative >= 10 and pose_loss <= 1),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "mode_margin_lopo_summary.csv", index=False)
    report_path = output_dir / "multimotion_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# 3RScan multi-motion pilot\n"
    marker = "## Source-only LOPO switch-margin probe"
    report = report.split(marker)[0].rstrip()
    lines = [
        "## Source-only LOPO switch-margin probe",
        "",
        "This diagnostic changes only query-to-hypothesis arbitration. A motion mode replaces the scene assignment when its spatial residual is lower by a source-selected margin; mode count and margin are selected on the other 15 source relations. The target setting is selected once from all 16 sources.",
        "",
        "| Arm | Source LOPO coverage | Change vs baseline | Nonnegative / eligible | Target K/margin | Target coverage | Gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.arm} | {100 * row.source_lopo_coverage:.2f}% | {row.coverage_delta_pp:+.2f} pp | "
            f"{row.source_nonnegative_pairs}/{row.source_eligible_pairs} | "
            f"{row.target_mode_count} / {row.target_switch_margin_m:.2f} m | "
            f"{100 * row.target_coverage:.2f}% | {'yes' if row.stage1_gate_pass else 'no'} |"
        )
    report_path.write_text(report + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return selections, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, aggregate = run_lopo(args.output_dir)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
