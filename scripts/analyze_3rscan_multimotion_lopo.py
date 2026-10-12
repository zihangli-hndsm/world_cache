"""Source-only leave-one-relation-out selection of motion-mode count."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _bool_column(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy()
    return values.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def evaluate_mode_prefix(frame: pd.DataFrame, mode_count: int, threshold: float = 0.10) -> dict[str, float | int]:
    """Reassign each query using only the first `mode_count` hypotheses."""
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
        raise ValueError("diagnostics contain no per-hypothesis residuals")
    take = present[: max(1, min(mode_count, len(present)))]
    residuals = frame[[residual_columns[i] for i in take]].to_numpy(dtype=np.float64)
    object_ids = frame[[object_columns[i] for i in take]].to_numpy(dtype=np.float64)
    valid = np.isfinite(residuals) & (residuals <= threshold)
    masked = np.where(valid, residuals, np.inf)
    slots = np.argmin(masked, axis=1)
    rows = np.arange(len(frame))
    best_residual = masked[rows, slots]
    selected_object = object_ids[rows, slots].astype(np.int64)
    assigned = np.isfinite(best_residual)
    query_evaluable = _bool_column(frame.query_evaluable)
    expected_object = frame.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
    moved = query_evaluable & (expected_object > 0)
    correct = moved & assigned & (selected_object == expected_object)
    return {
        "moved_rigid_queries": int(moved.sum()),
        "correct_transfers": int(correct.sum()),
        "coverage": float(correct.sum() / moved.sum()) if moved.any() else float("nan"),
        "assigned_moved_queries": int((moved & assigned).sum()),
        "precision": float(correct.sum() / (moved & assigned).sum()) if (moved & assigned).any() else float("nan"),
        "selected_mode_count": int(len(take)),
    }


def assign_mode_prefix(frame: pd.DataFrame, mode_count: int, threshold: float = 0.10) -> pd.DataFrame:
    """Return post-hoc query scores under a truncated motion set."""
    result = frame.copy()
    residual_columns = sorted(
        (column for column in result if column.startswith("hypothesis_") and column.endswith("_residual_m")),
        key=lambda column: int(column.split("_")[1]),
    )
    object_columns = [
        f"hypothesis_{int(column.split('_')[1])}_selected_reference_instance_id"
        for column in residual_columns
    ]
    present = [
        i for i, column in enumerate(residual_columns)
        if result[column].notna().any() and result[object_columns[i]].notna().any()
    ]
    take = present[: max(1, min(mode_count, len(present)))]
    residuals = result[[residual_columns[i] for i in take]].to_numpy(dtype=np.float64)
    object_ids = result[[object_columns[i] for i in take]].to_numpy(dtype=np.float64)
    valid = np.isfinite(residuals) & (residuals <= threshold)
    masked = np.where(valid, residuals, np.inf)
    slots = np.argmin(masked, axis=1)
    rows = np.arange(len(result))
    best_residual = masked[rows, slots]
    result["lopo_selected_mode_count"] = len(take)
    result["lopo_assigned_mode"] = np.where(np.isfinite(best_residual), np.asarray(take)[slots], -1)
    result["lopo_residual_m"] = best_residual
    result["lopo_selected_reference_instance_id"] = object_ids[rows, slots].astype(np.int64)
    result["lopo_consensus_inlier"] = np.isfinite(best_residual)
    query_evaluable = _bool_column(result.query_evaluable)
    expected_object = result.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
    moved = query_evaluable & (expected_object > 0)
    result["lopo_moved_rigid_query"] = moved
    result["lopo_correct_moved_rigid_transfer"] = (
        moved
        & result.lopo_consensus_inlier.to_numpy()
        & (result.lopo_selected_reference_instance_id.to_numpy() == expected_object)
    )
    return result


def select_mode_count(
    diagnostics: pd.DataFrame,
    train_pairs: list[str],
    arm: str,
    max_modes: int = 4,
    threshold: float = 0.10,
) -> tuple[int, dict[int, int]]:
    scores: dict[int, int] = {}
    for mode_count in range(1, max_modes + 1):
        score = 0
        for pair in train_pairs:
            frame = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == arm)]
            score += int(evaluate_mode_prefix(frame, mode_count, threshold)["correct_transfers"])
        scores[mode_count] = score
    # Prefer fewer modes when source weighted transfer counts tie.
    best = min(mode for mode, score in scores.items() if score == max(scores.values()))
    return best, scores


def run_lopo(diagnostics: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_pairs = sorted(summary.loc[summary.role == "source", "pair"].unique().tolist())
    target_pairs = summary.loc[summary.role == "target", "pair"].unique().tolist()
    if len(source_pairs) != 16 or target_pairs != ["target"]:
        raise ValueError("LOPO mode selection requires all 16 source relations and the single held-out target")
    arms = ["multi_motion", "combined"]
    selected_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    query_frames: list[pd.DataFrame] = []

    for arm in arms:
        for heldout in source_pairs:
            training_pairs = [pair for pair in source_pairs if pair != heldout]
            selected, train_scores = select_mode_count(diagnostics, training_pairs, arm)
            frame = diagnostics[(diagnostics.pair == heldout) & (diagnostics.arm == arm)]
            metric = evaluate_mode_prefix(frame, selected)
            base_frame = diagnostics[(diagnostics.pair == heldout) & (diagnostics.arm == "global_ransac")]
            baseline_metric = evaluate_mode_prefix(base_frame, 1)
            selected_rows.append({
                "arm": arm,
                "role": "source_lopo",
                "heldout_pair": heldout,
                "training_relations": len(training_pairs),
                "selected_mode_count": selected,
                "training_correct_transfers_at_selected_count": train_scores[selected],
                **{f"training_mode_{mode}_correct_transfers": count for mode, count in train_scores.items()},
                "heldout_moved_rigid_queries": metric["moved_rigid_queries"],
                "heldout_correct_transfers": metric["correct_transfers"],
                "heldout_coverage": metric["coverage"],
                "baseline_heldout_correct_transfers": baseline_metric["correct_transfers"],
                "baseline_heldout_coverage": baseline_metric["coverage"],
            })
            query_frames.append(assign_mode_prefix(frame, selected))

        selected, train_scores = select_mode_count(diagnostics, source_pairs, arm)
        frame = diagnostics[(diagnostics.pair == "target") & (diagnostics.arm == arm)]
        metric = evaluate_mode_prefix(frame, selected)
        selected_rows.append({
            "arm": arm,
            "role": "target_once",
            "heldout_pair": "target",
            "training_relations": len(source_pairs),
            "selected_mode_count": selected,
            "training_correct_transfers_at_selected_count": train_scores[selected],
            **{f"training_mode_{mode}_correct_transfers": count for mode, count in train_scores.items()},
            "heldout_moved_rigid_queries": metric["moved_rigid_queries"],
            "heldout_correct_transfers": metric["correct_transfers"],
            "heldout_coverage": metric["coverage"],
            "baseline_heldout_correct_transfers": np.nan,
            "baseline_heldout_coverage": np.nan,
        })
        query_frames.append(assign_mode_prefix(frame, selected))

    selections = pd.DataFrame(selected_rows)
    selections.to_csv(output_dir / "lopo_mode_selection.csv", index=False)
    query_output = pd.concat(query_frames, ignore_index=True)
    query_output.to_csv(output_dir / "lopo_query_diagnostics.csv", index=False)

    aggregate_rows: list[dict[str, object]] = []
    for arm in arms:
        source_result = selections[(selections.arm == arm) & (selections.role == "source_lopo")]
        denominator = int(source_result.heldout_moved_rigid_queries.sum())
        correct = int(source_result.heldout_correct_transfers.sum())
        comparable = source_result.heldout_moved_rigid_queries > 0
        nonnegative = int((
            source_result.loc[comparable, "heldout_correct_transfers"].to_numpy()
            / source_result.loc[comparable, "heldout_moved_rigid_queries"].to_numpy()
            >= source_result.loc[comparable, "baseline_heldout_coverage"].to_numpy()
        ).sum())
        base_correct = int(source_result.baseline_heldout_correct_transfers.sum())
        base_coverage = base_correct / denominator if denominator else float("nan")
        coverage = correct / denominator if denominator else float("nan")
        method_pose = summary[(summary.role == "source") & (summary.arm == arm)]
        baseline_pose = summary[(summary.role == "source") & (summary.arm == "global_ransac")]
        pose_loss = int(baseline_pose.strict_scene_pose_success.sum() - method_pose.strict_scene_pose_success.sum())
        aggregate_rows.append({
            "arm": arm,
            "source_pairs": len(source_pairs),
            "source_eligible_pairs": int(comparable.sum()),
            "source_lopo_correct_transfers": correct,
            "source_moved_rigid_queries": denominator,
            "source_lopo_coverage": coverage,
            "baseline_correct_transfers": base_correct,
            "baseline_coverage": base_coverage,
            "source_lopo_coverage_delta_pp": 100.0 * (coverage - base_coverage),
            "source_pairs_nonnegative": nonnegative,
            "source_pairs_negative": int(comparable.sum()) - nonnegative,
            "source_pairs_not_comparable": len(source_pairs) - int(comparable.sum()),
            "source_strict_scene_pose_successes": int(method_pose.strict_scene_pose_success.sum()),
            "scene_pose_success_loss": pose_loss,
            "stage1_gate_pass": bool(
                coverage - base_coverage >= 0.10 and nonnegative >= 10 and pose_loss <= 1
            ),
        })
    target_rows = selections[selections.role == "target_once"]
    for row in aggregate_rows:
        target = target_rows[target_rows.arm == row["arm"]].iloc[0]
        row["target_selected_mode_count"] = int(target.selected_mode_count)
        row["target_correct_transfers"] = int(target.heldout_correct_transfers)
        row["target_moved_rigid_queries"] = int(target.heldout_moved_rigid_queries)
        row["target_coverage"] = float(target.heldout_coverage)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "lopo_mode_summary.csv", index=False)

    lines = [
        "## Source-only LOPO mode-count selection",
        "",
        "For each source relation, the number of retained motion hypotheses was selected by maximizing source weighted correct moved-rigid transfer count on the other 15 relations; ties favor fewer modes. The target mode count was selected once using all 16 sources.",
        "",
        "| Arm | LOPO source coverage | Change vs baseline | Nonnegative / eligible source pairs | N/A pairs | Scene-pose loss | Target mode count | Target coverage | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.arm} | {100 * row.source_lopo_coverage:.2f}% | "
            f"{row.source_lopo_coverage_delta_pp:+.2f} pp | "
            f"{row.source_pairs_nonnegative}/{row.source_eligible_pairs} | "
            f"{row.source_pairs_not_comparable}/16 | {row.scene_pose_success_loss}/16 | "
            f"{row.target_selected_mode_count} | {100 * row.target_coverage:.2f}% | "
            f"{'yes' if row.stage1_gate_pass else 'no'} |"
        )
    report_path = output_dir / "multimotion_report.md"
    prior = report_path.read_text(encoding="utf-8") if report_path.exists() else "# 3RScan multi-motion pilot\n"
    marker = "## Source-only LOPO mode-count selection"
    prior = prior.split(marker)[0].rstrip()
    relation_effects = selections[selections.role == "source_lopo"].copy()
    relation_effects["coverage_delta_pp"] = 100.0 * (
        relation_effects.heldout_coverage - relation_effects.baseline_heldout_coverage
    )
    effect_sentences = []
    for arm in arms:
        eligible = relation_effects[(relation_effects.arm == arm) & relation_effects.coverage_delta_pp.notna()]
        best = eligible.loc[eligible.coverage_delta_pp.idxmax()]
        worst = eligible.loc[eligible.coverage_delta_pp.idxmin()]
        effect_sentences.append(
            f"{arm}: largest gain {best.heldout_pair} ({best.coverage_delta_pp:+.1f} pp, "
            f"{int(best.heldout_correct_transfers)}/{int(best.heldout_moved_rigid_queries)} vs "
            f"{int(best.baseline_heldout_correct_transfers)}/{int(best.heldout_moved_rigid_queries)}); "
            f"largest decline {worst.heldout_pair} ({worst.coverage_delta_pp:+.1f} pp)."
        )
    detail_lines = [
        "",
        "Relation effects are concentrated and heterogeneous. " + " ".join(effect_sentences) + " The opposing room-level outcomes explain why the weighted source gain remains below the 10 pp gate.",
    ]
    report_path.write_text(prior + "\n\n" + "\n".join(lines + detail_lines) + "\n", encoding="utf-8")
    (output_dir / "lopo_status.json").write_text(
        '{\n'
        '  "status": "complete",\n'
        '  "selection": "maximise source weighted correct moved-rigid transfers across all eligible source queries; ties favour fewer modes",\n'
        '  "validation": "leave one source relation out",\n'
        '  "target": "selected once using all 16 sources and scored once",\n'
        '  "annotations_used_for_selection": true,\n'
        '  "target_annotations_used_for_selection": false,\n'
        '  "independent_environment_validation": false\n'
        '}\n',
        encoding="utf-8",
    )
    return selections, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    diagnostics = pd.read_csv(args.output_dir / "query_diagnostics.csv")
    summary = pd.read_csv(args.output_dir / "summary.csv")
    _, aggregate = run_lopo(diagnostics, summary, args.output_dir)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
