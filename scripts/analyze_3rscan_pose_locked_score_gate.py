#!/usr/bin/env python3
"""Test a source-fit DINO score gate on pose-locked motion assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_N = 20_000
BOOTSTRAP_SEED = 20261014
JOINT_BOOTSTRAP_N = 5_000
JOINT_BOOTSTRAP_SEED = 20261015
SCORE_PROTOCOL = "source_train_fixed"
SCORE_SIGNAL = "DINO top-1 similarity"
SCORE_TARGET = 0.25
ARMS = [
    "scene_ransac",
    "pose_locked_motion",
    "score_gated_motion",
    "baseline_preserving_score_gated_additions",
]


def bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    draws = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_N, len(delta)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high)


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    q025, median, q975 = np.quantile(np.asarray(values, dtype=float), [0.025, 0.5, 0.975])
    return {"q025": float(q025), "median": float(median), "q975": float(q975)}


def load_source_scores(root: Path) -> list[np.ndarray]:
    paths = sorted(
        (root / "outputs/runs").glob(
            "20260922_3rscan_consensus_*_vitg_diag_clean/query_diagnostics.csv"
        )
    )
    paths = [p for p in paths if p.parent.name != "20260922_3rscan_consensus_target_vitg_diag_clean"]
    if len(paths) != 16:
        raise ValueError(f"expected 16 source relations for threshold bootstrap, found {len(paths)}")
    relation_scores = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[
            frame.geometry_weight.eq(0.0)
            & frame.top_k.eq(20)
            & frame.query_evaluable.astype(bool)
        ]
        if frame.empty or frame.top1_descriptor_score.isna().any():
            raise ValueError(f"invalid source score calibration rows in {path}")
        relation_scores.append(frame.top1_descriptor_score.to_numpy(dtype=float))
    return relation_scores


def joint_source_environment_bootstrap(
    source_relation_scores: list[np.ndarray],
    environment_groups: dict[str, pd.DataFrame],
    replicates: int = JOINT_BOOTSTRAP_N,
) -> pd.DataFrame:
    """Refit the source score quantile and resample validation environments."""
    rng = np.random.default_rng(JOINT_BOOTSTRAP_SEED)
    env_names = np.array(sorted(environment_groups))
    n_rel = len(source_relation_scores)
    rows = []
    for _ in range(replicates):
        rel_ids = rng.integers(0, n_rel, size=n_rel)
        threshold = float(
            np.quantile(np.concatenate([source_relation_scores[i] for i in rel_ids]), 0.75)
        )
        sampled_envs = rng.integers(0, len(env_names), size=len(env_names))
        metric_lists = {
            arm: {"coverage": [], "precision": []}
            for arm in ARMS
        }
        paired_precision_deltas = {
            "pose_locked_motion_minus_scene_ransac": [],
            "score_gated_motion_minus_pose_locked_motion": [],
            "baseline_preserving_score_gated_additions_minus_scene_ransac": [],
            "baseline_preserving_score_gated_additions_minus_pose_locked_motion": [],
        }
        for env_idx in sampled_envs:
            group = environment_groups[env_names[env_idx]]
            score = group.top1_descriptor_score.to_numpy(dtype=float)
            score_accept = score >= threshold
            assign_base = group.assign_base.to_numpy(dtype=bool)
            assign_motion = group.assign_motion.to_numpy(dtype=bool)
            correct_base = group.correct_base.to_numpy(dtype=bool)
            correct_motion = group.correct_motion.to_numpy(dtype=bool)

            assignments = {
                "scene_ransac": assign_base,
                "pose_locked_motion": assign_motion,
                "score_gated_motion": assign_motion & score_accept,
                "baseline_preserving_score_gated_additions": assign_base | (
                    assign_motion & ~assign_base & score_accept
                ),
            }
            correct = {
                "scene_ransac": assign_base & correct_base,
                "pose_locked_motion": assign_motion & correct_motion,
                "score_gated_motion": assign_motion & score_accept & correct_motion,
                "baseline_preserving_score_gated_additions": (
                    assign_base & correct_base
                ) | (assign_motion & ~assign_base & score_accept & correct_motion),
            }
            for arm in ARMS:
                assigned_n = int(assignments[arm].sum())
                metric_lists[arm]["coverage"].append(float(correct[arm].mean()))
                metric_lists[arm]["precision"].append(
                    float(correct[arm].sum() / assigned_n) if assigned_n else np.nan
                )
            for key, candidate, reference in (
                ("pose_locked_motion_minus_scene_ransac", "pose_locked_motion", "scene_ransac"),
                ("score_gated_motion_minus_pose_locked_motion", "score_gated_motion", "pose_locked_motion"),
                (
                    "baseline_preserving_score_gated_additions_minus_scene_ransac",
                    "baseline_preserving_score_gated_additions",
                    "scene_ransac",
                ),
                (
                    "baseline_preserving_score_gated_additions_minus_pose_locked_motion",
                    "baseline_preserving_score_gated_additions",
                    "pose_locked_motion",
                ),
            ):
                if assignments[candidate].any() and assignments[reference].any():
                    paired_precision_deltas[key].append(
                        float(
                            correct[candidate].sum() / assignments[candidate].sum()
                            - correct[reference].sum() / assignments[reference].sum()
                        )
                    )
        row: dict[str, float] = {"threshold": threshold}
        for arm in ARMS:
            for metric in ("coverage", "precision"):
                vals = np.asarray(metric_lists[arm][metric], dtype=float)
                row[f"{arm}_{metric}"] = float(np.nanmean(vals))
        for key, values in paired_precision_deltas.items():
            row[key] = float(np.mean(values)) if values else np.nan
        row["baseline_preserving_minus_scene_coverage"] = (
            row["baseline_preserving_score_gated_additions_coverage"] - row["scene_ransac_coverage"]
        )
        row["baseline_preserving_minus_motion_coverage"] = (
            row["baseline_preserving_score_gated_additions_coverage"] - row["pose_locked_motion_coverage"]
        )
        row["score_gated_minus_motion_coverage"] = (
            row["score_gated_motion_coverage"] - row["pose_locked_motion_coverage"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--pose-output",
        type=Path,
        default=Path("outputs/reports/20261006_vitg_pose_locked_motion_validation"),
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        default=Path("outputs/reports/20261006_3rscan_score_selectivity_lopo"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/20261006_pose_locked_score_gate"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    pose_dir = args.pose_output if args.pose_output.is_absolute() else root / args.pose_output
    score_dir = args.score_output if args.score_output.is_absolute() else root / args.score_output
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(pose_dir / "run_manifest.csv")
    if len(manifest) != 110 or manifest.status.ne("complete").any():
        raise ValueError("expected 110 completed official validation relations")
    branch_chunks = []
    for row in manifest.itertuples(index=False):
        relation_dir = Path(row.output_dir)
        if not relation_dir.is_absolute():
            relation_dir = root / relation_dir
        path = relation_dir / "query_diagnostics.csv"
        frame = pd.read_csv(path)
        frame["environment_id"] = row.environment_id
        branch_chunks.append(frame)
    branch = pd.concat(branch_chunks, ignore_index=True)
    branch = branch[branch.moved_rigid.astype(bool)].copy()
    if branch.duplicated(["pair", "query_index", "arm"]).any():
        raise ValueError("duplicate pose-branch query/arm rows")
    if set(branch.arm.unique()) != {"global_ransac", "combined"}:
        raise ValueError(f"unexpected pose-branch arms: {sorted(branch.arm.unique())}")

    scores = pd.read_csv(score_dir / "crossfit_predictions.csv.gz")
    scores = scores[
        scores.calibration_protocol.eq(SCORE_PROTOCOL)
        & scores.signal.eq(SCORE_SIGNAL)
        & scores.train_acceptance_target.eq(SCORE_TARGET)
    ].copy()
    if scores.duplicated(["run_id", "query_index"]).any():
        raise ValueError("duplicate source-fit score rows")
    thresholds = scores.threshold.unique()
    if len(thresholds) != 1:
        raise ValueError(f"source-fit threshold should be fixed, found {len(thresholds)} values")
    threshold = float(thresholds[0])
    scores["score_accept"] = scores.top1_descriptor_score.ge(threshold)

    base = branch[branch.arm.eq("global_ransac")].copy()
    motion = branch[branch.arm.eq("combined")].copy()
    keys = ["pair", "query_index"]
    if len(base) != len(motion) or not base[keys].sort_values(keys).reset_index(drop=True).equals(
        motion[keys].sort_values(keys).reset_index(drop=True)
    ):
        raise ValueError("scene and motion arms do not contain identical moved-rigid queries")
    if not base.frame_id.reset_index(drop=True).equals(motion.frame_id.reset_index(drop=True)):
        raise ValueError("scene and motion arm frame IDs differ")

    keep = ["pair", "query_index", "frame_id", "environment_id", "query_evaluable",
            "official_rigid_instance_correct", "consensus_inlier"]
    joined = base[keep].rename(
        columns={
            "query_evaluable": "query_evaluable_base",
            "official_rigid_instance_correct": "correct_base",
            "consensus_inlier": "assign_base",
        }
    ).merge(
        motion[keys + ["frame_id", "official_rigid_instance_correct", "consensus_inlier"]].rename(
            columns={
                "official_rigid_instance_correct": "correct_motion",
                "consensus_inlier": "assign_motion",
            }
        ),
        on=keys + ["frame_id"],
        validate="one_to_one",
    ).merge(
        scores[["run_id", "query_index", "frame_id", "threshold", "top1_descriptor_score", "score_accept"]],
        left_on=["pair", "query_index", "frame_id"],
        right_on=["run_id", "query_index", "frame_id"],
        validate="one_to_one",
    )
    if joined.environment_id.nunique() != 42 or len(joined) != 14_088:
        raise ValueError("moved-rigid query join is incomplete or has unexpected scope")
    if not joined.query_evaluable_base.astype(bool).all():
        raise ValueError("moved-rigid set includes unevaluable queries")

    joined["correct_base"] = joined.correct_base.astype(bool)
    joined["correct_motion"] = joined.correct_motion.astype(bool)
    joined["assign_base"] = joined.assign_base.astype(bool)
    joined["assign_motion"] = joined.assign_motion.astype(bool)
    joined["score_accept"] = joined.score_accept.astype(bool)
    if int((joined.assign_base & joined.correct_base).sum()) != 2_876:
        raise ValueError("scene baseline does not reproduce the saved 2,876 correct transfers")
    if int((joined.assign_motion & joined.correct_motion).sum()) != 3_808:
        raise ValueError("motion branch does not reproduce the saved 3,808 correct transfers")

    joined["assign_score_gated_motion"] = joined.assign_motion & joined.score_accept
    joined["correct_score_gated_motion"] = joined.assign_score_gated_motion & joined.correct_motion
    joined["assign_preserve_base_gate_add"] = joined.assign_base | (
        joined.assign_motion & ~joined.assign_base & joined.score_accept
    )
    joined["correct_scene_ransac"] = joined.assign_base & joined.correct_base
    joined["correct_pose_locked_motion"] = joined.assign_motion & joined.correct_motion
    joined["correct_preserve_base_gate_add"] = joined.correct_scene_ransac | (
        joined.assign_motion & ~joined.assign_base & joined.score_accept & joined.correct_motion
    )
    joined.to_csv(out / "query_outcomes.csv.gz", index=False, compression="gzip")

    arm_cols = {
        "scene_ransac": ("assign_base", "correct_scene_ransac"),
        "pose_locked_motion": ("assign_motion", "correct_pose_locked_motion"),
        "score_gated_motion": ("assign_score_gated_motion", "correct_score_gated_motion"),
        "baseline_preserving_score_gated_additions": (
            "assign_preserve_base_gate_add",
            "correct_preserve_base_gate_add",
        ),
    }
    env_rows = []
    for env, group in joined.groupby("environment_id", sort=True):
        for arm in ARMS:
            assigned_col, correct_col = arm_cols[arm]
            assigned = group[assigned_col].astype(bool)
            correct = group[correct_col].astype(bool)
            n_assigned = int(assigned.sum())
            env_rows.append(
                {
                    "environment_id": env,
                    "arm": arm,
                    "queries": len(group),
                    "assigned": n_assigned,
                    "correct": int(correct.sum()),
                    "coverage": float(correct.mean()),
                    "precision": float(correct.sum() / n_assigned) if n_assigned else np.nan,
                }
            )
    env_metrics = pd.DataFrame(env_rows)
    env_metrics.to_csv(out / "environment_metrics.csv", index=False)

    pooled_rows = []
    for arm in ARMS:
        assigned_col, correct_col = arm_cols[arm]
        assigned = joined[assigned_col].astype(bool)
        correct = joined[correct_col].astype(bool)
        pooled_rows.append(
            {
                "arm": arm,
                "queries": len(joined),
                "assigned": int(assigned.sum()),
                "correct": int(correct.sum()),
                "pooled_coverage": float(correct.mean()),
                "pooled_precision": float(correct.sum() / assigned.sum()) if assigned.any() else np.nan,
                "environment_macro_coverage": float(env_metrics[env_metrics.arm.eq(arm)].coverage.mean()),
                "environment_macro_precision": float(env_metrics[env_metrics.arm.eq(arm)].precision.mean()),
                "environments_with_assignments": int(env_metrics.loc[env_metrics.arm.eq(arm), "precision"].notna().sum()),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(out / "arm_summary.csv", index=False)

    pivot_cov = env_metrics.pivot(index="environment_id", columns="arm", values="coverage")
    pivot_prec = env_metrics.pivot(index="environment_id", columns="arm", values="precision")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    contrasts = []
    references = {
        "scene_ransac": ("pose_locked_motion", "score_gated_motion", "baseline_preserving_score_gated_additions"),
        "pose_locked_motion": ("score_gated_motion", "baseline_preserving_score_gated_additions"),
    }
    for reference, candidates in references.items():
        for candidate in candidates:
            cov = bootstrap_difference(
                pivot_cov[candidate].to_numpy(), pivot_cov[reference].to_numpy(), rng
            )
            common = pivot_prec[[candidate, reference]].dropna()
            precision = bootstrap_difference(
                common[candidate].to_numpy(), common[reference].to_numpy(), rng
            )
            contrasts.append(
                {
                    "candidate_arm": candidate,
                    "reference_arm": reference,
                    "coverage_macro_difference_pp": [value * 100 for value in cov],
                    "precision_macro_difference_pp": [value * 100 for value in precision],
                    "precision_paired_environments": len(common),
                    "bootstrap_environments": len(pivot_cov),
                    "bootstrap_samples": BOOTSTRAP_N,
                }
            )
    original_additions = joined.assign_motion & ~joined.assign_base
    gated_additions = original_additions & joined.score_accept
    incremental = {
        "original_motion_additions_assigned": int(original_additions.sum()),
        "original_motion_additions_correct": int((original_additions & joined.correct_motion).sum()),
        "score_gated_additions_assigned": int(gated_additions.sum()),
        "score_gated_additions_correct": int((gated_additions & joined.correct_motion).sum()),
        "score_gate_retained_fraction_of_original_additions": float(gated_additions.sum() / original_additions.sum()),
    }
    source_relation_scores = load_source_scores(root)
    refit_threshold = float(np.quantile(np.concatenate(source_relation_scores), 0.75))
    if not np.isclose(refit_threshold, threshold, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"threshold mismatch: score audit={threshold:.12f}, source files={refit_threshold:.12f}"
        )
    joint_draws = joint_source_environment_bootstrap(
        source_relation_scores,
        {name: group.copy() for name, group in joined.groupby("environment_id", sort=True)},
    )
    joint_draws.to_csv(out / "joint_bootstrap_draws.csv.gz", index=False, compression="gzip")
    joint_columns = {
        "threshold": ("threshold", 1.0),
        "baseline_preserving_minus_scene_coverage": ("baseline_preserving_minus_scene_coverage", 100.0),
        "baseline_preserving_score_gated_additions_minus_scene_ransac": (
            "baseline_preserving_score_gated_additions_minus_scene_ransac",
            100.0,
        ),
        "baseline_preserving_minus_motion_coverage": ("baseline_preserving_minus_motion_coverage", 100.0),
        "baseline_preserving_score_gated_additions_minus_pose_locked_motion": (
            "baseline_preserving_score_gated_additions_minus_pose_locked_motion",
            100.0,
        ),
    }
    joint_intervals = {
        label: quantile_summary(joint_draws[column].to_numpy(dtype=float) * scale)
        for label, (column, scale) in joint_columns.items()
    }
    summary = {
        "status": "complete",
        "scope": "110 official 3RScan validation relations; moved-rigid query subset; 42 environments",
        "calibration": {
            "protocol": SCORE_PROTOCOL,
            "signal": SCORE_SIGNAL,
            "training_target_acceptance": SCORE_TARGET,
            "source_relations": 16,
            "threshold": threshold,
            "threshold_source": "existing fixed source-fit score analysis; validation labels and score distribution do not set the threshold",
            "threshold_fit_uncertainty": "propagated by resampling the 16 source relations and refitting the 75th-percentile score threshold in each joint bootstrap draw",
        },
        "query_join": {
            "rows": len(joined),
            "run_ids": int(joined.pair.nunique()),
            "environment_ids": int(joined.environment_id.nunique()),
            "frame_id_mismatches": 0,
            "base_correct_reproduced": 2_876,
            "motion_correct_reproduced": 3_808,
        },
        "arms": pooled_rows,
        "incremental_motion_additions": incremental,
        "paired_environment_bootstrap_contrasts": contrasts,
        "joint_source_relation_environment_bootstrap": {
            "source_relations": 16,
            "validation_environments": int(joined.environment_id.nunique()),
            "bootstrap_samples": JOINT_BOOTSTRAP_N,
            "seed": JOINT_BOOTSTRAP_SEED,
            "intervals": joint_intervals,
        },
        "interpretation": "post-hoc fixed-split sensitivity; score threshold is source-fit but the official validation split and motion branch have already been explored; not independent-site validation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Source-fit score gate on pose-locked motion assignments",
        "",
        f"Applied the fixed DINO top-1 threshold {threshold:.8f}, calibrated to a 25% training-score acceptance quantile on 16 source relations, to saved pose-locked assignments for all 110 official validation relations. The joined moved-rigid subset contains {len(joined):,} queries across {joined.environment_id.nunique()} environments; relation/query/frame IDs align exactly. Baseline and motion correct-transfer totals reproduce 2,876 and 3,808.",
        "",
        "The baseline-preserving arm retains every scene-RANSAC assignment and gates only additional pose-locked assignments by the score threshold. The all-assignment gate filters every motion assignment, including baseline inliers. All estimates are post-hoc on an already inspected validation split.",
        "",
        "| Arm | Assigned | Correct | Pooled coverage | Pooled precision | Environment-macro coverage | Environment-macro precision | Environments with assignments |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pooled_rows:
        lines.append(
            f"| {row['arm']} | {row['assigned']:,} | {row['correct']:,} | {100*row['pooled_coverage']:.2f}% | {100*row['pooled_precision']:.2f}% | {100*row['environment_macro_coverage']:.2f}% | {100*row['environment_macro_precision']:.2f}% | {row['environments_with_assignments']}/42 |"
        )
    lines.extend(
        [
            "",
            f"The motion branch originally added {incremental['original_motion_additions_assigned']:,} assignments, of which {incremental['original_motion_additions_correct']:,} were correct. Preserving scene-RANSAC assignments and applying the source gate only to those additions retains {incremental['score_gated_additions_assigned']:,} additions, of which {incremental['score_gated_additions_correct']:,} are correct ({100*incremental['score_gate_retained_fraction_of_original_additions']:.1f}% of the original added assignments).",
            "",
            "Paired environment-bootstrap differences (20,000 resamples; coverage uses all 42 environments, precision uses environments with assignments in both arms):",
            "",
            "| Arm minus reference | Coverage difference, pp [95% CI] | Precision difference, pp [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for contrast in contrasts:
        cov = contrast["coverage_macro_difference_pp"]
        prec = contrast["precision_macro_difference_pp"]
        lines.append(
            f"| {contrast['candidate_arm']} − {contrast['reference_arm']} | {cov[0]:+.2f} [{cov[1]:+.2f},{cov[2]:+.2f}] | {prec[0]:+.2f} [{prec[1]:+.2f},{prec[2]:+.2f}] (n={contrast['precision_paired_environments']}) |"
        )
    lines.extend(
        [
            "",
            "A second 5,000-draw bootstrap independently resamples the 16 source relations and 42 validation environments, refitting the score threshold in every draw. Joint intervals:",
            "",
            f"- Threshold: [{joint_intervals['threshold']['q025']:.5f}, {joint_intervals['threshold']['q975']:.5f}].",
            f"- Baseline-preserving gate minus scene RANSAC macro coverage: [{joint_intervals['baseline_preserving_minus_scene_coverage']['q025']:+.2f}, {joint_intervals['baseline_preserving_minus_scene_coverage']['q975']:+.2f}] pp.",
            f"- Baseline-preserving gate minus scene RANSAC macro precision: [{joint_intervals['baseline_preserving_score_gated_additions_minus_scene_ransac']['q025']:+.2f}, {joint_intervals['baseline_preserving_score_gated_additions_minus_scene_ransac']['q975']:+.2f}] pp.",
            f"- Baseline-preserving gate minus ungated motion macro coverage: [{joint_intervals['baseline_preserving_minus_motion_coverage']['q025']:+.2f}, {joint_intervals['baseline_preserving_minus_motion_coverage']['q975']:+.2f}] pp.",
            f"- Baseline-preserving gate minus ungated motion macro precision: [{joint_intervals['baseline_preserving_score_gated_additions_minus_pose_locked_motion']['q025']:+.2f}, {joint_intervals['baseline_preserving_score_gated_additions_minus_pose_locked_motion']['q975']:+.2f}] pp.",
            "",
            "This is a threshold-transfer sensitivity for the motion branch, not independent-site validation. The threshold was fit without validation labels, but both the validation outcomes and the branch have been inspected. Query-level outcomes and environment metrics are saved alongside this report.",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines))
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
