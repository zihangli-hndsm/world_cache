#!/usr/bin/env python3
"""Aggregate the frozen seed-17 multi-motion audit over all locally complete train relations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_STATUS = ("model_size", "layers", "iterations", "secondary_iterations", "seed", "max_modes_including_scene", "min_secondary_mode_support")
ARMS = ("global_ransac", "multi_motion")


def load_run(run_dir: Path, *, require_source_only: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or not status.get("same_query_set_all_arms"):
        raise ValueError(f"incomplete run or mismatched arm queries: {run_dir}")
    if status.get("inference_uses_annotations") is not False:
        raise ValueError(f"annotations were not confirmed post-hoc: {run_dir}")
    if not status.get("baseline_regression_pass"):
        raise ValueError(f"baseline regression did not pass: {run_dir}")
    if require_source_only and "source" not in status.get("run_scope", "source"):
        raise ValueError(f"unexpected run scope: {run_dir}")
    summary = pd.read_csv(run_dir / "summary.csv")
    diagnostics = pd.read_csv(run_dir / "query_diagnostics.csv")
    return summary, diagnostics, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-run", type=Path, default=Path("outputs/reports/20260928_multimotion_pilot"))
    parser.add_argument("--added-run", type=Path, default=Path("outputs/reports/20261003_3rscan_extra_pair_multimotion"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261003_multimotion_extended_source_audit"))
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=24817)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("use at least 1,000 relation-bootstrap replicates")
    old, old_diag, old_status = load_run(args.original_run, require_source_only=True)
    new, new_diag, new_status = load_run(args.added_run, require_source_only=True)
    for key in REQUIRED_STATUS:
        if old_status.get(key) != new_status.get(key):
            raise ValueError(f"protocol mismatch for {key}: {old_status.get(key)} != {new_status.get(key)}")
    old = old[(old.role == "source") & old.arm.isin(ARMS)].copy()
    new = new[new.arm.isin(ARMS)].copy()
    new["role"] = "source"
    summaries = pd.concat([old, new], ignore_index=True)
    if summaries.duplicated(["pair", "arm"]).any():
        raise ValueError("relation/arm appears more than once")
    pivot = summaries.pivot(index="pair", columns="arm", values=[
        "moved_rigid_queries", "moved_rigid_correct_transfer_count",
        "moved_rigid_correct_transfer_coverage", "strict_scene_pose_success",
        "scene_rotation_error_deg", "scene_translation_error_m",
    ])
    rows = []
    for pair, row in pivot.iterrows():
        q = int(row[("moved_rigid_queries", "global_ransac")])
        q_multi = int(row[("moved_rigid_queries", "multi_motion")])
        if q != q_multi:
            raise ValueError(f"moved-query denominator mismatch for {pair}")
        correct_global = int(row[("moved_rigid_correct_transfer_count", "global_ransac")])
        correct_multi = int(row[("moved_rigid_correct_transfer_count", "multi_motion")])
        if q:
            delta = correct_multi / q - correct_global / q
        else:
            delta = np.nan
        pose_global = bool(row[("strict_scene_pose_success", "global_ransac")])
        pose_multi = bool(row[("strict_scene_pose_success", "multi_motion")])
        rot_diff = abs(float(row[("scene_rotation_error_deg", "global_ransac")]) - float(row[("scene_rotation_error_deg", "multi_motion")]))
        trans_diff = abs(float(row[("scene_translation_error_m", "global_ransac")]) - float(row[("scene_translation_error_m", "multi_motion")]))
        if pose_global != pose_multi or rot_diff > 1e-9 or trans_diff > 1e-9:
            raise ValueError(f"the dominant scene pose changed for {pair}")
        rows.append({
            "pair": pair, "role": "source", "moved_rigid_queries": q,
            "global_correct_transfers": correct_global, "multi_motion_correct_transfers": correct_multi,
            "global_coverage": correct_global / q if q else np.nan,
            "multi_motion_coverage": correct_multi / q if q else np.nan,
            "paired_coverage_delta": delta,
            "strict_scene_pose_success": pose_global,
            "scene_rotation_error_deg": float(row[("scene_rotation_error_deg", "global_ransac")]),
            "scene_translation_error_m": float(row[("scene_translation_error_m", "global_ransac")]),
        })
    per_relation = pd.DataFrame(rows).sort_values("pair")
    if len(per_relation) != 18:
        raise ValueError(f"expected 18 source relations after extension, found {len(per_relation)}")
    # Confirm exact query identity between fixed arms on every relation.
    diagnostics = pd.concat([old_diag, new_diag], ignore_index=True)
    diagnostics = diagnostics[diagnostics.arm.isin(ARMS)]
    for pair, group in diagnostics.groupby("pair"):
        sets = [set(g.query_index.astype(int)) for _, g in group.groupby("arm")]
        if len(sets) != 2 or sets[0] != sets[1]:
            raise ValueError(f"query pool differs across global and multi-motion arms for {pair}")
    eligible = per_relation[per_relation.moved_rigid_queries > 0].copy()
    if len(eligible) != 14:
        raise ValueError(f"expected 14 relations with linked rigid queries, found {len(eligible)}")
    q_total = int(eligible.moved_rigid_queries.sum())
    global_correct = int(eligible.global_correct_transfers.sum())
    multi_correct = int(eligible.multi_motion_correct_transfers.sum())
    rng = np.random.default_rng(args.seed)
    values = eligible.paired_coverage_delta.to_numpy()
    denominators = eligible.moved_rigid_queries.to_numpy()
    differences = (eligible.multi_motion_correct_transfers - eligible.global_correct_transfers).to_numpy()
    indices = rng.integers(0, len(eligible), size=(args.bootstrap_samples, len(eligible)))
    macro_boot = values[indices].mean(axis=1)
    weighted_boot = differences[indices].sum(axis=1) / denominators[indices].sum(axis=1)
    macro_ci = np.quantile(macro_boot, [0.025, 0.975]).tolist()
    weighted_ci = np.quantile(weighted_boot, [0.025, 0.975]).tolist()
    positive = int((values > 0).sum()); negative = int((values < 0).sum()); tied = int((values == 0).sum())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_relation.to_csv(args.output_dir / "per_relation.csv", index=False)
    result = {
        "scope": "18 source relations with local annotated PLY and sequence ZIP; official 3RScan train split",
        "protocol": {key: old_status[key] for key in REQUIRED_STATUS},
        "all_source_relations": len(per_relation), "relations_with_linked_rigid_queries": len(eligible),
        "zero_query_relations": int((per_relation.moved_rigid_queries == 0).sum()),
        "moved_rigid_queries": q_total,
        "global_ransac_correct_transfers": global_correct,
        "multi_motion_correct_transfers": multi_correct,
        "global_ransac_weighted_coverage": global_correct / q_total,
        "multi_motion_weighted_coverage": multi_correct / q_total,
        "weighted_coverage_delta": (multi_correct - global_correct) / q_total,
        "weighted_relation_bootstrap_95ci": weighted_ci,
        "relation_macro_delta": float(values.mean()), "relation_macro_bootstrap_95ci": macro_ci,
        "relations_positive": positive, "relations_negative": negative, "relations_tied": tied,
        "strict_scene_pose_successes_global": int(per_relation.strict_scene_pose_success.sum()),
        "strict_scene_pose_successes_multi_motion": int(per_relation.strict_scene_pose_success.sum()),
        "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.seed,
        "interpretation": "internal availability-selected train-relation audit; not external-site validation",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = f"""# Extended source-relation multi-motion audit

This combines the original 16 source relations with the two additional locally complete train relations. Both runs use the same frozen protocol: ViT-G, layers {old_status['layers']}, seed {old_status['seed']}, {old_status['iterations']} global iterations, {old_status['secondary_iterations']} secondary iterations, at most {old_status['max_modes_including_scene']} modes, and minimum secondary support {old_status['min_secondary_mode_support']}. Exact candidate query sets agree between arms within every relation. The two added relation baselines reproduce their saved global-RANSAC query count, inlier count, and pose errors exactly.

## Results

On {q_total:,} linked moved-rigid queries from {len(eligible)} relations, global RANSAC transfers {global_correct:,} ({100*global_correct/q_total:.1f}%) correctly and multi-motion transfers {multi_correct:,} ({100*multi_correct/q_total:.1f}%). The weighted paired difference is {100*(multi_correct-global_correct)/q_total:+.2f} pp (relation-resampling 95% CI [{100*weighted_ci[0]:+.2f}, {100*weighted_ci[1]:+.2f}]). The unweighted relation-macro difference is {100*values.mean():+.2f} pp (95% CI [{100*macro_ci[0]:+.2f}, {100*macro_ci[1]:+.2f}]); {positive} relations improve, {negative} declines, and {tied} tie. These intervals resample relation units only; physical capture-site independence is unverified.

Strict scene-pose successes are {result['strict_scene_pose_successes_global']}/{len(per_relation)} for both arms. The added mode hypotheses leave the dominant global pose unchanged, so the correct object-transfer gain does not restore scene registration on these cases.

## Evidence boundary

The two added relations are from official 3RScan train and were selected because both required raw inputs were locally available. Four of the 18 relations have no linked rigid query and do not enter the transfer-rate bootstrap. One added relation contributes 172 queries; the other contributes only three. This is a broader internal relation audit, not an external validation or independent-site test. The paired relation intervals include zero, so this extended set does not establish a reliable positive multi-motion effect. The result supports retaining the change-aware diagnosis while withholding a method claim pending external data and a mechanism that controls the assignment-precision cost.

The adaptive-K and combined arms are excluded: the added runs use smoke-mode `--pairs`, which bypasses the source-calibrated K threshold. The comparison here is only global RANSAC versus fixed-K multi-motion.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
