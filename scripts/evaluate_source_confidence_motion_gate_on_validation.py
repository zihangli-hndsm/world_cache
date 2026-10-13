#!/usr/bin/env python3
"""Apply the final source-only motion-confidence selector to 3RScan validation.

The geometric configuration and top1-minus-top20 descriptor-gap threshold
come from the 16 official-train source relations. This reuses the completed
seed-17 motion hypotheses; it does not fit on validation labels or rerun the
backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_3rscan_instance_correspondence import read_annotated_mesh


INPUT = Path("outputs/reports/20261006_vitg_pose_locked_motion_validation")
SOURCE_FIT = Path("outputs/reports/20261003_source_confidence_motion_lopo/summary.json")
OUTPUT = Path("outputs/reports/20261004_source_confidence_motion_validation")
PAIR_MANIFEST = Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def rates(counts: dict[str, int]) -> dict[str, float]:
    return {
        "coverage": counts["correct"] / counts["moved"] if counts["moved"] else float("nan"),
        "precision": counts["correct"] / counts["assigned"] if counts["assigned"] else float("nan"),
    }


def macro_bootstrap(
    env_counts: dict[str, dict[str, int]], left: str, right: str,
    metric: str, samples: int, seed: int,
) -> dict[str, object]:
    envs = sorted(env_counts)
    def value(counts: dict[str, int]) -> float:
        if metric == "coverage":
            return counts["correct"] / counts["moved"] if counts["moved"] else float("nan")
        return counts["correct"] / counts["assigned"] if counts["assigned"] else float("nan")
    deltas = np.asarray([value(env_counts[e][left]) - value(env_counts[e][right]) for e in envs], dtype=float)
    valid = np.isfinite(deltas)
    paired = deltas[valid]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(paired), size=(samples, len(paired)))
    boot = paired[draws].mean(axis=1)
    return {"estimate": float(paired.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])], "environments": int(valid.sum())}


def stratified_bootstrap(
    query_frame: pd.DataFrame, available: bool, candidate_arm: str, reference_arm: str,
    metric: str, samples: int, seed: int,
) -> dict[str, object]:
    subset = query_frame[query_frame.official_target_in_top20.astype(bool) == available]
    envs = sorted(subset.environment_id.unique())
    values = []
    for env in envs:
        per_arm = {}
        for arm in (candidate_arm, reference_arm):
            group = subset[(subset.environment_id == env) & (subset.arm == arm)]
            moved = int(group.moved_rigid.sum())
            assigned = int(group.assigned.sum())
            correct = int(group.correct.sum())
            denominator = moved if metric == "coverage" else assigned
            per_arm[arm] = correct / denominator if denominator else float("nan")
        values.append(per_arm[candidate_arm] - per_arm[reference_arm])
    paired = np.asarray(values, dtype=float)
    paired = paired[np.isfinite(paired)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(paired), size=(samples, len(paired)))
    boot = paired[draws].mean(axis=1)
    return {"estimate": float(paired.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])], "environments": int(len(paired))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--source-fit", type=Path, default=SOURCE_FIT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--pair-manifest", type=Path, default=PAIR_MANIFEST)
    parser.add_argument("--annotation-root", type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=10_042_028)
    args = parser.parse_args()
    fit = json.loads((args.root / args.source_fit).read_text())
    source = fit["target_train_diagnostic"]
    config = {
        "modes": int(source["modes"]),
        "margin_m": float(source["residual_margin_m"]),
        "min_independent_support": int(source["min_independent_support"]),
        "descriptor_gap_threshold": float(source["descriptor_gap_threshold"]),
        "source_training_precision": float(source["training_precision"]),
        "source_training_relations": 16,
        "selection_rule": "jointly selected on all 16 train source relations under a 90% pooled physical-instance precision floor",
    }
    if config["modes"] != 2 or config["min_independent_support"] != 128 or not np.isclose(config["margin_m"], 0.15):
        raise ValueError(f"unexpected source-fitted motion configuration: {config}")
    pairs = read_csv(args.root / args.pair_manifest)
    if len(pairs) != 110 or len({p["environment_id"] for p in pairs}) != 47:
        raise ValueError("expected 110 official relations and 47 environments")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    query_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    env_counts: dict[str, dict[str, dict[str, int]]] = {}
    alignment_checks = 0
    for i, pair in enumerate(pairs):
        run_id = f"{i:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        path = args.root / args.input_dir / "relations" / run_id / "query_diagnostics.csv"
        diagnostic = pd.read_csv(path)
        base = diagnostic[diagnostic.arm == "global_ransac"].sort_values("query_index").reset_index(drop=True)
        motion = diagnostic[diagnostic.arm == "combined"].sort_values("query_index").reset_index(drop=True)
        if not np.array_equal(base.query_index.to_numpy(), motion.query_index.to_numpy()) or not np.array_equal(base.frame_id.to_numpy(), motion.frame_id.to_numpy()):
            raise ValueError(f"baseline/motion query mismatch for {run_id}")
        if not np.array_equal(base.descriptor_gap_top1_top20.to_numpy(), motion.descriptor_gap_top1_top20.to_numpy(), equal_nan=True):
            raise ValueError(f"descriptor confidence mismatch for {run_id}")
        alignment_checks += 1

        n = len(base)
        expected = base.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
        moved = base.moved_rigid.astype(bool).to_numpy() & (expected > 0)
        cache_path = args.root / args.input_dir / "descriptor_candidates" / f"{run_id}.npz"
        with np.load(cache_path) as cache:
            if str(cache["pair"].item()) != run_id:
                raise ValueError(f"descriptor candidate cache identity mismatch: {run_id}")
            ref_points = cache["reference_points"].astype(np.float32)
            candidate_indices = cache["candidate_indices"][:, :20].astype(np.int64)
            cache_frames = cache["query_frame_ids"].astype(np.int64)
        if len(cache_frames) != n or not np.array_equal(cache_frames, base.frame_id.to_numpy(dtype=np.int64)):
            raise ValueError(f"candidate-cache query/frame mismatch: {run_id}")
        tree, reference_object_ids, _, _, _ = read_annotated_mesh(args.annotation_root, pair["reference_scan"])
        _, nearest = tree.query(ref_points, k=1)
        candidate_object_ids = reference_object_ids[nearest]
        target_candidate_hit = np.zeros(n, dtype=bool)
        for qi in np.flatnonzero(moved):
            target_candidate_hit[qi] = np.any(candidate_object_ids[candidate_indices[qi]] == expected[qi])
        base_residual = base.spatial_residual_m.to_numpy(dtype=float)
        base_object = base.selected_reference_instance_id.to_numpy(dtype=np.int64)
        # The final source selector uses only one secondary mode (mode 1).
        mode_residual = motion.hypothesis_1_residual_m.to_numpy(dtype=float)
        mode_object = motion.hypothesis_1_selected_reference_instance_id.fillna(-1).to_numpy(dtype=np.int64)
        independent_support = int(np.sum((base_residual > 0.10) & np.isfinite(mode_residual) & (mode_residual <= 0.10)))
        status = json.loads((args.root / args.input_dir / "relations" / run_id / "status.json").read_text())
        saved_support = int(status["selector_diagnostics"]["secondary_independent_support"][0])
        if independent_support != saved_support:
            raise ValueError(f"mode-1 support mismatch for {run_id}: {independent_support} != {saved_support}")

        secondary_eligible = independent_support >= config["min_independent_support"]
        switch = np.zeros(n, dtype=bool)
        if secondary_eligible:
            switch = (
                np.isfinite(mode_residual)
                & (mode_residual <= 0.10)
                & ((base_residual > 0.10) | ((base_residual - mode_residual) >= config["margin_m"]))
            )
        unfiltered_assigned = (base_residual <= 0.10) | switch
        chosen_object = base_object.copy()
        chosen_object[switch] = mode_object[switch]
        gap = base.descriptor_gap_top1_top20.to_numpy(dtype=float)
        confidence_keep = gap >= config["descriptor_gap_threshold"]
        gated_assigned = unfiltered_assigned & confidence_keep
        base_assigned = base.consensus_inlier.astype(bool).to_numpy()
        base_correct = base.baseline_correct.astype(bool).to_numpy()
        unfiltered_correct = moved & unfiltered_assigned & (chosen_object == expected)
        gated_correct = moved & gated_assigned & (chosen_object == expected)
        if not np.array_equal(unfiltered_correct & (base_residual <= 0.10), base_correct):
            raise ValueError(f"unfiltered source geometry's baseline inliers failed to reproduce: {run_id}")
        if np.any((base_correct | gated_correct) & moved & ~target_candidate_hit):
            raise ValueError(f"correct transfer without its official top-20 object candidate: {run_id}")

        arms = {
            "baseline": {"assigned_mask": moved & base_assigned, "correct_mask": base_correct},
            "motion_config": {"assigned_mask": moved & unfiltered_assigned, "correct_mask": unfiltered_correct},
            "confidence_guard": {"assigned_mask": moved & gated_assigned, "correct_mask": gated_correct},
        }
        counts: dict[str, dict[str, int]] = {}
        for arm, masks in arms.items():
            counts[arm] = {
                "moved": int(moved.sum()),
                "assigned": int(masks["assigned_mask"].sum()),
                "correct": int(masks["correct_mask"].sum()),
            }
            for query_index in np.flatnonzero(moved):
                query_rows.append({
                    "run_id": run_id, "environment_id": pair["environment_id"],
                    "query_index": int(query_index), "frame_id": int(base.frame_id.iloc[query_index]),
                    "arm": arm, "moved_rigid": True,
                    "official_target_in_top20": bool(target_candidate_hit[query_index]),
                    "descriptor_gap": float(gap[query_index]),
                    "assigned": bool(masks["assigned_mask"][query_index]),
                    "correct": bool(masks["correct_mask"][query_index]),
                })
        relation = {"run_id": run_id, "environment_id": pair["environment_id"], "mode1_independent_support": independent_support}
        for arm, values in counts.items():
            relation.update({f"{arm}_{key}": value for key, value in values.items()})
            relation.update({f"{arm}_{key}": value for key, value in rates(values).items()})
        relation["confidence_retained_fraction_of_motion_assignments"] = counts["confidence_guard"]["assigned"] / counts["motion_config"]["assigned"] if counts["motion_config"]["assigned"] else float("nan")
        relation_rows.append(relation)
        env = env_counts.setdefault(pair["environment_id"], {})
        for arm, values in counts.items():
            target = env.setdefault(arm, {"moved": 0, "assigned": 0, "correct": 0})
            for key, value in values.items():
                target[key] += value
        if (i + 1) % 20 == 0:
            print(f"processed {i+1}/110 relations", flush=True)

    relation_frame = pd.DataFrame(relation_rows)
    query_frame = pd.DataFrame(query_rows)
    relation_frame.to_csv(args.output_dir / "per_relation.csv", index=False)
    query_frame.to_csv(args.output_dir / "per_query.csv", index=False)
    environment_rows = []
    for env, arms in env_counts.items():
        for arm, values in arms.items():
            environment_rows.append({"environment_id": env, "arm": arm, **values, **rates(values)})
    pd.DataFrame(environment_rows).to_csv(args.output_dir / "per_environment.csv", index=False)
    pooled: dict[str, object] = {}
    pooled_counts: dict[str, dict[str, int]] = {}
    for arm in ("baseline", "motion_config", "confidence_guard"):
        sub = relation_frame
        counts = {k: int(sub[f"{arm}_{k}"].sum()) for k in ("moved", "assigned", "correct")}
        pooled_counts[arm] = counts
        pooled[arm] = {"counts": counts, **rates(counts)}
    deltas = {}
    for candidate_arm in ("motion_config", "confidence_guard"):
        for metric in ("coverage", "precision"):
            deltas[f"{candidate_arm}_minus_baseline_{metric}"] = macro_bootstrap(
                env_counts, candidate_arm, "baseline", metric,
                args.bootstrap_samples, args.bootstrap_seed + (metric == "precision"),
            )
    for metric in ("coverage", "precision"):
        deltas[f"confidence_guard_minus_motion_config_{metric}"] = macro_bootstrap(
            env_counts, "confidence_guard", "motion_config", metric,
            args.bootstrap_samples, args.bootstrap_seed + 10 + (metric == "precision"),
        )
    keep = pooled_counts["confidence_guard"]["assigned"] / pooled_counts["motion_config"]["assigned"] if pooled_counts["motion_config"]["assigned"] else float("nan")
    stratified: dict[str, object] = {}
    for available, label in ((True, "candidate_present"), (False, "candidate_absent")):
        subset = query_frame[query_frame.official_target_in_top20.astype(bool) == available]
        arm_summary = {}
        for arm in ("baseline", "motion_config", "confidence_guard"):
            group = subset[subset.arm == arm]
            assigned = int(group.assigned.sum())
            correct = int(group.correct.sum())
            arm_summary[arm] = {"moved": int(group.moved_rigid.sum()), "assigned": assigned, "correct": correct, "false_assignments": assigned - correct, "coverage": correct / len(group) if len(group) else float("nan"), "precision": correct / assigned if assigned else None}
        motion = subset[subset.arm == "motion_config"].set_index(["run_id", "query_index"])
        guard = subset[subset.arm == "confidence_guard"].set_index(["run_id", "query_index"])
        rejected = motion.assigned.astype(bool) & ~guard.assigned.astype(bool)
        stratified[label] = {
            "queries": int(len(subset[subset.arm == "baseline"])),
            "arms": arm_summary,
            "confidence_gate_removed_from_motion_config": int(rejected.sum()),
            "removed_correct_assignments": int((rejected & motion.correct.astype(bool)).sum()),
            "removed_false_assignments": int((rejected & ~motion.correct.astype(bool)).sum()),
            "confidence_guard_minus_motion_config_macro_coverage": stratified_bootstrap(query_frame, available, "confidence_guard", "motion_config", "coverage", args.bootstrap_samples, args.bootstrap_seed + int(available)),
            "confidence_guard_minus_motion_config_macro_precision": stratified_bootstrap(query_frame, available, "confidence_guard", "motion_config", "precision", args.bootstrap_samples, args.bootstrap_seed + 20 + int(available)),
        }
    summary = {
        "status": "complete", "relations": 110, "environments": 47,
        "source_frozen_selector": config,
        "validation_labels_used_for_selector_or_threshold": False,
        "posthoc_application": True,
        "query_identity_relations_checked": alignment_checks,
        "mode1_support_reconstructed_exactly": True,
        "pooled": pooled,
        "environment_macro_paired_deltas_vs_baseline": deltas,
        "candidate_availability_strata": stratified,
        "confidence_gate_fraction_of_unfiltered_motion_assignments_retained": keep,
        "bootstrap": {"unit": "environment; paired resampling of 47 environment clusters", "samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "scope": "The validation split has already been inspected. This is one fixed source-trained selector application, not independent-site generalization.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Source-fitted descriptor-confidence gate on full validation",
        "",
        f"Applied the final selector chosen on 16 official-train source relations: {config['modes']} modes, {config['margin_m']:.2f} m residual margin, {config['min_independent_support']} independent secondary inliers, and descriptor gap >= {config['descriptor_gap_threshold']:.5f}. Its source pooled precision was {config['source_training_precision']:.1%} under a 90% source floor. No validation label selected the rule. The saved mode-1 independent support was reconstructed exactly for all 110 relations.",
        "",
        "| Arm | Correct / moved | Correct coverage pooled | Assigned precision pooled | Assigned / moved |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, title in (("baseline", "Global RANSAC baseline"), ("motion_config", "Same 2-mode config, no descriptor gate"), ("confidence_guard", "Source-fitted descriptor-gap gate")):
        p = pooled[arm]
        lines.append(f"| {title} | {p['counts']['correct']:,} / {p['counts']['moved']:,} | {p['coverage']:.1%} | {p['precision']:.1%} | {p['counts']['assigned']:,} / {p['counts']['moved']:,} |")
    lines.append("")
    for arm in ("motion_config", "confidence_guard"):
        for metric in ("coverage", "precision"):
            d = deltas[f"{arm}_minus_baseline_{metric}"]
            lines.append(f"- {arm} minus baseline, environment-macro {metric} change: {100*d['estimate']:+.1f} pp (95% CI [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}], {d['environments']} paired environments).")
    for metric in ("coverage", "precision"):
        d = deltas[f"confidence_guard_minus_motion_config_{metric}"]
        lines.append(f"- Confidence guard minus same-geometry motion configuration, environment-macro {metric}: {100*d['estimate']:+.1f} pp (95% CI [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}]).")
    lines.extend([
        f"- The descriptor gate retains {keep:.1%} of the assignments made by its same-geometry ungated configuration.",
        "",
        "## Stratified by whether the official target object appears in top-20",
        "",
        "| Candidate availability | Queries | Ungated motion assignments (correct) | Confidence-gated assignments (correct) | Removed correct / false assignments |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, item in stratified.items():
        ungated, gated = item["arms"]["motion_config"], item["arms"]["confidence_guard"]
        lines.append(f"| {label.replace('_', ' ')} | {item['queries']:,} | {ungated['assigned']:,} ({ungated['correct']:,}) | {gated['assigned']:,} ({gated['correct']:,}) | {item['removed_correct_assignments']:,} / {item['removed_false_assignments']:,} |")
    lines.extend([
        "",
        "The gate's rejection behavior is label-stratified post hoc: rejection of unavailable-target assignments can remove false positives, while rejection in the available-target stratum can also remove correct transfers. This table uses candidate availability only for analysis; the inference rule does not see it.",
        "",
        "The confidence threshold is frozen from train-only fitting, but this is a post-hoc transfer to a validation split whose aggregate matching outcomes were previously inspected. Full relation/query outputs and bootstrap summary are saved with this report.",
        "",
        "Reproduce with `scripts/evaluate_source_confidence_motion_gate_on_validation.py`.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote source-confidence transfer audit to {args.output_dir}")


if __name__ == "__main__":
    main()
