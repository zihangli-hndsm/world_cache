#!/usr/bin/env python3
"""Audit accepted/rejected ICP poses against target-present object IoU post hoc."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_rescan_icp_gate_object_transfer import transfer_iou, transform
from rerank_rescan_fpfh_by_color import read_scene_ply


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def cluster_bootstrap(values: np.ndarray, scenes: list[str], seed: int,
                      samples: int = 20_000) -> list[float]:
    names = sorted(set(scenes))
    rows_by_scene = {name: np.flatnonzero(np.asarray(scenes) == name) for name in names}
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(names), size=len(names))
        rows = np.concatenate([rows_by_scene[names[i]] for i in draw])
        boot[index] = float(np.mean(values[rows]))
    return [float(value) for value in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--refined-csv", type=Path, required=True)
    parser.add_argument("--gate-csv", type=Path, required=True)
    parser.add_argument("--eligible-transfer-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_042_026)
    args = parser.parse_args()
    if args.max_distance <= 0 or args.bootstrap_samples < 1:
        raise ValueError("max distance and bootstrap sample count must be positive")

    refined_rows = read_csv(args.refined_csv)
    gate_rows = read_csv(args.gate_csv)
    eligible_rows = read_csv(args.eligible_transfer_csv)
    key_fn = lambda row: (row["scene"], row["target_scan"], str(row["source_instance_id"]))
    refined = {key_fn(row): row for row in refined_rows}
    gate = {key_fn(row): row for row in gate_rows}
    eligible = {key_fn(row): row for row in eligible_rows}
    if not eligible or not set(eligible).issubset(refined) or not set(eligible).issubset(gate):
        raise ValueError("eligible queries do not align with refined and gate rows")

    records: list[dict[str, object]] = []
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_cache_key: tuple[str, str] | None = None
    target_points: np.ndarray | None = None
    target_ids: np.ndarray | None = None
    target_tree: cKDTree | None = None
    for key in sorted(eligible):
        scene, target_scan, raw_id = key
        instance_id = int(raw_id)
        if scene not in source_cache:
            points, _, ids, _ = read_scene_ply(
                args.data_root / scene / "gt_segmentation" / f"{scene}_0.ply"
            )
            source_cache[scene] = points, ids
        if target_cache_key != key[:2]:
            target_points, _, target_ids, _ = read_scene_ply(
                args.data_root / scene / "gt_segmentation" / f"{target_scan}.ply"
            )
            target_tree = cKDTree(target_points)
            target_cache_key = key[:2]
        source_points_all, source_ids = source_cache[scene]
        source_points = source_points_all[source_ids == instance_id]
        if len(source_points) == 0 or not np.any(target_ids == instance_id):
            raise ValueError(f"eligible query missing source or target instance: {key}")
        gate_row = gate[key]
        used = gate_row["used_icp_pose"].lower() == "true"
        identity_iou = float(eligible[key]["identity_instance_iou"])
        policy_iou = float(eligible[key]["gated_icp_instance_iou"])
        if used:
            raw_icp_iou = policy_iou
        else:
            score = transfer_iou(
                source_points, target_ids, target_tree, instance_id,
                transform(refined[key]), args.max_distance,
            )
            raw_icp_iou = score["instance_iou"]
        records.append({
            "scene": scene,
            "target_scan": target_scan,
            "source_instance_id": raw_id,
            "used_icp_pose": used,
            "icp_fitness": float(gate_row["icp_fitness"]),
            "icp_inlier_rmse_m": float(gate_row["icp_inlier_rmse_m"]),
            "identity_iou": identity_iou,
            "refined_icp_counterfactual_iou": raw_icp_iou,
            "gate_policy_iou": policy_iou,
            "counterfactual_icp_delta": raw_icp_iou - identity_iou,
            "gate_policy_delta": policy_iou - identity_iou,
            "oracle_regret": max(raw_icp_iou, identity_iou) - policy_iou,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "icp_gate_selection_utility.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def summarize(subset: list[dict[str, object]]) -> dict[str, object]:
        identity = np.asarray([float(row["identity_iou"]) for row in subset])
        raw = np.asarray([float(row["refined_icp_counterfactual_iou"]) for row in subset])
        policy = np.asarray([float(row["gate_policy_iou"]) for row in subset])
        regret = np.asarray([float(row["oracle_regret"]) for row in subset])
        return {
            "queries": len(subset),
            "identity_iou": float(identity.mean()),
            "refined_icp_iou": float(raw.mean()),
            "gate_policy_iou": float(policy.mean()),
            "refined_icp_minus_identity": float((raw - identity).mean()),
            "gate_policy_minus_identity": float((policy - identity).mean()),
            "oracle_regret_iou": float(regret.mean()),
            "icp_improves_over_identity": int(np.sum(raw > identity)),
            "icp_harms_vs_identity": int(np.sum(raw < identity)),
        }

    used_records = [row for row in records if row["used_icp_pose"]]
    rejected_records = [row for row in records if not row["used_icp_pose"]]
    accepted_deltas = np.asarray([float(row["counterfactual_icp_delta"]) for row in used_records])
    rejected_deltas = np.asarray([float(row["counterfactual_icp_delta"]) for row in rejected_records])
    all_scenes = [str(row["scene"]) for row in records]
    all_delta = np.asarray([float(row["gate_policy_delta"]) for row in records])
    raw_delta = np.asarray([float(row["counterfactual_icp_delta"]) for row in records])
    regret = np.asarray([float(row["oracle_regret"]) for row in records])
    summary = {
        "queries": len(records),
        "scenes": len(set(all_scenes)),
        "eligibility": "source instance annotation present in target; target labels used only for eligibility and post-hoc scoring",
        "gate_frozen_before_b_m": {"icp_fitness_min": 0.8, "icp_inlier_rmse_m_max": 0.05},
        "accepted": summarize(used_records),
        "rejected": summarize(rejected_records),
        "overall_gate_policy": summarize(records),
        "accepted_harm_fraction": float(np.mean(accepted_deltas < 0)) if len(accepted_deltas) else None,
        "rejected_benefit_fraction": float(np.mean(rejected_deltas > 0)) if len(rejected_deltas) else None,
        "gate_policy_delta_scene_cluster_bootstrap_95pct": cluster_bootstrap(
            all_delta, all_scenes, args.seed, args.bootstrap_samples
        ),
        "counterfactual_raw_icp_delta_scene_cluster_bootstrap_95pct": cluster_bootstrap(
            raw_delta, all_scenes, args.seed + 1, args.bootstrap_samples
        ),
        "oracle_regret_scene_cluster_bootstrap_95pct": cluster_bootstrap(
            regret, all_scenes, args.seed + 2, args.bootstrap_samples
        ),
        "bootstrap_unit": "scene, query-weighted within each sampled scene set",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    (args.output_dir / "icp_gate_selection_utility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Post-hoc utility of the frozen ICP quality gate", "",
        f"The scene_a-frozen gate accepts {len(used_records)}/{len(records)} target-present queries. Target labels are used only for query eligibility and post-hoc instance IoU scoring; the gate itself uses ICP fitness and RMSE.",
        "", "| Gate subset | Queries | Identity IoU | Refined ICP IoU | Policy IoU | ICP better than identity | ICP worse than identity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, subset in (("Accepted", used_records), ("Rejected", rejected_records), ("All", records)):
        stat = summarize(subset)
        lines.append(
            f"| {name} | {stat['queries']} | {stat['identity_iou']:.3f} | {stat['refined_icp_iou']:.3f} | {stat['gate_policy_iou']:.3f} | {stat['icp_improves_over_identity']} | {stat['icp_harms_vs_identity']} |"
        )
    gate_ci = summary["gate_policy_delta_scene_cluster_bootstrap_95pct"]
    raw_ci = summary["counterfactual_raw_icp_delta_scene_cluster_bootstrap_95pct"]
    regret_ci = summary["oracle_regret_scene_cluster_bootstrap_95pct"]
    lines.extend([
        "", f"Accepted-pose harm fraction: {summary['accepted_harm_fraction']:.3f}. Rejected-pose benefit fraction (counterfactual): {summary['rejected_benefit_fraction']:.3f}.",
        f"Policy-minus-identity delta: {summary['overall_gate_policy']['gate_policy_minus_identity']:+.3f} (scene-cluster 95% CI [{gate_ci[0]:+.3f},{gate_ci[1]:+.3f}]).",
        f"Applying raw ICP to every eligible query would change IoU by {summary['overall_gate_policy']['refined_icp_minus_identity']:+.3f} (CI [{raw_ci[0]:+.3f},{raw_ci[1]:+.3f}]). The per-query oracle gate has mean regret {summary['overall_gate_policy']['oracle_regret_iou']:.3f} IoU (CI [{regret_ci[0]:.3f},{regret_ci[1]:.3f}]).",
        "", "All results are post-hoc mechanism diagnostics on held-out Rescan B–M scans, not thresholds to tune on these scenes.",
    ])
    (args.output_dir / "icp_gate_selection_utility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
