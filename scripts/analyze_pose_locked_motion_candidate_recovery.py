#!/usr/bin/env python3
"""Decompose frozen motion-branch transfer by official top-20 object availability.

Candidate availability is reconstructed from each query's official rigid
reference objectId and the cached descriptor-ranked top-20 reference points.
This avoids relying on the separate cross-scan link table, which is missing for
some rigid queries. No matching, threshold, or hypothesis is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_3rscan_instance_correspondence import read_annotated_mesh


DEFAULT_INPUT = Path("outputs/reports/20261006_vitg_pose_locked_motion_validation")
DEFAULT_OUTPUT = Path("outputs/reports/20261004_pose_locked_candidate_recovery")
DEFAULT_ANNOTATIONS = (Path.home() / 'data/worldcache_datasets/3rscan')
DEFAULT_MANIFEST = Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)


def bootstrap_category(env: pd.DataFrame, category: bool, samples: int, seed: int) -> dict[str, object]:
    # env is already pooled by environment; resample the same 47 clusters.
    part = env[env.candidate_hit == category].set_index("environment_id")
    envs = sorted(env.environment_id.unique())
    full = pd.DataFrame(0, index=envs, columns=[
        "moved_queries", "baseline_correct", "method_correct",
        "baseline_assigned", "method_assigned", "new_assigned", "new_correct",
    ], dtype=np.int64)
    full.loc[part.index, full.columns] = part[full.columns].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(envs), size=(samples, len(envs)))
    matrix = full.to_numpy(dtype=float)
    # columns: N, base-correct, method-correct, base-assigned,
    # method-assigned, newly-assigned, newly-correct.
    per_env_base = ratio(matrix[:, 1], matrix[:, 0])
    per_env_method = ratio(matrix[:, 2], matrix[:, 0])
    per_env_new_precision = ratio(matrix[:, 6], matrix[:, 5])
    base_macro = float(np.nanmean(per_env_base))
    method_macro = float(np.nanmean(per_env_method))
    delta = per_env_method - per_env_base
    new_macro = float(np.nanmean(per_env_new_precision))
    base_draw = np.nanmean(per_env_base[draws], axis=1)
    method_draw = np.nanmean(per_env_method[draws], axis=1)
    delta_draw = np.nanmean(delta[draws], axis=1)
    new_draw = np.nanmean(per_env_new_precision[draws], axis=1)
    return {
        "environment_macro_baseline_coverage": base_macro,
        "environment_macro_method_coverage": method_macro,
        "environment_macro_coverage_delta": float(np.nanmean(delta)),
        "environment_macro_coverage_delta_ci95": [float(x) for x in np.nanquantile(delta_draw, [0.025, 0.975])],
        "environment_macro_new_assignment_precision": new_macro,
        "environment_macro_new_assignment_precision_ci95": [float(x) for x in np.nanquantile(new_draw, [0.025, 0.975])],
        "environment_macro_baseline_coverage_ci95": [float(x) for x in np.nanquantile(base_draw, [0.025, 0.975])],
        "environment_macro_method_coverage_ci95": [float(x) for x in np.nanquantile(method_draw, [0.025, 0.975])],
        "environments_with_moved_queries": int(np.isfinite(per_env_base).sum()),
        "environments_with_new_assignments": int(np.isfinite(per_env_new_precision).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--pair-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=10_042_027)
    args = parser.parse_args()
    pairs = csv_rows(args.root / args.pair_manifest)
    if len(pairs) != 110 or len({p["environment_id"] for p in pairs}) != 47:
        raise ValueError("expected the full 110-relation / 47-environment validation manifest")

    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        cache_path = args.root / args.input_dir / "descriptor_candidates" / f"{run_id}.npz"
        with np.load(cache_path) as cache:
            if str(cache["pair"].item()) != run_id:
                raise ValueError(f"candidate cache identity mismatch: {run_id}")
            ref_points = cache["reference_points"].astype(np.float32)
            candidates = cache["candidate_indices"][:, :20].astype(np.int64)
            query_frame_ids = cache["query_frame_ids"].astype(np.int64)
        relation_dir = args.root / args.input_dir / "relations" / run_id
        diagnostic = pd.read_csv(relation_dir / "query_diagnostics.csv")
        base = diagnostic[diagnostic.arm == "global_ransac"].sort_values("query_index").reset_index(drop=True)
        method = diagnostic[diagnostic.arm == "combined"].sort_values("query_index").reset_index(drop=True)
        if not np.array_equal(base.query_index.to_numpy(), method.query_index.to_numpy()):
            raise ValueError(f"motion-arm query mismatch: {run_id}")
        if len(base) != len(candidates) or not np.array_equal(base.frame_id.to_numpy(dtype=np.int64), query_frame_ids):
            raise ValueError(f"query/cache frame identity mismatch: {run_id}")

        tree, reference_object_ids, _, _, _ = read_annotated_mesh(args.annotation_root, pair["reference_scan"])
        _, nearest = tree.query(ref_points, k=1)
        patch_object_ids = reference_object_ids[nearest]
        expected_ids = base.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
        moved = base.moved_rigid.astype(bool).to_numpy() & (expected_ids > 0)
        candidate_hit = np.zeros(len(base), dtype=bool)
        for query_index in np.flatnonzero(moved):
            candidate_hit[query_index] = np.any(patch_object_ids[candidates[query_index]] == expected_ids[query_index])
        base_assigned = base.consensus_inlier.astype(bool).to_numpy()
        method_assigned = method.consensus_inlier.astype(bool).to_numpy()
        base_correct = base.baseline_correct.astype(bool).to_numpy()
        method_correct = method.method_correct.astype(bool).to_numpy()
        added_assigned = moved & method_assigned & ~base_assigned
        added_correct = moved & method_correct & ~base_correct
        if np.any((base_correct | method_correct) & moved & ~candidate_hit):
            raise ValueError(f"a physically correct transfer lacks its official top-20 candidate: {run_id}")
        if np.any(added_correct & ~added_assigned):
            raise ValueError(f"correct motion transfer is not a newly assigned baseline outlier: {run_id}")

        for availability in (True, False):
            mask = moved & (candidate_hit == availability)
            rows.append({
                "run_id": run_id,
                "environment_id": pair["environment_id"],
                "candidate_hit": availability,
                "moved_queries": int(mask.sum()),
                "baseline_correct": int((mask & base_correct).sum()),
                "method_correct": int((mask & method_correct).sum()),
                "baseline_assigned": int((mask & base_assigned).sum()),
                "method_assigned": int((mask & method_assigned).sum()),
                "new_assigned": int((mask & added_assigned).sum()),
                "new_correct": int((mask & added_correct).sum()),
            })
        if (index + 1) % 20 == 0:
            print(f"audited {index + 1}/110 relations", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_relation = pd.DataFrame(rows)
    per_relation.to_csv(args.output_dir / "per_relation_candidate_availability.csv", index=False)
    per_environment = per_relation.groupby(["environment_id", "candidate_hit"], as_index=False).sum(numeric_only=True)
    per_environment.to_csv(args.output_dir / "per_environment_candidate_availability.csv", index=False)
    summary: dict[str, object] = {
        "status": "complete",
        "relations": 110,
        "environments": 47,
        "moved_rigid_queries": int(per_relation.moved_queries.sum()),
        "candidate_hit_queries": int(per_relation.loc[per_relation.candidate_hit, "moved_queries"].sum()),
        "candidate_availability_pooled": float(per_relation.loc[per_relation.candidate_hit, "moved_queries"].sum() / per_relation.moved_queries.sum()),
        "scoring_target": "official rigid reference objectId, reconstructed from each relation's annotation mesh and saved descriptor-ranked top-20 point candidates",
        "query_alignment": "all 110 relation cache/frame/query identities checked against seed-17 motion diagnostics",
        "correct_transfer_without_candidate": 0,
        "bootstrap": {"unit": "environment cluster; pool relations per environment before rates", "samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "by_candidate_availability": {},
    }
    for avail, label in ((True, "candidate_present"), (False, "candidate_absent")):
        sub = per_relation[per_relation.candidate_hit == avail]
        counts = {k: int(sub[k].sum()) for k in ("moved_queries", "baseline_correct", "method_correct", "baseline_assigned", "method_assigned", "new_assigned", "new_correct")}
        pooled = {
            "baseline_coverage": counts["baseline_correct"] / counts["moved_queries"] if counts["moved_queries"] else None,
            "method_coverage": counts["method_correct"] / counts["moved_queries"] if counts["moved_queries"] else None,
            "new_assignment_precision": counts["new_correct"] / counts["new_assigned"] if counts["new_assigned"] else None,
            "candidate_share_of_all_moved_queries": counts["moved_queries"] / summary["moved_rigid_queries"],
        }
        env = per_environment[per_environment.candidate_hit == avail]
        uncertainty = bootstrap_category(env, avail, args.bootstrap_samples, args.bootstrap_seed + int(avail))
        summary["by_candidate_availability"][label] = {"counts": counts, "pooled": pooled, **uncertainty}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    present = summary["by_candidate_availability"]["candidate_present"]
    absent = summary["by_candidate_availability"]["candidate_absent"]
    lines = [
        "# Pose-locked motion: candidate availability versus consensus recovery",
        "",
        "For each moved-rigid query, the reference objectId comes from the official rigid relation mapping. Candidate availability is reconstructed by mapping each saved top-20 reference 3-D point to the annotated reference mesh and checking whether the expected object appears. This avoids dependence on the separate object-link field, which is absent for some official rigid queries. Candidate annotations are used only for this post-hoc decomposition; matching and mode selection remain label-free.",
        "",
        f"Across {summary['moved_rigid_queries']:,} moved-rigid queries, the correct physical instance appears in the top-20 candidate set for {summary['candidate_hit_queries']:,} ({summary['candidate_availability_pooled']:.1%}). All 110 query/cache identity and frame-order checks pass. No correct baseline or motion transfer occurs when the official target object is absent from top-20.",
        "",
        "| Top-20 contains official target object | Queries | Baseline correct coverage | Motion correct coverage | New assignments | New correct / assignments | Added-assignment precision |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("Yes", present), ("No", absent)):
        c, p = item["counts"], item["pooled"]
        precision = "—" if p["new_assignment_precision"] is None else f"{p['new_assignment_precision']:.1%}"
        lines.append(f"| {label} | {c['moved_queries']:,} | {p['baseline_coverage']:.1%} | {p['method_coverage']:.1%} | {c['new_assigned']:,} | {c['new_correct']:,} / {c['new_assigned']:,} | {precision} |")
    lines.extend([
        "",
        f"When the target object is available in top-20, environment-macro correct-transfer coverage changes by {100*present['environment_macro_coverage_delta']:+.1f} pp (95% environment-bootstrap CI [{100*present['environment_macro_coverage_delta_ci95'][0]:+.1f}, {100*present['environment_macro_coverage_delta_ci95'][1]:+.1f}]); when unavailable, the change is {100*absent['environment_macro_coverage_delta']:+.1f} pp (CI [{100*absent['environment_macro_coverage_delta_ci95'][0]:+.1f}, {100*absent['environment_macro_coverage_delta_ci95'][1]:+.1f}]). The absent-candidate stratum is a hard ceiling for all top-20 selection rules. The present-candidate stratum measures the remaining consensus/selection problem, but its selected assignments remain label-free and the result is post-hoc on the already-inspected validation split.",
        "",
        "Reproduce with `scripts/analyze_pose_locked_motion_candidate_recovery.py`. Machine-readable relation and environment tables are alongside this report.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote candidate-recovery decomposition to {args.output_dir}")


if __name__ == "__main__":
    main()
