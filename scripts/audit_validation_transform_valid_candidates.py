#!/usr/bin/env python3
"""Measure physical and transform-valid top-K candidates on official validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from audit_transform_valid_candidate_availability import (
    apply,
    object_matrix,
    object_labels,
    read_ply,
    scene_matrix,
)


BOOTSTRAP_N = 20_000
SEED = 20261016
K_VALUES = (1, 5, 10, 20, 50)


def transform_points(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", transforms[:, :3, :3], points) + transforms[:, :3, 3]


def environment_bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument("--pose-output", type=Path, default=Path("outputs/reports/20261006_vitg_pose_locked_motion_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261008_validation_transform_valid_candidate_availability"))
    parser.add_argument("--distance-threshold", type=float, default=0.10)
    args = parser.parse_args()
    root = Path(".").resolve()
    annotation_root = args.annotation_root if args.annotation_root.is_absolute() else root / args.annotation_root
    pose_root = args.pose_output if args.pose_output.is_absolute() else root / args.pose_output
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((annotation_root / "3RScan.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(pose_root / "run_manifest.csv")
    if len(manifest) != 110 or manifest.status.ne("complete").any():
        raise ValueError("expected all 110 official validation relations to be complete")
    if manifest.environment_id.nunique() != 47:
        raise ValueError("expected 47 official validation environments")

    ref_mesh_cache: dict[str, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    all_queries: list[pd.DataFrame] = []
    relation_rows: list[dict[str, object]] = []
    total_candidate_bytes = 0

    for rel in manifest.itertuples(index=False):
        cache_path = pose_root / "descriptor_candidates" / f"{rel.run_id}.npz"
        diag_path = Path(rel.output_dir)
        if not diag_path.is_absolute():
            diag_path = root / diag_path
        diag_path = diag_path / "query_diagnostics.csv"
        if not cache_path.is_file() or not diag_path.is_file():
            raise FileNotFoundError(f"missing candidate or diagnostic cache for {rel.run_id}")
        total_candidate_bytes += cache_path.stat().st_size
        with np.load(cache_path) as z:
            reference_id = str(z["reference_id"].item())
            rescan_id = str(z["rescan_id"].item())
            ref_points = z["reference_points"].astype(np.float32)
            query_points = z["query_points"].astype(np.float64)
            frame_ids = z["query_frame_ids"].astype(np.int64)
            candidate_indices = z["candidate_indices"].astype(np.int64)
        if candidate_indices.shape != (len(query_points), 50):
            raise ValueError(f"{rel.run_id}: expected exact top-50 candidate cache")

        diag_all = pd.read_csv(diag_path)
        diag = diag_all[diag_all.arm.eq("global_ransac")].sort_values("query_index").reset_index(drop=True)
        motion_diag = diag_all[diag_all.arm.eq("combined")].sort_values("query_index").reset_index(drop=True)
        if len(diag) != len(query_points) or diag.query_index.duplicated().any():
            raise ValueError(f"{rel.run_id}: query diagnostics do not match candidate cache")
        if len(motion_diag) != len(diag) or not diag[["query_index", "frame_id"]].equals(
            motion_diag[["query_index", "frame_id"]].reset_index(drop=True)
        ):
            raise ValueError(f"{rel.run_id}: baseline/motion query rows are not aligned")
        if not np.array_equal(diag.query_index.to_numpy(dtype=np.int64), np.arange(len(query_points))):
            raise ValueError(f"{rel.run_id}: query indices are not dense cache indices")
        if not np.array_equal(frame_ids, diag.frame_id.to_numpy(dtype=np.int64)):
            raise ValueError(f"{rel.run_id}: frame IDs differ from saved diagnostics")

        rel_meta = next((r for r in metadata if r.get("reference") == reference_id), None)
        if rel_meta is None:
            raise ValueError(f"{rel.run_id}: reference missing from 3RScan metadata")
        scan = next((s for s in rel_meta.get("scans", []) if s.get("reference") == rescan_id), None)
        if scan is None:
            raise ValueError(f"{rel.run_id}: rescan relation missing from 3RScan metadata")
        scene = scene_matrix(scan["transform"])
        rigid_map: dict[int, tuple[int, np.ndarray]] = {}
        for item in scan.get("rigid", []):
            qid = int(item.get("instance_rescan", -1))
            rid = int(item.get("instance_reference", -1))
            if qid > 0 and rid > 0:
                rigid_map[qid] = (rid, object_matrix(item["transform"]))

        moved = diag.moved_rigid.astype(bool).to_numpy()
        qrows = diag[moved].copy().reset_index(drop=True)
        motion_qrows = motion_diag[moved].copy().reset_index(drop=True)
        qidx = qrows.query_index.to_numpy(dtype=np.int64)
        if len(qrows) == 0:
            continue
        qoid = qrows.query_instance_id.to_numpy(dtype=np.int64)
        mapped_oid = qrows.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
        if np.any(mapped_oid <= 0) or np.any(qoid <= 0):
            raise ValueError(f"{rel.run_id}: moved-rigid query lacks linked instance IDs")
        target_xyz, target_oid_mesh, _ = read_ply(
            annotation_root / rescan_id / "labels.instances.annotated.v2.ply"
        )
        target_oid_at_query = object_labels(
            query_points[qidx], cKDTree(target_xyz), target_oid_mesh
        )
        if not np.array_equal(target_oid_at_query, qoid):
            raise ValueError(f"{rel.run_id}: query ObjectIds do not reproduce target annotation labels")
        transforms = []
        moved_differs = np.zeros(len(qrows), dtype=bool)
        for i, (source_oid, target_oid) in enumerate(zip(qoid, mapped_oid)):
            if int(source_oid) not in rigid_map or rigid_map[int(source_oid)][0] != int(target_oid):
                raise ValueError(f"{rel.run_id}: official rigid map disagrees for object {source_oid}")
            ref_to_rescan = rigid_map[int(source_oid)][1]
            rescan_to_ref = np.linalg.inv(ref_to_rescan)
            transforms.append(rescan_to_ref)
            relative = rescan_to_ref @ np.linalg.inv(scene)
            angle = np.arccos(np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
            moved_differs[i] = np.linalg.norm(relative[:3, 3]) > 0.05 or angle > np.deg2rad(5.0)
        transforms = np.stack(transforms)

        if reference_id not in ref_mesh_cache:
            mesh_path = annotation_root / reference_id / "labels.instances.annotated.v2.ply"
            xyz, object_ids, _ = read_ply(mesh_path)
            ref_mesh_cache[reference_id] = (xyz.astype(np.float32), object_ids, cKDTree(xyz))
        mesh_xyz, mesh_oid, mesh_tree = ref_mesh_cache[reference_id]
        candidate_object_ids = mesh_oid[mesh_tree.query(ref_points, k=1)[1]][candidate_indices[qidx]]
        candidate_xyz = ref_points[candidate_indices[qidx]]
        candidate_same_object = candidate_object_ids == mapped_oid[:, None]

        query_xyz = query_points[qidx]
        scene_query = apply(query_xyz, scene)
        object_query = transform_points(query_xyz, transforms)
        scene_dist = np.linalg.norm(candidate_xyz - scene_query[:, None, :], axis=2)
        object_dist = np.linalg.norm(candidate_xyz - object_query[:, None, :], axis=2)
        scene_valid = candidate_same_object & (scene_dist <= args.distance_threshold)
        object_valid = candidate_same_object & (object_dist <= args.distance_threshold)

        object_trees = {
            int(oid): cKDTree(mesh_xyz[mesh_oid == oid])
            for oid in np.unique(mapped_oid)
            if np.any(mesh_oid == oid)
        }
        if any(int(oid) not in object_trees for oid in np.unique(mapped_oid)):
            raise ValueError(f"{rel.run_id}: linked reference object has no mesh support")
        scene_surface = np.fromiter(
            (object_trees[int(oid)].query(scene_query[i], k=1)[0] <= args.distance_threshold
             for i, oid in enumerate(mapped_oid)),
            dtype=bool,
            count=len(mapped_oid),
        )
        object_surface = np.fromiter(
            (object_trees[int(oid)].query(object_query[i], k=1)[0] <= args.distance_threshold
             for i, oid in enumerate(mapped_oid)),
            dtype=bool,
            count=len(mapped_oid),
        )

        q = pd.DataFrame(
            {
                "run_id": rel.run_id,
                "environment_id": rel.environment_id,
                "query_index": qidx,
                "frame_id": qrows.frame_id.to_numpy(dtype=np.int64),
                "query_object_id": qoid,
                "reference_object_id": mapped_oid,
                "object_transform_differs_from_scene": moved_differs,
                "scene_surface_support": scene_surface,
                "object_surface_support": object_surface,
                "physical_candidate_hit_k50": candidate_same_object.any(axis=1),
                "global_correct": qrows.baseline_correct.astype(bool).to_numpy(),
                "motion_correct": motion_qrows.method_correct.astype(bool).to_numpy(),
            }
        )
        for k in K_VALUES:
            q[f"physical_hit_k{k}"] = candidate_same_object[:, :k].any(axis=1)
            q[f"scene_valid_k{k}"] = scene_valid[:, :k].any(axis=1)
            q[f"object_valid_k{k}"] = object_valid[:, :k].any(axis=1)
        all_queries.append(q)
        row: dict[str, object] = {
            "run_id": rel.run_id,
            "environment_id": rel.environment_id,
            "queries": len(q),
            "object_transform_differs_queries": int(moved_differs.sum()),
        }
        for metric in ("physical_hit", "scene_valid", "object_valid"):
            for k in K_VALUES:
                row[f"{metric}_k{k}"] = float(q[f"{metric}_k{k}"].mean())
        row["scene_surface_support"] = float(scene_surface.mean())
        row["object_surface_support"] = float(object_surface.mean())
        row["global_correct_given_object_valid_k20"] = (
            float(q.loc[q.object_valid_k20, "global_correct"].mean()) if q.object_valid_k20.any() else np.nan
        )
        row["motion_correct_given_object_valid_k20"] = (
            float(q.loc[q.object_valid_k20, "motion_correct"].mean()) if q.object_valid_k20.any() else np.nan
        )
        relation_rows.append(row)

    queries = pd.concat(all_queries, ignore_index=True)
    relations = pd.DataFrame(relation_rows)
    if len(queries) != 14_088 or queries.run_id.nunique() != len(relation_rows) or queries.environment_id.nunique() != 42:
        raise ValueError(
            f"unexpected moved-rigid scope: {len(queries)} queries, {queries.run_id.nunique()} relations, {queries.environment_id.nunique()} environments"
        )
    if int(queries.physical_hit_k20.sum()) != 7_626:
        raise ValueError("physical-instance top-20 availability fails the saved 7,626-query cross-check")
    if int(queries.global_correct.sum()) != 2_876 or int(queries.motion_correct.sum()) != 3_808:
        raise ValueError("baseline or motion correct-transfer count fails the saved validation cross-check")
    if (queries.global_correct & ~queries.physical_hit_k20).any() or (
        queries.motion_correct & ~queries.physical_hit_k20
    ).any():
        raise ValueError("a correct physical transfer has no linked-object candidate in top-20")
    queries.to_csv(output / "query_diagnostics.csv.gz", index=False, compression="gzip")
    relations.to_csv(output / "per_relation.csv", index=False)

    env = queries.groupby("environment_id", sort=True).agg(
        queries=("query_index", "size"),
        physical_hit_k20=("physical_hit_k20", "mean"),
        scene_valid_k20=("scene_valid_k20", "mean"),
        object_valid_k20=("object_valid_k20", "mean"),
        physical_hit_k50=("physical_hit_k50", "mean"),
        scene_valid_k50=("scene_valid_k50", "mean"),
        object_valid_k50=("object_valid_k50", "mean"),
        scene_surface_support=("scene_surface_support", "mean"),
        object_surface_support=("object_surface_support", "mean"),
        global_correct=("global_correct", "mean"),
        motion_correct=("motion_correct", "mean"),
    ).reset_index()
    env.to_csv(output / "per_environment.csv", index=False)
    rng = np.random.default_rng(SEED)
    macro_rows = []
    for metric in ("physical_hit", "scene_valid", "object_valid"):
        for k in (20, 50):
            name = f"{metric}_k{k}"
            estimate, low, high = environment_bootstrap(env[name].to_numpy(), rng)
            macro_rows.append({"metric": name, "macro": estimate, "ci_low": low, "ci_high": high})
    delta_metrics = {}
    for k in (20, 50):
        delta = env[f"object_valid_k{k}"].to_numpy() - env[f"scene_valid_k{k}"].to_numpy()
        estimate, low, high = environment_bootstrap(delta, rng)
        delta_metrics[f"object_minus_scene_valid_k{k}"] = {
            "estimate": estimate,
            "ci95": [low, high],
            "environments": len(delta),
        }
    conditional_rows = []
    for environment_id, group in queries.groupby("environment_id", sort=True):
        valid = group.object_valid_k20.astype(bool)
        if not valid.any():
            continue
        conditional_rows.append(
            {
                "environment_id": environment_id,
                "queries_with_object_valid_k20": int(valid.sum()),
                "global_correct_rate": float(group.loc[valid, "global_correct"].mean()),
                "motion_correct_rate": float(group.loc[valid, "motion_correct"].mean()),
            }
        )
    conditional_env = pd.DataFrame(conditional_rows)
    conditional_env.to_csv(output / "conditional_recovery_by_environment.csv", index=False)
    conditional_delta = conditional_env.motion_correct_rate.to_numpy() - conditional_env.global_correct_rate.to_numpy()
    macro_global, global_lo, global_hi = environment_bootstrap(
        conditional_env.global_correct_rate.to_numpy(), rng
    )
    macro_motion, motion_lo, motion_hi = environment_bootstrap(
        conditional_env.motion_correct_rate.to_numpy(), rng
    )
    macro_delta, delta_lo, delta_hi = environment_bootstrap(conditional_delta, rng)
    conditional_recovery = {
        "object_valid_k20_queries": int(queries.object_valid_k20.sum()),
        "environment_count_with_object_valid_k20": len(conditional_env),
        "global_correct_count": int(queries.loc[queries.object_valid_k20, "global_correct"].sum()),
        "motion_correct_count": int(queries.loc[queries.object_valid_k20, "motion_correct"].sum()),
        "pooled_global_correct_rate": float(queries.loc[queries.object_valid_k20, "global_correct"].mean()),
        "pooled_motion_correct_rate": float(queries.loc[queries.object_valid_k20, "motion_correct"].mean()),
        "environment_macro_global_correct_rate": macro_global,
        "environment_macro_global_correct_rate_ci95": [global_lo, global_hi],
        "environment_macro_motion_correct_rate": macro_motion,
        "environment_macro_motion_correct_rate_ci95": [motion_lo, motion_hi],
        "environment_macro_motion_minus_global_delta": macro_delta,
        "environment_macro_motion_minus_global_delta_ci95": [delta_lo, delta_hi],
    }
    summary = {
        "status": "complete",
        "relations": int(queries.run_id.nunique()),
        "environments_with_moved_rigid_queries": int(queries.environment_id.nunique()),
        "moved_rigid_queries": int(len(queries)),
        "threshold_m": args.distance_threshold,
        "candidate_archives": int(manifest.run_id.nunique()),
        "candidate_archive_bytes": int(total_candidate_bytes),
        "query_frame_id_join": "exact for all 110 validation relations",
        "object_transform_semantics": "official reference-to-rescan rigid-object transforms inverted to map rescan queries into reference coordinates",
        "pooled_counts": {
            column: int(queries[column].sum())
            for column in queries.columns
            if column.endswith(("_k1", "_k5", "_k10", "_k20", "_k50"))
        },
        "pooled_rates": {
            column: float(queries[column].mean())
            for column in queries.columns
            if column.endswith(("_k1", "_k5", "_k10", "_k20", "_k50"))
        },
        "environment_macro": macro_rows,
        "environment_bootstrap_object_vs_scene": delta_metrics,
        "object_transform_differs_from_scene_queries": int(queries.object_transform_differs_from_scene.sum()),
        "surface_support": {
            "scene_pose": int(queries.scene_surface_support.sum()),
            "object_pose": int(queries.object_surface_support.sum()),
        },
        "physical_transfer": {
            "global_correct": int(queries.global_correct.sum()),
            "motion_correct": int(queries.motion_correct.sum()),
            "global_correct_given_object_valid_k20": float(queries.loc[queries.object_valid_k20, "global_correct"].mean()),
            "motion_correct_given_object_valid_k20": float(queries.loc[queries.object_valid_k20, "motion_correct"].mean()),
        },
        "conditional_recovery_given_object_valid_k20": conditional_recovery,
        "evaluation_only": "official object IDs and transforms define the oracle; they do not affect feature ranking, candidate selection, or RANSAC",
        "caveat": "post-hoc fixed-split diagnostic; environment bootstrap does not establish external generalization",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    macro = {row["metric"]: row for row in macro_rows}
    d20 = delta_metrics["object_minus_scene_valid_k20"]
    d50 = delta_metrics["object_minus_scene_valid_k50"]
    report = f"""# Validation transform-valid candidate availability

Measured top-50 candidate availability for all {len(queries):,} official moved-rigid query patches across {queries.run_id.nunique()} validation relations and {queries.environment_id.nunique()} environments. Candidate caches and frame IDs match all 110 saved relation diagnostics. For each query, candidate validity requires both the linked reference `objectId` and distance within {args.distance_threshold:.2f} m after the relevant transform. The global scene transform and the official object transform are both evaluation-only oracles; neither changes the cached DINO ranking. The object-specific transform differs from the scene transform for {int(queries.object_transform_differs_from_scene.sum()):,}/{len(queries):,} queries.

| K | Physical-instance hit | Scene-pose valid | Object-pose valid |
|---:|---:|---:|---:|
""" + "\n".join(
        f"| {k} | {int(queries[f'physical_hit_k{k}'].sum()):,} ({queries[f'physical_hit_k{k}'].mean():.1%}) | {int(queries[f'scene_valid_k{k}'].sum()):,} ({queries[f'scene_valid_k{k}'].mean():.1%}) | {int(queries[f'object_valid_k{k}'].sum()):,} ({queries[f'object_valid_k{k}'].mean():.1%}) |"
        for k in K_VALUES
    ) + f"""

At K=20, environment-macro candidate availability is {macro['physical_hit_k20']['macro']:.1%} for the linked physical instance, {macro['scene_valid_k20']['macro']:.1%} for a scene-pose-valid candidate, and {macro['object_valid_k20']['macro']:.1%} for an object-pose-valid candidate. Replacing the scene transform with the object oracle changes valid-candidate recall by {d20['estimate']*100:+.1f} pp (environment-bootstrap 95% CI [{d20['ci95'][0]*100:+.1f},{d20['ci95'][1]*100:+.1f}]). At K=50 the paired difference is {d50['estimate']*100:+.1f} pp [{d50['ci95'][0]*100:+.1f},{d50['ci95'][1]*100:+.1f}].

Among the {int(queries.object_valid_k20.sum()):,} queries with an object-pose-valid top-20 candidate, scene RANSAC correctly transfers {int(queries.loc[queries.object_valid_k20, 'global_correct'].sum()):,} ({queries.loc[queries.object_valid_k20, 'global_correct'].mean():.1%}), while the pose-locked branch transfers {int(queries.loc[queries.object_valid_k20, 'motion_correct'].sum()):,} ({queries.loc[queries.object_valid_k20, 'motion_correct'].mean():.1%}). Across {len(conditional_env)} environments containing such queries, environment-macro conditional correctness is {macro_global:.1%} for scene RANSAC and {macro_motion:.1%} for pose-locked motion (paired delta {macro_delta*100:+.1f} pp, environment-bootstrap 95% CI [{delta_lo*100:+.1f},{delta_hi*100:+.1f}]). This conditional diagnostic scores physical-instance identity, not point-level geometric accuracy or estimated object-pose accuracy. Geometric candidate availability is not a ceiling on identity correctness: a selected point can belong to the linked object yet lie outside the distance tolerance. The pose-locked branch makes its predictions independently of this annotation-defined subset.

Full reference-mesh support within {args.distance_threshold:.2f} m is {int(queries.scene_surface_support.sum()):,}/{len(queries):,} under the scene pose and {int(queries.object_surface_support.sum()):,}/{len(queries):,} under the object pose. These query-level upper bounds are post-hoc on the explored validation split; the environment bootstrap quantifies only this split and is not independent-site generalization.

Artifacts: `query_diagnostics.csv.gz`, `per_relation.csv`, `per_environment.csv`, `conditional_recovery_by_environment.csv`, and `summary.json` in this directory. Reproducer: `scripts/audit_validation_transform_valid_candidates.py`.
"""
    (output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
