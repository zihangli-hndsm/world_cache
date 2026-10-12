#!/usr/bin/env python3
"""Audit whether top-K contains geometrically valid tracked-object candidates.

All labels, relation links, and transforms are post-hoc scoring inputs. Candidate
selection remains the cached descriptor ranking. For moved rigid objects this
compares scene-pose and object-pose oracle candidate availability.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from compare_vitl_vitg_candidate_audit import PAIR_TO_G_AUDIT, as_bool


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii")
    count_match = re.search(r"element vertex (\d+)", header)
    if count_match is None:
        raise ValueError(f"missing vertex count in {path}")
    count = int(count_match.group(1))
    values = np.fromstring(data[end:].decode("ascii"), sep=" ", dtype=np.float64)
    if values.size < count * 11:
        raise ValueError(f"truncated vertex data in {path}")
    v = values[: count * 11].reshape(count, 11)
    return v[:, :3].astype(np.float32), v[:, 6].astype(np.int64), v[:, 7].astype(np.int64)


def scene_matrix(flat: list[float]) -> np.ndarray:
    m = np.asarray(flat, dtype=np.float64).reshape(4, 4).T
    return m


def object_matrix(flat: list[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(4, 4).T


def apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def object_labels(points: np.ndarray, tree: cKDTree, ids: np.ndarray) -> np.ndarray:
    # Match the float32 precision used by the source object-motion audit when
    # resolving exact annotated-vertex ties.
    return ids[tree.query(points.astype(np.float32), k=1)[1]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--audit-root", type=Path, required=True, help="ViT-G K-sweep relation folders")
    p.add_argument("--candidate-root", type=Path, required=True, help="saved descriptor_candidates/*.npz")
    p.add_argument("--runs-root", type=Path, required=True, help="native-frame pair cache root")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--distance-threshold", type=float, default=0.10)
    p.add_argument("--bootstrap-samples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261002)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.annotation_root / "3RScan.json").read_text(encoding="utf-8"))
    pair_names = [x for x in PAIR_TO_G_AUDIT if "target" not in x]
    all_queries: list[pd.DataFrame] = []
    relation_rows: list[dict[str, object]] = []

    for pair_path in pair_names:
        pair = PAIR_TO_G_AUDIT[pair_path]
        cache_path = args.candidate_root / f"{pair}.npz"
        diag_path = args.audit_root / pair_path / "query_diagnostics.csv"
        if not (cache_path.is_file() and diag_path.is_file()):
            raise FileNotFoundError(f"missing cache or diagnostics for {pair}")
        with np.load(cache_path) as z:
            ref_id, rescan_id = str(z["reference_id"].item()), str(z["rescan_id"].item())
            ref_points = z["reference_points"].astype(np.float64)
            query_points = z["query_points"].astype(np.float64)
            frame_ids = z["query_frame_ids"].astype(np.int64)
            candidate_indices = z["candidate_indices"].astype(np.int64)
        diag = pd.read_csv(diag_path)
        d20 = diag[diag.top_k == 20].sort_values(["query_index", "frame_id"]).reset_index(drop=True)
        d50 = diag[diag.top_k == 50].sort_values(["query_index", "frame_id"]).reset_index(drop=True)
        if len(query_points) != len(d20) or candidate_indices.shape != (len(d20), 50):
            raise ValueError(f"{pair}: query/candidate cache dimension mismatch")
        if not np.array_equal(frame_ids, d20.frame_id.to_numpy(dtype=np.int64)):
            raise ValueError(f"{pair}: frame IDs differ from saved K-sweep diagnostics")
        if not d20[["query_index", "frame_id"]].equals(d50[["query_index", "frame_id"]]):
            raise ValueError(f"{pair}: K20/K50 query rows differ")

        rel = next((r for r in metadata if r.get("reference") == ref_id), None)
        if rel is None:
            raise ValueError(f"{pair}: reference missing in metadata")
        scan = next((s for s in rel.get("scans", []) if s.get("reference") == rescan_id), None)
        if scan is None:
            raise ValueError(f"{pair}: rescan relation missing in metadata")
        scene = scene_matrix(scan["transform"])
        ref_xyz, ref_oid_mesh, _ = read_ply(args.annotation_root / ref_id / "labels.instances.annotated.v2.ply")
        res_xyz, res_oid_mesh, _ = read_ply(args.annotation_root / rescan_id / "labels.instances.annotated.v2.ply")
        ref_tree, res_tree = cKDTree(ref_xyz), cKDTree(res_xyz)
        ref_object_trees = {
            int(oid): cKDTree(ref_xyz[ref_oid_mesh == oid])
            for oid in np.unique(ref_oid_mesh)
            if oid > 0 and np.any(ref_oid_mesh == oid)
        }
        ref_oid_at_candidate = object_labels(ref_points, ref_tree, ref_oid_mesh)
        query_oid = object_labels(query_points, res_tree, res_oid_mesh)
        diag_query_oid = d20.query_object_id.to_numpy(dtype=np.int64)
        if not np.array_equal(query_oid, diag_query_oid):
            raise ValueError(f"{pair}: query object IDs do not reproduce saved physical audit")

        # Official rigid mappings take precedence over same-ID links. Mirror the
        # candidate audit's ambiguity exclusions exactly.
        ambiguity: set[int] = set()
        for group in rel.get("ambiguity", []):
            for item in group:
                for key in ("instance_source", "instance_target"):
                    if key in item:
                        ambiguity.add(int(item[key]))
        rigid_map: dict[int, tuple[int, np.ndarray]] = {}
        for item in scan.get("rigid", []):
            src, dst = int(item.get("instance_rescan", -1)), int(item.get("instance_reference", -1))
            if src < 0 or dst < 0 or src in ambiguity or dst in ambiguity:
                continue
            rigid_map[src] = (dst, object_matrix(item["transform"]))
        # Same-ID unchanged objects are included only when both meshes have
        # nonzero support and each object maps to one nonzero semantic label.
        # The saved audit provides the authoritative link mask and mapped ID.
        linked = as_bool(d20.physical_instance_evaluable)
        mapped_ref_oid = d20.mapped_reference_object_id.to_numpy(dtype=np.int64)
        saved_query_oid = d20.query_object_id.to_numpy(dtype=np.int64)
        if not np.array_equal(saved_query_oid, query_oid):
            raise ValueError(f"{pair}: saved query object ID mismatch")

        transforms: list[np.ndarray] = []
        differs_from_scene = np.zeros(len(d20), dtype=bool)
        mapped_from_meta = np.full(len(d20), -1, dtype=np.int64)
        for i, qoid in enumerate(query_oid):
            if int(qoid) in rigid_map:
                roid, ref_to_res = rigid_map[int(qoid)]
                mapped_from_meta[i] = roid
                transforms.append(np.linalg.inv(ref_to_res))
                # A rigid relation is considered moved if its object-to-scene
                # rescan transform differs by >5 cm or >5 degrees.
                obj_to_ref = np.linalg.inv(ref_to_res)
                relative = obj_to_ref @ np.linalg.inv(scene)
                angle = np.arccos(np.clip((np.trace(relative[:3, :3]) - 1) / 2, -1, 1))
                differs_from_scene[i] = (np.linalg.norm(relative[:3, 3]) > 0.05) or (angle > np.deg2rad(5))
            else:
                transforms.append(scene)
        mapped_meta_mask = mapped_from_meta >= 0
        if np.any(mapped_meta_mask & (mapped_ref_oid != mapped_from_meta)):
            raise ValueError(f"{pair}: official rigid object links disagree with saved audit")
        if np.any(linked & (mapped_ref_oid < 0)):
            # Same-ID unchanged links use the scene pose as their valid motion model.
            mapped_ref_oid = mapped_ref_oid.copy()

        scene_transformed = apply(query_points, scene)
        oracle_transformed = np.stack([apply(query_points[i:i+1], transforms[i])[0] for i in range(len(query_points))])
        phys_hit = np.zeros((len(query_points), 50), dtype=bool)
        scene_valid = np.zeros_like(phys_hit)
        oracle_valid = np.zeros_like(phys_hit)
        scene_surface_support = np.zeros(len(query_points), dtype=bool)
        object_surface_support = np.zeros(len(query_points), dtype=bool)
        candidate_xyz = ref_points[candidate_indices]
        candidate_obj = ref_oid_at_candidate[candidate_indices]
        for i in range(len(query_points)):
            if not linked[i] or mapped_ref_oid[i] <= 0:
                continue
            same = candidate_obj[i] == mapped_ref_oid[i]
            phys_hit[i] = same
            object_tree = ref_object_trees.get(int(mapped_ref_oid[i]))
            if object_tree is None:
                continue
            scene_surface_support[i] = object_tree.query(scene_transformed[i], k=1)[0] <= args.distance_threshold
            object_surface_support[i] = object_tree.query(oracle_transformed[i], k=1)[0] <= args.distance_threshold
            scene_dist = np.linalg.norm(candidate_xyz[i] - scene_transformed[i], axis=1)
            oracle_dist = np.linalg.norm(candidate_xyz[i] - oracle_transformed[i], axis=1)
            scene_valid[i] = same & (scene_dist <= args.distance_threshold)
            oracle_valid[i] = same & (oracle_dist <= args.distance_threshold)

        # Cross-check physical candidate availability against the independent
        # audit at K=20 and K=50 before interpreting transform-aware metrics.
        for k, d in ((20, d20), (50, d50)):
            stored = as_bool(d.physical_instance_candidate_hit)
            computed = phys_hit[:, :k].any(axis=1) & linked
            if not np.array_equal(stored, computed):
                raise ValueError(f"{pair}: candidate objectId hit fails K={k} reproduction")

        q = pd.DataFrame({
            "pair": pair,
            "query_index": np.arange(len(query_points)),
            "query_object_id": query_oid,
            "mapped_reference_object_id": mapped_ref_oid,
            "linked": linked,
            "official_rigid": mapped_meta_mask & linked,
            "object_transform_differs_from_scene": differs_from_scene & mapped_meta_mask & linked,
            "physical_hit_k20": phys_hit[:, :20].any(axis=1),
            "physical_hit_k50": phys_hit.any(axis=1),
            "scene_surface_support": scene_surface_support,
            "object_oracle_surface_support": object_surface_support,
            "scene_valid_k20": scene_valid[:, :20].any(axis=1),
            "scene_valid_k50": scene_valid.any(axis=1),
            "object_oracle_valid_k20": oracle_valid[:, :20].any(axis=1),
            "object_oracle_valid_k50": oracle_valid.any(axis=1),
            "scene_new_valid_21_50": scene_valid[:, 20:].any(axis=1) & ~scene_valid[:, :20].any(axis=1),
            "object_new_valid_21_50": oracle_valid[:, 20:].any(axis=1) & ~oracle_valid[:, :20].any(axis=1),
        })
        all_queries.append(q)
        for group_name, mask in (("linked", linked), ("official_rigid", mapped_meta_mask & linked)):
            row: dict[str, object] = {"pair": pair, "group": group_name, "queries": int(mask.sum())}
            for name, values in (("physical", phys_hit), ("scene_valid", scene_valid), ("object_oracle_valid", oracle_valid)):
                for k in (20, 50):
                    row[f"{name}_recall_k{k}"] = float(values[mask, :k].any(axis=1).mean()) if mask.any() else np.nan
            if group_name == "official_rigid":
                for k in (20, 50):
                    row[f"scene_valid_recall_k{k}"] = float(scene_valid[mask, :k].any(axis=1).mean()) if mask.any() else np.nan
                    row[f"object_oracle_valid_recall_k{k}"] = float(oracle_valid[mask, :k].any(axis=1).mean()) if mask.any() else np.nan
                row["scene_surface_support"] = float(scene_surface_support[mask].mean()) if mask.any() else np.nan
                row["object_oracle_surface_support"] = float(object_surface_support[mask].mean()) if mask.any() else np.nan
                row["object_oracle_valid_k20_given_surface_support"] = (
                    float(oracle_valid[mask, :20].any(axis=1)[object_surface_support[mask]].mean())
                    if object_surface_support[mask].any() else np.nan
                )
                row["object_oracle_valid_k50_given_surface_support"] = (
                    float(oracle_valid[mask].any(axis=1)[object_surface_support[mask]].mean())
                    if object_surface_support[mask].any() else np.nan
                )
            relation_rows.append(row)

    queries = pd.concat(all_queries, ignore_index=True)
    rel = pd.DataFrame(relation_rows)
    queries.to_csv(args.output_dir / "query_diagnostics.csv", index=False)
    rel.to_csv(args.output_dir / "per_relation.csv", index=False)
    rng = np.random.default_rng(args.seed)
    macro: dict[str, dict[str, float]] = {}
    for group in ("linked", "official_rigid"):
        sub = rel[rel.group == group].set_index("pair")
        names = ["physical", "scene_valid", "object_oracle_valid"]
        macro[group] = {}
        for name in names:
            for k in (20, 50):
                col = f"{name}_recall_k{k}"
                v = sub[col].dropna().to_numpy(dtype=np.float64)
                if not len(v):
                    continue
                macro[group][col] = float(v.mean())
        if group == "official_rigid":
            for col in ("scene_surface_support", "object_oracle_surface_support", "object_oracle_valid_k20_given_surface_support", "object_oracle_valid_k50_given_surface_support"):
                macro[group][col] = float(sub[col].dropna().mean())
        if group == "official_rigid":
            delta = (sub["object_oracle_valid_recall_k20"] - sub["scene_valid_recall_k20"]).dropna().to_numpy()
            idx = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
            boot = delta[idx].mean(axis=1)
            macro[group]["object_minus_scene_k20"] = float(delta.mean())
            macro[group]["object_minus_scene_k20_ci_low"] = float(np.quantile(boot, .025))
            macro[group]["object_minus_scene_k20_ci_high"] = float(np.quantile(boot, .975))
            delta50 = (sub["object_oracle_valid_recall_k50"] - sub["scene_valid_recall_k50"]).dropna().to_numpy()
            idx50 = rng.integers(0, len(delta50), size=(args.bootstrap_samples, len(delta50)))
            boot50 = delta50[idx50].mean(axis=1)
            macro[group]["object_minus_scene_k50"] = float(delta50.mean())
            macro[group]["object_minus_scene_k50_ci_low"] = float(np.quantile(boot50, .025))
            macro[group]["object_minus_scene_k50_ci_high"] = float(np.quantile(boot50, .975))

    summary = {
        "relations": len(pair_names),
        "distance_threshold_m": args.distance_threshold,
        "relation_macro": macro,
        "linked_queries": int(queries.linked.sum()),
        "official_rigid_queries": int(queries.official_rigid.sum()),
        "object_transform_differs_from_scene_queries": int(queries.object_transform_differs_from_scene.sum()),
        "official_rigid_relations_with_queries": int((rel[(rel.group == "official_rigid")].queries > 0).sum()),
        "official_rigid_scene_valid_k20_hits": int((queries.official_rigid & queries.scene_valid_k20).sum()),
        "official_rigid_object_valid_k20_hits": int((queries.official_rigid & queries.object_oracle_valid_k20).sum()),
        "official_rigid_object_valid_k50_hits": int((queries.official_rigid & queries.object_oracle_valid_k50).sum()),
        "official_rigid_object_surface_supported_queries": int((queries.official_rigid & queries.object_oracle_surface_support).sum()),
        "official_rigid_physical_candidate_k20_hits": int((queries.official_rigid & queries.physical_hit_k20).sum()),
        "official_rigid_physical_candidate_k50_hits": int((queries.official_rigid & queries.physical_hit_k50).sum()),
        "object_transform_semantics": "official reference-to-rescan transform inverted to map rescan query into reference coordinates",
        "selection": "cached descriptor top-50; labels and transforms post-hoc only",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Transform-valid candidate availability audit",
        "",
        f"Post-hoc audit of ViT-G/14 descriptor top-50 candidates on {len(pair_names)} source relations. A query has a valid candidate at K when at least one cached candidate is assigned to its officially linked reference object and lies within {args.distance_threshold:.2f} m of the query after applying the relevant oracle transform. Candidate ranking is unchanged; labels and transforms are evaluation-only. The global scene transform is used for scene support. For moved rigid objects, the official per-object transform is inverted to map rescan points into reference coordinates.",
        "",
        f"There are {summary['official_rigid_queries']} officially rigid-linked queries in {summary['official_rigid_relations_with_queries']} relations ({summary['object_transform_differs_from_scene_queries']} have object transforms differing from scene pose by >5 cm or >5 degrees). Relation-macro K20 valid-candidate recall is {macro['official_rigid'].get('scene_valid_recall_k20', float('nan')):.1%} under scene pose and {macro['official_rigid'].get('object_oracle_valid_recall_k20', float('nan')):.1%} under object-pose oracle; the paired difference is {macro['official_rigid'].get('object_minus_scene_k20', float('nan'))*100:+.1f} pp (relation-bootstrap 95% CI [{macro['official_rigid'].get('object_minus_scene_k20_ci_low', float('nan'))*100:+.1f}, {macro['official_rigid'].get('object_minus_scene_k20_ci_high', float('nan'))*100:+.1f}] pp). Pooled K20 valid hits are {summary['official_rigid_scene_valid_k20_hits']}/{summary['official_rigid_queries']} under scene pose and {summary['official_rigid_object_valid_k20_hits']}/{summary['official_rigid_queries']} under object pose. At K50, object-pose valid candidates increase to {summary['official_rigid_object_valid_k50_hits']}/{summary['official_rigid_queries']}. Full reference-mesh support under object pose is {summary['official_rigid_object_surface_supported_queries']}/{summary['official_rigid_queries']}; among those surface-supported queries, {summary['official_rigid_object_valid_k20_hits']}/{summary['official_rigid_object_surface_supported_queries']} have a valid top-20 candidate and {summary['official_rigid_object_valid_k50_hits']}/{summary['official_rigid_object_surface_supported_queries']} have one at top-50. The gains are concentrated in a small number of relations; see the per-relation CSV and bootstrap interval.",
        "",
        "This is a candidate-ceiling diagnostic, not a label-free method or pose-recovery result. Object-oracle performance uses ground-truth instance transforms and cannot be claimed as achievable inference performance. Relation bootstrap reflects this fixed source pool only; it does not establish independent-site generalization.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
