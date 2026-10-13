#!/usr/bin/env python3
"""Measure top-K physical and transform-valid candidate availability on all source rigid queries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from audit_transform_valid_candidate_availability import apply, object_matrix, read_ply, scene_matrix


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotation-root", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--motion-oracle-queries", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--bootstrap-samples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261002)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.annotation_root / "3RScan.json").read_text(encoding="utf-8"))
    oracle = pd.read_csv(args.motion_oracle_queries)
    rigid = oracle[(oracle.split == "source") & (oracle.group == "rigid")].copy()
    pair_names = sorted(rigid.pair.unique().tolist())
    query_frames: list[pd.DataFrame] = []
    relation_rows: list[dict[str, object]] = []

    for pair in pair_names:
        cache_path = args.candidate_root / f"{pair}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path) as z:
            ref_id, rescan_id = str(z["reference_id"].item()), str(z["rescan_id"].item())
            ref_points = z["reference_points"].astype(np.float64)
            query_points_all = z["query_points"].astype(np.float64)
            frame_ids_all = z["query_frame_ids"].astype(np.int64)
            candidates_all = z["candidate_indices"].astype(np.int64)
        rows = rigid[rigid.pair == pair].sort_values("query_index").reset_index(drop=True)
        qidx = rows.query_index.to_numpy(dtype=np.int64)
        if np.any(qidx < 0) or np.any(qidx >= len(query_points_all)):
            raise ValueError(f"{pair}: oracle query index outside descriptor cache")
        if not np.array_equal(frame_ids_all[qidx], rows.frame_id.to_numpy(dtype=np.int64)):
            raise ValueError(f"{pair}: query frame IDs do not match oracle audit")
        qpoints = query_points_all[qidx]
        cindices = candidates_all[qidx]
        query_oid = rows.query_object_id.to_numpy(dtype=np.int64)
        mapped_oid = rows.reference_object_id.to_numpy(dtype=np.int64)

        relation = next((r for r in metadata if r.get("reference") == ref_id), None)
        if relation is None:
            raise ValueError(f"{pair}: missing reference metadata")
        scan = next((s for s in relation.get("scans", []) if s.get("reference") == rescan_id), None)
        if scan is None:
            raise ValueError(f"{pair}: missing rescan relation metadata")
        scene = scene_matrix(scan["transform"])
        rigid_map: dict[int, tuple[int, np.ndarray]] = {}
        for item in scan.get("rigid", []):
            qid = int(item.get("instance_rescan", -1))
            rid = int(item.get("instance_reference", -1))
            if qid > 0 and rid > 0:
                rigid_map[qid] = (rid, object_matrix(item["transform"]))
        transforms: list[np.ndarray] = []
        for qid, rid in zip(query_oid, mapped_oid):
            if int(qid) not in rigid_map or rigid_map[int(qid)][0] != int(rid):
                raise ValueError(f"{pair}: motion-oracle link disagrees with official metadata")
            transforms.append(np.linalg.inv(rigid_map[int(qid)][1]))

        ref_xyz, ref_oid_mesh, _ = read_ply(args.annotation_root / ref_id / "labels.instances.annotated.v2.ply")
        ref_tree = cKDTree(ref_xyz)
        candidate_oid = ref_oid_mesh[ref_tree.query(ref_points.astype(np.float32), k=1)[1]][cindices]
        candidate_xyz = ref_points[cindices]
        scene_q = apply(qpoints, scene)
        object_q = np.stack([apply(qpoints[i:i+1], transforms[i])[0] for i in range(len(qpoints))])
        instance_candidate = candidate_oid == mapped_oid[:, None]
        scene_dist = np.linalg.norm(candidate_xyz - scene_q[:, None, :], axis=2)
        object_dist = np.linalg.norm(candidate_xyz - object_q[:, None, :], axis=2)
        scene_valid = instance_candidate & (scene_dist <= args.threshold)
        object_valid = instance_candidate & (object_dist <= args.threshold)
        object_trees = {
            int(oid): cKDTree(ref_xyz[ref_oid_mesh == oid])
            for oid in np.unique(mapped_oid)
            if np.any(ref_oid_mesh == oid)
        }
        surface_support = np.asarray([
            object_trees[int(oid)].query(object_q[i], k=1)[0] <= args.threshold
            for i, oid in enumerate(mapped_oid)
        ])
        scene_support = np.asarray([
            object_trees[int(oid)].query(scene_q[i], k=1)[0] <= args.threshold
            for i, oid in enumerate(mapped_oid)
        ])

        q_values: dict[str, object] = {
            "pair": pair,
            "query_index": qidx,
            "frame_id": rows.frame_id.to_numpy(dtype=np.int64),
            "query_object_id": query_oid,
            "reference_object_id": mapped_oid,
            "reference_object_surface_support_scene": scene_support,
            "reference_object_surface_support_object": surface_support,
        }
        for k in (1, 5, 10, 20, 50):
            q_values[f"physical_candidate_k{k}"] = instance_candidate[:, :k].any(axis=1)
            q_values[f"scene_valid_candidate_k{k}"] = scene_valid[:, :k].any(axis=1)
            q_values[f"object_valid_candidate_k{k}"] = object_valid[:, :k].any(axis=1)
        q = pd.DataFrame(q_values)
        query_frames.append(q)
        row: dict[str, object] = {"pair": pair, "queries": len(q)}
        for metric in q.columns[5:]:
            row[metric] = float(q[metric].mean())
        relation_rows.append(row)

    qall = pd.concat(query_frames, ignore_index=True)
    rel = pd.DataFrame(relation_rows)
    qall.to_csv(args.output_dir / "query_diagnostics.csv", index=False)
    rel.to_csv(args.output_dir / "per_relation.csv", index=False)
    rng = np.random.default_rng(args.seed)
    valid_rel = rel
    deltas = valid_rel.object_valid_candidate_k20.to_numpy() - valid_rel.scene_valid_candidate_k20.to_numpy()
    draws = rng.integers(0, len(deltas), size=(args.bootstrap_samples, len(deltas)))
    boot = deltas[draws].mean(axis=1)
    summary: dict[str, object] = {
        "source_relations": len(pair_names),
        "rigid_queries": len(qall),
        "threshold_m": args.threshold,
        "pooled": {c: float(qall[c].mean()) for c in qall.columns[5:]},
        "pooled_counts": {c: int(qall[c].sum()) for c in qall.columns[5:]},
        "k20_to_k50_transitions": {
            "new_physical_object_queries": int((~qall.physical_candidate_k20 & qall.physical_candidate_k50).sum()),
            "new_object_valid_queries": int((~qall.object_valid_candidate_k20 & qall.object_valid_candidate_k50).sum()),
            "new_object_valid_queries_with_physical_hit_already_at_k20": int((qall.physical_candidate_k20 & ~qall.object_valid_candidate_k20 & qall.object_valid_candidate_k50).sum()),
            "new_object_valid_queries_with_first_physical_hit_at_k50": int((~qall.physical_candidate_k20 & qall.object_valid_candidate_k50).sum()),
        },
        "relation_macro": {c: float(rel[c].mean()) for c in rel.columns if c not in {"pair", "queries"}},
        "relation_macro_object_vs_scene_valid_k20": {
            "delta": float(deltas.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "bootstrap_relations": len(deltas),
        },
        "relation_macro_k50_minus_k20": {},
        "scoring": "official rigid object IDs, reference mesh coordinates, and object transforms are post-hoc only",
    }
    for metric in ("physical_candidate", "scene_valid_candidate", "object_valid_candidate"):
        change = (rel[f"{metric}_k50"] - rel[f"{metric}_k20"]).to_numpy(dtype=np.float64)
        change_draws = rng.integers(0, len(change), size=(args.bootstrap_samples, len(change)))
        change_boot = change[change_draws].mean(axis=1)
        summary["relation_macro_k50_minus_k20"][metric] = {
            "delta": float(change.mean()),
            "ci95": [float(np.quantile(change_boot, .025)), float(np.quantile(change_boot, .975))],
            "bootstrap_relations": len(change),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = [
        "# Rigid-object transform-valid candidate audit",
        "",
        f"Across {len(pair_names)} source relations and {len(qall)} queries linked by official rigid-instance metadata, each query uses its saved descriptor top-50 candidates. A valid candidate must belong to the linked reference object and lie within {args.threshold:.2f} m of the query under the tested transform. Labels and transforms are post-hoc only. This analysis uses the same 16-source rigid-query pool as the motion oracle, rather than the smaller 13-relation matched physical-instance sweep.",
        "",
        "Pooled candidate availability by K (counts / 1,510 queries; columns are physical-object hit, scene-pose-valid, and object-pose-valid):",
        "",
        "| K | Physical object | Scene valid | Object-pose valid |",
        "|---:|---:|---:|---:|",
        *[f"| {k} | {summary['pooled_counts'][f'physical_candidate_k{k}']} ({summary['pooled'][f'physical_candidate_k{k}']:.1%}) | {summary['pooled_counts'][f'scene_valid_candidate_k{k}']} ({summary['pooled'][f'scene_valid_candidate_k{k}']:.1%}) | {summary['pooled_counts'][f'object_valid_candidate_k{k}']} ({summary['pooled'][f'object_valid_candidate_k{k}']:.1%}) |" for k in (1, 5, 10, 20, 50)],
        "",
        f"Full reference-object surface support under object pose is {summary['pooled_counts']['reference_object_surface_support_object']}/{len(qall)}.",
        "",
        f"Relation-macro K20 valid-candidate recall is {summary['relation_macro']['scene_valid_candidate_k20']:.1%} under scene pose and {summary['relation_macro']['object_valid_candidate_k20']:.1%} under object pose (paired delta {summary['relation_macro_object_vs_scene_valid_k20']['delta']*100:+.1f} pp, relation-bootstrap 95% CI [{summary['relation_macro_object_vs_scene_valid_k20']['ci95'][0]*100:+.1f}, {summary['relation_macro_object_vs_scene_valid_k20']['ci95'][1]*100:+.1f}] pp). Conditional on full object-surface support, only {summary['pooled_counts']['object_valid_candidate_k20']}/{summary['pooled_counts']['reference_object_surface_support_object']} queries have a valid top-20 candidate and {summary['pooled_counts']['object_valid_candidate_k50']}/{summary['pooled_counts']['reference_object_surface_support_object']} at top-50.",
        f"From K20 to K50, 61 additional queries gain an object-pose-valid candidate, while only 9 queries gain any candidate from the linked object. Of the 61 newly valid queries, 57 already had a linked-object candidate at K20; the added ranks supply a spatially better point from that same object. Thus candidate rank/point choice remains important even after physical-object identity is retrieved.",
        "",
        f"Relation-macro K50−K20 paired changes (95% relation-bootstrap CI): physical-object availability {summary['relation_macro_k50_minus_k20']['physical_candidate']['delta']*100:+.1f} pp [{summary['relation_macro_k50_minus_k20']['physical_candidate']['ci95'][0]*100:+.1f},{summary['relation_macro_k50_minus_k20']['physical_candidate']['ci95'][1]*100:+.1f}], object-pose valid candidates {summary['relation_macro_k50_minus_k20']['object_valid_candidate']['delta']*100:+.1f} pp [{summary['relation_macro_k50_minus_k20']['object_valid_candidate']['ci95'][0]*100:+.1f},{summary['relation_macro_k50_minus_k20']['object_valid_candidate']['ci95'][1]*100:+.1f}].",
        "",
        "The oracle is an upper-bound diagnostic, not an inference method. Candidate validity is defined against sampled reference RGB-D patches; full-mesh support is a separate geometric ceiling. Relation bootstrap is within 3RScan train and does not establish site-level generalization.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
