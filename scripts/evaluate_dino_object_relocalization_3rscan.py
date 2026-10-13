#!/usr/bin/env python3
"""Score frozen DINO correspondence hypotheses as per-object 6DoF relocalization.

The source object mask is used only to pool its already-computed query
correspondences, as in object-instance relocalization. Candidate indices,
mode assignments and inlier flags are fixed outputs. Official instance links,
object transforms and symmetry labels are opened only for post-hoc scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def matrix(flat: list[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(4, 4).T


def invert_se3(t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3].T @ t[:3, 3]
    return out


def fit_rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 3:
        raise ValueError("at least three correspondences required")
    sx, sy = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - sx).T @ (target - sy))
    sign = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rot = u @ np.diag([1.0, 1.0, sign]) @ vt
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = sy - rot @ sx
    return out


def pose_errors(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    angle = Rotation.from_matrix(truth[:3, :3].T @ pred[:3, :3]).magnitude()
    translation = np.linalg.norm(pred[:3, 3] - truth[:3, 3])
    return float(np.degrees(angle)), float(translation)


def bool_col(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = frame[name]
    if values.dtype == object:
        return values.fillna(False).astype(str).str.lower().isin({"true", "1"}).to_numpy()
    return values.fillna(False).to_numpy(dtype=bool)


def majority(values: np.ndarray) -> int:
    values = values.astype(np.int64)
    values = values[values > 0]
    if not len(values):
        return -1
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--annotation-root", type=Path, default=Path("data/3rscan"))
    ap.add_argument("--audit-root", type=Path, default=Path("outputs/reports/20261006_vitg_pose_locked_motion_validation"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_dino_object_pose_relocalization"))
    ap.add_argument("--min-object-inliers", type=int, default=8)
    ap.add_argument("--bootstrap-draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    root = args.root.resolve()
    audit_root = root / args.audit_root
    annotation_root = args.annotation_root
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((annotation_root / "3RScan.json").read_text())
    by_reference = {row["reference"]: row for row in metadata}
    records: list[dict] = []

    for cache_path in sorted((audit_root / "descriptor_candidates").glob("*.npz")):
        with np.load(cache_path) as z:
            reference_id = str(z["reference_id"].item())
            rescan_id = str(z["rescan_id"].item())
            query_xyz = z["query_points"].astype(np.float64)
            reference_xyz = z["reference_points"].astype(np.float64)
        relation = by_reference.get(reference_id)
        if relation is None:
            raise ValueError(f"missing metadata reference {reference_id}")
        scan = next((x for x in relation.get("scans", []) if x.get("reference") == rescan_id), None)
        if scan is None:
            raise ValueError(f"missing rescan {rescan_id} under {reference_id}")
        folder = audit_root / "relations" / cache_path.stem
        diagnostic_path = folder / "query_diagnostics.csv"
        if not diagnostic_path.is_file():
            raise FileNotFoundError(diagnostic_path)
        diag = pd.read_csv(diagnostic_path)
        baseline = diag[diag.arm.eq("global_ransac")].sort_values("query_index").reset_index(drop=True)
        method = diag[diag.arm.eq("combined")].sort_values("query_index").reset_index(drop=True)
        if len(baseline) != len(query_xyz) or len(method) != len(query_xyz):
            raise ValueError(f"query rows do not align with candidate cache: {cache_path.stem}")
        if not np.array_equal(baseline.query_index.to_numpy(), np.arange(len(query_xyz))) or not np.array_equal(method.query_index.to_numpy(), np.arange(len(query_xyz))):
            raise ValueError(f"query indices are not complete/ordered for {cache_path.stem}")
        if not np.array_equal(
            baseline.query_instance_id.to_numpy(dtype=np.int64),
            method.query_instance_id.to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"query instance IDs differ between arms for {cache_path.stem}")

        expected = {}
        ambiguous: set[int] = set()
        for group in relation.get("ambiguity", []):
            for entry in group:
                for key in ("instance_source", "instance_target"):
                    if key in entry:
                        ambiguous.add(int(entry[key]))
        for item in scan.get("rigid", []):
            qid = int(item.get("instance_rescan", -1))
            rid = int(item.get("instance_reference", -1))
            if qid <= 0 or rid <= 0 or qid in ambiguous or rid in ambiguous:
                continue
            if qid in expected and expected[qid]["reference_object_id"] != rid:
                raise ValueError(f"ambiguous rigid mapping for {reference_id}/{rescan_id}/{qid}")
            expected[qid] = {
                "reference_object_id": rid,
                "truth_rescan_to_reference": invert_se3(matrix(item["transform"])),
                "symmetry": int(item.get("symmetry", -1)),
            }
        scene_to_reference = matrix(scan["transform"])

        # Reconstruct the shared scene pose from the frozen global RANSAC
        # inliers. This uses only saved geometric correspondences, not labels.
        base_mask = bool_col(baseline, "hypothesis_0_inlier")
        base_indices = baseline.loc[base_mask, "hypothesis_0_selected_reference_index"].to_numpy(dtype=np.int64)
        base_pose = fit_rigid(query_xyz[base_mask], reference_xyz[base_indices]) if len(base_indices) >= 3 else None

        qid_values = baseline.query_instance_id.to_numpy(dtype=np.int64)
        for qid, link in expected.items():
            object_mask = qid_values == qid
            observed_queries = int(object_mask.sum())
            if observed_queries == 0:
                continue
            for arm, frame in (("global_ransac", baseline), ("combined", method)):
                if arm == "global_ransac":
                    pred_pose = base_pose
                    support = int((object_mask & base_mask).sum())
                    assigned_mode = 0 if pred_pose is not None else -1
                    selected_ref_ids = frame.loc[object_mask & base_mask, "hypothesis_0_selected_reference_instance_id"].to_numpy(dtype=np.int64)
                else:
                    mode = frame.assigned_mode.to_numpy(dtype=np.int64)
                    inlier = bool_col(frame, "consensus_inlier")
                    candidate_counts = [int((object_mask & inlier & (mode == m)).sum()) for m in range(4)]
                    assigned_mode = int(np.argmax(candidate_counts)) if max(candidate_counts, default=0) else -1
                    use = object_mask & inlier & (mode == assigned_mode) if assigned_mode >= 0 else np.zeros(len(query_xyz), dtype=bool)
                    support = int(use.sum())
                    pred_pose = None
                    selected_ref_ids = np.empty(0, dtype=np.int64)
                    if support >= 3:
                        ref_indices = frame.loc[use, f"hypothesis_{assigned_mode}_selected_reference_index"].to_numpy(dtype=np.int64)
                        valid = (ref_indices >= 0) & (ref_indices < len(reference_xyz))
                        use_indices = np.flatnonzero(use)[valid]
                        ref_indices = ref_indices[valid]
                        support = len(ref_indices)
                        if support >= 3:
                            pred_pose = fit_rigid(query_xyz[use_indices], reference_xyz[ref_indices])
                            selected_ref_ids = frame.loc[use, f"hypothesis_{assigned_mode}_selected_reference_instance_id"].to_numpy(dtype=np.int64)[valid]
                pred_reference_object_id = majority(selected_ref_ids)
                if pred_pose is None:
                    rot_error, trans_error = np.inf, np.inf
                else:
                    rot_error, trans_error = pose_errors(pred_pose, link["truth_rescan_to_reference"])
                strict = bool(rot_error <= 10.0 and trans_error <= 0.10)
                relaxed = bool(rot_error <= 20.0 and trans_error <= 0.20)
                records.append({
                    "environment_id": reference_id,
                    "pair": cache_path.stem,
                    "reference_scan_id": reference_id,
                    "rescan_scan_id": rescan_id,
                    "query_object_id": qid,
                    "reference_object_id_gt": link["reference_object_id"],
                    "symmetry": link["symmetry"],
                    "query_patches": observed_queries,
                    "predicted_inlier_support": support,
                    "predicted_mode": assigned_mode,
                    "predicted_reference_object_id": pred_reference_object_id,
                    "identity_correct": bool(pred_reference_object_id == link["reference_object_id"]),
                    "pose_available": bool(pred_pose is not None and support >= args.min_object_inliers),
                    "rotation_error_deg": rot_error,
                    "translation_error_m": trans_error,
                    "predicted_pose_rescan_to_reference": json.dumps(pred_pose.tolist()) if pred_pose is not None else "",
                    "strict_10cm_10deg": bool(strict and pred_pose is not None and support >= args.min_object_inliers),
                    "relaxed_20cm_20deg": bool(relaxed and pred_pose is not None and support >= args.min_object_inliers),
                    "arm": arm,
                })
    objects = pd.DataFrame(records)
    if objects.empty:
        raise RuntimeError("no moved-object instances scored")
    # Symmetry classes are reported but excluded from rotation-threshold claims;
    # this avoids silently penalizing valid equivalent orientations.
    objects.to_csv(output_dir / "per_object.csv", index=False)
    nonsym = objects[objects.symmetry.eq(0)].copy()
    envs = sorted(nonsym.environment_id.unique())
    if not envs:
        raise RuntimeError("no non-symmetric moved objects with query support")
    rng = np.random.default_rng(args.seed)
    draw = rng.integers(0, len(envs), size=(args.bootstrap_draws, len(envs)))
    metrics = ["strict_10cm_10deg", "relaxed_20cm_20deg", "identity_correct", "pose_available"]
    summaries = {}
    for arm, frame in nonsym.groupby("arm"):
        summaries[arm] = {"objects_with_query_support": int(len(frame)), "environments": int(frame.environment_id.nunique())}
        for metric in metrics:
            per_env = frame.groupby("environment_id")[metric].mean().reindex(envs).fillna(0.0).to_numpy(dtype=float)
            samples = per_env[draw].mean(axis=1)
            summaries[arm][metric] = {
                "environment_macro": float(per_env.mean()),
                "ci95_cluster_bootstrap": np.quantile(samples, [.025, .975]).tolist(),
            }
    paired = nonsym.pivot_table(index=["environment_id", "pair", "query_object_id"], columns="arm", values="strict_10cm_10deg", aggfunc="first").fillna(False)
    if {"global_ransac", "combined"}.issubset(paired.columns):
        paired_delta = paired["combined"].astype(float) - paired["global_ransac"].astype(float)
        delta_by_env = paired_delta.groupby(level="environment_id").mean().reindex(envs).fillna(0.0).to_numpy()
        delta_samples = delta_by_env[draw].mean(axis=1)
        delta = {"combined_minus_global_strict_pose_macro": float(delta_by_env.mean()),
                 "ci95_environment_cluster_bootstrap": np.quantile(delta_samples, [.025, .975]).tolist(),
                 "objects": int(len(paired))}
    else:
        delta = {}
    summary = {
        "status": "complete",
        "dataset": "3RScan official validation split",
        "relations": int(objects.pair.nunique()),
        "environments_with_scored_objects": int(nonsym.environment_id.nunique()),
        "all_object_rows": int(len(objects)),
        "non_symmetric_object_rows": int(len(nonsym)),
        "symmetric_object_rows_excluded_from_pose_success": int(objects.symmetry.ne(0).sum()),
        "minimum_object_inliers_for_pose_available": args.min_object_inliers,
        "pose_thresholds_from_RIO": {"strict": {"translation_m": .10, "rotation_deg": 10}, "relaxed": {"translation_m": .20, "rotation_deg": 20}},
        "mode_selection": "for each provided source-instance mask, majority assigned frozen hypothesis among its label-free inlier correspondences; fit a rigid transform from that object's inlier query/candidate point pairs",
        "baseline": "one reconstructed global RANSAC pose shared by all moved objects in the pair",
        "scoring": "official reference-to-rescan object transform inverted to rescan-to-reference; GT links/transforms opened only after frozen predictions; nonzero symmetry classes excluded from rotation-threshold success",
        "arms": summaries,
        "paired_environment_bootstrap": delta,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
        "limitations": [
            "This adapts frozen DINO correspondences and the existing pose-locked multi-motion predictions; it is not the learned RIO model reproduction.",
            "Source object instance masks are supplied to group correspondences; this follows the object-relocation task input, but no predicted segmentation is evaluated.",
            "Symmetric objects are excluded from pose success because symmetry-aware rotation equivalences are not implemented in this evaluator.",
            "The predictions and validation population were previously explored; this is a post-hoc task-level re-scoring, not independent-site evidence.",
            "The combined arm includes source-selected adaptive candidate limits; this is not an equal-compute or descriptor-only comparison.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
