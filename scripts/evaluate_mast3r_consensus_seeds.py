#!/usr/bin/env python3
"""Rerun dense MASt3R geometric consensus seeds from saved candidate caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from evaluate_3rscan_pose_free_registration import transform
from evaluate_3rscan_spatial_consensus import consensus


def evaluate(cache_path: Path, output_dir: Path, seed: int, threshold: float, iterations: int,
             ground_truth_override: np.ndarray | None = None) -> dict:
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    query_points = arrays["query_points"]
    reference_points = arrays["reference_points"]
    candidate_indices = arrays["candidate_indices"]
    candidate_count = arrays["candidate_count"]
    frame_ids = arrays["query_frame_ids"]
    eligible = candidate_count > 0
    selected = np.full(len(query_points), len(reference_points) - 1, dtype=np.int64)
    inliers = np.zeros(len(query_points), dtype=bool)
    if int(eligible.sum()) >= 3:
        matrix, chosen, chosen_inliers = consensus(
            query_points[eligible], reference_points, candidate_indices[eligible],
            threshold=threshold, iterations=iterations, seed=seed,
            group_ids=frame_ids[eligible],
        )
        selected[eligible] = chosen
        inliers[eligible] = chosen_inliers
    else:
        matrix = np.eye(4, dtype=np.float64)

    ref_obj = arrays["reference_object_ids"]
    ref_global = arrays["reference_global_ids"]
    ref_sem = arrays["reference_semantics"]
    query_obj = arrays["query_object_ids"]
    query_global = arrays["query_global_ids"]
    query_sem = arrays["query_semantics"]
    query_eval = arrays["query_evaluable"].astype(bool)
    physical_eval = arrays["physical_evaluable"].astype(bool)
    mapped_ref = arrays["mapped_reference_ids"]
    residual = np.linalg.norm(transform(query_points, matrix) - reference_points[selected], axis=1)
    gt = arrays["ground_truth_transform"] if ground_truth_override is None else ground_truth_override
    rot_error = float(np.degrees(Rotation.from_matrix(matrix[:3, :3].T @ gt[:3, :3]).magnitude()))
    trans_error = float(np.linalg.norm(matrix[:3, 3] - gt[:3, 3]))
    physical_correct = ref_obj[selected] == mapped_ref
    physical_transfer = inliers & physical_eval & physical_correct
    semantic_correct = ref_sem[selected] == query_sem
    semantic_inliers = inliers & query_eval
    assigned_physical = inliers & physical_eval
    summary = {
        "method": "dense MASt3R candidate cache with seed-varying spatial consensus",
        "candidate_cache": cache_path.name,
        "query_points": int(len(query_points)),
        "semantic_evaluable_queries": int(query_eval.sum()),
        "linked_physical_instance_queries": int(physical_eval.sum()),
        "mean_actual_candidates_per_query": float(candidate_count.mean()),
        "queries_eligible_for_consensus": int(eligible.sum()),
        "physical_instance_candidate_recall_at_k": float((arrays["physical_candidate_hit"].astype(bool) & physical_eval).sum() / physical_eval.sum()) if physical_eval.any() else None,
        "semantic_candidate_recall_at_k": float((arrays["semantic_candidate_hit"].astype(bool) & query_eval).sum() / query_eval.sum()) if query_eval.any() else None,
        "consensus_assigned_queries": int(inliers.sum()),
        "semantic_inliers": int(semantic_inliers.sum()),
        "semantic_correct_inliers": int((semantic_inliers & semantic_correct).sum()),
        "semantic_inlier_coverage": float(semantic_inliers.sum() / query_eval.sum()) if query_eval.any() else None,
        "semantic_accuracy_among_inliers": float(semantic_correct[semantic_inliers].mean()) if semantic_inliers.any() else None,
        "physical_correct_transfers": int(physical_transfer.sum()),
        "physical_correct_transfer_coverage": float(physical_transfer.sum() / physical_eval.sum()) if physical_eval.any() else None,
        "physical_assigned_precision": float(physical_transfer.sum() / assigned_physical.sum()) if assigned_physical.any() else None,
        "scene_rotation_error_deg": rot_error,
        "scene_translation_error_m": trans_error,
        "strict_scene_pose_success": bool(rot_error < 5.0 and trans_error < 0.25),
        "ransac_seed": int(seed),
        "ransac_iterations": int(iterations),
        "candidate_top_k": int(candidate_indices.shape[1]),
        "pair_run": cache_path.parent.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predicted_rescan_to_reference.npy", matrix)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({
        "query_index": np.arange(len(query_points)), "frame_id": frame_ids,
        "query_object_id": query_obj, "query_global_id": query_global,
        "query_semantic_id": query_sem, "candidate_count": candidate_count,
        "semantic_candidate_hit": arrays["semantic_candidate_hit"].astype(bool) & query_eval,
        "physical_instance_evaluable": physical_eval,
        "physical_instance_candidate_hit": arrays["physical_candidate_hit"].astype(bool) & physical_eval,
        "consensus_inlier": inliers,
        "selected_reference_object_id": ref_obj[selected],
        "physical_instance_correct": physical_correct,
        "physical_correct_transfer": physical_transfer,
        "semantic_correct": semantic_correct, "spatial_residual_m": residual,
    }).to_csv(output_dir / "query_diagnostics.csv", index=False)
    (output_dir / "status.json").write_text(json.dumps({
        "status": "complete", "ransac_seed": int(seed),
        "ransac_iterations": int(iterations), "candidate_cache": cache_path.name,
        "queries_eligible_for_consensus": int(eligible.sum()),
        "scene_rotation_error_deg": rot_error,
        "scene_translation_error_m": trans_error,
        "strict_scene_pose_success": bool(rot_error < 5.0 and trans_error < 0.25),
    }, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/runs/20261004_mast3r_seed_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs/20261004_mast3r_dense_seed_sweep"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--expected-relations", type=int, default=14,
                        help="number of candidate caches to require (default: 14 for the train audit)")
    args = parser.parse_args()
    caches = sorted(args.cache_root.glob("*/candidate_cache.npz"))
    if len(caches) != args.expected_relations:
        raise ValueError(f"expected {args.expected_relations} relation candidate caches; found {len(caches)} under {args.cache_root}")
    results = []
    for index, cache_path in enumerate(caches, start=1):
        for seed in args.seeds:
            output = args.output_root / cache_path.parent.name / f"seed_{seed}"
            summary = evaluate(cache_path, output, seed, args.threshold, args.iterations)
            results.append(summary)
        print(f"completed consensus seeds for relation {index}/{len(caches)}: {cache_path.parent.name}", flush=True)
    frame = pd.DataFrame(results)
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_root / "per_relation_seed.csv", index=False)
    aggregation = frame.groupby("ransac_seed").apply(
        lambda rows: pd.Series({
            "relations": len(rows),
            "linked_queries": int(rows.linked_physical_instance_queries.sum()),
            "weighted_candidate_recall": float((rows.physical_instance_candidate_recall_at_k * rows.linked_physical_instance_queries).sum() / rows.linked_physical_instance_queries.sum()),
            "weighted_transfer_coverage": float(rows.physical_correct_transfers.sum() / rows.linked_physical_instance_queries.sum()),
            "strict_pose_successes": int(rows.strict_scene_pose_success.sum()),
            "mean_pose_rotation_deg": float(rows.scene_rotation_error_deg.mean()),
            "median_pose_translation_m": float(rows.scene_translation_error_m.median()),
        }), include_groups=False,
    ).reset_index()
    aggregation.to_csv(args.output_root / "aggregate_by_seed.csv", index=False)
    print(aggregation.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
