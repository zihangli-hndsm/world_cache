#!/usr/bin/env python3
"""Refine RGB-ranked Rescan object poses with label-free scene ICP.

Only the already selected top-1 spatial+RGB proposal for each source
instance/target pair is refined. ICP aligns the annotated source object points
to the complete, unsegmented target cloud. Target instance IDs are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

from evaluate_rescan_full_scene_pose_transfer import get_transform, rank_group
from evaluate_rescan_xyz_transfer import read_annotated_ply


def cloud(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    if voxel_size > 0:
        result = result.voxel_down_sample(voxel_size)
    return result


def rotation_delta_degrees(before: np.ndarray, after: np.ndarray) -> float:
    relative = after[:3, :3] @ before[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-correspondence-distance", type=float, default=0.12)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()
    if args.max_correspondence_distance <= 0 or args.voxel_size <= 0 or args.max_iterations < 1:
        raise ValueError("ICP radius, voxel size, and iteration count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)

    by_target: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in groups:
        by_target[key[:2]].append(key)

    source_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    target_cache: dict[tuple[str, str], o3d.geometry.PointCloud] = {}
    diagnostics: list[dict[str, object]] = []
    refined_rows: list[dict[str, str]] = []
    for (scene, target_scan), keys in sorted(by_target.items()):
        annotation_dir = args.data_root / scene / "gt_segmentation"
        source_key = (scene, f"{scene}_0")
        if source_key not in source_cache:
            points, ids, _ = read_annotated_ply(annotation_dir / f"{scene}_0.ply")
            source_cache[source_key] = (points, ids)
        if (scene, target_scan) not in target_cache:
            target_points, _, _ = read_annotated_ply(annotation_dir / f"{target_scan}.ply")
            target_cache[(scene, target_scan)] = cloud(target_points, args.voxel_size)
        source_points, source_ids = source_cache[source_key]
        target_cloud = target_cache[(scene, target_scan)]

        for key in keys:
            selected = rank_group(groups[key], "color")
            source_id = int(key[2])
            object_points = source_points[source_ids == source_id]
            source_cloud = cloud(object_points, args.voxel_size)
            before = get_transform(selected)
            result = o3d.pipelines.registration.registration_icp(
                source_cloud,
                target_cloud,
                args.max_correspondence_distance,
                before,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=args.max_iterations
                ),
            )
            after = np.asarray(result.transformation, dtype=np.float64)
            for r in range(4):
                for c in range(4):
                    selected[f"transform_{r}{c}"] = repr(float(after[r, c]))
            refined_rows.append(selected)
            diagnostics.append({
                "scene": scene,
                "target_scan": target_scan,
                "source_instance_id": source_id,
                "source_points": len(object_points),
                "source_voxel_points": len(source_cloud.points),
                "target_voxel_points": len(target_cloud.points),
                "initial_fpfh_fitness": float(selected["fitness"]),
                "icp_fitness": float(result.fitness),
                "icp_inlier_rmse_m": float(result.inlier_rmse),
                "translation_update_m": float(np.linalg.norm(after[:3, 3] - before[:3, 3])),
                "rotation_update_degrees": rotation_delta_degrees(before, after),
            })
        print(f"refined {target_scan}; objects={len(keys)}", flush=True)

    output_candidates = args.output_dir / "color_rerank_selected_icp.csv"
    with output_candidates.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(refined_rows)
    with (args.output_dir / "icp_refinement_per_object.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    summary = {
        "source_object_target_pairs": len(diagnostics),
        "target_scans": len(by_target),
        "target_instance_labels_used": False,
        "initialized_from": "top-1 candidate under frozen spatial+RGB ranking",
        "target_geometry": "complete unsegmented target cloud",
        "max_correspondence_distance_m": args.max_correspondence_distance,
        "voxel_size_m": args.voxel_size,
        "max_iterations": args.max_iterations,
        "mean_icp_fitness": float(np.mean([r["icp_fitness"] for r in diagnostics])),
        "median_icp_inlier_rmse_m": float(np.median([r["icp_inlier_rmse_m"] for r in diagnostics])),
        "median_translation_update_m": float(np.median([r["translation_update_m"] for r in diagnostics])),
        "median_rotation_update_degrees": float(np.median([r["rotation_update_degrees"] for r in diagnostics])),
        "limitations": [
            "ICP can converge to nearby scene surfaces and assumes the initial proposal is in its basin of attraction",
            "the experiment refines one frozen top-1 proposal per object-target pair and does not change candidate ranking",
            "source instance masks are available, while the target remains unsegmented",
        ],
    }
    (args.output_dir / "icp_refinement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
