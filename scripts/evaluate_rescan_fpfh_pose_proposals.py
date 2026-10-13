#!/usr/bin/env python3
"""Evaluate multi-seed FPFH object-to-scene pose proposal recall on Rescan.

Source object masks come from the first scan. Target geometry is unsegmented
for feature extraction and RANSAC. Target labels are used only afterward to
measure exact-instance and same-class center recall at the paper's 20 cm pose
proposal threshold. Repeated seeds form a proposal ensemble, not a learned
method or an independent final test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d

from evaluate_rescan_xyz_transfer import read_annotated_ply


def to_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    return cloud


def descriptors(points: np.ndarray, voxel: float) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    cloud = to_cloud(points).voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=50))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=100),
    )
    return cloud, feature


def stable_seed(scene: str, target: str, instance_id: int, repeat: int) -> int:
    digest = hashlib.sha256(f"{scene}/{target}/{instance_id}/{repeat}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", help="optional scene subset for pilot runs")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--source-voxel", type=float, default=0.04)
    parser.add_argument("--target-voxel", type=float, default=0.06)
    parser.add_argument("--ransac-distance", type=float, default=0.12)
    parser.add_argument("--ransac-iterations", type=int, default=20000)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--pose-threshold", type=float, default=0.20)
    parser.add_argument(
        "--include-target-absent",
        action="store_true",
        help="generate proposals for every eligible source instance, even if its raw ID is absent from the target annotations",
    )
    args = parser.parse_args()
    if not 0.0 < args.ransac_confidence <= 1.0:
        raise ValueError("--ransac-confidence must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_list = [
        line.strip()
        for line in (args.data_root / "scene_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scenes = args.scenes if args.scenes else scene_list
    missing = sorted(set(scenes) - set(scene_list))
    if missing:
        raise ValueError(f"unknown scene names: {missing}")
    candidate_path = args.output_dir / "pose_candidates.partial.csv"
    object_path = args.output_dir / "object_pose_recall.partial.csv"
    candidate_rows: list[dict[str, float | int | str | bool]] = []
    object_rows: list[dict[str, float | int | str | bool]] = []

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)

    if candidate_path.exists():
        with candidate_path.open(newline="", encoding="utf-8") as stream:
            candidate_rows = list(csv.DictReader(stream))
    if object_path.exists():
        with object_path.open(newline="", encoding="utf-8") as stream:
            object_rows = list(csv.DictReader(stream))
    candidate_counts: dict[tuple[str, str, str], int] = {}
    for row in candidate_rows:
        key = (row["scene"], row["target_scan"], str(row["source_instance_id"]))
        candidate_counts[key] = candidate_counts.get(key, 0) + 1
    complete_pairs = {key for key, count in candidate_counts.items() if count == args.repeats}
    object_keys = {
        (row["scene"], row["target_scan"], str(row["source_instance_id"]))
        for row in object_rows
    }
    complete_pairs &= object_keys
    if complete_pairs:
        candidate_rows = [
            row for row in candidate_rows
            if (row["scene"], row["target_scan"], str(row["source_instance_id"])) in complete_pairs
        ]
        object_rows = [
            row for row in object_rows
            if (row["scene"], row["target_scan"], str(row["source_instance_id"])) in complete_pairs
        ]
        print(f"resuming from {len(complete_pairs)} complete object-target pairs", flush=True)
    else:
        candidate_rows = []
        object_rows = []
    completed_targets = {(row["scene"], row["target_scan"]) for row in object_rows}

    for scene in scenes:
        ann = args.data_root / scene / "gt_segmentation"
        source_path = ann / f"{scene}_0.ply"
        source_points, source_ids, source_classes = read_annotated_ply(source_path)
        targets = sorted(ann.glob(f"{scene}_[1-9]*.ply"))
        for target_path in targets:
            if (scene, target_path.stem) in completed_targets:
                continue
            target_points, target_ids, target_classes = read_annotated_ply(target_path)
            target_cloud, target_features = descriptors(target_points, args.target_voxel)
            target_ids_present = np.unique(target_ids)
            source_ids_present = np.unique(source_ids)
            for raw_id in source_ids_present:
                if raw_id < 0 or raw_id >= 1024:
                    continue
                source_mask = source_ids == raw_id
                source_object = source_points[source_mask]
                if len(source_object) < 300:
                    continue
                source_class = int(np.bincount(source_classes[source_mask].clip(min=0)).argmax())
                target_instance_present = bool(raw_id in target_ids_present)
                if not target_instance_present and not args.include_target_absent:
                    continue
                target_mask = target_ids == raw_id
                target_center = target_points[target_mask].mean(axis=0) if target_instance_present else None
                target_class = (
                    int(np.bincount(target_classes[target_mask].clip(min=0)).argmax())
                    if target_instance_present else None
                )
                same_class_centers = []
                for other_id in target_ids_present:
                    if other_id < 0 or other_id >= 1024:
                        continue
                    other_mask = target_ids == other_id
                    other_class = int(np.bincount(target_classes[other_mask].clip(min=0)).argmax())
                    if other_class == source_class:
                        same_class_centers.append(target_points[other_mask].mean(axis=0))
                same_class_centers_array = np.asarray(same_class_centers, dtype=np.float64).reshape(-1, 3)

                source_cloud, source_features = descriptors(source_object, args.source_voxel)
                if len(source_cloud.points) < 20:
                    continue
                proposals = []
                for repeat in range(args.repeats):
                    o3d.utility.random.seed(stable_seed(scene, target_path.stem, int(raw_id), repeat))
                    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source_cloud,
                        target_cloud,
                        source_features,
                        target_features,
                        True,
                        args.ransac_distance,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                        3,
                        [
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(args.ransac_distance),
                        ],
                        o3d.pipelines.registration.RANSACConvergenceCriteria(
                            args.ransac_iterations, args.ransac_confidence
                        ),
                    )
                    source_center = source_object.mean(axis=0)
                    predicted_center = result.transformation[:3, :3] @ source_center + result.transformation[:3, 3]
                    exact_error = (
                        float(np.linalg.norm(predicted_center - target_center))
                        if target_center is not None else float("inf")
                    )
                    class_error = (
                        float(np.min(np.linalg.norm(same_class_centers_array - predicted_center, axis=1)))
                        if len(same_class_centers_array) else float("inf")
                    )
                    proposal = {
                        "repeat": repeat,
                        "fitness": float(result.fitness),
                        "rmse_m": float(result.inlier_rmse),
                        "exact_instance_center_error_m": exact_error,
                        "same_class_nearest_center_error_m": class_error,
                        "exact_instance_hit": target_instance_present and exact_error < args.pose_threshold and source_class == target_class,
                        "same_class_hit": class_error < args.pose_threshold,
                    }
                    proposals.append(proposal)
                    candidate_rows.append({
                        "scene": scene,
                        "source_scan": source_path.stem,
                        "target_scan": target_path.stem,
                        "source_instance_id": int(raw_id),
                        "source_class_id": source_class,
                        "target_instance_present": target_instance_present,
                        **proposal,
                        **{
                            f"transform_{r}{c}": float(result.transformation[r, c])
                            for r in range(4)
                            for c in range(4)
                        },
                    })
                proposals.sort(key=lambda row: (-float(row["fitness"]), int(row["repeat"])))
                row: dict[str, float | int | str | bool] = {
                    "scene": scene,
                    "source_scan": source_path.stem,
                    "target_scan": target_path.stem,
                    "source_instance_id": int(raw_id),
                    "source_class_id": source_class,
                    "target_instance_present": target_instance_present,
                    "candidate_count": len(proposals),
                }
                for k in (1, 3, 5):
                    top = proposals[:k]
                    row[f"exact_instance_recall_at_{k}"] = any(bool(item["exact_instance_hit"]) for item in top)
                    row[f"same_class_recall_at_{k}"] = any(bool(item["same_class_hit"]) for item in top)
                row["rank1_fitness"] = float(proposals[0]["fitness"])
                row["rank1_exact_center_error_m"] = float(proposals[0]["exact_instance_center_error_m"])
                row["best_exact_center_error_m"] = float(min(float(item["exact_instance_center_error_m"]) for item in proposals))
                object_rows.append(row)
            print(f"completed {target_path.stem}; evaluated_objects={sum(r['target_scan']==target_path.stem and r['scene']==scene for r in object_rows)}", flush=True)

            # Preserve completed scans during long full-dataset runs. The
            # canonical reports below are still written only after completion.
            write_csv(object_path, object_rows)
            write_csv(candidate_path, candidate_rows)
            completed_targets.add((scene, target_path.stem))

    if not object_rows:
        raise RuntimeError("no source objects with matching target identities were evaluated")
    write_csv(args.output_dir / "object_pose_recall.csv", object_rows)
    write_csv(args.output_dir / "pose_candidates.csv", candidate_rows)

    def hit_mean(rows: list[dict[str, object]], key: str) -> float:
        values = []
        for row in rows:
            value = row[key]
            values.append(value if isinstance(value, bool) else str(value).lower() == "true")
        return float(np.mean(values))

    summary: dict[str, object] = {
        "scenes": len(scenes),
        "target_scans": len({(row["scene"], row["target_scan"]) for row in object_rows}),
        "source_object_target_pairs": len(object_rows),
        "candidate_repeats_per_object": args.repeats,
        "ransac_max_iterations": args.ransac_iterations,
        "ransac_confidence": args.ransac_confidence,
        "open3d_version": o3d.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "proposal_rank_key": "Open3D RANSAC fitness, descending",
        "positive_instance_centroid_hit_definition": f"transformed source object centroid within {args.pose_threshold:.2f} m of target instance centroid; same class required for exact-instance hit",
        "metric_scope": "instance-centroid localization recall at the stated threshold; not full 6-DoF pose accuracy",
        "pair_eligibility": (
            "all source instances with sufficient source geometry are included"
            if args.include_target_absent else
            "source raw instance ID must be present in target annotations; all metrics are conditional on annotated target presence"
        ),
        "target_instance_labels_used_for_pair_eligibility": not args.include_target_absent,
        "target_instance_labels_used_for_proposal_generation": False,
        "target_instance_labels_used_for_scoring": True,
        "mean_exact_instance_recall_at_1": hit_mean(object_rows, "exact_instance_recall_at_1"),
        "mean_exact_instance_recall_at_5": hit_mean(object_rows, "exact_instance_recall_at_5"),
        "mean_same_class_recall_at_1": hit_mean(object_rows, "same_class_recall_at_1"),
        "mean_same_class_recall_at_5": hit_mean(object_rows, "same_class_recall_at_5"),
        "per_scene": {},
        "limitations": [
            "geometry-only point-cloud FPFH proposals, not the RGB-D/DINO method",
            "source instance masks are provided; target labels score outcomes and optionally filter to present IDs, but do not rank proposals",
            "independent randomized RANSAC runs form a candidate ensemble and may contain duplicate poses",
            "this is a centroid-localization proposal diagnostic, not an end-to-end segmentation-transfer score",
            "Open3D may fall back to unfiltered feature correspondences when mutual filtering leaves too few",
        ],
    }
    for scene in scenes:
        group = [row for row in object_rows if row["scene"] == scene]
        if group:
            summary["per_scene"][scene] = {
                "object_target_pairs": len(group),
                "exact_instance_recall_at_1": hit_mean(group, "exact_instance_recall_at_1"),
                "exact_instance_recall_at_5": hit_mean(group, "exact_instance_recall_at_5"),
                "same_class_recall_at_1": hit_mean(group, "same_class_recall_at_1"),
                "same_class_recall_at_5": hit_mean(group, "same_class_recall_at_5"),
            }
    (args.output_dir / "fpfh_pose_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rescan FPFH object-centroid localization proposal diagnostic",
        "",
        f"For each source instance, {args.repeats} seeded FPFH/RANSAC runs align its annotated source geometry to the unsegmented target scene. Target labels never generate or rank proposals. {'All eligible source objects are included regardless of target presence.' if args.include_target_absent else 'Evaluation includes only source raw IDs present in the target annotations, so metrics are conditional on annotated target presence.'} Target labels score proposals. Proposals are ranked by RANSAC fitness. A hit requires the transformed source-object centroid to be within {args.pose_threshold:.2f} m of the target instance centroid; exact-instance hits also require the same class. This is centroid-localization recall, not full 6-DoF pose accuracy.",
        "",
        f"- Scene / target scans / object-target pairs: {len(scenes)} / {summary['target_scans']} / {len(object_rows)}",
        f"- RANSAC maximum iterations / confidence: {args.ransac_iterations} / {args.ransac_confidence:.3f}",
        f"- Open3D / OMP_NUM_THREADS: {o3d.__version__} / {os.environ.get('OMP_NUM_THREADS', 'unset')}",
        f"- Exact-identity proposal recall@1 / @5: {summary['mean_exact_instance_recall_at_1']:.3f} / {summary['mean_exact_instance_recall_at_5']:.3f}",
        f"- Same-class proposal recall@1 / @5: {summary['mean_same_class_recall_at_1']:.3f} / {summary['mean_same_class_recall_at_5']:.3f}",
        "",
        "| Scene | Object-target pairs | Exact ID R@1 | Exact ID R@5 | Same-class R@1 | Same-class R@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scene, values in summary["per_scene"].items():
        lines.append(
            f"| {scene} | {values['object_target_pairs']} | {values['exact_instance_recall_at_1']:.3f} | "
            f"{values['exact_instance_recall_at_5']:.3f} | {values['same_class_recall_at_1']:.3f} | {values['same_class_recall_at_5']:.3f} |"
        )
    lines.extend([
        "",
        "Open3D warned that mutual filtering left too few correspondences and fell back to unfiltered feature matches on `scene_b_3`, `scene_e_2`, and `scene_i_2`.",
        "",
        f"The {args.repeats} candidates are repeated stochastic fits, not distinct modes by construction; some may duplicate the same pose. This probe tests whether geometry-only global transforms place the source-object centroid near the annotated target instance and whether fitness ranking resolves identity. It does not measure full 6-DoF pose accuracy, transfer labels, or validate the project's RGB-D/DINO method.",
        "",
    ])
    (args.output_dir / "fpfh_pose_report.md").write_text("\n".join(lines), encoding="utf-8")
    object_path.unlink(missing_ok=True)
    candidate_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
