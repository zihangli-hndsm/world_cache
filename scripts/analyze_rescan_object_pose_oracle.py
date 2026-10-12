#!/usr/bin/env python3
"""Measure an identity-oracle ICP ceiling for Rescan instance transfer.

The shared Rescan instance IDs are used only to pair source and target object
point sets before ICP. This deliberately leaks target instance segmentation and
identity, so the result is an upper-bound diagnostic rather than a method.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from evaluate_rescan_xyz_transfer import (
    UNKNOWN_ID,
    instance_transfer_miou,
    read_annotated_ply,
    rescan_eval_instance_ids,
)


def make_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    return cloud


def evaluate_pair(
    target_path: Path,
    source_points: np.ndarray,
    source_ids_raw: np.ndarray,
    target_points: np.ndarray,
    target_ids_raw: np.ndarray,
    transfer_radius: float,
    voxel_size: float,
    icp_radius: float,
) -> dict[str, float | int | str]:
    source_ids = rescan_eval_instance_ids(source_ids_raw)
    target_ids = rescan_eval_instance_ids(target_ids_raw)
    shared_ids = sorted(
        set(int(v) for v in np.unique(source_ids_raw) if 0 <= v < 1024)
        & set(int(v) for v in np.unique(target_ids_raw) if 0 <= v < 1024)
    )

    moved_points: list[np.ndarray] = []
    moved_ids: list[np.ndarray] = []
    icp_fitness: list[float] = []
    icp_rmse: list[float] = []
    target_tree = None
    for instance_id in shared_ids:
        src_obj = source_points[source_ids_raw == instance_id]
        tgt_obj = target_points[target_ids_raw == instance_id]
        if len(src_obj) < 10 or len(tgt_obj) < 10:
            continue
        source_cloud = make_cloud(src_obj).voxel_down_sample(voxel_size)
        target_cloud = make_cloud(tgt_obj).voxel_down_sample(voxel_size)
        transform = np.eye(4)
        transform[:3, 3] = tgt_obj.mean(axis=0) - src_obj.mean(axis=0)
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            icp_radius,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        aligned = (result.transformation[:3, :3] @ src_obj.T).T + result.transformation[:3, 3]
        eval_id = int(rescan_eval_instance_ids(np.asarray([instance_id]))[0])
        moved_points.append(aligned)
        moved_ids.append(np.full(len(aligned), eval_id, dtype=np.int32))
        icp_fitness.append(float(result.fitness))
        icp_rmse.append(float(result.inlier_rmse))

    if not moved_points:
        raise ValueError(f"no shared instances to align for {target_path}")
    template = np.concatenate(moved_points, axis=0)
    template_ids = np.concatenate(moved_ids)
    distances, nearest = cKDTree(template).query(target_points, k=1, workers=-1)
    prediction = np.where(distances <= transfer_radius, template_ids[nearest], UNKNOWN_ID)
    miou, scored_instances = instance_transfer_miou(
        target_ids, prediction, target_path.with_suffix(".txt")
    )
    return {
        "target_scan": target_path.stem,
        "shared_source_target_instances": len(moved_points),
        "icp_mean_fitness": float(np.mean(icp_fitness)),
        "icp_mean_inlier_rmse_m": float(np.mean(icp_rmse)),
        "transformed_template_point_coverage": float(np.mean(distances <= transfer_radius)),
        "instance_transfer_miou": miou,
        "scored_target_instances": scored_instances,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transfer-radius", type=float, default=0.05)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--icp-radius", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = [
        line.strip()
        for line in (args.data_root / "scene_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, float | int | str]] = []
    per_scene: dict[str, dict[str, float | int]] = {}
    for scene in scenes:
        ann = args.data_root / scene / "gt_segmentation"
        source_path = ann / f"{scene}_0.ply"
        source_points, source_ids_raw, _ = read_annotated_ply(source_path)
        scene_rows: list[dict[str, float | int | str]] = []
        for target_path in sorted(ann.glob(f"{scene}_[1-9]*.ply")):
            target_points, target_ids_raw, _ = read_annotated_ply(target_path)
            row = evaluate_pair(
                target_path,
                source_points,
                source_ids_raw,
                target_points,
                target_ids_raw,
                args.transfer_radius,
                args.voxel_size,
                args.icp_radius,
            )
            row["scene"] = scene
            rows.append(row)
            scene_rows.append(row)
            print(
                f"{target_path.stem}: ICP-oracle mIoU={row['instance_transfer_miou']:.4f} "
                f"fit={row['icp_mean_fitness']:.3f} ids={row['shared_source_target_instances']}",
                flush=True,
            )
        per_scene[scene] = {
            "target_scans": len(scene_rows),
            "mean_instance_transfer_miou": float(np.nanmean([r["instance_transfer_miou"] for r in scene_rows])),
            "mean_icp_fitness": float(np.mean([r["icp_mean_fitness"] for r in scene_rows])),
        }
    with (args.output_dir / "icp_oracle_per_scan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scenes": len(scenes),
        "future_scans": len(rows),
        "pose_source": "ground-truth target instance membership; centroid initialized per-instance ICP",
        "target_labels_used_for_object_association_and_pose": True,
        "target_labels_used_for_pointwise_prediction": False,
        "transfer_radius_m": args.transfer_radius,
        "voxel_size_m": args.voxel_size,
        "icp_radius_m": args.icp_radius,
        "mean_instance_transfer_miou": float(np.nanmean([r["instance_transfer_miou"] for r in rows])),
        "mean_icp_fitness": float(np.mean([r["icp_mean_fitness"] for r in rows])),
        "per_scene": per_scene,
        "interpretation": "Identity-oracle geometric upper bound; not a deployable method or independent test of the RGB-D/DINO pipeline.",
    }
    (args.output_dir / "icp_oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rescan per-object ICP oracle",
        "",
        "For each future scan, the target ground-truth instance ID is used to segment and associate each object with its source geometry. The source object is initialized by the target/source centroid difference, then aligned by point-to-point ICP. Transformed source points transfer labels to the target within 5 cm. Target labels do not choose point predictions, but they provide object segmentation, identity, and pose initialization; this is an oracle ceiling only.",
        "",
        f"- Scenes / future scans: {len(scenes)} / {len(rows)}",
        f"- Mean Instance Transfer mIoU: {summary['mean_instance_transfer_miou']:.4f}",
        f"- Mean per-object ICP fitness: {summary['mean_icp_fitness']:.3f}",
        "",
        "| Scene | Future scans | Oracle Instance Transfer mIoU | ICP fitness |",
        "|---|---:|---:|---:|",
    ]
    for scene, values in per_scene.items():
        lines.append(
            f"| {scene} | {values['target_scans']} | {values['mean_instance_transfer_miou']:.4f} | "
            f"{values['mean_icp_fitness']:.3f} |"
        )
    lines.extend([
        "",
        "This diagnoses whether geometry plus a per-object rigid transform can recover the transfer signal when object identity is known. It does not demonstrate that source-only descriptors or RANSAC can recover the same poses. Future learned-method evaluation should select all settings on scene-level folds and reserve untouched scenes.",
        "",
    ])
    (args.output_dir / "icp_oracle_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
