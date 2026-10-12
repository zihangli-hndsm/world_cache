#!/usr/bin/env python3
"""Measure how a geometry-only Rescan map-to-mesh fit changes with ICP tolerance.

This is a diagnostic for the accumulated depth-only trajectory, not a pose
validation. The frame intrinsics and sensor-to-mesh relation are uncalibrated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from probe_rescan_depth_odometry import make_cloud, read_pgm16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgm-dir", type=Path, required=True)
    parser.add_argument("--odometry-json", type=Path, required=True)
    parser.add_argument("--mesh-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--fx", type=float, default=571.62)
    parser.add_argument("--fy", type=float, default=571.62)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.075, 0.10, 0.15, 0.20, 0.30])
    args = parser.parse_args()
    if args.frame_step < 1:
        parser.error("--frame-step must be positive")

    odometry = json.loads(args.odometry_json.read_text())
    pair_rows = odometry["sequential_icp"]["pairs"]
    poses = [np.eye(4)]
    for row in pair_rows:
        poses.append(poses[-1] @ np.asarray(row["relative_transform"], dtype=float))
    indices = list(range(0, len(poses), args.frame_step))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        640, 480, args.fx, args.fy, args.cx, args.cy
    )
    source = o3d.geometry.PointCloud()
    for index in indices:
        cloud = make_cloud(
            read_pgm16(args.pgm_dir / f"frame_{index}.pgm"),
            intrinsic,
            args.voxel_size,
            7.0,
        )
        cloud.transform(poses[index])
        source += cloud
    source = source.voxel_down_sample(args.voxel_size)
    target = o3d.io.read_point_cloud(str(args.mesh_ply)).voxel_down_sample(args.voxel_size)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size * 2, max_nn=40
    )
    for cloud in (source, target):
        cloud.estimate_normals(normal_search)
    feature_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size * 5, max_nn=100
    )
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, feature_search)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, feature_search)
    global_fit = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            use_absolute_scale=True,
            maximum_correspondence_distance=args.voxel_size * 2.5,
        ),
    )

    rows = []
    for threshold in args.thresholds:
        fit = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            global_fit.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
        )
        aligned = o3d.geometry.PointCloud(source)
        aligned.transform(fit.transformation)
        distances = np.asarray(aligned.compute_point_cloud_distance(target))
        rows.append({
            "max_correspondence_m": threshold,
            "fitness": fit.fitness,
            "inlier_rmse_m": fit.inlier_rmse,
            "source_distance_p50_m": float(np.quantile(distances, 0.50)),
            "source_distance_p90_m": float(np.quantile(distances, 0.90)),
            "source_fraction_within_0_05m": float(np.mean(distances < 0.05)),
            "source_fraction_within_0_10m": float(np.mean(distances < 0.10)),
            "source_fraction_within_0_20m": float(np.mean(distances < 0.20)),
            "transform": fit.transformation.tolist(),
        })

    result = {
        "status": "complete",
        "frame_count_in_odometry": len(poses),
        "integrated_frame_indices": indices,
        "source_frame_step": args.frame_step,
        "source_points": len(source.points),
        "target_points": len(target.points),
        "mesh_ply": str(args.mesh_ply),
        "intrinsics_assumed": {
            "fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy,
        },
        "fgr_initialization": {
            "fitness": global_fit.fitness,
            "inlier_rmse_m": global_fit.inlier_rmse,
            "transform": global_fit.transformation.tolist(),
        },
        "threshold_sweep": rows,
        "limitations": [
            "Depth intrinsics are assumed, not scan-specific calibration.",
            "The trajectory is sequential depth ICP without ground truth and can drift.",
            "The annotated mesh is used as geometry only; instance labels are not read.",
            "Map-to-mesh fit does not validate per-frame or RGB projection accuracy.",
            "FPFH/FGR initialization may vary between runs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "source_points": result["source_points"],
        "target_points": result["target_points"],
        "fgr_initialization": result["fgr_initialization"],
        "threshold_sweep": [
            {k: row[k] for k in row if k != "transform"} for row in rows
        ],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
