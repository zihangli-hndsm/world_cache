#!/usr/bin/env python3
"""Probe independent global geometry alignment for sparse Rescan depth frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from probe_rescan_depth_odometry import make_cloud, read_pgm16


def rotation_delta_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgm-dir", type=Path, required=True)
    parser.add_argument("--mesh-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fx", type=float, default=575.0)
    parser.add_argument("--fy", type=float, default=575.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=7.0)
    parser.add_argument("--icp-thresholds", type=float, nargs="+", default=[0.05, 0.10])
    args = parser.parse_args()

    paths = sorted(args.pgm_dir.glob("frame_*.pgm"), key=lambda p: int(p.stem.split("_")[-1]))
    if not paths:
        raise FileNotFoundError(f"no frame_*.pgm under {args.pgm_dir}")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        640, 480, args.fx, args.fy, args.cx, args.cy
    )
    target = o3d.io.read_point_cloud(str(args.mesh_ply)).voxel_down_sample(args.voxel_size)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size * 2, max_nn=40
    )
    target.estimate_normals(normal_search)
    feature_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size * 5, max_nn=100
    )
    target_features = o3d.pipelines.registration.compute_fpfh_feature(target, feature_search)

    rows = []
    fitted_poses = []
    for path in paths:
        source = make_cloud(read_pgm16(path), intrinsic, args.voxel_size, args.max_depth_m)
        source.estimate_normals(normal_search)
        source_features = o3d.pipelines.registration.compute_fpfh_feature(source, feature_search)
        global_fit = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source,
            target,
            source_features,
            target_features,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                use_absolute_scale=True,
                maximum_correspondence_distance=args.voxel_size * 2.5,
            ),
        )
        sweep = []
        for threshold in args.icp_thresholds:
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
            sweep.append({
                "max_correspondence_m": threshold,
                "fitness": fit.fitness,
                "inlier_rmse_m": fit.inlier_rmse,
                "fraction_within_0_05m": float(np.mean(distances < 0.05)),
                "fraction_within_0_10m": float(np.mean(distances < 0.10)),
                "median_distance_m": float(np.median(distances)),
                "transform": fit.transformation.tolist(),
            })
        selected = next((row for row in sweep if row["max_correspondence_m"] == 0.05), sweep[0])
        pose = np.asarray(selected["transform"], dtype=float)
        fitted_poses.append(pose)
        rows.append({
            "frame_index_in_sampled_sequence": int(path.stem.split("_")[-1]),
            "source_points": len(source.points),
            "fgr_fitness": global_fit.fitness,
            "fgr_rmse_m": global_fit.inlier_rmse,
            "icp_threshold_sweep": sweep,
        })

    for index in range(1, len(rows)):
        previous = fitted_poses[index - 1]
        current = fitted_poses[index]
        rows[index]["independent_fit_step_translation_m"] = float(
            np.linalg.norm(current[:3, 3] - previous[:3, 3])
        )
        rows[index]["independent_fit_step_rotation_deg"] = rotation_delta_degrees(previous, current)

    result = {
        "status": "complete",
        "frame_count": len(rows),
        "mesh_ply": str(args.mesh_ply),
        "intrinsics_assumed": {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy},
        "target_points": len(target.points),
        "voxel_size_m": args.voxel_size,
        "method": "Independent FPFH/FGR initialization and point-to-plane ICP to the full mesh for each sparse depth frame.",
        "frames": rows,
        "limitations": [
            "Intrinsics are an assumption, not a scan-specific calibration.",
            "The geometric mesh fit uses no instance identities and does not validate RGB-depth extrinsics.",
            "Independent fits may jump between repeated geometric structures; they are not a continuous camera trajectory.",
            "FGR initialization may vary between runs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "frame_count": result["frame_count"],
        "target_points": result["target_points"],
        "frames": [{
            "frame_index_in_sampled_sequence": row["frame_index_in_sampled_sequence"],
            "source_points": row["source_points"],
            "fgr_fitness": row["fgr_fitness"],
            "fgr_rmse_m": row["fgr_rmse_m"],
            "selected_icp": {k: v for k, v in row["icp_threshold_sweep"][0].items()
                              if k != "transform"},
            "independent_fit_step_translation_m": row.get("independent_fit_step_translation_m"),
            "independent_fit_step_rotation_deg": row.get("independent_fit_step_rotation_deg"),
        } for row in rows],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
