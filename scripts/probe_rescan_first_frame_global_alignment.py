#!/usr/bin/env python3
"""Try geometry-only global alignment of a Rescan depth frame to its mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-u16", type=Path, required=True,
                        help="raw 640x480 uint16 depth image in millimeters")
    parser.add_argument("--mesh-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--focals", type=float, nargs="+",
                        default=[500, 575, 650, 800])
    args = parser.parse_args()

    raw = np.fromfile(args.depth_u16, dtype="<u2")
    if raw.size != 640 * 480:
        raise ValueError(f"expected 640x480 uint16 depth; found {raw.size} values")
    depth = raw.reshape(480, 640)
    target = o3d.io.read_point_cloud(str(args.mesh_ply)).voxel_down_sample(args.voxel_size)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size * 2, max_nn=40
    )
    target.estimate_normals(normal_search)
    feature_search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size * 5, max_nn=100
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, feature_search)

    results = []
    for focal in args.focals:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            640, 480, focal, focal, 319.5, 239.5
        )
        source = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth), intrinsic, depth_scale=1000.0,
            depth_trunc=7.0, stride=3,
        ).voxel_down_sample(args.voxel_size)
        source.estimate_normals(normal_search)
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source, feature_search
        )
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
        for threshold in (0.05, 0.10, 0.20):
            fit = o3d.pipelines.registration.registration_icp(
                source, target, threshold, global_fit.transformation,
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
                "fraction_within_0_05m": float(np.mean(distances < 0.05)),
                "fraction_within_0_10m": float(np.mean(distances < 0.10)),
                "median_distance_m": float(np.median(distances)),
                "transform": fit.transformation.tolist(),
            })
        results.append({
            "focal_px": focal,
            "source_points": len(source.points),
            "fgr_fitness": global_fit.fitness,
            "fgr_rmse_m": global_fit.inlier_rmse,
            "fgr_transform": global_fit.transformation.tolist(),
            "icp_threshold_sweep": rows,
        })

    output = {
        "status": "complete",
        "depth_frame": str(args.depth_u16),
        "mesh_ply": str(args.mesh_ply),
        "target_points": len(target.points),
        "voxel_size_m": args.voxel_size,
        "method": "FPFH/FGR global initialization followed by point-to-plane ICP; geometry only, no identity labels.",
        "hypotheses": results,
        "limitations": [
            "Focal lengths are hypotheses; scan-specific intrinsics are unavailable.",
            "A high geometric fit on one frame would not establish temporal trajectory or RGB-depth extrinsics.",
            "FGR may vary between runs and repeated room geometry can produce ambiguous fits.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"],
        "target_points": output["target_points"],
        "hypotheses": [{
            "focal_px": row["focal_px"],
            "source_points": row["source_points"],
            "fgr_fitness": row["fgr_fitness"],
            "fgr_rmse_m": row["fgr_rmse_m"],
            "icp_threshold_sweep": [
                {k: sweep[k] for k in sweep if k != "transform"}
                for sweep in row["icp_threshold_sweep"]
            ],
        } for row in results],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
