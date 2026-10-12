#!/usr/bin/env python3
"""Test whether a shared depth focal length improves sparse Rescan-to-mesh fits.

This is a calibration-recoverability diagnostic. It uses geometry only and
does not claim camera-pose ground truth or RGB-depth registration.
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
    parser.add_argument("--mesh-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focals", type=float, nargs="+", default=list(range(400, 1001, 50)))
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=7.0)
    args = parser.parse_args()

    paths = sorted(args.pgm_dir.glob("frame_*.pgm"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(paths) != 6:
        raise ValueError(f"expected six sparse frames, found {len(paths)}")
    o3d.utility.random.seed(args.seed)
    voxel = args.voxel_size
    target = o3d.io.read_point_cloud(str(args.mesh_ply)).voxel_down_sample(voxel)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=40)
    target.estimate_normals(normal_search)
    feature_search = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100)
    target_features = o3d.pipelines.registration.compute_fpfh_feature(target, feature_search)

    results = []
    for focal in args.focals:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, focal, focal, 319.5, 239.5)
        frame_rows = []
        poses = []
        for path in paths:
            source = make_cloud(read_pgm16(path), intrinsic, voxel, args.max_depth_m)
            source.estimate_normals(normal_search)
            source_features = o3d.pipelines.registration.compute_fpfh_feature(source, feature_search)
            global_fit = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source, target, source_features, target_features,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    use_absolute_scale=True,
                    maximum_correspondence_distance=voxel * 2.5,
                ),
            )
            thresholds = []
            for distance in (0.05, 0.10):
                fit = o3d.pipelines.registration.registration_icp(
                    source, target, distance, global_fit.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
                )
                aligned = o3d.geometry.PointCloud(source)
                aligned.transform(fit.transformation)
                nearest = np.asarray(aligned.compute_point_cloud_distance(target))
                thresholds.append({
                    "max_correspondence_m": distance,
                    "fitness": float(fit.fitness),
                    "inlier_rmse_m": float(fit.inlier_rmse),
                    "fraction_within_0_05m": float(np.mean(nearest < 0.05)),
                    "fraction_within_0_10m": float(np.mean(nearest < 0.10)),
                    "median_nearest_distance_m": float(np.median(nearest)),
                    "transform": fit.transformation.tolist(),
                })
            poses.append(np.asarray(thresholds[0]["transform"], dtype=float))
            frame_rows.append({
                "frame_index_in_sampled_sequence": int(path.stem.split("_")[-1]),
                "source_points": len(source.points),
                "fgr_fitness": float(global_fit.fitness),
                "fgr_rmse_m": float(global_fit.inlier_rmse),
                "icp_thresholds": thresholds,
            })
        results.append({"focal_px": focal, "frames": frame_rows})
        print(json.dumps({
            "focal_px": focal,
            "median_fitness_5cm": float(np.median([r["icp_thresholds"][0]["fitness"] for r in frame_rows])),
            "median_fitness_10cm": float(np.median([r["icp_thresholds"][1]["fitness"] for r in frame_rows])),
            "median_rmse_5cm": float(np.median([r["icp_thresholds"][0]["inlier_rmse_m"] for r in frame_rows])),
            "median_within_5cm": float(np.median([r["icp_thresholds"][0]["fraction_within_0_05m"] for r in frame_rows])),
            "median_within_10cm": float(np.median([r["icp_thresholds"][1]["fraction_within_0_10m"] for r in frame_rows])),
        }), flush=True)

    output = {
        "status": "complete",
        "scan": "scene_i_1",
        "sampled_frames": [int(p.stem.split("_")[-1]) for p in paths],
        "method": "Per-frame FPFH/FGR plus ICP to annotated scan mesh, repeated over a shared focal-length grid.",
        "fixed_intrinsics_assumptions": {"resolution": [640, 480], "principal_point": [319.5, 239.5], "fx_equals_fy": True},
        "identity_targets_used": False,
        "camera_poses_used": False,
        "mesh_labels_used": False,
        "registration_seed": args.seed,
        "focal_sweep": results,
        "limits": [
            "A common focal-length sweep does not recover per-frame camera-pose ground truth.",
            "FGR/ICP fits use the annotated mesh geometry and can match repeated structures.",
            "RGB-depth extrinsics and color-frame projection are not measured.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
