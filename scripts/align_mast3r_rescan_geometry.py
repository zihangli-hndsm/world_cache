#!/usr/bin/env python3
"""Align an RGB-only MASt3R cloud to Rescan scan geometry (XYZ only)."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", required=True, type=Path)
    ap.add_argument("--mesh-ply", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--voxel", type=float, default=0.04)
    args = ap.parse_args()
    pred = np.load(args.prediction)
    xyz = pred["xyz"].astype(np.float64)
    confidence = pred["confidence"]
    xyz = xyz[np.isfinite(xyz).all(1) & np.isfinite(confidence)]
    source_raw = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    source = source_raw.voxel_down_sample(args.voxel)

    # Use XYZ from the PLY only. Vertex instance/class properties are ignored.
    target_raw = o3d.io.read_point_cloud(str(args.mesh_ply), remove_nan_points=True,
                                         remove_infinite_points=True)
    target = target_raw.voxel_down_sample(args.voxel)
    normal_search = o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2.0, max_nn=40)
    feature_search = o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 5.0, max_nn=100)
    source.estimate_normals(normal_search)
    target.estimate_normals(normal_search)
    sf = o3d.pipelines.registration.compute_fpfh_feature(source, feature_search)
    tf = o3d.pipelines.registration.compute_fpfh_feature(target, feature_search)
    reg = o3d.pipelines.registration
    init = reg.registration_fgr_based_on_feature_matching(
        source, target, sf, tf,
        reg.FastGlobalRegistrationOption(use_absolute_scale=True,
                                         maximum_correspondence_distance=args.voxel * 2.5))
    rows = []
    for distance in (0.05, 0.10, 0.20):
        fit = reg.registration_icp(
            source, target, distance, init.transformation,
            reg.TransformationEstimationPointToPoint(),
            reg.ICPConvergenceCriteria(max_iteration=100))
        aligned = o3d.geometry.PointCloud(source)
        aligned.transform(fit.transformation)
        d = np.asarray(aligned.compute_point_cloud_distance(target))
        rows.append({
            "max_correspondence_m": distance,
            "fitness": float(fit.fitness),
            "inlier_rmse_m": float(fit.inlier_rmse),
            "fraction_within_0_05m": float(np.mean(d < .05)),
            "fraction_within_0_10m": float(np.mean(d < .10)),
            "median_source_to_target_distance_m": float(np.median(d)),
            "p90_source_to_target_distance_m": float(np.quantile(d, .9)),
            "transform": fit.transformation.tolist(),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "RGB-only MASt3R sparse global alignment; XYZ-only FPFH/FGR+ICP geometry check",
        "source_raw_points": int(len(source_raw.points)),
        "source_voxel_points": int(len(source.points)),
        "target_raw_points": int(len(target_raw.points)),
        "target_voxel_points": int(len(target.points)),
        "voxel_m": args.voxel,
        "fgr_fitness": float(init.fitness),
        "fgr_rmse_m": float(init.inlier_rmse),
        "icp": rows,
        "interpretation": "Feasibility only; ICP scores are against target geometry and do not validate RGB-D projection or object identity.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "icp"}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
