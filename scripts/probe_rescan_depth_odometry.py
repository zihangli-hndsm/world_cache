#!/usr/bin/env python3
"""Probe sequential depth ICP on a bounded Rescan frame prefix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d


def read_pgm16(path: Path) -> np.ndarray:
    data = path.read_bytes()
    tokens: list[bytes] = []
    pos = 0
    while len(tokens) < 4:
        while pos < len(data) and data[pos] in b" \t\r\n":
            pos += 1
        if pos < len(data) and data[pos] == ord("#"):
            end = data.find(b"\n", pos)
            if end < 0:
                raise ValueError(f"unterminated PGM comment in {path}")
            pos = end + 1
            continue
        end = pos
        while end < len(data) and data[end] not in b" \t\r\n":
            end += 1
        if end == pos:
            raise ValueError(f"malformed PGM header in {path}")
        tokens.append(data[pos:end])
        pos = end
    if tokens[0] != b"P5" or int(tokens[3]) != 65535:
        raise ValueError(f"expected binary 16-bit PGM in {path}; header={tokens}")
    width, height = int(tokens[1]), int(tokens[2])
    # Skip exactly the whitespace separator after maxval (comments were handled above).
    while pos < len(data) and data[pos] in b" \t\r\n":
        pos += 1
        break
    pixels = np.frombuffer(data, dtype=">u2", offset=pos)
    if pixels.size != width * height:
        raise ValueError(f"PGM pixel count mismatch in {path}: {pixels.size}")
    return pixels.reshape(height, width).astype(np.uint16)


def make_cloud(depth: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic,
               voxel: float, depth_trunc: float) -> o3d.geometry.PointCloud:
    image = o3d.geometry.Image(depth)
    cloud = o3d.geometry.PointCloud.create_from_depth_image(
        image, intrinsic, depth_scale=1000.0, depth_trunc=depth_trunc, stride=3
    ).voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    return cloud


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgm-dir", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--mesh-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fx", type=float, default=571.62)
    parser.add_argument("--fy", type=float, default=571.62)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=7.0)
    parser.add_argument("--max-correspondence-m", type=float, default=0.10)
    args = parser.parse_args()

    paths = [args.pgm_dir / f"frame_{i}.pgm" for i in range(args.frame_count)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing decoded depth frames: {missing[:3]}")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, args.fx, args.fy, args.cx, args.cy)
    clouds = [make_cloud(read_pgm16(p), intrinsic, args.voxel_size, args.max_depth_m) for p in paths]

    poses = [np.eye(4)]
    rows = []
    for index, (previous, current) in enumerate(zip(clouds, clouds[1:]), start=1):
        result = o3d.pipelines.registration.registration_icp(
            current, previous, args.max_correspondence_m, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        relative = result.transformation
        pose = poses[-1] @ relative
        poses.append(pose)
        rows.append({
            "frame": index,
            "fitness": result.fitness,
            "inlier_rmse_m": result.inlier_rmse,
            "relative_translation_m": float(np.linalg.norm(relative[:3, 3])),
            "cumulative_translation_m": float(np.linalg.norm(pose[:3, 3])),
            "relative_transform": relative.tolist(),
        })

    aligned = o3d.geometry.PointCloud()
    for cloud, pose in zip(clouds, poses):
        current = o3d.geometry.PointCloud(cloud)
        current.transform(pose)
        aligned += current
    aligned = aligned.voxel_down_sample(args.voxel_size)
    target = o3d.io.read_point_cloud(str(args.mesh_ply)).voxel_down_sample(args.voxel_size)
    for cloud in (aligned, target):
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.voxel_size * 2, max_nn=40
        ))
    f1 = o3d.pipelines.registration.compute_fpfh_feature(
        aligned, o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 5, max_nn=100)
    )
    f2 = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 5, max_nn=100)
    )
    global_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        aligned, target, f1, f2,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            use_absolute_scale=True,
            maximum_correspondence_distance=args.voxel_size * 2.5,
        ),
    )
    refined = o3d.pipelines.registration.registration_icp(
        aligned, target, args.voxel_size * 2.5, global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    result = {
        "status": "complete",
        "frame_count": len(paths),
        "mesh_ply": str(args.mesh_ply),
        "intrinsics_assumed": {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy},
        "depth_scale_assumed": "millimeters",
        "identity_targets_read": False,
        "mesh_instance_labels_used": False,
        "sequential_icp": {
            "pair_count": len(rows),
            "median_fitness": float(np.median([r["fitness"] for r in rows])) if rows else None,
            "p05_fitness": float(np.quantile([r["fitness"] for r in rows], 0.05)) if rows else None,
            "median_rmse_m": float(np.median([r["inlier_rmse_m"] for r in rows])) if rows else None,
            "failed_pairs_fitness_below_0_1": sum(r["fitness"] < 0.1 for r in rows),
            "trajectory_endpoint_m": poses[-1][:3, 3].tolist(),
            "trajectory_path_length_m": float(sum(r["relative_translation_m"] for r in rows)),
            "pairs": rows,
        },
        "integrated_cloud_points": len(aligned.points),
        "mesh_cloud_points": len(target.points),
        "integrated_cloud_to_mesh_registration": {
            "fgr_fitness": global_result.fitness,
            "fgr_rmse_m": global_result.inlier_rmse,
            "icp_fitness": refined.fitness,
            "icp_rmse_m": refined.inlier_rmse,
            "transform": refined.transformation.tolist(),
        },
        "limitations": [
            "Intrinsics are an initialization, not scan-specific calibration.",
            "Sequential ICP has no ground-truth trajectory and can drift.",
            "Global mesh alignment uses geometry only and is not proof of per-frame projection accuracy.",
            "RGB-depth extrinsics and color-frame projections are not evaluated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "frame_count": result["frame_count"],
                      "sequential_icp": {k: v for k, v in result["sequential_icp"].items() if k != "pairs"},
                      "integrated_cloud_to_mesh_registration": result["integrated_cloud_to_mesh_registration"],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
