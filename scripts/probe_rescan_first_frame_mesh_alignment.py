#!/usr/bin/env python3
"""Probe whether one Rescan depth frame registers stably to its scan mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


INTRINSIC_HYPOTHESES = [
    (500.0, 500.0, 320.0, 240.0),
    (575.0, 575.0, 320.0, 240.0),
    (650.0, 650.0, 320.0, 240.0),
    (575.0, 575.0, 319.5, 239.5),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-raw-uint16", type=Path, required=True)
    parser.add_argument("--mesh-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-depth-m", type=float, default=7.0)
    args = parser.parse_args()

    raw = np.fromfile(args.depth_raw_uint16, dtype=np.uint16)
    if raw.size != 640 * 480:
        raise ValueError(f"expected one 640x480 uint16 frame, got {raw.size} values")
    image = o3d.geometry.Image(raw.reshape(480, 640).copy())
    target = o3d.io.read_point_cloud(str(args.mesh_ply))
    target = target.voxel_down_sample(args.voxel_size)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel_size * 2, max_nn=40
    ))
    results = []
    for fx, fy, cx, cy in INTRINSIC_HYPOTHESES:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)
        source = o3d.geometry.PointCloud.create_from_depth_image(
            image, intrinsic, depth_scale=1000.0,
            depth_trunc=args.max_depth_m, stride=2,
        ).voxel_down_sample(args.voxel_size)
        result = o3d.pipelines.registration.registration_icp(
            source, target, args.voxel_size * 3.33, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        transform = result.transformation
        angle = np.degrees(np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1)))
        results.append({
            "intrinsics_hypothesis": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "source_points": len(source.points),
            "fitness": result.fitness,
            "inlier_rmse_m": result.inlier_rmse,
            "icp_translation_m": transform[:3, 3].tolist(),
            "icp_rotation_angle_deg": float(angle),
            "transform": transform.tolist(),
        })
    output = {
        "status": "complete",
        "depth_frame": str(args.depth_raw_uint16),
        "scan_mesh": str(args.mesh_ply),
        "depth_resolution_assumed": [640, 480],
        "depth_scale_assumed": "millimeters",
        "identity_targets_read": False,
        "mesh_instance_labels_used": False,
        "voxel_size_m": args.voxel_size,
        "initial_transform": "identity for every ICP hypothesis",
        "intrinsic_hypotheses": results,
        "interpretation_limit": (
            "This is an exploratory single-frame registration probe. Low or unstable fitness/transforms "
            "do not establish a valid trajectory; mesh color, object labels, and cross-scan identities "
            "are not used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "hypotheses": results,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
