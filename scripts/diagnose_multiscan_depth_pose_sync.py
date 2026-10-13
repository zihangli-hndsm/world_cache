#!/usr/bin/env python3
"""Diagnostic only: test small RGB-pose/depth index offsets on MultiScan."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d

from download_multiscan_pilot_pair import FILES, REVISION
from audit_multiscan_pilot_annotations import unique_member
from audit_multiscan_rgbd_projection import load_frames, read_mesh, scaled_intrinsic_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/multiscan_scene_00069"))
    parser.add_argument("--max-offset", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scans = []
    for rel in sorted(FILES):
        scan_id = Path(rel).stem
        with zipfile.ZipFile(args.data_dir / Path(rel).name) as archive:
            info, poses, depths, confidences = load_frames(archive, scan_id)
            vertices, triangles, _ = read_mesh(archive, scan_id)
            alignment = json.loads(archive.read(unique_member(archive, scan_id + ".align.json")))
        T = np.asarray(alignment["coordinate_transform"], dtype=np.float64).reshape(4, 4, order="F")
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        mesh = o3d.t.geometry.TriangleMesh(
            o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        h, w = info["depth"]["resolution"]
        color_h, color_w = info["color"]["resolution"]
        sx, sy = w / color_w, h / color_h
        yy, xx = np.mgrid[0:h, 0:w]
        frame_ids = sorted(set([0, info["count"] // 2, info["count"] - 1]))
        frame_results = []
        for depth_id in frame_ids:
            candidate_rows = []
            obs = depths[depth_id]
            valid = np.isfinite(obs) & (obs > 0) & (obs < 8) & (confidences[depth_id] == 2)
            for offset in range(-args.max_offset, args.max_offset + 1):
                pose_id = depth_id + offset
                if not 0 <= pose_id < info["count"]:
                    continue
                pose = poses[pose_id]
                T_arkit = np.asarray(pose["transform"], dtype=np.float64).reshape(4, 4, order="F")
                T_cw = T_arkit @ np.diag([1.0, -1.0, -1.0, 1.0])
                fx, fy, cx, cy = scaled_intrinsic_parameters(pose["intrinsics"], sx, sy)
                dcam = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1).reshape(-1, 3)
                dworld = dcam @ T_cw[:3, :3].T
                origins = np.broadcast_to(T_cw[:3, 3], dworld.shape)
                rays = np.concatenate((origins, dworld), axis=1).astype(np.float32)
                hit = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy().reshape(h, w)
                common = valid & np.isfinite(hit) & (hit > 0)
                residual = np.abs(hit[common] - obs[common])
                candidate_rows.append({
                    "pose_offset_frames": offset,
                    "pose_frame_id": pose_id,
                    "overlap_pixels": int(common.sum()),
                    "median_abs_depth_residual_m": float(np.median(residual)) if residual.size else None,
                    "within_5cm_fraction": float(np.mean(residual <= 0.05)) if residual.size else None,
                })
            best = min(candidate_rows, key=lambda r: r["median_abs_depth_residual_m"] if r["median_abs_depth_residual_m"] is not None else float("inf"))
            frame_results.append({"depth_frame_id": depth_id, "best_offset_diagnostic_only": best, "offset_sweep": candidate_rows})
            print(f"{scan_id} depth {depth_id}: best offset {best['pose_offset_frames']:+d}, median abs error {best['median_abs_depth_residual_m']:.3f}m", flush=True)
        scans.append({"scan_id": scan_id, "sampled_depth_frame_ids": frame_ids, "frames": frame_results})
    output = args.output or args.data_dir / "pose_sync_diagnostic.json"
    result = {
        "status": "diagnostic_complete",
        "dataset_revision": REVISION,
        "scene_id": "scene_00069",
        "max_pose_offset_frames": args.max_offset,
        "depth_pixels_require_confidence_2": True,
        "purpose": "diagnose possible RGB-pose/depth frame-index offset only; not used to select or score held-out validation frames",
        "scans": scans,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
