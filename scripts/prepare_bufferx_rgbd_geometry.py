#!/usr/bin/env python3
"""Prepare dense native-scan geometry from the same saved RGB-D observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def backproject(observation: dict[str, np.ndarray], stride: int) -> np.ndarray:
    depth = observation["depth"]
    intrinsics = observation["intrinsics"]
    pose = observation["camera_to_world"]
    if depth.ndim != 2 or intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("invalid RGB-D geometry shapes")
    if not np.isfinite(intrinsics).all() or not np.isfinite(pose).all() or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0:
        raise ValueError("invalid camera calibration")
    yy, xx = np.meshgrid(np.arange(0, depth.shape[0], stride), np.arange(0, depth.shape[1], stride), indexing="ij")
    z = depth[yy, xx].reshape(-1)
    valid = np.isfinite(z) & (z > 0)
    z, xx, yy = z[valid], xx.reshape(-1)[valid], yy.reshape(-1)[valid]
    camera = np.stack(((xx-intrinsics[0, 2])*z/intrinsics[0, 0],
                       (yy-intrinsics[1, 2])*z/intrinsics[1, 1], z), axis=1)
    native = camera @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(native).all():
        raise ValueError("nonfinite backprojected coordinates")
    return native.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=2)
    args = parser.parse_args()
    if args.pixel_stride < 1:
        raise ValueError("pixel stride must be positive")
    config = json.loads((args.pair_run / "config.json").read_text())
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("native-scan RGB-D observations required; do not use hidden cross-scan alignment")
    manifest = json.loads((args.pair_run / "manifest.json").read_text())
    clouds = {}
    frames = {}
    for arm in ("reference", "rescan"):
        values, records = [], []
        for entry in manifest[arm]:
            path = args.pair_run / entry["path"]
            with np.load(path) as archive:
                observation = {k: archive[k] for k in ("depth", "intrinsics", "camera_to_world")}
            points = backproject(observation, args.pixel_stride)
            values.append(points)
            records.append({"frame_id": entry["frame_id"], "path": str(path),
                            "points": len(points), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if not values:
            raise ValueError(f"no {arm} observations")
        clouds[arm + "_points"] = np.concatenate(values)
        frames[arm] = records
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "native_geometry.npz", **clouds)
    result = {"status": "complete", "pair_run": str(args.pair_run), "reference": config["reference"],
              "rescan": config["rescan"], "coordinate_convention": "native_scan_frames",
              "pixel_stride": args.pixel_stride, "points": {k: len(v) for k, v in clouds.items()},
              "frames": frames, "annotation_or_cross_scan_transform_used": False,
              "scope": "denser geometry from the same stored observations; upstream adaptive voxelization still required"}
    (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
