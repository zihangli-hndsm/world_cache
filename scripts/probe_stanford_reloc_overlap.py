#!/usr/bin/env python3
"""Bounded cross-sequence RGB-D geometry overlap check for Stanford Reloc apt1.

Uses only a handful of official depth frames, their released camera-to-world
poses, and the archive's default depth intrinsics. It does not use color,
semantic labels, or physical-instance IDs.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from stream_rescan_archive_subset import HTTPRangeReader


DEFAULT_URL = "https://graphics.stanford.edu/projects/reloc/data/apt1.zip"
SAMPLES = {"kitchen": [100, 500, 900], "living": [120, 750, 1300]}


def parse_info(text: str) -> tuple[int, int, float, float, float, float, float]:
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    w, h = int(values["m_depthWidth"]), int(values["m_depthHeight"])
    intr = [float(x) for x in values["m_calibrationDepthIntrinsic"].split()]
    fx, fy, cx, cy = intr[0], intr[5], intr[2], intr[6]
    return w, h, fx, fy, cx, cy, float(values.get("m_depthShift", "1000"))


def parse_pose(text: str) -> np.ndarray:
    vals = [float(x) for x in text.split()]
    if len(vals) != 16 or not all(math.isfinite(x) for x in vals):
        raise ValueError("pose is missing or non-finite")
    return np.asarray(vals, dtype=np.float64).reshape(4, 4)


def depth_points(png: bytes, intr: tuple[int, int, float, float, float, float, float], pose: np.ndarray) -> np.ndarray:
    w, h, fx, fy, cx, cy, depth_shift = intr
    depth = np.asarray(Image.open(io.BytesIO(png)), dtype=np.float32)
    if depth.shape != (h, w):
        raise ValueError(f"depth size mismatch: {depth.shape} vs {(h, w)}")
    # The official format page says depth is millimetres. depthShift is retained
    # in the report as metadata, but is not applied as a scale factor.
    yy, xx = np.mgrid[0:h:4, 0:w:4]
    z = depth[::4, ::4] * 0.001
    valid = (z > 0.15) & (z < 8.0)
    cam = np.column_stack(((xx[valid] - cx) * z[valid] / fx,
                           (yy[valid] - cy) * z[valid] / fy,
                           z[valid]))
    world = cam @ pose[:3, :3].T + pose[:3, 3]
    return world.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rooms", nargs="+", choices=sorted(SAMPLES), default=sorted(SAMPLES))
    args = parser.parse_args()

    remote = HTTPRangeReader(args.archive_url)
    frames = {}
    intrinsics = {}
    with ZipFile(remote) as archive:
        for room in args.rooms:
            root = f"apt1/{room}"
            intrinsics[room] = parse_info(archive.read(f"{root}/info.txt").decode("utf-8"))
            for frame_id in SAMPLES[room]:
                base = f"{root}/data/frame-{frame_id:06d}"
                pose = parse_pose(archive.read(base + ".pose.txt").decode("utf-8"))
                points = depth_points(archive.read(base + ".depth.png"), intrinsics[room], pose)
                frames[(room, frame_id)] = points

    pairs = []
    for room in args.rooms:
        ids = SAMPLES[room]
        for a, b in zip(ids, ids[1:]):
            pa, pb = frames[(room, a)], frames[(room, b)]
            da = cKDTree(pb).query(pa, k=1)[0]
            db = cKDTree(pa).query(pb, k=1)[0]
            pairs.append({
                "room": room,
                "frames": [a, b],
                "points": [len(pa), len(pb)],
                "directed_within_2cm": [float(np.mean(da < 0.02)), float(np.mean(db < 0.02))],
                "directed_within_5cm": [float(np.mean(da < 0.05)), float(np.mean(db < 0.05))],
                "median_nearest_m": [float(np.median(da)), float(np.median(db))],
            })
    output = {
        "status": "complete",
        "dataset": "Stanford RGB Camera Relocalization Dataset, Apartment 1",
        "archive_url": args.archive_url,
        "archive_bytes": remote.size,
        "range_bytes_fetched": remote.bytes_downloaded,
        "frame_ids": {room: SAMPLES[room] for room in args.rooms},
        "intrinsics_depth": {room: {"width": v[0], "height": v[1], "fx": v[2], "fy": v[3], "cx": v[4], "cy": v[5], "depth_shift_metadata": v[6]} for room, v in intrinsics.items()},
        "pose_convention": "released camera-to-world 4x4 matrices interpreted row-major",
        "depth_units": "millimetres converted to metres",
        "point_stride": 4,
        "pairs": pairs,
        "limits": [
            "The website says room sequences were globally bundle-adjusted, but the pose ground truth is not independently validated here.",
            "The website calls the supplied intrinsics defaults and says no calibration was performed.",
            "This geometry-only probe uses no color, semantic labels, or persistent object identities.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"range_bytes_fetched": remote.bytes_downloaded, "pairs": pairs}, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
