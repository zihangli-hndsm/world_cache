#!/usr/bin/env python3
"""Check color/depth frame counts and embedded timestamps for two Rescan scans."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def depth_stream_metadata(path: Path) -> dict:
    timestamps_offset = 0
    frame_count = 0
    file_bytes = path.stat().st_size
    with path.open("rb") as stream:
        while True:
            size_bytes = stream.read(4)
            if not size_bytes:
                timestamps_offset = stream.tell()
                break
            if len(size_bytes) != 4:
                raise ValueError(f"truncated frame-size field in {path}")
            payload_size = struct.unpack("<I", size_bytes)[0]
            if payload_size == 0:
                raise ValueError(f"zero-sized depth frame in {path}")
            payload = stream.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError(f"truncated depth payload in {path}")
            frame_count += 1
            timestamps_offset = stream.tell()
            if file_bytes - timestamps_offset == frame_count * 16:
                break

        remaining = file_bytes - timestamps_offset
        if remaining != frame_count * 16:
            raise ValueError(
                f"timestamp trailer mismatch in {path}: {remaining} bytes for {frame_count} frames"
            )
        stream.seek(timestamps_offset)
        raw = stream.read(remaining)
    timestamps = struct.unpack("<" + "d" * (2 * frame_count), raw)
    depth_ts, color_ts = timestamps[:frame_count], timestamps[frame_count:]

    def series_stats(values: tuple[float, ...]) -> dict:
        deltas = [b - a for a, b in zip(values, values[1:])]
        sorted_deltas = sorted(deltas)
        return {
            "start_seconds": values[0],
            "end_seconds": values[-1],
            "strictly_increasing": all(delta > 0 for delta in deltas),
            "median_interval_ms": statistics.median(deltas) * 1000 if deltas else None,
            "interval_p05_ms": sorted_deltas[max(0, int(0.05 * (len(sorted_deltas) - 1)))] * 1000 if deltas else None,
            "interval_p95_ms": sorted_deltas[min(len(sorted_deltas) - 1, int(0.95 * (len(sorted_deltas) - 1)))] * 1000 if deltas else None,
        }

    skew = [abs(a - b) * 1000 for a, b in zip(depth_ts, color_ts)]
    return {
        "frame_count": frame_count,
        "file_bytes": file_bytes,
        "sha256": file_sha256(path),
        "timestamp_bytes": remaining,
        "depth_timestamps": series_stats(depth_ts),
        "color_timestamps": series_stats(color_ts),
        "color_depth_skew_ms": {
            "median": statistics.median(skew),
            "p95": sorted(skew)[int(0.95 * (len(skew) - 1))],
            "max": max(skew),
        },
    }


def color_stream_metadata(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(proc.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_read_frames"]),
        "file_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.extraction_manifest.read_text())
    root = args.extraction_manifest.parent
    scans = {}
    for scan in manifest["scans"]:
        color_path = root / manifest["scene"] / "color" / f"{scan}.h264"
        depth_path = root / manifest["scene"] / "depth" / f"{scan}.depth"
        color = color_stream_metadata(color_path)
        depth = depth_stream_metadata(depth_path)
        scans[scan] = {
            "color": color,
            "depth": depth,
            "frame_count_match": color["frame_count"] == depth["frame_count"],
            "depth_resolution_assumed_width_height": [640, 480],
            "color_depth_resolutions_differ": (color["width"], color["height"]) != (640, 480),
        }
    result = {
        "status": "complete",
        "scene": manifest["scene"],
        "scans": scans,
        "scope_limit": (
            "Equal color/depth counts and low embedded timestamp skew support frame-order pairing. "
            "They do not establish pixel registration, camera intrinsics, sensor extrinsics, "
            "per-frame camera poses, or mesh projection accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
