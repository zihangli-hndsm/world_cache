#!/usr/bin/env python3
"""Range-inspect Rescan stream headers without downloading a sequence file."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from stream_rescan_archive_subset import HTTPRangeReader


URL = "https://rescan.cs.princeton.edu/assets/rescan_dataset.zip"
COLOR_MEMBER = "scene_a/color/scene_a_0.h264"
DEPTH_MEMBER = "scene_a/depth/scene_a_0.depth"
PREFIX_BYTES = 2 * 1024 * 1024


def annexe_b_nals(data: bytes) -> list[bytes]:
    starts: list[tuple[int, int]] = []
    index = 0
    while index < len(data) - 3:
        if data[index:index + 4] == b"\0\0\0\1":
            starts.append((index, 4))
            index += 4
        elif data[index:index + 3] == b"\0\0\1":
            starts.append((index, 3))
            index += 3
        else:
            index += 1
    result = []
    for position, (start, size) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(data)
        nal = data[start + size:end]
        if nal:
            result.append(nal)
    return result


def ffprobe(path: Path, format_name: str) -> dict[str, object]:
    command = [
        "ffprobe", "-v", "error", "-f", format_name, "-show_streams",
        "-select_streams", "v:0", "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except FileNotFoundError:
        return {"available": False, "error": "ffprobe not installed"}
    except subprocess.TimeoutExpired:
        return {"available": True, "error": "timeout"}
    try:
        streams = json.loads(proc.stdout).get("streams", [])
    except json.JSONDecodeError:
        streams = []
    selected = streams[0] if streams else {}
    return {
        "available": True,
        "returncode": proc.returncode,
        "codec_name": selected.get("codec_name"),
        "width": selected.get("width"),
        "height": selected.get("height"),
        "stderr": proc.stderr[:500],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-url", default=URL)
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/20261006_rescan_rawrgbd_feasibility/stream_probe.json"))
    args = parser.parse_args()

    remote = HTTPRangeReader(args.archive_url)
    with ZipFile(remote) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        index_bytes = remote.bytes_downloaded
        names = [item.filename for item in members]
        pose_like = [name for name in names if re.search(r"pose|traj|intrin|camera|calib", name, re.I)]
        by_name = {item.filename: item for item in members}
        scene_a = {
            key: {"compressed_bytes": by_name[key].compress_size, "uncompressed_bytes": by_name[key].file_size}
            for key in (COLOR_MEMBER, DEPTH_MEMBER)
        }
        with archive.open(by_name[COLOR_MEMBER]) as stream:
            color_head = stream.read(PREFIX_BYTES)
        with archive.open(by_name[DEPTH_MEMBER]) as stream:
            depth_head = stream.read(PREFIX_BYTES)
        nals = annexe_b_nals(color_head)
        types = Counter(nal[0] & 0x1F for nal in nals)
        suffix_bytes = remote.bytes_downloaded

    with tempfile.TemporaryDirectory(prefix="rescan_stream_probe_") as temporary:
        temp = Path(temporary)
        color_path = temp / "color.h264"
        depth_path = temp / "depth.prefix"
        color_path.write_bytes(color_head)
        depth_path.write_bytes(depth_head)
        color_info = ffprobe(color_path, "h264")
        depth_info = {name: ffprobe(depth_path, name) for name in ("h264", "hevc", "mpegvideo")}

    result = {
        "archive_url": args.archive_url,
        "archive_bytes": remote.size,
        "archive_files": len(members),
        "root_entries": [name for name in names if "/" not in name],
        "extensions": sorted({name.rsplit("/", 1)[-1].rsplit(".", 1)[-1] for name in names}),
        "pose_camera_metadata_named_files": pose_like,
        "scene_a_scan0_stream_sizes": scene_a,
        "prefix_bytes_each": PREFIX_BYTES,
        "http_range_bytes_for_zip_index_and_prefixes": suffix_bytes,
        "http_range_bytes_after_zip_index": suffix_bytes - index_bytes,
        "color_h264_annex_b": {
            "nal_count": len(nals),
            "nal_type_counts": {str(key): int(value) for key, value in sorted(types.items())},
            "sei_nal_count": int(types.get(6, 0)),
            "vcl_nal_count": int(sum(value for key, value in types.items() if key in (1, 5))),
        },
        "ffprobe_color": color_info,
        "ffprobe_depth": depth_info,
        "temporary_prefixes_deleted": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
