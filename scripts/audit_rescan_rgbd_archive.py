#!/usr/bin/env python3
"""Inventory the public Rescan ZIP central directory without downloading scans."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from stream_rescan_archive_subset import DEFAULT_URL, HTTPRangeReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    remote = HTTPRangeReader(args.archive_url)
    with ZipFile(remote) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]

    by_scene: dict[str, dict] = defaultdict(lambda: {
        "member_count": 0,
        "compressed_bytes": 0,
        "uncompressed_bytes": 0,
        "types": defaultdict(lambda: {
            "member_count": 0,
            "compressed_bytes": 0,
            "uncompressed_bytes": 0,
        }),
        "members": [],
    })
    other = []
    for info in files:
        path = PurePosixPath(info.filename)
        if len(path.parts) >= 3 and path.parts[0].startswith("scene_"):
            scene, modality = path.parts[:2]
            ext = path.suffix.lower() or "[none]"
            entry = by_scene[scene]
            entry["member_count"] += 1
            entry["compressed_bytes"] += info.compress_size
            entry["uncompressed_bytes"] += info.file_size
            typ = entry["types"][f"{modality}{ext}"]
            typ["member_count"] += 1
            typ["compressed_bytes"] += info.compress_size
            typ["uncompressed_bytes"] += info.file_size
            entry["members"].append({
                "name": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
            })
        else:
            other.append({
                "name": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
            })

    scenes = {}
    for scene, entry in sorted(by_scene.items()):
        entry["types"] = dict(sorted(entry["types"].items()))
        entry["members"].sort(key=lambda item: item["name"])
        scenes[scene] = entry
    result = {
        "status": "complete",
        "source_url": args.archive_url,
        "archive_bytes": remote.size,
        "archive_etag": remote.etag,
        "zip_member_count_including_directories": len(files) + sum(
            1 for info in archive.infolist() if info.is_dir()
        ),
        "file_member_count": len(files),
        "central_directory_bytes_fetched": remote.bytes_downloaded,
        "scene_count": len(scenes),
        "file_bytes_downloaded": 0,
        "scenes": scenes,
        "non_scene_members": other,
        "pose_or_intrinsics_member_names": [
            item.filename for item in files
            if any(token in item.filename.lower() for token in
                   ("pose", "intrinsic", "calib", "trajectory", "transform"))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "archive_bytes": result["archive_bytes"],
        "file_member_count": result["file_member_count"],
        "central_directory_bytes_fetched": result["central_directory_bytes_fetched"],
        "scene_count": result["scene_count"],
        "pose_or_intrinsics_member_names": result["pose_or_intrinsics_member_names"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
