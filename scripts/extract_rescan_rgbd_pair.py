#!/usr/bin/env python3
"""Extract only two named Rescan color/depth streams from the public ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

from stream_rescan_archive_subset import DEFAULT_URL, HTTPRangeReader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-url", default=DEFAULT_URL)
    parser.add_argument("--scene", required=True, choices=[f"scene_{c}" for c in "abcdefghijklm"])
    parser.add_argument("--scans", nargs=2, required=True, help="two scan IDs, e.g. scene_i_1 scene_i_2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-uncompressed-gib", type=float, default=1.0)
    args = parser.parse_args()

    for scan in args.scans:
        if not scan.startswith(args.scene + "_") or not scan[len(args.scene) + 1:].isdigit():
            raise SystemExit(f"scan {scan!r} does not belong to {args.scene}")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = {}
    remote = HTTPRangeReader(args.archive_url)
    with ZipFile(remote) as archive:
        by_name = {info.filename: info for info in archive.infolist()}
        for scan in args.scans:
            for modality, suffix in (("color", ".h264"), ("depth", ".depth")):
                name = f"{args.scene}/{modality}/{scan}{suffix}"
                if name not in by_name:
                    raise FileNotFoundError(f"required archive member missing: {name}")
                selected[name] = by_name[name]

        uncompressed_total = sum(info.file_size for info in selected.values())
        max_bytes = int(args.max_uncompressed_gib * (1024**3))
        if uncompressed_total > max_bytes:
            raise SystemExit(
                f"selected streams need {uncompressed_total} bytes, above the "
                f"{args.max_uncompressed_gib:.3f} GiB extraction limit"
            )
        if shutil.disk_usage(root).free < int(uncompressed_total * 1.2):
            raise SystemExit("insufficient free space for the selected streams")
        existing = [root / name for name in selected]
        if any(path.exists() for path in existing):
            raise FileExistsError("refusing to overwrite existing stream files")

        records = []
        for name, info in sorted(selected.items()):
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            try:
                with archive.open(info) as source, temporary.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size:
                    raise IOError(f"size mismatch extracting {name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            records.append({
                "archive_member": name,
                "path": str(destination),
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": info.file_size,
                "sha256": sha256(destination),
            })

        manifest = {
            "status": "complete",
            "source_url": args.archive_url,
            "scene": args.scene,
            "scans": args.scans,
            "archive_bytes": remote.size,
            "archive_directory_bytes_fetched": remote.bytes_downloaded,
            "streams_uncompressed_bytes": uncompressed_total,
            "files": records,
            "mesh_files_extracted": False,
        }
    manifest_path = root / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "streams_uncompressed_bytes": uncompressed_total,
                      "manifest": str(manifest_path), "files": records}, indent=2))


if __name__ == "__main__":
    main()
