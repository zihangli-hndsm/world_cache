#!/usr/bin/env python3
"""Extract one Rescan depth stream for a bounded camera-tracking pilot."""

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
    parser.add_argument("--scan", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-uncompressed-mib", type=float, default=256.0)
    args = parser.parse_args()
    if not args.scan.startswith(args.scene + "_") or not args.scan[len(args.scene) + 1:].isdigit():
        raise SystemExit("scan ID must belong to the selected scene")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    name = f"{args.scene}/depth/{args.scan}.depth"
    remote = HTTPRangeReader(args.archive_url)
    with ZipFile(remote) as archive:
        info = next((item for item in archive.infolist() if item.filename == name), None)
        if info is None:
            raise FileNotFoundError(f"archive member missing: {name}")
        max_bytes = int(args.max_uncompressed_mib * 1024**2)
        if info.file_size > max_bytes:
            raise SystemExit(f"stream is {info.file_size} bytes, over extraction limit {max_bytes}")
        if shutil.disk_usage(root).free < int(info.file_size * 1.2):
            raise SystemExit("insufficient free space for depth stream")
        destination = root / f"{args.scan}.depth"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        temporary = destination.with_suffix(".depth.partial")
        try:
            with archive.open(info) as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise IOError("extracted depth stream size mismatch")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "status": "complete",
            "archive_url": args.archive_url,
            "archive_member": name,
            "archive_bytes": remote.size,
            "archive_directory_bytes_fetched": remote.bytes_downloaded,
            "file_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "output": str(destination),
        }
    (root / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
