#!/usr/bin/env python3
"""Fetch only missing annotated PLYs for official 3RScan train references.

The official root-level ``download.py`` implementation is reused. This avoids
the full release download and does not fetch RGB-D sequences or rescans.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


PLY_NAME = "labels.instances.annotated.v2.ply"


def load_official_downloader(path: Path):
    spec = importlib.util.spec_from_file_location("official_3rscan_downloader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official downloader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--downloader", type=Path, default=Path("download.py"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    scenes = json.loads(args.metadata.read_text(encoding="utf-8"))
    train_refs = sorted({scene["reference"] for scene in scenes if scene.get("type") == "train"})
    missing = [scan_id for scan_id in train_refs if not (args.dataset_root / scan_id / PLY_NAME).is_file()]
    downloader = load_official_downloader(args.downloader.resolve())
    completed = []
    for index, scan_id in enumerate(missing, start=1):
        target = args.dataset_root / scan_id / PLY_NAME
        print(f"[{index}/{len(missing)}] {scan_id}", flush=True)
        downloader.download_scan(scan_id, str(target.parent), [PLY_NAME])
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"download did not produce a nonempty annotation PLY: {target}")
        completed.append({"scan_id": scan_id, "bytes": target.stat().st_size})

    remaining = [scan_id for scan_id in train_refs if not (args.dataset_root / scan_id / PLY_NAME).is_file()]
    result = {
        "status": "complete" if not remaining else "incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_file": args.metadata.name,
        "dataset_root": "<authorized-local-3RScan-root>",
        "official_downloader": "download.py",
        "official_train_environment_count": len(train_refs),
        "downloaded_missing_reference_plys": len(completed),
        "downloaded_bytes": sum(row["bytes"] for row in completed),
        "available_train_reference_plys_after_run": len(train_refs) - len(remaining),
        "remaining_missing_ids": remaining,
        "downloaded": completed,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in [
        "status", "official_train_environment_count", "downloaded_missing_reference_plys",
        "downloaded_bytes", "available_train_reference_plys_after_run", "remaining_missing_ids",
    ]}, indent=2))
    if remaining:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
