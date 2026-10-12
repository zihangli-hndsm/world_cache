#!/usr/bin/env python3
"""Download only the files required by the official 3RScan validation run.

The dataset author's downloader is kept at the repository root. This helper
uses its functions so the validation manifest is fetched in one targeted,
resumable pass instead of invoking the upstream CLI hundreds of times.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "outputs/reports/20260929_validation_access/validation_scan_manifest.csv"
)
REQUIRED_FILES = (
    "sequence.zip",
    "labels.instances.annotated.v2.ply",
    "semseg.v2.json",
)


def read_scan_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "scan_id" not in rows[0]:
        raise ValueError(f"manifest has no scan_id column or rows: {path}")
    scan_ids = [row["scan_id"].strip() for row in rows]
    if len(scan_ids) != len(set(scan_ids)):
        raise ValueError("validation manifest contains duplicate scan IDs")
    if any(not scan_id for scan_id in scan_ids):
        raise ValueError("validation manifest contains an empty scan ID")
    return scan_ids


def load_upstream_downloader():
    path = ROOT / "download.py"
    spec = importlib.util.spec_from_file_location("worldcache_3rscan_upstream_download", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load dataset downloader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the three required files for each validation scan."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scan_ids = read_scan_ids(args.manifest)
    downloader = load_upstream_downloader()
    release_ids = set(downloader.get_scans(downloader.BASE_URL + downloader.RELEASE))
    hidden_ids = set(
        downloader.get_scans(downloader.BASE_URL + downloader.HIDDEN_RELEASE)
    )
    absent = sorted(set(scan_ids) - release_ids)
    hidden = sorted(set(scan_ids) & hidden_ids)
    if absent or hidden:
        raise ValueError(
            "manifest does not match the public validation release list; "
            f"not in release={absent[:5]}, hidden-test IDs={hidden[:5]}"
        )

    existing = sum(
        all((args.data_root / scan_id / filename).is_file()
            for filename in REQUIRED_FILES)
        for scan_id in scan_ids
    )
    print(
        f"Validation manifest: {len(scan_ids)} scans; {existing} already complete; "
        f"{len(scan_ids) - existing} need one or more files."
    )
    print(f"Files per scan: {', '.join(REQUIRED_FILES)}")
    print(f"Destination: {args.data_root}")
    if args.dry_run:
        return

    for index, scan_id in enumerate(scan_ids, start=1):
        print(f"[{index}/{len(scan_ids)}] {scan_id}", flush=True)
        downloader.download_scan(
            scan_id,
            str(args.data_root / scan_id),
            list(REQUIRED_FILES),
        )

    missing = [
        f"{scan_id}/{filename}"
        for scan_id in scan_ids
        for filename in REQUIRED_FILES
        if not (args.data_root / scan_id / filename).is_file()
        or (args.data_root / scan_id / filename).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"download pass left missing/empty files: {missing[:10]}")
    print(f"Validation inputs present for all {len(scan_ids)} scans.")


if __name__ == "__main__":
    main()
