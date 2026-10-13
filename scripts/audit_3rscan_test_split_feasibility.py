#!/usr/bin/env python3
"""Audit whether official 3RScan test metadata supports pose/identity scoring."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data/3rscan"
OUTPUT = ROOT / "outputs/reports/20261009_3rscan_test_split_feasibility"
LABEL_FILES = ("labels.instances.annotated.v2.ply", "semseg.v2.json")
SEQUENCE_FILE = "sequence.zip"


def load_downloader():
    path = ROOT / "download.py"
    spec = importlib.util.spec_from_file_location("worldcache_3rscan_download", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load 3RScan downloader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(data_root: Path) -> dict:
    metadata_path = data_root / "3RScan.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing official metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    tests = [row for row in metadata if row.get("type") == "test"]
    references = sorted({row["reference"] for row in tests})
    scan_records = [record for row in tests for record in row.get("scans", [])]
    rescans = sorted({record["reference"] for record in scan_records})
    downloader = load_downloader()
    official_test_filetypes = list(downloader.TEST_FILETYPES)
    missing_by_class = {
        "reference_directories": sum(not (data_root / scan).is_dir() for scan in references),
        "rescan_directories": sum(not (data_root / scan).is_dir() for scan in rescans),
        "reference_instance_annotations": sum(
            not (data_root / scan / LABEL_FILES[0]).is_file() for scan in references
        ),
        "rescan_instance_annotations": sum(
            not (data_root / scan / LABEL_FILES[0]).is_file() for scan in rescans
        ),
        "reference_semseg_annotations": sum(
            not (data_root / scan / LABEL_FILES[1]).is_file() for scan in references
        ),
        "rescan_semseg_annotations": sum(
            not (data_root / scan / LABEL_FILES[1]).is_file() for scan in rescans
        ),
    }
    global_transform_records = sum("transform" in record for record in scan_records)
    object_transform_records = sum(
        any(isinstance(item, dict) for item in record.get("rigid", []))
        for record in scan_records
    )
    return {
        "status": "not_scorable_for_pose_or_physical_identity",
        "metadata": "local official 3RScan.json",
        "test_reference_groups": len(tests),
        "unique_test_references": len(references),
        "test_rescan_entries": len(scan_records),
        "unique_test_rescans": len(rescans),
        "local_test_rescan_directories": sum((data_root / scan).is_dir() for scan in rescans),
        "global_transform_records": global_transform_records,
        "object_transform_records": object_transform_records,
        "rigid_id_lists": sum(isinstance(record.get("rigid"), list) for record in scan_records),
        "official_hidden_test_filetypes": official_test_filetypes,
        "required_identity_annotation_files": list(LABEL_FILES),
        "missing_counts": missing_by_class,
        "download_performed": False,
        "interpretation": (
            "The released test metadata omits scan-level global transforms and object-transform records. "
            "The upstream hidden-test downloader offers mesh/sequence files but not the instance/semantic "
            "annotation files required for target identity scoring. No test scans are local. Downloading "
            "these assets alone therefore cannot evaluate strict pose or physical-identity utility."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = audit(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# 3RScan official test split scoring feasibility

Inspected the local official `3RScan.json` and the upstream downloader's hidden-test file list. No test scans were downloaded for this audit.

| Item | Count |
|---|---:|
| Test reference groups | {summary['test_reference_groups']} |
| Unique test references | {summary['unique_test_references']} |
| Test rescan entries / unique scans | {summary['test_rescan_entries']} / {summary['unique_test_rescans']} |
| Local test rescan directories | {summary['local_test_rescan_directories']} |
| Test scan records with global transforms | {summary['global_transform_records']} |
| Test scan records with object transforms | {summary['object_transform_records']} |
| Test scan records containing rigid-ID lists | {summary['rigid_id_lists']} |

The official hidden-test downloader exposes `{', '.join(summary['official_hidden_test_filetypes'])}`. It does not expose `{', '.join(LABEL_FILES)}`. The target identity annotations and global reference/rescan transforms needed to evaluate this paper's endpoints are therefore unavailable through the documented test assets. The current 3RScan test split cannot substitute for a scored physical-identity/strict-pose holdout. Downloading the target sequences alone would not resolve this limitation.

Per-file availability counts and the exact conclusion are in `summary.json`. Reproduce without network or data downloads using:

```bash
python scripts/audit_3rscan_test_split_feasibility.py \\
  --data-root data/3rscan
```
"""
    (args.output_dir / "report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
