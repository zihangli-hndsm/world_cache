#!/usr/bin/env python3
"""Qualify per-scene persistent identity labels and PLY object-ID mappings."""

from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from audit_multiscan_pilot_annotations import labels_for_scan  # noqa: E402
from download_multiscan_pilot_pair import REVISION  # noqa: E402


def main() -> None:
    subset = json.loads((REPORT_DIR / "quota_subset_manifest.json").read_text())
    state = json.loads((REPORT_DIR / "download_state.json").read_text())
    scans: dict[str, dict] = {}
    timings = {}
    for item in subset["archives"]:
        scan_id = Path(item["filename"]).stem
        cache_path = ROOT / state["downloaded_archives"][item["filename"]]["cache_path"]
        started = time.monotonic()
        with zipfile.ZipFile(cache_path) as archive:
            result = labels_for_scan(archive, scan_id)
        scans[scan_id] = result
        timings[scan_id] = time.monotonic() - started

    rows = []
    detailed = []
    for relation in subset["relations"]:
        ref = scans[relation["reference_scan_id"]]
        query = scans[relation["query_scan_id"]]
        ref_ids = set(ref["persistent_object_labels"])
        query_ids = set(query["persistent_object_labels"])
        shared = sorted(ref_ids & query_ids)
        categories = {label.rsplit(".", 1)[0] for label in shared}
        row = {
            "environment_id": relation["environment_id"],
            "reference_scan_id": relation["reference_scan_id"],
            "query_scan_id": relation["query_scan_id"],
            "reference_persistent_objects": len(ref_ids),
            "query_persistent_objects": len(query_ids),
            "shared_physical_instance_ids": len(shared),
            "reference_only_ids": len(ref_ids - query_ids),
            "query_only_ids": len(query_ids - ref_ids),
            "shared_semantic_categories": len(categories),
            "raw_scan_local_object_id_equality_used": False,
            "ply_nonzero_object_ids_mapped": True,
            "identity_gate_pass": bool(shared),
        }
        rows.append(row)
        detailed.append({**row, "shared_persistent_object_labels": shared})

    output_dir = REPORT_DIR / "identity_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_environment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "details.json").write_text(json.dumps({
        "dataset_revision": REVISION,
        "selection": "all 17 relations in frozen quota subset; no retrieval predictions inspected",
        "relations": detailed,
        "scan_audit_seconds": timings,
        "runtime_seconds_total": sum(timings.values()),
    }, indent=2) + "\n")
    summary = {
        "dataset_revision": REVISION,
        "environment_count": len(rows),
        "identity_gate_pass_count": sum(row["identity_gate_pass"] for row in rows),
        "identity_gate_fail_count": sum(not row["identity_gate_pass"] for row in rows),
        "shared_identity_count_min": min(row["shared_physical_instance_ids"] for row in rows),
        "shared_identity_count_median": sorted(row["shared_physical_instance_ids"] for row in rows)[len(rows) // 2],
        "runtime_seconds_total": sum(timings.values()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
