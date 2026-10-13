#!/usr/bin/env python3
"""Inventory the official 3RScan validation inputs needed for evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


RAW_FILES = (
    "labels.instances.annotated.v2.ply",
    "semseg.v2.json",
    "sequence.zip",
    "mesh.refined.v2.obj",
)
PROCESSED_FILES = (
    "pointcloud.instances.align.ply",
    "pointcloud.labels.npz",
)


def present(scan_root: Path, scan_id: str, filename: str) -> bool:
    return (scan_root / scan_id / filename).is_file()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/3rscan/3RScan.json"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/3rscan"),
    )
    parser.add_argument("--official-val-split", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/20260929_validation_access"),
    )
    args = parser.parse_args()

    scenes = json.loads(args.metadata.read_text(encoding="utf-8"))
    validation = [scene for scene in scenes if scene.get("type") == "validation"]
    if len(validation) != 47:
        raise SystemExit(f"Expected 47 validation environments, found {len(validation)}")
    local_val_refs = {scene["reference"] for scene in validation}
    split_check = "not supplied"
    if args.official_val_split is not None:
        official_val_refs = {
            line.strip()
            for line in args.official_val_split.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if official_val_refs != local_val_refs:
            raise SystemExit(
                "Local validation references do not match the supplied official split: "
                f"official_only={sorted(official_val_refs - local_val_refs)}, "
                f"metadata_only={sorted(local_val_refs - official_val_refs)}"
            )
        split_check = f"exact match ({len(official_val_refs)}/{len(local_val_refs)} refs)"

    pair_rows: list[dict[str, object]] = []
    scan_roles: dict[str, set[str]] = defaultdict(set)
    val_ref_ids = {scene["reference"] for scene in validation}

    for scene in validation:
        env_id = scene["reference"]
        scan_roles[env_id].add("reference")
        for relation_index, rescan in enumerate(scene.get("scans", [])):
            scan_id = rescan["reference"]
            scan_roles[scan_id].add("rescan")
            pair_rows.append(
                {
                    "environment_id": env_id,
                    "reference_scan": env_id,
                    "rescan_scan": scan_id,
                    "relation_index": relation_index,
                    "rigid_instances": len(rescan.get("rigid", [])),
                    "nonrigid_instances": len(rescan.get("nonrigid", [])),
                    "removed_instances": len(rescan.get("removed", [])),
                    "has_rigid_transfer": bool(rescan.get("rigid", [])),
                    "reference_annotated_ply": present(
                        args.dataset_root, env_id, "labels.instances.annotated.v2.ply"
                    ),
                    "rescan_annotated_ply": present(
                        args.dataset_root, scan_id, "labels.instances.annotated.v2.ply"
                    ),
                    "reference_sequence_zip": present(args.dataset_root, env_id, "sequence.zip"),
                    "rescan_sequence_zip": present(args.dataset_root, scan_id, "sequence.zip"),
                }
            )

    scan_rows: list[dict[str, object]] = []
    for scan_id in sorted(scan_roles):
        row: dict[str, object] = {
            "scan_id": scan_id,
            "environment_id": scan_id if scan_id in val_ref_ids else next(
                scene["reference"]
                for scene in validation
                if any(r["reference"] == scan_id for r in scene.get("scans", []))
            ),
            "role": "+".join(sorted(scan_roles[scan_id])),
        }
        for filename in RAW_FILES + PROCESSED_FILES:
            row[filename] = present(args.dataset_root, scan_id, filename)
        sequence_path = args.dataset_root / scan_id / "sequence.zip"
        ply_path = args.dataset_root / scan_id / "labels.instances.annotated.v2.ply"
        row["sequence_zip_bytes"] = sequence_path.stat().st_size if sequence_path.is_file() else 0
        row["annotated_ply_bytes"] = ply_path.stat().st_size if ply_path.is_file() else 0
        scan_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = args.output_dir / "validation_pair_manifest.csv"
    scan_csv = args.output_dir / "validation_scan_manifest.csv"
    write_csv(
        pair_csv,
        pair_rows,
        [
            "environment_id",
            "reference_scan",
            "rescan_scan",
            "relation_index",
            "rigid_instances",
            "nonrigid_instances",
            "removed_instances",
            "has_rigid_transfer",
            "reference_annotated_ply",
            "rescan_annotated_ply",
            "reference_sequence_zip",
            "rescan_sequence_zip",
        ],
    )
    write_csv(
        scan_csv,
        scan_rows,
        ["scan_id", "environment_id", "role", *RAW_FILES, *PROCESSED_FILES,
         "sequence_zip_bytes", "annotated_ply_bytes"],
    )

    rigid_pairs = [row for row in pair_rows if row["has_rigid_transfer"]]
    rigid_instance_count = sum(int(row["rigid_instances"]) for row in rigid_pairs)
    raw_annotation_count = sum(bool(row["labels.instances.annotated.v2.ply"]) for row in scan_rows)
    raw_semseg_count = sum(bool(row["semseg.v2.json"]) for row in scan_rows)
    sequence_count = sum(bool(row["sequence.zip"]) for row in scan_rows)
    mesh_count = sum(bool(row["mesh.refined.v2.obj"]) for row in scan_rows)
    processed_pc_count = sum(bool(row["pointcloud.instances.align.ply"]) for row in scan_rows)
    processed_label_count = sum(bool(row["pointcloud.labels.npz"]) for row in scan_rows)
    all_local_sequences = [
        p.stat().st_size
        for p in args.dataset_root.glob("*/sequence.zip")
        if p.is_file()
    ]
    all_local_plys = [
        p.stat().st_size
        for p in args.dataset_root.glob("*/labels.instances.annotated.v2.ply")
        if p.is_file()
    ]
    estimated_sequence_and_ply_gb = len(scan_rows) * (
        (median(all_local_sequences) if all_local_sequences else 0)
        + (median(all_local_plys) if all_local_plys else 0)
    ) / 1e9
    report_date = datetime.now(timezone.utc).date().isoformat()
    report = f"""# 3RScan validation access and input inventory — {report_date}

## Official validation scope

- Validation environments: {len(validation)}
- Local metadata vs. official `splits/val.txt`: {split_check}
- Reference/rescan relations: {len(pair_rows)}
- Relations with official rigid-instance mappings: {len(rigid_pairs)}
- Mapped rigid instances across those relations: {rigid_instance_count}
- Unique reference and rescan scans required: {len(scan_rows)}

The official Living Scenes 3RScan evaluator reads every reference and rescan
listed for its configured split. Its validation configuration uses `split: val`
and ground-truth masks. The full validation comparison therefore requires the
processed instance point clouds and labels for every validation scan, with the
evaluator applying its category and minimum-point filters. The RGB-D transfer
experiment additionally requires the scan annotations and sequence archives
for whichever validation pairs are evaluated.

## Local availability

- Scans with `labels.instances.annotated.v2.ply`: {raw_annotation_count}/{len(scan_rows)}
- Scans with `semseg.v2.json`: {raw_semseg_count}/{len(scan_rows)}
- Scans with `sequence.zip`: {sequence_count}/{len(scan_rows)}
- Scans with `mesh.refined.v2.obj`: {mesh_count}/{len(scan_rows)}
- Scans with Living Scenes `pointcloud.instances.align.ply`: {processed_pc_count}/{len(scan_rows)}
- Scans with Living Scenes `pointcloud.labels.npz`: {processed_label_count}/{len(scan_rows)}

## Acquisition estimate

Across the existing local dataset, {len(all_local_sequences)} sequence archives have a median size of
{median(all_local_sequences) / 1e6:.1f} MB, and {len(all_local_plys)} annotated
PLY files have a median size of {median(all_local_plys) / 1e6:.1f} MB. Those
medians imply about {estimated_sequence_and_ply_gb:.1f} GB for sequence
archives plus annotated PLYs across {len(scan_rows)} scans. This is a planning
estimate: per-scan sizes vary, the existing subset may not represent
validation, and other raw files or Living Scenes preprocessing outputs add
storage.

The 3RScan project page directs users to submit its Terms of Use form before
downloading the dataset. The public split file exposes validation reference
IDs, and the toolkit FAQ says its download script can filter by scan ID or file
name. This filesystem inventory does not establish the account's authorization
status. Once authorized inputs are available locally, these manifests identify
the exact {len(scan_rows)} scan IDs and {len(pair_rows)} validation relations to
stage.

## Source references

- [3RScan project and download terms](https://waldjohannau.github.io/RIO/)
- [Official 3RScan validation split](https://github.com/WaldJohannaU/3RScan/blob/master/splits/val.txt)
- [3RScan toolkit FAQ and subset-download note](https://github.com/WaldJohannaU/3RScan/blob/master/FAQ.md)
- [Living Scenes 3RScan config](https://github.com/GradientSpaces/LivingScenes/blob/main/configs/3rscan.yaml)
- [Living Scenes 3RScan evaluator](https://github.com/GradientSpaces/LivingScenes/blob/main/eval_3rscan.py)
- [Living Scenes preparation instructions](https://github.com/GradientSpaces/LivingScenes)

## Artifacts

- `{pair_csv.name}` — all validation reference/rescan relations and local raw
  file availability.
- `{scan_csv.name}` — unique validation scans, roles, and raw/processed file
  availability.
"""
    (args.output_dir / "validation_access_inventory.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
