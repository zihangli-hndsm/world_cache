#!/usr/bin/env python3
"""Audit how much of official 3RScan validation survives LivingScenes filters.

This is a feasibility proxy, not the LivingScenes benchmark evaluation: it
counts annotated-mesh vertices, whereas LivingScenes builds scene point clouds
by back-projecting raw RGB-D measurements. It applies the official category
list and 1,024-point threshold to the local annotation PLYs, then counts
official rigid instance mappings whose endpoints survive on both scans.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


LIVINGSCENES_CATEGORIES = {
    "dining chair",
    "couch table",
    "dining table",
    "footstool",
    "cushion",
    "sofa",
    "bench",
    "table",
    "ottoman",
    "computer desk",
    "rocking chair",
    "armchair",
    "round table",
    "pillow",
    "side table",
    "stand",
    "sofa chair",
    "desk",
    "trash can",
    "couch",
    "coffee table",
    "chair",
    "bed",
}


def read_ascii_ply_object_ids(path: Path) -> tuple[np.ndarray, int]:
    """Read only the objectId vertex property from an ASCII 3RScan PLY."""
    header: list[bytes] = []
    vertex_count: int | None = None
    in_vertex = False
    object_id_column: int | None = None
    vertex_column = 0
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"truncated PLY header: {path}")
            header.append(line)
            fields = line.decode("ascii").strip().split()
            if fields[:2] == ["format", "ascii"]:
                pass
            elif fields[:2] == ["format", "binary_little_endian"] or fields[:2] == ["format", "binary_big_endian"]:
                raise ValueError(f"expected ASCII annotation PLY, got {fields[1]}: {path}")
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
                continue
            if fields[:2] == ["element"] and fields[1] != "vertex":
                in_vertex = False
            if in_vertex and fields[:1] == ["property"]:
                if fields[-1] == "objectId":
                    object_id_column = vertex_column
                vertex_column += 1
            if line.strip() == b"end_header":
                break

    if vertex_count is None or object_id_column is None:
        raise ValueError(f"missing vertex count or objectId property: {path}")
    values = np.loadtxt(
        path,
        dtype=np.int64,
        skiprows=len(header),
        max_rows=vertex_count,
        usecols=(object_id_column,),
        ndmin=1,
    )
    if len(values) != vertex_count:
        raise ValueError(f"read {len(values)} of {vertex_count} vertices: {path}")
    return values, vertex_count


def scan_inventory(scan_dir: Path, min_points: int) -> dict[str, Any]:
    ply = scan_dir / "labels.instances.annotated.v2.ply"
    semseg_path = scan_dir / "semseg.v2.json"
    object_ids, vertex_count = read_ascii_ply_object_ids(ply)
    segmentation = json.loads(semseg_path.read_text(encoding="utf-8"))["segGroups"]
    labels = {int(item["objectId"]): item.get("label", "") for item in segmentation}
    counts = Counter(map(int, object_ids))
    eligible = {
        object_id: {"points": count, "label": labels.get(object_id, "")}
        for object_id, count in counts.items()
        if count >= min_points and labels.get(object_id, "") in LIVINGSCENES_CATEGORIES
    }
    return {
        "vertex_count": vertex_count,
        "object_count": len(counts),
        "eligible": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--min-points", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.min_points <= 0:
        raise ValueError("--min-points must be positive")

    scan_rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    scan_ids = {row["scan_id"] for row in scan_rows}
    role_by_scan = {row["scan_id"]: row["role"] for row in scan_rows}
    inventory: dict[str, dict[str, Any]] = {}
    for index, scan_id in enumerate(sorted(scan_ids), start=1):
        inventory[scan_id] = scan_inventory(args.data_root / scan_id, args.min_points)
        if index % 20 == 0 or index == len(scan_ids):
            print(f"scans {index}/{len(scan_ids)}", flush=True)

    metadata = json.loads((args.data_root / "3RScan.json").read_text(encoding="utf-8"))
    validation_refs = {
        scan_id for scan_id, role in role_by_scan.items() if role == "reference"
    }
    relations: list[dict[str, Any]] = []
    for scene in metadata:
        reference_id = scene["reference"]
        if reference_id not in validation_refs:
            continue
        for scan in scene.get("scans", []):
            rescan_id = scan["reference"]
            if rescan_id not in inventory:
                continue
            mappings = scan.get("rigid", [])
            evaluated_mappings = 0
            eligible_mappings = 0
            for mapping in mappings:
                if not isinstance(mapping, dict):
                    continue
                evaluated_mappings += 1
                ref_id = int(mapping["instance_reference"])
                rescan_object_id = int(mapping["instance_rescan"])
                ref_obj = inventory[reference_id]["eligible"].get(ref_id)
                rescan_obj = inventory[rescan_id]["eligible"].get(rescan_object_id)
                if ref_obj and rescan_obj:
                    eligible_mappings += 1
            relations.append(
                {
                    "reference_id": reference_id,
                    "rescan_id": rescan_id,
                    "official_rigid_mappings": len(mappings),
                    "mapping_records_with_transforms": evaluated_mappings,
                    "eligible_linked_instances": eligible_mappings,
                    "reference_eligible_instances": len(inventory[reference_id]["eligible"]),
                    "rescan_eligible_instances": len(inventory[rescan_id]["eligible"]),
                    "eligible_vertices_reference": sum(
                        item["points"] for item in inventory[reference_id]["eligible"].values()
                    ),
                    "eligible_vertices_rescan": sum(
                        item["points"] for item in inventory[rescan_id]["eligible"].values()
                    ),
                }
            )

    if len(scan_ids) != 157 or len(relations) != 110:
        raise ValueError(f"expected 157 scans / 110 validation relations; got {len(scan_ids)} / {len(relations)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    relation_path = args.output_dir / "relation_feasibility.csv"
    with relation_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(relations[0]))
        writer.writeheader()
        writer.writerows(relations)

    scan_path = args.output_dir / "scan_inventory.csv"
    with scan_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["scan_id", "role", "vertices", "instances", "eligible_instances", "eligible_vertices"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scan_id in sorted(inventory):
            item = inventory[scan_id]
            writer.writerow(
                {
                    "scan_id": scan_id,
                    "role": role_by_scan[scan_id],
                    "vertices": item["vertex_count"],
                    "instances": item["object_count"],
                    "eligible_instances": len(item["eligible"]),
                    "eligible_vertices": sum(obj["points"] for obj in item["eligible"].values()),
                }
            )

    eligible_per_relation = [row["eligible_linked_instances"] for row in relations]
    summary = {
        "audit_type": "feasibility_proxy_not_official_livingscenes_eval",
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "scans": len(inventory),
        "relations": len(relations),
        "min_points_per_instance": args.min_points,
        "living_scenes_category_count": len(LIVINGSCENES_CATEGORIES),
        "annotation_mesh_vertices_total": sum(item["vertex_count"] for item in inventory.values()),
        "qualifying_instances_total": sum(len(item["eligible"]) for item in inventory.values()),
        "qualifying_scan_count": sum(bool(item["eligible"]) for item in inventory.values()),
        "official_rigid_mappings_total": sum(row["official_rigid_mappings"] for row in relations),
        "rigid_mapping_records_with_transforms": sum(row["mapping_records_with_transforms"] for row in relations),
        "eligible_linked_instances_total": sum(eligible_per_relation),
        "relations_with_ge1_eligible_linked_instance": sum(count >= 1 for count in eligible_per_relation),
        "relations_with_ge3_eligible_linked_instances": sum(count >= 3 for count in eligible_per_relation),
        "relations_with_ge5_eligible_linked_instances": sum(count >= 5 for count in eligible_per_relation),
        "relations_with_zero_eligible_linked_instances": sum(count == 0 for count in eligible_per_relation),
        "eligible_linked_instances_median": float(np.median(eligible_per_relation)),
        "eligible_linked_instances_p90": float(np.quantile(eligible_per_relation, 0.9)),
        "category_list_source": "GradientSpaces/LivingScenes cate_to_use.txt (official repository)",
        "protocol_source": "GradientSpaces/LivingScenes configs/3rscan.yaml (split=val, use_gt_mask=True, n_point_per_instance=1024)",
        "caveat": "Counts use labels.instances.annotated.v2.ply mesh vertices; official LivingScenes preparation back-projects raw RGB-D measurements into point clouds, so this audit is a lower-fidelity feasibility proxy and not baseline performance.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# LivingScenes validation input feasibility audit

This is a local feasibility proxy for the official LivingScenes / MORE² 3RScan
validation procedure. The official config uses the validation split, GT masks,
seven mapped shape categories represented by the 23 labels in `cate_to_use.txt`,
and a minimum of {args.min_points:,} points per instance. This audit applies
that category list and point threshold to the vertex-level labels in the
available 3RScan annotation meshes and then counts official rigid instance
mappings whose endpoints survive in both scans.

## Results

- Scans and relations audited: {len(inventory)} / {len(relations)}.
- Annotation mesh vertices: {summary['annotation_mesh_vertices_total']:,} total;
  {summary['qualifying_instances_total']:,} instances pass the category and
  point-count filters across {summary['qualifying_scan_count']} scans.
- Official rigid mappings: {summary['official_rigid_mappings_total']:,};
  {summary['rigid_mapping_records_with_transforms']:,} have the detailed mapping
  records required for object-level scoring.
- Mapped instances passing both endpoint filters: {summary['eligible_linked_instances_total']:,}.
- Relations with at least one qualifying linked instance: {summary['relations_with_ge1_eligible_linked_instance']};
  with at least three: {summary['relations_with_ge3_eligible_linked_instances']};
  with at least five: {summary['relations_with_ge5_eligible_linked_instances']}.
- The median number of qualifying linked instances per relation is
  {summary['eligible_linked_instances_median']:.1f} (90th percentile
  {summary['eligible_linked_instances_p90']:.1f}).

## Interpretation and next step

This does not establish that the benchmark is infeasible. The annotation PLYs
are labeled meshes, while LivingScenes documents point clouds back-projected
from raw RGB-D measurements. The latter can provide more points per object.
The audit shows that an evaluator run directly on annotation-mesh vertices
would cover only a restricted subset of the official validation relations.
The next valid feasibility step is to reproduce the authors' RGB-D
back-projection and GT-mask preparation for a small set of relations, then
recompute this coverage audit before attempting the full evaluator.

## Artifacts

- `relation_feasibility.csv`: one row per official validation relation.
- `scan_inventory.csv`: point and eligible-instance counts per scan.
- `summary.json`: machine-readable totals and protocol caveats.
- Reproduce with `scripts/audit_livingscenes_validation_feasibility.py`.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
