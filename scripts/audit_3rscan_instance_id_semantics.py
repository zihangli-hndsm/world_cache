"""Audit whether 3RScan PLY globalId can serve as a physical instance ID."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit_motion_stratified_failures import read_pair_table


def read_scan(annotation_root: Path, scan_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    scan_root = annotation_root / scan_id
    payload = (scan_root / "labels.instances.annotated.v2.ply").read_bytes()
    offset = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:offset].decode("ascii")
    match = re.search(r"element vertex (\d+)", header)
    if match is None:
        raise ValueError(f"no vertex count in {scan_id}")
    count = int(match.group(1))
    values = np.fromstring(payload[offset:].decode("ascii"), sep=" ", dtype=np.float64)
    vertices = values[: count * 11].reshape(count, 11)
    semseg = json.loads((scan_root / "semseg.v2.json").read_text(encoding="utf-8"))
    labels = {int(item["objectId"]): str(item["label"]) for item in semseg["segGroups"]}
    return (
        vertices[:, 6].astype(np.int64),
        vertices[:, 7].astype(np.int64),
        vertices[:, 8].astype(np.int64),
        labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    scan_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []

    for pair, reference_id, rescan_id in read_pair_table(args.pair_table):
        relation = next(item for item in metadata if item.get("reference") == reference_id)
        rescan_relation = next(
            item for item in relation.get("scans", [])
            if item.get("reference") == rescan_id
        )
        for side, scan_id in (("reference", reference_id), ("rescan", rescan_id)):
            object_ids, global_ids, nyu40_ids, labels = read_scan(args.annotation_root, scan_id)
            by_global: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
            for object_id in np.unique(object_ids):
                if object_id <= 0:
                    continue
                mask = object_ids == object_id
                gid_values, gid_counts = np.unique(global_ids[mask], return_counts=True)
                valid = gid_values > 0
                gid_values, gid_counts = gid_values[valid], gid_counts[valid]
                if not len(gid_values):
                    continue
                gid = int(gid_values[np.argmax(gid_counts)])
                nyu_values, nyu_counts = np.unique(nyu40_ids[mask], return_counts=True)
                nyu = int(nyu_values[np.argmax(nyu_counts)])
                label = labels.get(int(object_id), "unknown")
                by_global[gid].append((int(object_id), nyu, label))
            repeated = {
                gid: values for gid, values in by_global.items()
                if len({item[0] for item in values}) > 1
            }
            scan_rows.append({
                "pair": pair,
                "side": side,
                "scan_id": scan_id,
                "positive_object_ids": int(np.count_nonzero(np.unique(object_ids) > 0)),
                "positive_global_ids": int(np.count_nonzero(np.unique(global_ids) > 0)),
                "global_ids_shared_by_multiple_object_ids": len(repeated),
                "extra_object_ids_collapsed_by_global_id": sum(len(values) - 1 for values in repeated.values()),
            })
            if repeated and len(examples) < 12:
                gid = max(repeated, key=lambda key: len(repeated[key]))
                examples.append({
                    "pair": pair,
                    "side": side,
                    "global_id": gid,
                    "object_ids_and_labels": repeated[gid],
                })

        reference_objects, _, _, reference_labels = read_scan(args.annotation_root, reference_id)
        rescan_objects, _, _, rescan_labels = read_scan(args.annotation_root, rescan_id)
        reference_ids = set(reference_labels)
        rescan_ids = set(rescan_labels)
        common_ids = reference_ids & rescan_ids
        same_category = sum(
            reference_labels[object_id] == rescan_labels[object_id]
            for object_id in common_ids
        )
        rigid_pairs = [
            (
                int(item.get("instance_reference", item.get("id", -1))),
                int(item.get("instance_rescan", item.get("id", -1))),
            )
            for item in rescan_relation.get("rigid", [])
            if isinstance(item, dict)
        ]
        pair_rows.append({
            "pair": pair,
            "reference_object_ids": len(reference_ids),
            "rescan_object_ids": len(rescan_ids),
            "common_object_ids": len(common_ids),
            "common_object_ids_same_semseg_label": same_category,
            "official_rigid_id_mappings": len(rigid_pairs),
            "rigid_mappings_with_different_local_ids": sum(a != b for a, b in rigid_pairs),
            "nonrigid_annotation_entries": len(rescan_relation.get("nonrigid", [])),
            "ambiguity_groups": len(relation.get("ambiguity", [])),
        })

    pd.DataFrame(scan_rows).to_csv(args.output_dir / "instance_id_scan_audit.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.output_dir / "instance_id_pair_audit.csv", index=False)
    summary = {
        "status": "complete",
        "pairs": len(pair_rows),
        "scans": len(scan_rows),
        "scans_with_global_id_collisions": sum(
            int(row["global_ids_shared_by_multiple_object_ids"] > 0)
            for row in scan_rows
        ),
        "global_id_collision_groups": sum(
            int(row["global_ids_shared_by_multiple_object_ids"])
            for row in scan_rows
        ),
        "extra_local_instance_ids_collapsed": sum(
            int(row["extra_object_ids_collapsed_by_global_id"])
            for row in scan_rows
        ),
        "common_local_object_ids": sum(int(row["common_object_ids"]) for row in pair_rows),
        "common_local_object_ids_with_same_semseg_label": sum(
            int(row["common_object_ids_same_semseg_label"]) for row in pair_rows
        ),
        "rigid_local_id_mappings": sum(int(row["official_rigid_id_mappings"]) for row in pair_rows),
        "rigid_mappings_changing_local_id": sum(
            int(row["rigid_mappings_with_different_local_ids"]) for row in pair_rows
        ),
        "examples": examples,
        "interpretation": (
            "PLY globalId collapses distinct objectId instances and is unsuitable as "
            "an exact physical-instance identity. Keep NYU40/globalId for semantic "
            "labels; use objectId plus relation mappings for instance identity."
        ),
    }
    (args.output_dir / "instance_id_semantics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# 3RScan instance-ID semantics audit",
        "",
        f"Audited {len(pair_rows)} reference/rescan pairs ({len(scan_rows)} scans) using PLY vertex labels and per-scan semseg annotations.",
        "",
        f"Within the 34 scans, {summary['scans_with_global_id_collisions']} contain at least one PLY globalId shared by distinct objectIds. There are {summary['global_id_collision_groups']} such (scan, globalId) groups, collapsing {summary['extra_local_instance_ids_collapsed']} additional local object IDs. This directly rules out treating PLY globalId as a unique physical instance label in these scans.",
        "",
        f"Across pairs, {summary['common_local_object_ids_with_same_semseg_label']}/{summary['common_local_object_ids']} common local objectIds have the same semseg label on both sides. The official rigid relation lists {summary['rigid_local_id_mappings']} explicit object mappings, of which {summary['rigid_mappings_changing_local_id']} change the local objectId.",
        "",
        "The official PLY separates objectId from globalId; the project FAQ describes objectId as the per-instance ID and maps global IDs to semantic labels. Example: in pair 0cac's reference scan, objectIds 4 and 17 are separate chair instances but both carry globalId 85 and NYU40 label 5. See the primary dataset [FAQ](https://github.com/WaldJohannaU/3RScan/blob/master/FAQ.md) and [metadata format](https://github.com/WaldJohannaU/3RScan#data-formats).",
        "",
        "Consequently, earlier fields named instance_correct, candidate_instance_hit, and instance_mIoU that compare PLY globalId are fine-grained semantic-label metrics, not physical-instance metrics. The motion audit must group query samples by local objectId and official relation annotations. Pose errors and NYU40 semantic fields are unaffected.",
        "",
        "| Pair | Common objectIds | Same semseg label | Rigid mappings | Mappings with different local IDs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        report.append(
            f"| {row['pair']} | {row['common_object_ids']} | "
            f"{row['common_object_ids_same_semseg_label']} | "
            f"{row['official_rigid_id_mappings']} | "
            f"{row['rigid_mappings_with_different_local_ids']} |"
        )
    report.append("")
    (args.output_dir / "instance_id_semantics_audit.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
