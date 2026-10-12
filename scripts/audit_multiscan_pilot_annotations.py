#!/usr/bin/env python3
"""Audit stable object/part labels and scan-local PLY IDs in a MultiScan pair.

Reads only annotation/alignment JSON and face IDs from the two pinned scan
archives. It does not extract image, depth, or mesh data to disk. The archives
must first be obtained after accepting the dataset's separate access terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import zipfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO

from download_multiscan_pilot_pair import FILES, REVISION


PLY_TYPES = {
    "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i", "uint": "I", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}
SCAN_RE = re.compile(r"scene_\d{5}_\d{2}")
PERSISTENT_LABEL_RE = re.compile(r".+\.\d+")
NON_IDENTITY_LABELS = {"remove"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(stream: BinaryIO, n: int) -> bytes:
    data = stream.read(n)
    if len(data) != n:
        raise ValueError(f"truncated PLY record: needed {n} bytes, got {len(data)}")
    return data


def skip_exact(stream: BinaryIO, n: int) -> None:
    remaining = n
    while remaining:
        chunk = stream.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise ValueError(f"truncated PLY while skipping {n} bytes")
        remaining -= len(chunk)


def read_ply_face_ids(stream: BinaryIO) -> tuple[Counter[int], Counter[int], dict]:
    header_lines = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY has no end_header")
        header_lines.append(line.decode("ascii").strip())
        if header_lines[-1] == "end_header":
            break

    fmt = next((line.split()[1] for line in header_lines if line.startswith("format ")), None)
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"unsupported PLY format: {fmt}")
    endian = "<" if fmt == "binary_little_endian" else ">"
    elements: list[dict] = []
    current = None
    for line in header_lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif current is not None and len(parts) == 3 and parts[0] == "property":
            current["properties"].append({"kind": "scalar", "type": parts[1], "name": parts[2]})
        elif current is not None and len(parts) == 5 and parts[:2] == ["property", "list"]:
            current["properties"].append({
                "kind": "list", "count_type": parts[2], "item_type": parts[3], "name": parts[4]
            })
    face = next((e for e in elements if e["name"] == "face"), None)
    if face is None:
        raise ValueError("PLY has no face element")
    names = {prop["name"] for prop in face["properties"]}
    if "objectId" not in names:
        raise ValueError("PLY face element has no objectId property")
    if any(p["name"] == "objectId" and p["kind"] != "scalar" for p in face["properties"]):
        raise ValueError("PLY objectId property must be scalar")

    object_counts: Counter[int] = Counter()
    part_counts: Counter[int] = Counter()

    def scalar(type_name: str) -> int | float:
        code = PLY_TYPES.get(type_name)
        if code is None:
            raise ValueError(f"unsupported PLY scalar type: {type_name}")
        size = struct.calcsize(code)
        return struct.unpack(endian + code, read_exact(stream, size))[0]

    def skip_property(prop: dict, capture: set[str] | None = None) -> int | float | None:
        if prop["kind"] == "scalar":
            value = scalar(prop["type"])
            return value if capture and prop["name"] in capture else None
        count = int(scalar(prop["count_type"]))
        code = PLY_TYPES.get(prop["item_type"])
        if code is None:
            raise ValueError(f"unsupported PLY list item type: {prop['item_type']}")
        read_exact(stream, count * struct.calcsize(code))
        return None

    if fmt == "ascii":
        # ASCII records are decoded element-by-element so only face IDs are kept.
        for element in elements:
            for _ in range(element["count"]):
                tokens = iter(stream.readline().decode("ascii").split())
                if element["name"] == "face":
                    object_id = part_id = None
                    for prop in element["properties"]:
                        if prop["kind"] == "scalar":
                            value = float(next(tokens))
                            if prop["name"] == "objectId": object_id = int(value)
                            if prop["name"] == "partId": part_id = int(value)
                        else:
                            n = int(next(tokens))
                            for _ in range(n): next(tokens)
                    if object_id is None: raise ValueError("face objectId missing")
                    object_counts[object_id] += 1
                    if part_id is not None: part_counts[part_id] += 1
                else:
                    for prop in element["properties"]:
                        if prop["kind"] == "list":
                            n = int(next(tokens))
                            for _ in range(n): next(tokens)
                        else:
                            next(tokens)
    else:
        for element in elements:
            if element["name"] != "face" and all(p["kind"] == "scalar" for p in element["properties"]):
                stride = sum(struct.calcsize(PLY_TYPES[p["type"]]) for p in element["properties"])
                skip_exact(stream, element["count"] * stride)
                continue
            face_props = element["properties"]
            standard_triangle_faces = (
                element["name"] == "face"
                and [p["name"] for p in face_props] == ["vertex_indices", "objectId", "partId"]
                and face_props[0] == {"kind": "list", "count_type": "uchar", "item_type": "int", "name": "vertex_indices"}
                and face_props[1:] == [
                    {"kind": "scalar", "type": "ushort", "name": "objectId"},
                    {"kind": "scalar", "type": "ushort", "name": "partId"},
                ]
            )
            if standard_triangle_faces:
                record = struct.Struct(endian + "BiiiHH")
                payload = read_exact(stream, element["count"] * record.size)
                for values in record.iter_unpack(payload):
                    if values[0] != 3:
                        raise ValueError("expected triangular MultiScan mesh faces")
                    object_counts[int(values[4])] += 1
                    part_counts[int(values[5])] += 1
                continue
            for _ in range(element["count"]):
                if element["name"] == "face":
                    object_id = part_id = None
                    for prop in element["properties"]:
                        value = skip_property(prop, {"objectId", "partId"})
                        if prop["name"] == "objectId": object_id = int(value)
                        if prop["name"] == "partId": part_id = int(value)
                    if object_id is None: raise ValueError("face objectId missing")
                    object_counts[object_id] += 1
                    if part_id is not None: part_counts[part_id] += 1
                else:
                    for prop in element["properties"]: skip_property(prop)

    return object_counts, part_counts, {"format": fmt, "face_count": face["count"], "face_properties": sorted(names)}


def unique_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name == suffix or name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one archive member ending {suffix!r}; found {matches[:4]}")
    return matches[0]


def labels_for_scan(archive: zipfile.ZipFile, scan_id: str) -> dict:
    ann_name = unique_member(archive, scan_id + ".annotations.json")
    align_name = unique_member(archive, scan_id + ".align.json")
    ply_name = unique_member(archive, scan_id + ".ply")
    annotations = json.loads(archive.read(ann_name))
    if annotations.get("scanId") != scan_id:
        raise ValueError(f"annotation scanId mismatch for {scan_id}")
    objects = annotations.get("objects", [])
    annotation_parts = annotations.get("parts", [])
    if not isinstance(objects, list) or not isinstance(annotation_parts, list):
        raise ValueError(f"objects/parts must be lists in {scan_id}")
    local_to_label: dict[int, str] = {}
    persistent_object_labels: set[str] = set()
    ignored_object_ids: list[int] = []
    for index, obj in enumerate(objects, start=1):
        local_id, label = int(obj["objectId"]), str(obj["label"])
        if local_id != index:
            raise ValueError(f"objectId must equal list index + 1 in {scan_id}: expected {index}, got {local_id}")
        if label not in NON_IDENTITY_LABELS and not PERSISTENT_LABEL_RE.fullmatch(label):
            raise ValueError(f"invalid persistent object label in {scan_id}: {label!r}")
        if local_id in local_to_label or (label not in NON_IDENTITY_LABELS and label in persistent_object_labels):
            raise ValueError(f"duplicate local objectId or persistent label in {scan_id}: {local_id}/{label}")
        local_to_label[local_id] = label
        if label in NON_IDENTITY_LABELS:
            ignored_object_ids.append(local_id)
        else:
            persistent_object_labels.add(label)
    parts: dict[int, str] = {}
    ignored_part_ids: list[int] = []
    for index, part in enumerate(annotation_parts, start=1):
        part_id, label = int(part["partId"]), str(part["label"])
        if part_id != index:
            raise ValueError(f"partId must equal list index + 1 in {scan_id}: expected {index}, got {part_id}")
        if label not in NON_IDENTITY_LABELS and not PERSISTENT_LABEL_RE.fullmatch(label):
            raise ValueError(f"invalid persistent part label in {scan_id}: {label!r}")
        if part_id in parts:
            raise ValueError(f"duplicate partId {part_id} in {scan_id}")
        parts[part_id] = label
        if label in NON_IDENTITY_LABELS:
            ignored_part_ids.append(part_id)
    part_to_persistent: dict[int, str] = {}
    owned_part_ids: set[int] = set()
    for obj in objects:
        object_label = str(obj["label"])
        object_part_ids = [int(value) for value in obj.get("partIds", [])]
        if len(object_part_ids) != len(set(object_part_ids)):
            raise ValueError(f"duplicate partIds in object {object_label} in {scan_id}")
        for part_id in object_part_ids:
            part_id = int(part_id)
            if part_id not in parts:
                raise ValueError(f"object references missing partId {part_id} in {scan_id}")
            if part_id in part_to_persistent:
                raise ValueError(f"partId {part_id} is assigned to multiple objects in {scan_id}")
            if part_id in owned_part_ids:
                raise ValueError(f"partId {part_id} is assigned to multiple objects in {scan_id}")
            owned_part_ids.add(part_id)
            part_label = parts[part_id]
            if object_label not in NON_IDENTITY_LABELS and part_label not in NON_IDENTITY_LABELS:
                part_to_persistent[part_id] = f"{object_label}:{part_label}"
    unowned_parts = sorted(set(parts) - owned_part_ids)
    if unowned_parts:
        raise ValueError(f"annotation parts are not assigned to an object in {scan_id}: {unowned_parts[:20]}")

    with archive.open(ply_name) as stream:
        object_face_counts, part_face_counts, ply_info = read_ply_face_ids(stream)
    unknown_objects = sorted(i for i, count in object_face_counts.items() if i != 0 and count and i not in local_to_label)
    unknown_parts = sorted(i for i, count in part_face_counts.items() if i != 0 and count and i not in parts)
    if unknown_objects or unknown_parts:
        raise ValueError(f"unmapped nonzero PLY IDs in {scan_id}: objects={unknown_objects[:20]}, parts={unknown_parts[:20]}")

    alignment = json.loads(archive.read(align_name))
    if not isinstance(alignment, dict):
        raise ValueError(f"alignment metadata must be an object in {scan_id}")
    coordinate = alignment.get("coordinate_transform")
    ref_align = alignment.get("reference_scan_alignment")
    transform_check = {"coordinate_transform_present": coordinate is not None}
    if coordinate is not None:
        if (not isinstance(coordinate, list) or len(coordinate) != 16
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in coordinate)):
            raise ValueError(f"invalid coordinate_transform in {scan_id}")
        transform_check["coordinate_transform_length"] = len(coordinate)
    if ref_align is not None:
        if not isinstance(ref_align, dict):
            raise ValueError(f"invalid reference_scan_alignment in {scan_id}")
        transformation = ref_align.get("transformation", [])
        if (not isinstance(transformation, list) or len(transformation) != 16
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in transformation)):
            raise ValueError(f"invalid reference alignment in {scan_id}")
        transform_check["reference_target_id"] = ref_align.get("target_id")

    return {
        "scan_id": scan_id,
        "annotation_member": ann_name,
        "alignment_member": align_name,
        "mesh_member": ply_name,
        "objects": len(objects),
        "parts": len(parts),
        "persistent_object_labels": sorted(persistent_object_labels),
        "persistent_part_labels": sorted(set(part_to_persistent.values())),
        "ignored_nonidentity_object_ids": ignored_object_ids,
        "ignored_nonidentity_part_ids": ignored_part_ids,
        "object_face_ids": {str(k): int(v) for k, v in sorted(object_face_counts.items())},
        "part_face_ids": {str(k): int(v) for k, v in sorted(part_face_counts.items())},
        "ply": ply_info,
        "alignment": transform_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/multiscan_scene_00069"))
    parser.add_argument("--manifest", type=Path, help="defaults to DATA_DIR/download_manifest.json")
    parser.add_argument("--output", type=Path, help="defaults to DATA_DIR/annotation_audit.json")
    args = parser.parse_args()
    manifest_path = args.manifest or args.data_dir / "download_manifest.json"
    output_path = args.output or args.data_dir / "annotation_audit.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != REVISION:
        raise ValueError(f"unexpected dataset revision {manifest.get('revision')!r}")
    if manifest.get("archives_extracted"):
        raise ValueError("expected unextracted pinned archives")
    files = {item["filename"]: item for item in manifest.get("files", [])}
    if set(files) != set(FILES):
        raise ValueError(f"manifest file set differs from pinned pilot: {sorted(files)}")

    scans = []
    for filename in sorted(FILES):
        item = files[filename]
        if item.get("repo_oid") != FILES[filename]["repo_oid"]:
            raise ValueError(f"manifest repository object ID differs from pinned release for {filename}")
        if item.get("filename") != filename:
            raise ValueError(f"manifest filename mismatch for {filename}")
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != FILES[filename]["size"]:
            raise ValueError(f"missing or wrong-sized pinned archive: {path}")
        digest = sha256(path)
        if digest != item.get("sha256"):
            raise ValueError(f"archive SHA-256 differs from download manifest: {path}")
        scan_id = SCAN_RE.search(Path(filename).name).group()
        with zipfile.ZipFile(path) as archive:
            scans.append({**labels_for_scan(archive, scan_id), "archive_sha256": digest})

    left, right = scans
    left_objects, right_objects = set(left["persistent_object_labels"]), set(right["persistent_object_labels"])
    shared_objects = sorted(left_objects & right_objects)
    shared_parts = sorted(set(left["persistent_part_labels"]) & set(right["persistent_part_labels"]))
    object_categories = {}
    for label in shared_objects:
        category = label.rsplit(".", 1)[0]
        object_categories[label] = category
    result = {
        "status": "complete",
        "dataset_revision": REVISION,
        "scene_id": "scene_00069",
        "scans": scans,
        "identity_overlap": {
            "shared_object_label_count": len(shared_objects),
            "shared_object_labels": shared_objects,
            "reference_only_labels": sorted(left_objects - right_objects),
            "rescan_only_labels": sorted(right_objects - left_objects),
            "shared_part_label_count": len(shared_parts),
            "shared_part_labels": shared_parts,
            "shared_label_category_by_id": object_categories,
            "raw_object_id_equality_used": False,
        },
        "interpretation_note": "Label agreement is a necessary pilot check, not proof that RGB-D frames project correctly onto the annotated meshes.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "shared_object_label_count": len(shared_objects),
        "reference_only_count": len(left_objects - right_objects),
        "rescan_only_count": len(right_objects - left_objects),
        "shared_part_label_count": len(shared_parts),
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
