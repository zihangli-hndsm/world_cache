#!/usr/bin/env python3
"""Measure persistent-object mesh agreement under MultiScan's supplied alignment."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from download_multiscan_pilot_pair import FILES, REVISION
from audit_multiscan_pilot_annotations import unique_member
from audit_multiscan_rgbd_projection import read_mesh


def scan_mesh(path: Path, scan_id: str) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    with zipfile.ZipFile(path) as archive:
        vertices, faces, object_ids = read_mesh(archive, scan_id)
        annotations = json.loads(archive.read(unique_member(archive, scan_id + ".annotations.json")))
        align = json.loads(archive.read(unique_member(archive, scan_id + ".align.json")))
    id_to_label = {int(obj["objectId"]): str(obj["label"]) for obj in annotations["objects"]}
    centers = vertices[faces].mean(axis=1)
    by_label: dict[str, list[np.ndarray]] = {}
    for center, object_id in zip(centers, object_ids):
        label = id_to_label.get(int(object_id))
        if label and label != "remove" and "." in label:
            by_label.setdefault(label, []).append(center)
    clouds = {label: np.asarray(points, dtype=np.float64) for label, points in by_label.items()}
    reference = align.get("reference_scan_alignment")
    transform = None
    if reference is not None:
        transform = np.asarray(reference["transformation"], dtype=np.float64).reshape(4, 4, order="F")
    return clouds, transform


def distances(a: np.ndarray, b: np.ndarray) -> dict:
    # Bound memory and keep each side representative of the annotated mesh surface.
    max_points = 5000
    if len(a) > max_points:
        a = a[np.linspace(0, len(a) - 1, max_points, dtype=int)]
    if len(b) > max_points:
        b = b[np.linspace(0, len(b) - 1, max_points, dtype=int)]
    ab = cKDTree(b).query(a, k=1, workers=-1)[0]
    ba = cKDTree(a).query(b, k=1, workers=-1)[0]
    both = np.concatenate((ab, ba))
    return {
        "reference_face_centers": int(len(a)),
        "rescan_face_centers": int(len(b)),
        "symmetric_chamfer_mean_m": float(np.mean(both)),
        "symmetric_chamfer_median_m": float(np.median(both)),
        "symmetric_chamfer_p90_m": float(np.quantile(both, 0.90)),
        "within_5cm_fraction": float(np.mean(both <= 0.05)),
        "within_10cm_fraction": float(np.mean(both <= 0.10)),
        "within_20cm_fraction": float(np.mean(both <= 0.20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/multiscan_scene_00069"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261010_multiscan_pilot"))
    args = parser.parse_args()
    scan_ids = [Path(rel).stem for rel in sorted(FILES)]
    ref, rescan = scan_ids
    ref_clouds, _ = scan_mesh(args.data_dir / (ref + ".zip"), ref)
    rescan_clouds, transform = scan_mesh(args.data_dir / (rescan + ".zip"), rescan)
    if transform is None:
        raise ValueError("rescan archive has no reference_scan_alignment")
    shared = sorted(set(ref_clouds) & set(rescan_clouds))
    rows = []
    for label in shared:
        points = rescan_clouds[label]
        moved = points @ transform[:3, :3].T + transform[:3, 3]
        rows.append({"persistent_object_label": label, **distances(ref_clouds[label], moved)})
        print(f"{label}: Chamfer median={rows[-1]['symmetric_chamfer_median_m']:.3f}m, within10={rows[-1]['within_10cm_fraction']:.3f}", flush=True)
    aggregate = {
        "status": "complete",
        "dataset_revision": REVISION,
        "scene_id": "scene_00069",
        "reference_scan": ref,
        "rescan": rescan,
        "rescan_to_reference_alignment_used": True,
        "object_count_reference": len(ref_clouds),
        "object_count_rescan": len(rescan_clouds),
        "shared_persistent_object_count": len(shared),
        "object_level_median_symmetric_chamfer_m": float(np.median([r["symmetric_chamfer_median_m"] for r in rows])) if rows else None,
        "object_level_median_within_10cm_fraction": float(np.median([r["within_10cm_fraction"] for r in rows])) if rows else None,
        "caveat": "Face-center nearest-neighbor agreement under the dataset-provided alignment; this is not a learned matching baseline or an independent-scene estimate.",
        "objects": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "registered_mesh_consistency.json"
    csv_path = args.output_dir / "registered_mesh_consistency.csv"
    json_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: aggregate[k] for k in ["shared_persistent_object_count", "object_level_median_symmetric_chamfer_m", "object_level_median_within_10cm_fraction"]}, indent=2))
    print(f"outputs: {json_path} ; {csv_path}")


if __name__ == "__main__":
    main()
