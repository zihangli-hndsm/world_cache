"""Audit sampled RGB-D points against the correct scan-local annotations.

The current manifests store rescan observations in native scan coordinates.
For annotation lookup, reference and rescan points are queried in their own
respective mesh coordinates. This audit reports nearest annotated-mesh
distances and non-zero label rates without changing evaluation labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_3rscan_instance_correspondence import load, patch_world_points, read_annotated_mesh, transform_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=14)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    distance_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []

    for pair_run in args.pair_run:
        config = json.loads((pair_run / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((pair_run / "manifest.json").read_text(encoding="utf-8"))
        rescan_to_reference = np.asarray(config["rescan_to_reference"], dtype=np.float64)
        reference_to_rescan = np.linalg.inv(rescan_to_reference)
        native_manifest = config.get("coordinate_convention") == "native_scan_frames"
        for side, scan_id, inverse_transform in (
            ("reference", config["reference"], np.eye(4, dtype=np.float64)),
            ("rescan", config["rescan"], np.eye(4, dtype=np.float64) if native_manifest else reference_to_rescan),
        ):
            tree, _, global_ids, semantic_ids, _ = read_annotated_mesh(args.annotation_root, scan_id)
            points = []
            for entry in manifest[side]:
                observation = load(pair_run / entry["path"])
                sampled, _ = patch_world_points(observation, patch_size=args.patch_size)
                points.append(sampled.numpy())
            points_native = np.concatenate(points, axis=0).astype(np.float64)
            points_annotation = transform_points(points_native.astype(np.float32), inverse_transform)
            distances, nearest = tree.query(points_annotation, k=1)
            distance_blocks.append(distances)
            label_blocks.append((global_ids[nearest] != 0) & (semantic_ids[nearest] != 0))
            rows.append(
                {
                    "pair": pair_run.name,
                    "side": side,
                    "samples": len(points_annotation),
                    "nonzero_instance_label_rate": float(np.mean(global_ids[nearest] != 0)),
                    "nonzero_semantic_label_rate": float(np.mean(semantic_ids[nearest] != 0)),
                    "median_nearest_mesh_distance_m": float(np.median(distances)),
                    "p95_nearest_mesh_distance_m": float(np.quantile(distances, 0.95)),
                    "fraction_distance_gt_0.10m": float(np.mean(distances > 0.10)),
                    "fraction_distance_gt_0.15m": float(np.mean(distances > 0.15)),
                    "fraction_distance_gt_0.30m": float(np.mean(distances > 0.30)),
                    "max_nearest_mesh_distance_m": float(np.max(distances)),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    all_distances = np.concatenate(distance_blocks)
    all_labels = np.concatenate(label_blocks)
    summary = {
        "status": "complete",
        "rows": len(rows),
        "samples": int(len(all_distances)),
        "nonzero_label_rate": float(np.mean(all_labels)),
        "median_nearest_mesh_distance_m": float(np.median(all_distances)),
        "p95_nearest_mesh_distance_m": float(np.quantile(all_distances, 0.95)),
        "fraction_distance_gt_0.10m": float(np.mean(all_distances > 0.10)),
        "fraction_distance_gt_0.15m": float(np.mean(all_distances > 0.15)),
        "fraction_distance_gt_0.30m": float(np.mean(all_distances > 0.30)),
        "max_nearest_mesh_distance_m": float(np.max(all_distances)),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
