#!/usr/bin/env python3
"""Evaluate a label-preserving XYZ nearest-neighbor baseline on Rescan.

The source is the first annotated reconstruction in each scene. Its instance
IDs are transferred to later scans by nearest XYZ within the fixed 5 cm radius
used by the Rescan paper's segmentation-transfer stage. Target instance labels
are used only after prediction to compute the official Instance Transfer mIoU
and ambiguity remapping from the Rescan evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PLY_TYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}
MAX_INSTANCE_ID = 255
UNKNOWN_ID = MAX_INSTANCE_ID + 1


def rescan_eval_instance_ids(raw_ids: np.ndarray) -> np.ndarray:
    """Match `create_eval_files::write_instance_transfer_file` ID encoding."""
    raw_ids = raw_ids.astype(np.int32, copy=False)
    result = raw_ids + 1
    result[(raw_ids < 0) | (raw_ids == 1024)] = 0
    return result


def read_annotated_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        properties: list[tuple[str, str]] = []
        vertex_count: int | None = None
        is_binary_little_endian = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"unterminated PLY header: {path}")
            text = line.decode("ascii", errors="strict").strip()
            if text == "format binary_little_endian 1.0":
                is_binary_little_endian = True
            elif text.startswith("element vertex "):
                vertex_count = int(text.split()[2])
            elif text.startswith("property ") and vertex_count is not None:
                _, scalar_type, name = text.split()
                if scalar_type not in PLY_TYPES:
                    raise ValueError(f"unsupported PLY type {scalar_type!r} in {path}")
                properties.append((name, PLY_TYPES[scalar_type]))
            elif text == "end_header":
                break
        if not is_binary_little_endian or vertex_count is None:
            raise ValueError(f"expected binary little-endian vertex PLY: {path}")
        dtype = np.dtype(properties)
        vertices = np.fromfile(stream, dtype=dtype, count=vertex_count)
    required = {"x", "y", "z", "class_idx", "instance_idx"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"missing required vertex fields in {path}")
    points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    return points, vertices["instance_idx"].astype(np.int32), vertices["class_idx"].astype(np.int32)


def apply_ambiguity_permutations(confusion: np.ndarray, path: Path) -> None:
    """Port `reorder_confusion` from the official Rescan evaluator."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        left, right = line.replace("|", " ").split()[0], line.replace("|", " ").split()[1:]
        base_id = int(left) + 1
        equivalent_ids = [int(value) + 1 for value in right]
        best_id = max(equivalent_ids, key=lambda value: confusion[base_id, value])
        if base_id != best_id:
            confusion[base_id, base_id], confusion[base_id, best_id] = (
                confusion[base_id, best_id],
                confusion[base_id, base_id],
            )


def instance_transfer_miou(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    permutation_file: Path,
) -> tuple[float, int]:
    gt = ground_truth.astype(np.int64, copy=False)
    pred = prediction.astype(np.int64, copy=True)
    valid_gt = (gt >= 0) & (gt <= MAX_INSTANCE_ID)
    pred[(pred < 0) | (pred > MAX_INSTANCE_ID)] = UNKNOWN_ID
    flat = gt[valid_gt] * (UNKNOWN_ID + 1) + pred[valid_gt]
    confusion = np.bincount(flat, minlength=(UNKNOWN_ID + 1) ** 2).reshape(
        UNKNOWN_ID + 1, UNKNOWN_ID + 1
    )
    apply_ambiguity_permutations(confusion, permutation_file)
    instance_ids = np.unique(gt[valid_gt])
    instance_ids = instance_ids[instance_ids != 0]
    if not len(instance_ids):
        return float("nan"), 0
    scores = []
    for instance_id in instance_ids:
        tp = confusion[instance_id, instance_id]
        fn = confusion[instance_id].sum() - tp
        fp = confusion[: MAX_INSTANCE_ID + 1, instance_id].sum() - tp
        denom = tp + fp + fn
        if denom:
            scores.append(tp / denom)
    return (float(np.mean(scores)) if scores else float("nan"), len(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True, help="extracted Rescan annotation archive")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_distance <= 0:
        raise ValueError("--max-distance must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_list = args.data_root / "scene_list.txt"
    if not scene_list.is_file():
        raise FileNotFoundError(scene_list)
    scenes = [line.strip() for line in scene_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, object]] = []
    for scene in scenes:
        annotation_dir = args.data_root / scene / "gt_segmentation"
        source_path = annotation_dir / f"{scene}_0.ply"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_points, source_ids_raw, _ = read_annotated_ply(source_path)
        source_ids = rescan_eval_instance_ids(source_ids_raw)
        tree = cKDTree(source_points)
        targets = sorted(annotation_dir.glob(f"{scene}_[1-9]*.ply"))
        if not targets:
            raise ValueError(f"no future scans found for {scene}")
        for target_path in targets:
            query_points, target_ids_raw, _ = read_annotated_ply(target_path)
            target_ids = rescan_eval_instance_ids(target_ids_raw)
            distances, nearest = tree.query(query_points, k=1, workers=-1)
            prediction = np.where(distances <= args.max_distance, source_ids[nearest], UNKNOWN_ID)
            miou, instance_count = instance_transfer_miou(
                target_ids, prediction, target_path.with_suffix(".txt")
            )
            rows.append({
                "scene": scene,
                "source_scan": source_path.stem,
                "target_scan": target_path.stem,
                "source_points": len(source_points),
                "target_points": len(query_points),
                "target_instances_scored": instance_count,
                "target_point_coverage_at_5cm": float(np.mean(distances <= args.max_distance)),
                "nearest_xyz_distance_median_m": float(np.median(distances)),
                "nearest_xyz_distance_p90_m": float(np.quantile(distances, 0.9)),
                "instance_transfer_miou": miou,
                "target_labels_used_for_prediction": False,
            })
            print(
                f"{target_path.stem}: coverage={rows[-1]['target_point_coverage_at_5cm']:.3f} "
                f"median_nn={rows[-1]['nearest_xyz_distance_median_m']:.3f}m "
                f"InstanceTransfer-mIoU={miou:.4f}",
                flush=True,
            )

    csv_path = args.output_dir / "per_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_scene: dict[str, dict[str, float | int]] = {}
    for scene in scenes:
        subset = [row for row in rows if row["scene"] == scene]
        per_scene[scene] = {
            "target_scans": len(subset),
            "mean_instance_transfer_miou": float(np.nanmean([row["instance_transfer_miou"] for row in subset])),
            "mean_target_point_coverage_at_5cm": float(np.mean([row["target_point_coverage_at_5cm"] for row in subset])),
        }
    summary = {
        "baseline": "first-scan XYZ nearest-neighbor label transfer",
        "source_scan": "scan _0 in each scene",
        "target_scans": len(rows),
        "scenes": len(per_scene),
        "max_distance_m": args.max_distance,
        "features_used": ["XYZ only"],
        "source_instance_labels_used": True,
        "target_instance_labels_used_for_prediction": False,
        "official_metric_port": "Rescan/scripts/evaluate_instance_transfer.py; ambiguity permutations applied per target scan",
        "mean_instance_transfer_miou_over_future_scans": float(np.nanmean([row["instance_transfer_miou"] for row in rows])),
        "mean_point_coverage_at_5cm_over_future_scans": float(np.mean([row["target_point_coverage_at_5cm"] for row in rows])),
        "per_scene": per_scene,
        "limitations": [
            "geometry-only static-coordinate nearest-neighbor baseline; does not test the RGB-D/DINO motion method",
            "uses the first scan only and does not update a temporal object model",
            "coordinate-frame behavior and instance-label equivalence follow the released Rescan annotations",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rescan static XYZ label-transfer baseline",
        "",
        "This diagnostic transfers source instance IDs from scan `_0` to each later reconstruction by nearest XYZ, accepting matches within 5 cm. It uses no RGB, learned descriptors, motion hypotheses, or target labels during prediction. Target annotations are used only for scoring. The Instance Transfer mIoU and ambiguity-permutation handling follow the official Rescan evaluator.",
        "",
        f"- Scenes: {summary['scenes']}",
        f"- Future scans: {summary['target_scans']}",
        f"- Mean future-scan Instance Transfer mIoU: {summary['mean_instance_transfer_miou_over_future_scans']:.4f}",
        f"- Mean future-point coverage within 5 cm: {summary['mean_point_coverage_at_5cm_over_future_scans']:.3f}",
        "",
        "## Per-scene results",
        "",
        "| Scene | Future scans | Instance Transfer mIoU | 5 cm point coverage |",
        "|---|---:|---:|---:|",
    ]
    for scene, values in per_scene.items():
        lines.append(
            f"| {scene} | {values['target_scans']} | {values['mean_instance_transfer_miou']:.4f} | "
            f"{values['mean_target_point_coverage_at_5cm']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "This is a geometry-only diagnostic baseline on an independent dataset. It does not test the project's RGB-D/DINO multi-motion method, and its score should not be compared directly with the Rescan paper's full temporal-model result. Use it to establish the coordinate-transfer floor and the feasibility of a leakage-controlled external evaluation.",
        "",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
