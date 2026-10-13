#!/usr/bin/env python3
"""Probe cross-scan instance color stability with target segmentation oracle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluate_rescan_xyz_transfer import PLY_TYPES


def read_labels_and_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    properties: list[tuple[str, str]] = []
    vertex_count: int | None = None
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"unterminated PLY header: {path}")
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[2])
            elif text.startswith("property ") and vertex_count is not None:
                _, scalar_type, name = text.split()
                if scalar_type not in PLY_TYPES:
                    raise ValueError(f"unsupported PLY type {scalar_type!r} in {path}")
                properties.append((name, PLY_TYPES[scalar_type]))
            elif text == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count in {path}")
        vertices = np.fromfile(stream, dtype=np.dtype(properties), count=vertex_count)
    required = {"class_idx", "instance_idx", "red", "green", "blue"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"missing required label/RGB fields in {path}")
    labels = vertices["instance_idx"].astype(np.int32)
    classes = vertices["class_idx"].astype(np.int32)
    rgb = np.column_stack([vertices[c] for c in ("red", "green", "blue")]).astype(np.float32) / 255.0
    return labels, classes, rgb


def dominant_class(classes: np.ndarray) -> int:
    return int(np.bincount(classes.clip(min=0)).argmax())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_list = [
        line.strip()
        for line in (args.data_root / "scene_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, object]] = []
    for scene in scene_list:
        ann = args.data_root / scene / "gt_segmentation"
        source_scan = f"{scene}_0"
        source_ids, source_classes, source_rgb = read_labels_and_rgb(ann / f"{source_scan}.ply")
        source_objects = [
            int(raw_id)
            for raw_id in np.unique(source_ids)
            if 0 <= raw_id < 1024 and np.count_nonzero(source_ids == raw_id) >= 300
        ]
        for target_path in sorted(ann.glob(f"{scene}_[1-9]*.ply")):
            target_ids, target_classes, target_rgb = read_labels_and_rgb(target_path)
            target_objects = [int(i) for i in np.unique(target_ids) if 0 <= i < 1024]
            target_color = {
                raw_id: target_rgb[target_ids == raw_id].mean(axis=0)
                for raw_id in target_objects
            }
            target_class = {
                raw_id: dominant_class(target_classes[target_ids == raw_id])
                for raw_id in target_objects
            }
            for raw_id in source_objects:
                if raw_id not in target_color:
                    continue
                source_mask = source_ids == raw_id
                source_mean = source_rgb[source_mask].mean(axis=0)
                source_class = dominant_class(source_classes[source_mask])
                exact_color_distance = float(np.linalg.norm(source_mean - target_color[raw_id]))
                all_ranked = sorted(
                    target_objects,
                    key=lambda candidate: (
                        float(np.linalg.norm(source_mean - target_color[candidate])), candidate
                    ),
                )
                same_class = [candidate for candidate in target_objects if target_class[candidate] == source_class]
                class_ranked = sorted(
                    same_class,
                    key=lambda candidate: (
                        float(np.linalg.norm(source_mean - target_color[candidate])), candidate
                    ),
                )
                all_rank = all_ranked.index(raw_id) + 1
                class_rank = class_ranked.index(raw_id) + 1 if raw_id in class_ranked else 10**9
                row: dict[str, object] = {
                    "scene": scene,
                    "source_scan": source_scan,
                    "target_scan": target_path.stem,
                    "source_instance_id": raw_id,
                    "source_class_id": source_class,
                    "target_instance_count": len(target_objects),
                    "same_class_target_count": len(same_class),
                    "same_instance_mean_rgb_l2": exact_color_distance,
                    "all_target_color_rank": all_rank,
                    "same_class_color_rank": class_rank,
                    "all_target_recall_at_1": all_rank == 1,
                    "all_target_recall_at_5": all_rank <= 5,
                    "same_class_recall_at_1": class_rank == 1,
                    "same_class_recall_at_5": class_rank <= 5,
                }
                rows.append(row)
            print(f"completed {target_path.stem}; evaluated_objects={sum(r['target_scan'] == target_path.stem for r in rows)}", flush=True)

    if not rows:
        raise RuntimeError("no source objects with matching target identities were evaluated")
    with (args.output_dir / "color_oracle_per_object.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "scenes": len({row["scene"] for row in rows}),
        "target_scans": len({row["target_scan"] for row in rows}),
        "object_target_pairs": len(rows),
        "descriptor": "mean RGB per target instance",
        "target_instance_segmentation_used_for_descriptor": True,
        "target_instance_labels_used_for_scoring": True,
        "all_target_recall_at_1": float(np.mean([row["all_target_recall_at_1"] for row in rows])),
        "all_target_recall_at_5": float(np.mean([row["all_target_recall_at_5"] for row in rows])),
        "same_class_recall_at_1": float(np.mean([row["same_class_recall_at_1"] for row in rows])),
        "same_class_recall_at_5": float(np.mean([row["same_class_recall_at_5"] for row in rows])),
        "limitations": [
            "target instances are given to compute target RGB descriptors, so this is an oracle retrieval ceiling, not a deployable method",
            "uses mean RGB only and does not estimate pose or transfer labels",
        ],
    }
    (args.output_dir / "color_oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Rescan instance color separability oracle",
        "",
        "Mean RGB of each target instance is computed using its target segmentation, then source instances are ranked by Euclidean mean-RGB distance. Target segmentation is therefore leaked into the descriptor and this is a diagnostic ceiling only.",
        "",
        f"- Scenes / target scans / object-target pairs: {summary['scenes']} / {summary['target_scans']} / {summary['object_target_pairs']}",
        f"- All-target color retrieval R@1 / R@5: {summary['all_target_recall_at_1']:.3f} / {summary['all_target_recall_at_5']:.3f}",
        f"- Same-class color retrieval R@1 / R@5: {summary['same_class_recall_at_1']:.3f} / {summary['same_class_recall_at_5']:.3f}",
        "",
        "These scores test whether RGB is stable/discriminative enough to merit an unsegmented-scene reranking method. They are not a label-transfer score.",
    ]
    (args.output_dir / "color_oracle_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
