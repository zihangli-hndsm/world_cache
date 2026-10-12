#!/usr/bin/env python3
"""Rerank FPFH pose proposals by unsegmented-scene RGB agreement."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from evaluate_rescan_xyz_transfer import PLY_TYPES


def read_scene_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    required = {"x", "y", "z", "red", "green", "blue", "class_idx", "instance_idx"}
    if not required.issubset(vertices.dtype.names or ()):
        raise ValueError(f"missing required fields in {path}")
    points = np.column_stack([vertices[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    colors = np.column_stack([vertices[axis] for axis in ("red", "green", "blue")]).astype(np.float32) / 255.0
    ids = vertices["instance_idx"].astype(np.int32)
    classes = vertices["class_idx"].astype(np.int32)
    return points, colors, ids, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spatial-threshold", type=float, default=0.12)
    parser.add_argument("--color-threshold", type=float, default=0.15)
    args = parser.parse_args()
    if args.spatial_threshold <= 0 or args.color_threshold <= 0:
        raise ValueError("spatial and color thresholds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        candidate_rows = list(csv.DictReader(stream))
    if not candidate_rows:
        raise ValueError(f"no candidates in {args.candidate_csv}")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)
    target_presence_known = all("target_instance_present" in row for row in candidate_rows)
    includes_target_absent = target_presence_known and any(
        row["target_instance_present"].lower() == "false" for row in candidate_rows
    )

    scored_candidates: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    partial_candidates_path = args.output_dir / "color_rerank_candidates.partial.csv"
    partial_objects_path = args.output_dir / "color_rerank_objects.partial.csv"

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if rows:
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)

    if partial_candidates_path.exists() != partial_objects_path.exists():
        raise ValueError("incomplete rerank checkpoint: candidate/object partial CSVs must both exist")
    if partial_candidates_path.exists():
        with partial_candidates_path.open(newline="", encoding="utf-8") as stream:
            scored_candidates = list(csv.DictReader(stream))
        with partial_objects_path.open(newline="", encoding="utf-8") as stream:
            object_rows = list(csv.DictReader(stream))
        if not object_rows or not scored_candidates:
            raise ValueError("empty rerank checkpoint; remove both partial CSVs to restart")
        object_keys = {
            (row["scene"], row["target_scan"], str(row["source_instance_id"]))
            for row in object_rows
        }
        candidate_counts = Counter(
            (row["scene"], row["target_scan"], row["source_instance_id"])
            for row in scored_candidates
        )
        if len(object_keys) != len(object_rows):
            raise ValueError("rerank checkpoint has duplicate object rows")
        complete_keys = {key for key, count in candidate_counts.items() if count == 5}
        resumable_keys = object_keys & complete_keys
        if not resumable_keys:
            raise ValueError("rerank checkpoint has no complete object/candidate pairs")
        if any(len(groups.get(key, [])) != 5 for key in resumable_keys):
            raise ValueError("rerank checkpoint does not match the current five-proposal input")
        if resumable_keys != object_keys or any(count != 5 for count in candidate_counts.values()):
            # Candidate and object checkpoints are replaced atomically one file
            # at a time. If interrupted between replacements, resume from their
            # complete intersection and discard the mismatched target tail.
            object_rows = [
                row for row in object_rows
                if (row["scene"], row["target_scan"], str(row["source_instance_id"])) in resumable_keys
            ]
            scored_candidates = [
                row for row in scored_candidates
                if (row["scene"], row["target_scan"], row["source_instance_id"]) in resumable_keys
            ]
            print(f"reconciled checkpoint to {len(object_rows)} complete pairs", flush=True)
        print(f"resuming from {len(object_rows)} completed object-target pairs", flush=True)
    completed_groups = {
        (row["scene"], row["target_scan"], str(row["source_instance_id"]))
        for row in object_rows
    }
    group_keys = sorted(groups)
    current_target: tuple[str, str] | None = None
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_tree: cKDTree | None = None
    target_points = target_colors = target_ids = target_classes = None
    source_scan = ""

    for group_index, group_key in enumerate(group_keys):
        if group_key in completed_groups:
            continue
        scene, target_scan, raw_id_string = group_key
        if current_target != (scene, target_scan):
            ann = args.data_root / scene / "gt_segmentation"
            source_scan = f"{scene}_0"
            source_points, source_colors, source_ids, _ = read_scene_ply(ann / f"{source_scan}.ply")
            source_cache = {
                str(raw_id): (source_points[source_ids == int(raw_id)], source_colors[source_ids == int(raw_id)])
                for raw_id in {key[2] for key in groups if key[0] == scene}
            }
            target_points, target_colors, target_ids, target_classes = read_scene_ply(
                ann / f"{target_scan}.ply"
            )
            target_tree = cKDTree(target_points)
            current_target = (scene, target_scan)
            print(f"loaded {target_scan}", flush=True)

        source_points, source_colors = source_cache[raw_id_string]
        if len(source_points) == 0:
            raise ValueError(f"source instance {raw_id_string} missing from {source_scan}")
        candidate_scores = []
        for row in groups[group_key]:
            transform = np.asarray(
                [float(row[f"transform_{r}{c}"]) for r in range(4) for c in range(4)],
                dtype=np.float64,
            ).reshape(4, 4)
            transformed = source_points @ transform[:3, :3].T + transform[:3, 3]
            distances, nearest = target_tree.query(transformed, k=1, workers=1)
            rgb_distance = np.linalg.norm(source_colors - target_colors[nearest], axis=1)
            spatial_inlier = distances <= args.spatial_threshold
            joint_inlier = spatial_inlier & (rgb_distance <= args.color_threshold)
            median_rgb = float(np.median(rgb_distance[spatial_inlier])) if np.any(spatial_inlier) else float("inf")
            scored = {
                **row,
                "spatial_coverage": float(np.mean(spatial_inlier)),
                "median_rgb_distance_on_spatial_inliers": median_rgb,
                "joint_spatial_rgb_inlier_fraction": float(np.mean(joint_inlier)),
            }
            candidate_scores.append(scored)
            scored_candidates.append(scored)
        candidate_scores.sort(
            key=lambda row: (
                -float(row["joint_spatial_rgb_inlier_fraction"]),
                -float(row["spatial_coverage"]),
                float(row["median_rgb_distance_on_spatial_inliers"]),
                -float(row["fitness"]),
                int(row["repeat"]),
            )
        )
        base = groups[group_key][0]
        result: dict[str, object] = {
            "scene": scene,
            "source_scan": source_scan,
            "target_scan": target_scan,
            "source_instance_id": int(raw_id_string),
            "source_class_id": int(base["source_class_id"]),
            "candidate_count": len(candidate_scores),
        }
        for k in (1, 3, 5):
            top_color = candidate_scores[:k]
            top_fitness = sorted(
                candidate_scores,
                key=lambda row: (-float(row["fitness"]), int(row["repeat"])),
            )[:k]
            result[f"fitness_exact_recall_at_{k}"] = any(row["exact_instance_hit"].lower() == "true" for row in top_fitness)
            result[f"color_exact_recall_at_{k}"] = any(row["exact_instance_hit"].lower() == "true" for row in top_color)
            result[f"fitness_same_class_recall_at_{k}"] = any(row["same_class_hit"].lower() == "true" for row in top_fitness)
            result[f"color_same_class_recall_at_{k}"] = any(row["same_class_hit"].lower() == "true" for row in top_color)
        result["color_rank1_joint_inlier_fraction"] = candidate_scores[0]["joint_spatial_rgb_inlier_fraction"]
        result["color_rank1_median_rgb_distance"] = candidate_scores[0]["median_rgb_distance_on_spatial_inliers"]
        object_rows.append(result)
        next_target = group_keys[group_index + 1][:2] if group_index + 1 < len(group_keys) else None
        if next_target != (scene, target_scan):
            write_csv(partial_candidates_path, scored_candidates)
            write_csv(partial_objects_path, object_rows)
            print(
                f"checkpointed {target_scan}; completed_pairs={len(object_rows)}",
                flush=True,
            )

    write_csv(args.output_dir / "color_rerank_candidates.csv", scored_candidates)
    write_csv(args.output_dir / "color_rerank_objects.csv", object_rows)
    partial_candidates_path.unlink(missing_ok=True)
    partial_objects_path.unlink(missing_ok=True)

    def mean_hit(column: str) -> float:
        hits = []
        for row in object_rows:
            value = row[column]
            hits.append(value if isinstance(value, bool) else str(value).lower() == "true")
        return float(np.mean(hits))

    exact_1 = mean_hit("color_exact_recall_at_1")
    exact_5 = mean_hit("color_exact_recall_at_5")
    same_1 = mean_hit("color_same_class_recall_at_1")
    same_5 = mean_hit("color_same_class_recall_at_5")
    summary = {
        "scene_count": len({row["scene"] for row in object_rows}),
        "target_scan_count": len({row["target_scan"] for row in object_rows}),
        "object_target_pairs": len(object_rows),
        "proposal_rank_key": "joint raw-point spatial inlier fraction with RGB distance below fixed threshold",
        "spatial_threshold_m": args.spatial_threshold,
        "rgb_l2_threshold_0_1": args.color_threshold,
        "target_instance_labels_used_for_ranking": False,
        "target_instance_labels_used_for_pair_eligibility": not includes_target_absent,
        "target_instance_labels_used_for_scoring": True,
        "pair_eligibility": (
            "candidate set includes target-absent source instances; source eligibility is independent of target IDs"
            if includes_target_absent else
            "inherited from candidate generation: source raw IDs must be present in target annotations; metrics are conditional on annotated target presence"
        ),
        "scoring_definition": "exact-instance hit: transformed source centroid within 0.20 m of the target centroid for the same raw instance ID and same class; same-class hit: within 0.20 m of any target instance centroid of the same class; not full 6-DoF pose accuracy",
        "development_scene_for_threshold_selection": "scene_a",
        "evaluation_split": "12 held-out Rescan scenes B-M; thresholds frozen before evaluation",
        "color_exact_recall_at_1": exact_1,
        "color_exact_recall_at_5": exact_5,
        "color_same_class_recall_at_1": same_1,
        "color_same_class_recall_at_5": same_5,
        "fitness_exact_recall_at_1": mean_hit("fitness_exact_recall_at_1"),
        "fitness_exact_recall_at_5": mean_hit("fitness_exact_recall_at_5"),
        "fitness_same_class_recall_at_1": mean_hit("fitness_same_class_recall_at_1"),
        "fitness_same_class_recall_at_5": mean_hit("fitness_same_class_recall_at_5"),
        "limitations": [
            "only ranks five existing FPFH proposals; does not improve candidate recall",
            "target point RGB is used but target instance masks are not used for ranking",
            "thresholds were frozen using the scene_a development pilot; no held-out threshold tuning was performed",
        ],
    }
    (args.output_dir / "color_rerank_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Rescan FPFH proposal reranking by spatial support and RGB",
        "",
        "Rank five geometry-generated FPFH proposals by the fraction of source object points that land within the spatial threshold of unsegmented target points and have RGB L2 distance below the fixed color threshold. Target instance labels do not rank proposals, but define eligible source-target pairs (source IDs present in target) and score recall; metrics are conditional on annotated target presence. A spatial-only control is reported separately because this score combines geometric support and RGB.",
        "",
        "Scoring uses the saved candidate flags: exact-instance recall means the transformed source centroid lies within 0.20 m of the target centroid with the same raw instance ID and class; same-class recall means within 0.20 m of any target centroid of that class. These are centroid-localization metrics, not full 6-DoF pose accuracy.",
        "",
        f"- Scene / target scan / object-target pair: {summary['scene_count']} / {summary['target_scan_count']} / {summary['object_target_pairs']}",
        f"- Spatial / RGB threshold: {args.spatial_threshold:.3f} m / {args.color_threshold:.3f}",
        f"- Fitness exact recall@1 / @5: {summary['fitness_exact_recall_at_1']:.3f} / {summary['fitness_exact_recall_at_5']:.3f}",
        f"- Color exact recall@1 / @5: {exact_1:.3f} / {exact_5:.3f}",
        f"- Fitness same-class recall@1 / @5: {summary['fitness_same_class_recall_at_1']:.3f} / {summary['fitness_same_class_recall_at_5']:.3f}",
        f"- Color same-class recall@1 / @5: {same_1:.3f} / {same_5:.3f}",
        "",
        "Held-out evaluation: scene_a was used only to freeze the spatial/RGB thresholds; scenes B-M were scored once with those fixed thresholds. This reranker only ranks five existing FPFH proposals and cannot recover a pose absent from them.",
    ]
    (args.output_dir / "color_rerank_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
