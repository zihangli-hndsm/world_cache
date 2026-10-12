#!/usr/bin/env python3
"""Combine frozen HILO identity pilots without pooling duplicate IDs as new IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [json.loads(path.read_text()) for path in args.input]
    invariant_keys = ("backbone", "descriptor", "matching", "mask_depth_tolerance_mm", "reference")
    for key in invariant_keys:
        if len({source.get(key) for source in sources}) != 1:
            raise ValueError(f"input cohorts disagree on {key}")
    scenes = [scene for source in sources for scene in source["scenes"]]
    if len(scenes) != len(set(scenes)):
        raise ValueError("input cohorts contain duplicate scene configurations")
    result_rows = [row for source in sources for row in source["results"]]
    query_rows = [row for source in sources for row in source["query_outcomes"]]
    scene_rows = [row for source in sources for row in source["scene_macro"]]
    query_count = sum(row["object_queries"] for row in result_rows)
    dino_correct = sum(row["top1_correct"] for row in result_rows)
    dino_mutual_accepts = sum(row["mutual_accepts"] for row in result_rows)
    dino_mutual_correct = sum(row["mutual_correct"] for row in result_rows)
    rgb_correct = sum(row["rgb_histogram_top1_correct"] for row in result_rows)
    rgb_mutual_accepts = sum(row["rgb_histogram_mutual_accepts"] for row in result_rows)
    rgb_mutual_correct = sum(row["rgb_histogram_mutual_correct"] for row in result_rows)
    summary = {
        **{key: sources[0][key] for key in invariant_keys},
        "dataset": "HILO",
        "license": "cc-by-4.0",
        "scenes": scenes,
        "cohort_inputs": [str(path) for path in args.input],
        "cohort_protocol": (
            "first eight configurations and the next eight in the predeclared "
            "sha256(hilo_pose_pilot_v1:scene_folder) order among official _c1 folders"
        ),
        "identity_labels_used_during_features_or_matching": False,
        "ground_truth_mesh_poses_used_for_evaluation_masks": True,
        "mask_depth_tolerance_mm": 20,
        "visible_mask_pixels_by_frame": {
            key: value for source in sources
            for key, value in source["visible_mask_pixels_by_frame"].items()
        },
        "feature_patch_counts_by_frame": {
            key: value for source in sources
            for key, value in source["feature_patch_counts_by_frame"].items()
        },
        "scene_macro": scene_rows,
        "pooled": {
            "object_queries": query_count,
            "dino_top1_correct": dino_correct,
            "dino_top1_accuracy": dino_correct / query_count,
            "dino_mutual_correct": dino_mutual_correct,
            "dino_mutual_accepts": dino_mutual_accepts,
            "dino_mutual_precision": dino_mutual_correct / dino_mutual_accepts,
            "dino_mutual_coverage": dino_mutual_accepts / query_count,
            "rgb_histogram_top1_correct": rgb_correct,
            "rgb_histogram_top1_accuracy": rgb_correct / query_count,
            "rgb_histogram_mutual_correct": rgb_mutual_correct,
            "rgb_histogram_mutual_accepts": rgb_mutual_accepts,
            "rgb_histogram_mutual_precision": rgb_mutual_correct / rgb_mutual_accepts,
            "rgb_histogram_mutual_coverage": rgb_mutual_accepts / query_count,
        },
        "results": result_rows,
        "query_outcomes": query_rows,
        "scope": (
            "16 hash-selected controlled tabletop configurations; object-centric "
            "view-transfer stress test, not room-rescan/site validation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["pooled"], indent=2))
    print(f"configurations={len(scenes)} unique_scene_ids={len(set(scenes))} query_rows={len(query_rows)}")


if __name__ == "__main__":
    main()
