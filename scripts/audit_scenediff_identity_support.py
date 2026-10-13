#!/usr/bin/env python3
"""Inventory linked-object and hard-negative support in SceneDiff test annotations."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from evaluate_scenediff_identity_transfer import ARCHIVE_BYTES, ARCHIVE_URL, REVISION, canonical_label
from stream_rescan_archive_subset import HTTPRangeReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    remote = HTTPRangeReader(ARCHIVE_URL)
    if remote.size != ARCHIVE_BYTES:
        raise ValueError("pinned SceneDiff archive size changed")
    rows = []
    with ZipFile(remote) as archive:
        split_path = f"scenediff_benchmark/splits/{args.split}_split.json"
        splits = json.loads(archive.read(split_path))
        for subset, pairs in splits.items():
            for pair in pairs:
                prefix = f"scenediff_benchmark/data/{pair}/"
                seg = pickle.loads(archive.read(prefix + "segments.pkl"))
                metadata = {
                    str(item.get("original_obj_idx", i)): item
                    for i, item in enumerate(seg["objects"])
                }
                source = {str(k) for k in seg["video1_objects"]}
                target = {str(k) for k in seg["video2_objects"]}
                common = source & target
                candidate_labels = [
                    canonical_label(str(metadata.get(key, {}).get("label", key)))
                    for key in source
                ]
                same_label = {}
                for key in common:
                    label = canonical_label(str(metadata.get(key, {}).get("label", key)))
                    same_label[key] = sum(candidate == label for candidate in candidate_labels)
                added = sum(bool(v.get("in_video2")) and not bool(v.get("in_video1")) for v in metadata.values())
                removed = sum(bool(v.get("in_video1")) and not bool(v.get("in_video2")) for v in metadata.values())
                moved = sum(bool(v.get("in_video1")) and bool(v.get("in_video2")) for v in metadata.values())
                rows.append({
                    "pair": pair,
                    "subset": subset,
                    "scene_type": seg.get("scene_type", ""),
                    "annotated_changed_objects": len(metadata),
                    "video1_object_ids": len(source),
                    "video2_object_ids": len(target),
                    "objects_present_both": len(common),
                    "identity_queries_with_candidate_negative": sum(len(source) >= 2 for _ in common),
                    "queries_with_same_label_distractor": sum(count >= 2 for count in same_label.values()),
                    "same_label_query_fraction": (
                        sum(count >= 2 for count in same_label.values()) / len(common) if common else 0.0
                    ),
                    "added": added,
                    "removed": removed,
                    "moved_or_persistent_identity": moved,
                })
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / f"{args.split}_annotation_support.csv", index=False)
    summary = {
        "dataset": "SceneDiff",
        "dataset_revision": REVISION,
        "archive_bytes": ARCHIVE_BYTES,
        "official_split": args.split,
        "pair_count": len(frame),
        "by_subset": {
            subset: {
                "pairs": int(len(group)),
                "annotated_changed_objects": int(group.annotated_changed_objects.sum()),
                "objects_present_both": int(group.objects_present_both.sum()),
                "queries_with_candidate_negative": int(group.identity_queries_with_candidate_negative.sum()),
                "queries_with_same_label_distractor": int(group.queries_with_same_label_distractor.sum()),
                "pairs_with_at_least_two_source_candidates": int((group.video1_object_ids >= 2).sum()),
                "median_source_object_count": float(group.video1_object_ids.median()),
            }
            for subset, group in frame.groupby("subset")
        },
        "totals": {
            "annotated_changed_objects": int(frame.annotated_changed_objects.sum()),
            "objects_present_both": int(frame.objects_present_both.sum()),
            "queries_with_candidate_negative": int(frame.identity_queries_with_candidate_negative.sum()),
            "queries_with_same_label_distractor": int(frame.queries_with_same_label_distractor.sum()),
            "pairs_with_at_least_two_source_candidates": int((frame.video1_object_ids >= 2).sum()),
        },
        "range_reader_bytes_fetched": remote.bytes_downloaded,
        "interpretation": "Annotation-level feasibility only; no video frames or model features were read. Candidate support assumes every labeled object is visible in sampled frames.",
    }
    (args.output_dir / f"{args.split}_annotation_support.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
