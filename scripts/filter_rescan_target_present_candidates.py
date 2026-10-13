#!/usr/bin/env python3
"""Keep only Rescan source-instance queries annotated in their target scan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rerank_rescan_fpfh_by_color import read_scene_ply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--excluded-csv", type=Path)
    args = parser.parse_args()

    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("candidate CSV is empty")

    target_ids: dict[tuple[str, str], set[int]] = {}
    eligible: set[tuple[str, str, str]] = set()
    excluded: set[tuple[str, str, str]] = set()
    for row in rows:
        scene, scan = row["scene"], row["target_scan"]
        target_key = (scene, scan)
        if target_key not in target_ids:
            _, _, instance_ids, _ = read_scene_ply(
                args.data_root / scene / "gt_segmentation" / f"{scan}.ply"
            )
            target_ids[target_key] = set(map(int, instance_ids))
        pair = (scene, scan, row["source_instance_id"])
        (eligible if int(pair[2]) in target_ids[target_key] else excluded).add(pair)

    keep = [row for row in rows if (row["scene"], row["target_scan"], row["source_instance_id"]) in eligible]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def write(path: Path, selected: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)

    write(args.output_csv, keep)
    if args.excluded_csv:
        rejected = [row for row in rows if (row["scene"], row["target_scan"], row["source_instance_id"]) in excluded]
        args.excluded_csv.parent.mkdir(parents=True, exist_ok=True)
        write(args.excluded_csv, rejected)

    if eligible & excluded:
        raise ValueError("a query was inconsistently classified as both eligible and excluded")
    print(
        f"candidate_groups={len(eligible) + len(excluded)} eligible_groups={len(eligible)} "
        f"excluded_groups={len(excluded)} candidate_rows={len(keep)}"
    )


if __name__ == "__main__":
    main()
