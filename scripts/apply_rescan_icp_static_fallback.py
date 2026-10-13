#!/usr/bin/env python3
"""Apply an ICP-quality gate, falling back to the source's static pose."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from evaluate_rescan_full_scene_pose_transfer import rank_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-fitness", type=float, default=0.0)
    parser.add_argument("--max-rmse", type=float, default=float("inf"))
    parser.add_argument("--max-translation-update", type=float, default=float("inf"))
    parser.add_argument("--max-rotation-update", type=float, default=float("inf"))
    args = parser.parse_args()
    with args.candidate_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with args.diagnostics_csv.open(newline="", encoding="utf-8") as stream:
        diagnostics = {
            (r["scene"], r["target_scan"], r["source_instance_id"]): r
            for r in csv.DictReader(stream)
        }
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["target_scan"], row["source_instance_id"])].append(row)

    accepted = 0
    selection_rows = []
    gated_rows: list[dict[str, str]] = []
    identity = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    for key, candidates in groups.items():
        if key not in diagnostics:
            raise ValueError(f"missing ICP diagnostics for {key}")
        diag = diagnostics[key]
        okay = (
            float(diag["icp_fitness"]) >= args.min_fitness
            and float(diag["icp_inlier_rmse_m"]) <= args.max_rmse
            and float(diag["translation_update_m"]) <= args.max_translation_update
            and float(diag["rotation_update_degrees"]) <= args.max_rotation_update
        )
        selected = rank_group(candidates, "color")
        if okay:
            accepted += 1
        else:
            for r in range(4):
                for c in range(4):
                    selected[f"transform_{r}{c}"] = repr(identity[r][c])
        selected["used_icp_pose"] = str(okay).lower()
        gated_rows.append(selected)
        selection_rows.append({
            "scene": key[0],
            "target_scan": key[1],
            "source_instance_id": key[2],
            "icp_fitness": diag["icp_fitness"],
            "icp_inlier_rmse_m": diag["icp_inlier_rmse_m"],
            "translation_update_m": diag["translation_update_m"],
            "rotation_update_degrees": diag["rotation_update_degrees"],
            "used_icp_pose": okay,
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gated_rows[0]))
        writer.writeheader()
        writer.writerows(gated_rows)
    report_path = args.output_csv.with_name(args.output_csv.stem + "_gate.csv")
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)
    print(f"used ICP for {accepted}/{len(groups)} pairs; static fallback for {len(groups)-accepted}")


if __name__ == "__main__":
    main()
