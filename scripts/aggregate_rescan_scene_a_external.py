#!/usr/bin/env python3
"""Aggregate the separately held-out single-site Rescan scene-A probe."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/reports/20261006_rescan_scene_a_external")


def metrics(frame: pd.DataFrame, method: str) -> dict[str, float | int]:
    selected = frame[frame.method == method]
    denominator = int(selected.target_present_queries.sum())
    physical = int(selected.physical_correct.sum())
    semantic = int(selected.semantic_correct_present.sum())
    return {
        "target_present_queries": denominator,
        "physical_correct": physical,
        "semantic_correct": semantic,
        "physical_recall": physical / denominator,
        "semantic_recall": semantic / denominator,
        "semantic_minus_physical_gap": (semantic - physical) / denominator,
    }


def main() -> None:
    small = pd.read_csv(ROOT / "dino_s/pair_metrics.csv")
    giant = pd.read_csv(ROOT / "dino_g/pair_metrics.csv")
    small = small[small.k == 20].copy()
    giant = giant[giant.k == 20].copy()
    keys = ["scene", "source_scan", "target_scan", "method"]
    merged = small.merge(giant, on=keys, suffixes=("_s", "_g"), validate="one_to_one")
    if len(merged) != 6 or merged.target_scan.nunique() != 3:
        raise ValueError("expected scene A's 3 target scans and two methods")
    for column in ("queries", "target_present_queries"):
        if not (merged[f"{column}_s"] == merged[f"{column}_g"]).all():
            raise ValueError(f"DINO-S/G {column} mismatch")
    color = merged[merged.method == "rgb_color"]
    for column in ("physical_correct", "semantic_correct_present"):
        if not (color[f"{column}_s"] == color[f"{column}_g"]).all():
            raise ValueError(f"RGB baseline mismatch in {column}")

    scores = {
        "dino_s": metrics(small, "dino_synthetic_views"),
        "dino_g": metrics(giant, "dino_synthetic_views"),
        "rgb_color": metrics(small, "rgb_color"),
    }
    for scale, values in scores.items():
        values["semantic_minus_physical_gap_pp"] = values["semantic_minus_physical_gap"] * 100
    per_scan_rows = []
    for target in sorted(merged.target_scan.unique()):
        row: dict[str, object] = {"target_scan": target}
        for scale, source in (("dino_s", small), ("dino_g", giant), ("rgb_color", small)):
            subset = source[(source.target_scan == target) & (source.method == ("rgb_color" if scale == "rgb_color" else "dino_synthetic_views"))]
            row[f"{scale}_queries"] = int(subset.queries.sum())
            row[f"{scale}_target_present"] = int(subset.target_present_queries.sum())
            row[f"{scale}_physical_recall"] = float(subset.physical_correct.sum() / subset.target_present_queries.sum())
            row[f"{scale}_semantic_recall"] = float(subset.semantic_correct_present.sum() / subset.target_present_queries.sum())
        per_scan_rows.append(row)
    result = {
        "status": "complete",
        "scene": "scene_a",
        "target_scans": int(merged.target_scan.nunique()),
        "statistical_unit": "one physical space; scan rows are repeated captures, not independent sites",
        "not_in_previous_b_to_m_audit": True,
        "protocol": "same synthetic-view DINO-S/G point retrieval, 5 cm candidate pool, top-20, and same-pool RGB baseline as B-M audit",
        "scores": scores,
        "relative_to_rgb_pp": {
            "dino_s_physical": (scores["dino_s"]["physical_recall"] - scores["rgb_color"]["physical_recall"]) * 100,
            "dino_s_semantic": (scores["dino_s"]["semantic_recall"] - scores["rgb_color"]["semantic_recall"]) * 100,
            "dino_g_physical": (scores["dino_g"]["physical_recall"] - scores["rgb_color"]["physical_recall"]) * 100,
            "dino_g_semantic": (scores["dino_g"]["semantic_recall"] - scores["rgb_color"]["semantic_recall"]) * 100,
        },
        "per_target_scan": per_scan_rows,
    }
    (ROOT / "scene_a_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(per_scan_rows).to_csv(ROOT / "scene_a_per_scan.csv", index=False)

    lines = [
        "# Separate Rescan scene-A check", "",
        "Scene A was absent from the original local B-M extraction and from the earlier Rescan geometry diagnostics and rendering pilot. Only its four labeled PLY reconstructions were range-extracted from the official 25 GB archive (35,829,882 compressed bytes); no raw video streams were downloaded. The three later scans were queried against scan 0 using the same point-cloud rendering, DINO-S/G, query construction, and RGB baseline as the B-M audit.", "",
        "| Descriptor | Physical Recall@20 | Semantic Recall@20 | Semantic minus physical |",
        "|---|---:|---:|---:|",
    ]
    for scale, label in (("dino_s", "DINOv2-S"), ("dino_g", "DINOv2-G"), ("rgb_color", "RGB distance")):
        value = scores[scale]
        lines.append(f"| {label} | {value['physical_recall']*100:.1f}% | {value['semantic_recall']*100:.1f}% | {value['semantic_minus_physical_gap']*100:.1f} pp |")
    lines.extend([
        "",
        f"DINO-G minus RGB is {result['relative_to_rgb_pp']['dino_g_physical']:+.1f} pp physical and {result['relative_to_rgb_pp']['dino_g_semantic']:+.1f} pp semantic Recall@20. Thus the semantic-over-instance gap persists in this single additional space, but DINO does not beat the color baseline here. DINO-S is also below RGB on both metrics.",
        "",
        "This is one physical site with three repeated captures, so it supports no site-level confidence interval and is not a confirmatory sample. It is reported separately from the B-M estimates; do not pool the three scans as independent sites or tune on this result. The check uses rendered annotated point clouds, not raw RGB-D streams, and evaluates no pose.",
        "",
        "The source archive inventory and minimal scan manifest are recorded alongside the probe outputs. Reproduce feature extraction with `scripts/probe_rescan_dino_point_candidates.py` and aggregation with `scripts/aggregate_rescan_scene_a_external.py`.",
    ])
    (ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote scene-A metrics to {ROOT}")


if __name__ == "__main__":
    main()
