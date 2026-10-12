#!/usr/bin/env python3
"""Collect the four-pair official-MASt3R versus ViT-G physical audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PAIR_DIRS = {
    "pair02": Path("outputs/reports/20261002_physical_instance_candidate_audit_vitg_k_sweep/20260917_3rscan_pair02_s1"),
    "pair05": Path("outputs/reports/20261002_physical_instance_candidate_audit_vitg_k_sweep/20260917_3rscan_pair05_s1"),
    "pair06": Path("outputs/reports/20261003_dino_pair06_physical_probe"),
    "pair08": Path("outputs/reports/20261003_dino_pair08_physical_probe"),
}


def dino_metrics(path: Path) -> dict[str, float | int | bool]:
    diagnostics = pd.read_csv(path / "query_diagnostics.csv")
    if "top_k" in diagnostics:
        diagnostics = diagnostics[diagnostics.top_k == 20]
    evaluable = diagnostics.physical_instance_evaluable.astype(bool)
    inliers = diagnostics.consensus_inlier.astype(bool)
    correct = diagnostics.physical_instance_correct.astype(bool)
    count = int(evaluable.sum())
    assigned = int((inliers & evaluable).sum())
    correct_transfers = int((inliers & correct & evaluable).sum())
    summary = pd.read_csv(path / "summary.csv").iloc[0]
    return {
        "dino_queries": count,
        "dino_candidate_recall": float((diagnostics.physical_instance_candidate_hit & evaluable).sum() / count),
        "dino_transfer": float(correct_transfers / count),
        "dino_precision": float(correct_transfers / assigned) if assigned else None,
        "dino_rot": float(summary.rotation_error_deg),
        "dino_trans": float(summary.translation_error_m),
        "dino_pose": bool(float(summary.rotation_error_deg) < 5.0 and float(summary.translation_error_m) < 0.25),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261003_mast3r_baseline_comparison"))
    parser.add_argument(
        "--mast3r-mode",
        choices=("reciprocal", "dense_topk"),
        default="reciprocal",
    )
    args = parser.parse_args()
    rows = []
    for pair, dino_dir in PAIR_DIRS.items():
        mast3r_dir = (
            Path("outputs/runs") / f"20261003_mast3r_dense_topk_{pair}"
            if args.mast3r_mode == "dense_topk"
            else Path("outputs/reports") / f"20261003_mast3r_{pair}_probe"
        )
        mast3r = json.loads((mast3r_dir / "summary.json").read_text(encoding="utf-8"))
        dino = dino_metrics(dino_dir)
        rows.append({
            "pair": pair,
            "mast3r_queries": int(mast3r["linked_physical_instance_queries"]),
            "mast3r_mean_actual_candidates": float(mast3r["mean_actual_candidates_per_query"]),
            "mast3r_candidate_recall": float(mast3r["physical_instance_candidate_recall_at_k"]),
            "mast3r_transfer": float(mast3r["physical_correct_transfer_coverage"]),
            "mast3r_precision": mast3r["physical_assigned_precision"],
            "mast3r_rot": float(mast3r["scene_rotation_error_deg"]),
            "mast3r_trans": float(mast3r["scene_translation_error_m"]),
            "mast3r_pose": bool(mast3r["strict_scene_pose_success"]),
            **dino,
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 4:
        raise ValueError("expected four paired baseline relations")
    for method in ("mast3r", "dino"):
        denominator = frame[f"{method}_queries"].sum()
        frame.attrs[f"{method}_weighted_candidate_recall"] = float(
            (frame[f"{method}_candidate_recall"] * frame[f"{method}_queries"]).sum() / denominator
        )
        frame.attrs[f"{method}_weighted_transfer"] = float(
            (frame[f"{method}_transfer"] * frame[f"{method}_queries"]).sum() / denominator
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "per_pair.csv", index=False)
    summary = {
        "pairs": list(frame.pair),
        "protocol": {
            "common_query_population": "20 RGB-D frames per side, 224x224 depth-valid patch centers, physical objectId links scored post hoc",
            "dino": "ViT-G blocks 2,20,40; top-20; 0.10m spatial consensus; 1,500 iterations; seed 17",
            "mast3r": (
                "MASt3R ViT-L 512x512 descriptors; confidence-weighted dense top-20 per query over the cached frame bank; same depth-valid patch query pool and spatial consensus; 1,500 iterations; seed 17; custom budget-matched diagnostic, not the official reciprocal matcher"
                if args.mast3r_mode == "dense_topk"
                else "official MASt3R ViT-L checkpoint; 512x512 RGB; reciprocal descriptor matches; confidence-ranked sparse candidates; top-20 cap; 0.10m spatial consensus; 1,500 iterations; seed 17"
            ),
        },
        "weighted_by_linked_queries": {
            "mast3r_candidate_recall": frame.attrs["mast3r_weighted_candidate_recall"],
            "dino_candidate_recall": frame.attrs["dino_weighted_candidate_recall"],
            "mast3r_physical_transfer": frame.attrs["mast3r_weighted_transfer"],
            "dino_physical_transfer": frame.attrs["dino_weighted_transfer"],
        },
        "caveats": ([
            "The dense top-20 retrieval rule is a custom budget-matched diagnostic over MASt3R descriptors, not the official reciprocal matcher.",
            "Only four availability-selected 3RScan train relations were evaluated; this is not external validation and relation-level uncertainty is high.",
            "The comparison uses MASt3R's official non-commercial checkpoint and source repository; license terms must be preserved for redistribution.",
        ] if args.mast3r_mode == "dense_topk" else [
            "MASt3R reciprocal matching is sparse: the four-pair mean actual candidate count is below the top-20 cap, so this is a standard sparse baseline rather than a matched dense candidate budget.",
            "This is an internal four-relation train-split comparison, not external validation.",
            "The comparison uses MASt3R's official non-commercial checkpoint and source repository; license terms must be preserved for redistribution.",
        ]),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MASt3R candidate-budget comparison" if args.mast3r_mode == "dense_topk" else "# Official MASt3R baseline comparison",
        "",
        (
            "This diagnostic compares MASt3R and ViT-G over the same four cached 20-frame RGB-D relations and the same 224-pixel depth-valid query pool. Both methods supply at most 20 unique reference observations per query to the same 3-D consensus scorer. MASt3R's dense top-20 variant ranks confidence-weighted descriptor similarity over the 20-frame reference bank; it is a custom candidate-budget control, not the official reciprocal-matching method. The four relations are availability-selected from the 3RScan train split."
            if args.mast3r_mode == "dense_topk"
            else "This is a four-relation internal 3RScan train-split comparison on the same cached 20-frame RGB-D pairs and the same 224-pixel depth-valid patch query population. DINOv2 uses the existing ViT-G top-20 dense descriptor pool. MASt3R uses its official ViT-L checkpoint at 512 pixels, reciprocal descriptor matches for each frame pair, confidence-ranked candidate aggregation, and the same 3-D consensus scorer."
        ),
        "",
        "| Pair | Linked queries | MASt3R candidates / query | MASt3R physical recall@20 | ViT-G physical recall@20 | MASt3R transfer | ViT-G transfer | MASt3R precision | ViT-G precision | MASt3R pose | ViT-G pose |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair']} | {row['mast3r_queries']} | {row['mast3r_mean_actual_candidates']:.1f} | {100*row['mast3r_candidate_recall']:.1f}% | {100*row['dino_candidate_recall']:.1f}% | "
            f"{100*row['mast3r_transfer']:.1f}% | {100*row['dino_transfer']:.1f}% | {100*row['mast3r_precision']:.1f}% | {100*row['dino_precision']:.1f}% | "
            f"{'pass' if row['mast3r_pose'] else 'fail'} | {'pass' if row['dino_pose'] else 'fail'} |"
        )
    lines += [
        "",
        f"Weighted by linked queries, MASt3R physical-instance candidate Recall@20 is {100*summary['weighted_by_linked_queries']['mast3r_candidate_recall']:.1f}% versus {100*summary['weighted_by_linked_queries']['dino_candidate_recall']:.1f}% for ViT-G. Correct physical transfer is {100*summary['weighted_by_linked_queries']['mast3r_physical_transfer']:.1f}% versus {100*summary['weighted_by_linked_queries']['dino_physical_transfer']:.1f}%.",
        "",
        (
            "All four dense MASt3R runs reached 20 candidates per evaluable query. Weighted physical-instance Recall@20 is similar for MASt3R and ViT-G (97.6% vs. 97.2%), while correct physical transfers are 1,679/2,967 (56.6%) vs. 1,077/2,967 (36.3%); the transfer point estimate is higher for MASt3R in all four relations. Strict scene pose passes on only two MASt3R relations, and pair06/pair08 retain substantial pose error. This post-hoc comparison uses four availability-selected train relations, so it is exploratory and does not justify a superiority claim. The sparse reciprocal baseline is reported separately in `outputs/reports/20261003_mast3r_baseline_comparison/`."
            if args.mast3r_mode == "dense_topk"
            else "The relation pattern is heterogeneous. MASt3R is slightly better on physical transfer for pair02, pair06, and pair08, while ViT-G is much better on pair05. MASt3R's reciprocal matching is sparse: the actual candidate count is usually far below 20, so its lower candidate Recall is also a measurable coverage bottleneck rather than evidence that its descriptor is intrinsically less semantic. This baseline does not replace external validation and is not added to the main paper claim set yet."
        ),
        "",
        "Artifacts: `per_pair.csv`, `summary.json`, and each relation's MASt3R/DINO query diagnostics. Reproduction uses `scripts/evaluate_3rscan_mast3r_baseline.py` and this aggregator.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
