#!/usr/bin/env python3
"""Apply the frozen 3RScan DINO cosine gate to UW RGB-D Object retrieval rows."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/reports/20261009_rgbd_object_eval_multiview/outcomes.jsonl"
DEFAULT_GATE = ROOT / "outputs/reports/20261009_dino_multiview_source_gate/source_frozen_gate.json"
DEFAULT_OUT = ROOT / "outputs/reports/20261009_uw_rgbd_source_gate_transfer"


def aggregate(rows: list[dict], threshold: float) -> dict:
    ids = sorted({row["object_id"] for row in rows})
    per_object: dict[str, dict] = {}
    for oid in ids:
        subset = [row for row in rows if row["object_id"] == oid]
        accepted = [row for row in subset if row["dino_top1_similarity"] >= threshold]
        correct = sum(bool(row["dino_instance_correct"]) for row in accepted)
        category_correct = sum(bool(row["dino_category_correct"]) for row in accepted)
        per_object[oid] = {
            "queries": len(subset),
            "accepted": len(accepted),
            "correct": correct,
            "category_correct": category_correct,
            "coverage": correct / len(subset),
            "precision": correct / len(accepted) if accepted else None,
            "category_precision": category_correct / len(accepted) if accepted else None,
            "unfiltered_correct": sum(bool(row["dino_instance_correct"]) for row in subset),
        }

    total_q = len(rows)
    total_a = sum(v["accepted"] for v in per_object.values())
    total_c = sum(v["correct"] for v in per_object.values())
    total_cat = sum(v["category_correct"] for v in per_object.values())
    supported = [v for v in per_object.values() if v["accepted"]]
    rng = np.random.default_rng(20261009)
    nboot = 20_000
    sample = rng.integers(0, len(ids), size=(nboot, len(ids)))
    query_n = np.asarray([per_object[x]["queries"] for x in ids], dtype=float)
    accept_n = np.asarray([per_object[x]["accepted"] for x in ids], dtype=float)
    correct_n = np.asarray([per_object[x]["correct"] for x in ids], dtype=float)
    coverage = np.asarray([per_object[x]["coverage"] for x in ids], dtype=float)
    precision = np.asarray([
        per_object[x]["precision"] if per_object[x]["precision"] is not None else np.nan
        for x in ids
    ], dtype=float)
    pooled_precision = correct_n[sample].sum(1) / np.maximum(accept_n[sample].sum(1), 1)
    pooled_coverage = correct_n[sample].sum(1) / np.maximum(query_n[sample].sum(1), 1)
    macro_coverage = coverage[sample].mean(1)
    macro_precision = np.nanmean(precision[sample], axis=1)

    return {
        "queries": total_q,
        "objects": len(ids),
        "threshold": threshold,
        "accepted": total_a,
        "correct": total_c,
        "category_correct": total_cat,
        "acceptance_rate": total_a / total_q,
        "pooled_precision": total_c / total_a if total_a else None,
        "pooled_category_precision": total_cat / total_a if total_a else None,
        "pooled_correct_coverage": total_c / total_q,
        "object_macro_coverage": float(np.mean(coverage)),
        "objects_with_any_accepts": len(supported),
        "object_macro_precision_among_supported": float(np.mean([v["precision"] for v in supported])) if supported else None,
        "unfiltered_object_macro_top1": float(np.mean([v["unfiltered_correct"] / v["queries"] for v in per_object.values()])),
        "object_cluster_bootstrap": {
            "draws": nboot,
            "seed": 20261009,
            "unit": "physical object ID",
            "pooled_precision_95": [float(x) for x in np.quantile(pooled_precision, [.025, .975])],
            "pooled_correct_coverage_95": [float(x) for x in np.quantile(pooled_coverage, [.025, .975])],
            "object_macro_coverage_95": [float(x) for x in np.quantile(macro_coverage, [.025, .975])],
            "supported_object_macro_precision_95": [float(x) for x in np.nanquantile(macro_precision, [.025, .975])],
        },
        "per_object": per_object,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    threshold = float(gate["frozen_threshold"])
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len(rows) != 21_600 or any("dino_top1_similarity" not in row for row in rows):
        raise ValueError("expected 21,600 UW rows with DINO top-1 similarities")
    overall = aggregate(rows, threshold)
    by_fold = {
        video: aggregate([row for row in rows if row["source_video"] == video], threshold)
        for video in sorted({row["source_video"] for row in rows})
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete",
        "source_gate": str(args.gate.relative_to(ROOT)) if args.gate.is_relative_to(ROOT) else str(args.gate),
        "frozen_threshold": threshold,
        "target": "UW RGB-D Object Dataset official evaluation set",
        "target_license": "official dataset page permits non-commercial research/education use",
        "model_and_score": "frozen DINOv2 ViT-G/14; maximum cosine over 12 source views per object",
        "protocol": "source threshold is transferred unchanged; query gallery contains all 300 physical objects; masks/crops are privileged",
        "overall": overall,
        "by_source_video_fold": by_fold,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Cross-dataset transfer of the frozen DINO acceptance threshold",
        "",
        f"The cosine cutoff {threshold:.12f}, fitted on eight 3RScan source groups for a 90% macro-precision point-estimate floor, is applied unchanged to the UW RGB-D Object Dataset evaluation set. No UW labels are used to choose the cutoff.",
        "",
        "| Outcome | Estimate | Object-cluster 95% interval |",
        "|---|---:|---:|",
        f"| Accepted | {overall['accepted']:,}/{overall['queries']:,} ({overall['acceptance_rate']*100:.1f}%) | — |",
        f"| Pooled physical-instance precision | {overall['pooled_precision']*100:.1f}% | [{overall['object_cluster_bootstrap']['pooled_precision_95'][0]*100:.1f},{overall['object_cluster_bootstrap']['pooled_precision_95'][1]*100:.1f}]% |",
        f"| Pooled correct coverage | {overall['pooled_correct_coverage']*100:.1f}% | [{overall['object_cluster_bootstrap']['pooled_correct_coverage_95'][0]*100:.1f},{overall['object_cluster_bootstrap']['pooled_correct_coverage_95'][1]*100:.1f}]% |",
        f"| Object-macro correct coverage | {overall['object_macro_coverage']*100:.1f}% | [{overall['object_cluster_bootstrap']['object_macro_coverage_95'][0]*100:.1f},{overall['object_cluster_bootstrap']['object_macro_coverage_95'][1]*100:.1f}]% |",
        f"| Supported-object macro precision ({overall['objects_with_any_accepts']}/{overall['objects']}) | {overall['object_macro_precision_among_supported']*100:.1f}% | [{overall['object_cluster_bootstrap']['supported_object_macro_precision_95'][0]*100:.1f},{overall['object_cluster_bootstrap']['supported_object_macro_precision_95'][1]*100:.1f}]% |",
        "",
        f"Unfiltered DINO object-macro top-1 is {overall['unfiltered_object_macro_top1']*100:.1f}% over the same queries. This is a cross-dataset score-threshold transfer, not a room-rescan or deployment result: the target is a controlled turntable set with privileged masks, and the candidate gallery contains all known objects.",
        "",
        "The interval resamples physical-object IDs, preserving all correlated query frames within each object. The threshold was previously selected on 3RScan, whose source and validation populations had already been explored; this result is an external dataset check but not independent preregistered confirmation. DINOv2 pretraining exposure is unknown.",
        "",
        "Reproduce by adding `dino_top1_similarity` to the saved UW rows with `scripts/evaluate_rgbd_object_eval_multiview.py`, then run `scripts/analyze_uw_rgbd_source_gate_transfer.py`.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
