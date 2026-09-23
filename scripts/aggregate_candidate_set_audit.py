"""Aggregate post-hoc top-K candidate-set recall audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pair_name(path: Path) -> str:
    name = path.name.replace("20261005_candidate_audit_", "").replace("_vitg_clean", "")
    return "target" if name == "target" else name


def read_run(path: Path, split: str) -> dict[str, object]:
    diagnostics = pd.read_csv(path / "query_diagnostics.csv")
    summary = pd.read_csv(path / "summary.csv")
    if len(summary) != 1:
        raise ValueError(f"expected one K=20 row in {path}")
    if set(diagnostics.top_k.unique()) != {20} or set(diagnostics.geometry_weight.unique()) != {0.0}:
        raise ValueError(f"unexpected audit grid in {path}")
    required = {
        "query_evaluable", "consensus_inlier", "semantic_correct", "instance_correct",
        "candidate_semantic_count", "candidate_instance_count",
        "candidate_semantic_hit", "candidate_instance_hit",
    }
    missing = required.difference(diagnostics.columns)
    if missing:
        raise ValueError(f"candidate audit fields missing in {path}: {sorted(missing)}")

    evaluated = diagnostics[diagnostics.query_evaluable.astype(bool)].copy()
    queries = len(evaluated)
    inliers = int(evaluated.consensus_inlier.astype(bool).sum())
    semantic_correct_inliers = int(
        (evaluated.consensus_inlier.astype(bool) & evaluated.semantic_correct.astype(bool)).sum()
    )
    instance_correct_inliers = int(
        (evaluated.consensus_inlier.astype(bool) & evaluated.instance_correct.astype(bool)).sum()
    )
    semantic_hits = int(evaluated.candidate_semantic_hit.astype(bool).sum())
    instance_hits = int(evaluated.candidate_instance_hit.astype(bool).sum())
    semantic_miss_inliers = int(
        (evaluated.consensus_inlier.astype(bool) & ~evaluated.candidate_semantic_hit.astype(bool)).sum()
    )
    return {
        "split": split,
        "pair": pair_name(path),
        "queries": queries,
        "candidate_semantic_hits": semantic_hits,
        "candidate_semantic_recall": semantic_hits / queries if queries else 0.0,
        "candidate_instance_hits": instance_hits,
        "candidate_instance_recall": instance_hits / queries if queries else 0.0,
        "inliers": inliers,
        "inlier_coverage": inliers / queries if queries else 0.0,
        "semantic_correct_inliers": semantic_correct_inliers,
        "semantic_coverage": semantic_correct_inliers / queries if queries else 0.0,
        "semantic_precision": semantic_correct_inliers / inliers if inliers else float("nan"),
        "instance_correct_inliers": instance_correct_inliers,
        "instance_precision": instance_correct_inliers / inliers if inliers else float("nan"),
        "semantic_recovery_given_candidate": semantic_correct_inliers / semantic_hits if semantic_hits else 0.0,
        "semantic_miss_inliers": semantic_miss_inliers,
    }


def summarize(frame: pd.DataFrame, split: str) -> dict[str, object]:
    group = frame[frame.split == split]
    queries = int(group.queries.sum())
    semantic_hits = int(group.candidate_semantic_hits.sum())
    instance_hits = int(group.candidate_instance_hits.sum())
    inliers = int(group.inliers.sum())
    semantic_correct = int(group.semantic_correct_inliers.sum())
    instance_correct = int(group.instance_correct_inliers.sum())
    return {
        "split": split,
        "pairs": int(len(group)),
        "queries": queries,
        "candidate_semantic_hits": semantic_hits,
        "candidate_semantic_recall": semantic_hits / queries if queries else 0.0,
        "candidate_instance_hits": instance_hits,
        "candidate_instance_recall": instance_hits / queries if queries else 0.0,
        "inliers": inliers,
        "inlier_coverage": inliers / queries if queries else 0.0,
        "semantic_correct_inliers": semantic_correct,
        "semantic_coverage": semantic_correct / queries if queries else 0.0,
        "semantic_precision": semantic_correct / inliers if inliers else float("nan"),
        "instance_correct_inliers": instance_correct,
        "instance_precision": instance_correct / inliers if inliers else float("nan"),
        "semantic_recovery_given_candidate": semantic_correct / semantic_hits if semantic_hits else 0.0,
        "semantic_miss_inliers": int(group.semantic_miss_inliers.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [read_run(path, "source") for path in args.source]
    rows.append(read_run(args.target, "target"))
    per_pair = pd.DataFrame(rows)
    summary = pd.DataFrame([summarize(per_pair, "source"), summarize(per_pair, "target")])
    per_pair.to_csv(args.output_dir / "per_pair.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    status = {
        "status": "complete",
        "protocol": {
            "model_size": "giant",
            "layers": [2, 20, 40],
            "top_k": 20,
            "threshold_m": 0.10,
            "iterations": 1500,
            "annotation_labels_used_for_matching": False,
            "known_alignment_used_for_selection": False,
            "candidate_set_fields_are_posthoc": True,
        },
        "source_runs": [str(path) for path in args.source],
        "target_run": str(args.target),
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
