"""Aggregate post-hoc RANSAC seed-sweep runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pair_name(path: Path) -> str:
    name = path.name.replace("20261007_seed_sweep_", "").replace("_vitg_clean", "")
    return "target" if name == "target" else name


def read_runs(paths: list[Path], split: str) -> pd.DataFrame:
    rows = []
    for path in paths:
        frame = pd.read_csv(path / "summary.csv")
        if set(frame.top_k.unique()) != {20} or set(frame.geometry_weight.unique()) != {0.0}:
            raise ValueError(f"unexpected seed-sweep grid in {path}")
        if set(frame.ransac_seed.unique()) != {17, 29, 41}:
            raise ValueError(f"unexpected seed set in {path}")
        frame.insert(0, "pair", pair_name(path))
        frame.insert(0, "split", split)
        rows.append(frame)
    if not rows:
        raise ValueError(f"no {split} runs supplied")
    return pd.concat(rows, ignore_index=True)


def summarize_seed_metrics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    for seed, group in frame[frame.split == split].groupby("ransac_seed", sort=True):
        inliers = group.inliers.sum()
        queries = group.queries.sum()
        rows.append({
            "split": split,
            "ransac_seed": int(seed),
            "pairs": int(len(group)),
            "weighted_coverage": float(inliers / queries),
            "weighted_semantic_precision": float(
                (group.inliers * group.inlier_semantic_accuracy).sum() / inliers
            ),
            "weighted_instance_precision": float(
                (group.inliers * group.inlier_instance_accuracy).sum() / inliers
            ),
            "macro_coverage": float(group.inlier_coverage.mean()),
            "macro_semantic_precision": float(group.inlier_semantic_accuracy.mean()),
            "strict_pose_successes": int(
                ((group.rotation_error_deg < 5.0) & (group.translation_error_m < 0.25)).sum()
            ),
        })
    return pd.DataFrame(rows)


def summarize_pair_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, pair), group in frame.groupby(["split", "pair"], sort=True):
        rows.append({
            "split": split,
            "pair": pair,
            "seeds": int(group.ransac_seed.nunique()),
            "semantic_min": float(group.inlier_semantic_accuracy.min()),
            "semantic_max": float(group.inlier_semantic_accuracy.max()),
            "semantic_range": float(group.inlier_semantic_accuracy.max() - group.inlier_semantic_accuracy.min()),
            "coverage_min": float(group.inlier_coverage.min()),
            "coverage_max": float(group.inlier_coverage.max()),
            "coverage_range": float(group.inlier_coverage.max() - group.inlier_coverage.min()),
            "rotation_min_deg": float(group.rotation_error_deg.min()),
            "rotation_max_deg": float(group.rotation_error_deg.max()),
            "strict_pose_successes": int(
                ((group.rotation_error_deg < 5.0) & (group.translation_error_m < 0.25)).sum()
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = read_runs(args.source, "source")
    target = read_runs([args.target], "target")
    per_seed = pd.concat([source, target], ignore_index=True)
    summary = pd.concat([
        summarize_seed_metrics(source, "source"),
        summarize_seed_metrics(target, "target"),
    ], ignore_index=True)
    stability = summarize_pair_stability(per_seed)
    per_seed.to_csv(args.output_dir / "per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    stability.to_csv(args.output_dir / "pair_stability.csv", index=False)
    status = {
        "status": "complete",
        "protocol": {
            "model_size": "giant",
            "layers": [2, 20, 40],
            "top_k": 20,
            "threshold_m": 0.10,
            "iterations": 1500,
            "ransac_seeds": [17, 29, 41],
            "annotation_labels_used_for_matching": False,
            "known_alignment_used_for_selection": False,
            "seed_sweep_is_posthoc": True,
        },
        "source_runs": [str(path) for path in args.source],
        "target_run": str(args.target),
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
