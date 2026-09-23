"""Aggregate clean ViT-G spatial-consensus top-K sensitivity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def pair_name(path: Path) -> str:
    name = path.name.replace("20261003_k_sweep_", "").replace("_vitg_clean", "")
    return "target" if name == "target" else name


def summarize(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for top_k, group in frame.groupby("top_k", sort=True):
        inliers = group["inliers"].sum()
        queries = group["queries"].sum()
        rows.append(
            {
                "split": split,
                "top_k": int(top_k),
                "pairs": int(len(group)),
                "weighted_coverage": float(inliers / queries),
                "weighted_semantic_precision": float(
                    (group["inliers"] * group["inlier_semantic_accuracy"]).sum() / inliers
                ),
                "weighted_instance_precision": float(
                    (group["inliers"] * group["inlier_instance_accuracy"]).sum() / inliers
                ),
                "macro_coverage": float(group["inlier_coverage"].mean()),
                "macro_semantic_precision": float(group["inlier_semantic_accuracy"].mean()),
                "macro_instance_precision": float(group["inlier_instance_accuracy"].mean()),
                "strict_pose_successes": int(
                    ((group["rotation_error_deg"] < 5.0) & (group["translation_error_m"] < 0.25)).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def read_runs(paths: list[Path], split: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path / "summary.csv").copy()
        frame.insert(0, "pair", pair_name(path))
        frame.insert(0, "split", split)
        rows.append(frame)
    if not rows:
        raise ValueError(f"no {split} runs supplied")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = read_runs(args.source, "source")
    target = read_runs([args.target], "target")
    per_pair = pd.concat([source, target], ignore_index=True)
    summary = pd.concat([summarize(source, "source"), summarize(target, "target")], ignore_index=True)
    per_pair.to_csv(args.output_dir / "per_pair.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    source_at_min_coverage = summary[
        (summary.split == "source") & (summary.weighted_coverage >= 0.20)
    ].sort_values(["weighted_semantic_precision", "weighted_coverage"], ascending=[False, True]).iloc[0]
    status = {
        "status": "complete",
        "protocol": {
            "model_size": "giant",
            "layers": [2, 20, 40],
            "threshold_m": 0.10,
            "iterations": 1500,
            "annotation_labels_used_for_matching": False,
            "known_alignment_used_for_selection": False,
        },
        "source_runs": [str(path) for path in args.source],
        "target_run": str(args.target),
        "source_best_weighted_semantic_at_min_20pct_coverage": {
            "top_k": int(source_at_min_coverage.top_k),
            "weighted_coverage": float(source_at_min_coverage.weighted_coverage),
            "weighted_semantic_precision": float(source_at_min_coverage.weighted_semantic_precision),
        },
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
