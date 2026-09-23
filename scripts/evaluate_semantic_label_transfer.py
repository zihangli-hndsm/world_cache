"""Evaluate semantic/instance label transfer from spatial correspondences."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def mean_iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute class-balanced IoU over labels present in either array."""
    labels = np.union1d(ground_truth, prediction)
    if len(labels) == 0:
        return float("nan")
    scores = []
    for label in labels:
        actual = ground_truth == label
        guessed = prediction == label
        union = np.logical_or(actual, guessed).sum()
        if union:
            scores.append(float(np.logical_and(actual, guessed).sum() / union))
    return float(np.mean(scores)) if scores else float("nan")


def metrics(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    count = int(mask.sum())
    if count == 0:
        return {"queries": 0, "coverage": 0.0, "semantic_accuracy": float("nan"), "semantic_mIoU": float("nan"), "instance_accuracy": float("nan"), "instance_mIoU": float("nan")}
    query_semantics = frame.loc[mask, "query_semantic_id"].to_numpy()
    predicted_semantics = frame.loc[mask, "selected_reference_semantic_id"].to_numpy()
    query_instances = frame.loc[mask, "query_instance_id"].to_numpy()
    predicted_instances = frame.loc[mask, "selected_reference_instance_id"].to_numpy()
    return {
        "queries": count,
        "coverage": float(mask.mean()),
        "semantic_accuracy": float(np.mean(query_semantics == predicted_semantics)),
        "semantic_mIoU": mean_iou(query_semantics, predicted_semantics),
        "instance_accuracy": float(np.mean(query_instances == predicted_instances)),
        "instance_mIoU": mean_iou(query_instances, predicted_instances),
    }


def read(path: Path) -> pd.DataFrame:
    raw_pair = re.sub(r"^\d+_3rscan_consensus_", "", path.parent.name).replace("_vitg_diag", "")
    match = re.search(r"(0cac|20c993|4aca|5630|6bde|751a|95be|c670|chairs|pair0[2-8])", raw_pair)
    pair = match.group(1) if match else ("target" if "target" in raw_pair else raw_pair)
    frame = pd.read_csv(path).assign(pair=pair)
    if "query_evaluable" in frame:
        frame = frame[frame["query_evaluable"].astype(bool)].reset_index(drop=True)
    required = {
        "query_semantic_id",
        "selected_reference_semantic_id",
        "query_instance_id",
        "selected_reference_instance_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks saved label IDs: {sorted(missing)}; rerun the spatial evaluator")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--residual-threshold", type=float, default=0.04034)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.concat([read(path).assign(dataset="source") for path in args.source], ignore_index=True)
    target = read(args.target).assign(dataset="target")
    all_rows = []
    per_pair_rows = []
    for dataset, frame in (("source", source), ("target", target)):
        selectors = {
            "all_correspondences": np.ones(len(frame), dtype=bool),
            "consensus_inlier": frame["consensus_inlier"].to_numpy(bool),
            "source_calibrated_residual": frame["spatial_residual_m"].to_numpy(float) <= args.residual_threshold,
        }
        for selector, mask in selectors.items():
            row = {"dataset": dataset, "selector": selector, **metrics(frame, mask)}
            all_rows.append(row)
            for pair, pair_frame in frame.groupby("pair", sort=True):
                pair_mask = mask[pair_frame.index.to_numpy()]
                per_pair_rows.append({"dataset": dataset, "pair": pair, "selector": selector, **metrics(pair_frame, pair_mask)})

    summary = pd.DataFrame(all_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    per_pair = pd.DataFrame(per_pair_rows)
    per_pair.to_csv(args.output_dir / "per_pair.csv", index=False)
    macro = per_pair.groupby(["dataset", "selector"], as_index=False).agg(
        pairs=("pair", "count"),
        macro_coverage=("coverage", "mean"),
        macro_semantic_accuracy=("semantic_accuracy", "mean"),
        macro_semantic_mIoU=("semantic_mIoU", "mean"),
        macro_instance_accuracy=("instance_accuracy", "mean"),
        macro_instance_mIoU=("instance_mIoU", "mean"),
    )
    macro.to_csv(args.output_dir / "macro_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
