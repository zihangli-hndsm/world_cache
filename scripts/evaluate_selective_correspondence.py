"""Build semantic risk-coverage curves from query-level consensus diagnostics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def risk_curve(frame: pd.DataFrame, dataset: str, score_name: str, values: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-values, kind="stable")
    fractions = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
    rows = []
    for fraction in fractions:
        count = max(1, int(np.ceil(len(frame) * fraction)))
        selected = frame.iloc[order[:count]]
        rows.append({
            "dataset": dataset,
            "score": score_name,
            "coverage": float(count / len(frame)),
            "queries": int(count),
            "semantic_precision": float(selected["semantic_correct"].mean()),
            "instance_precision": float(selected["instance_correct"].mean()),
            "consensus_inlier_fraction": float(selected["consensus_inlier"].mean()),
        })
    return pd.DataFrame(rows)


def score_map(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "descriptor_margin": frame["top1_top2_margin"].to_numpy(float),
        "top1_descriptor_score": frame["top1_descriptor_score"].to_numpy(float),
        "selected_descriptor_score": frame["selected_descriptor_score"].to_numpy(float),
        "low_spatial_residual": -frame["spatial_residual_m"].to_numpy(float),
        "consensus_inlier": frame["consensus_inlier"].astype(float).to_numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def read(path: Path) -> pd.DataFrame:
        raw_pair = re.sub(r"^\d+_3rscan_consensus_", "", path.parent.name).replace("_vitg_diag", "")
        match = re.search(r"(0cac|20c993|4aca|5630|6bde|751a|95be|c670|chairs|pair0[2-8])", raw_pair)
        pair = match.group(1) if match else ("target" if "target" in raw_pair else raw_pair)
        frame = pd.read_csv(path).assign(pair=pair)
        if "query_evaluable" in frame:
            frame = frame[frame["query_evaluable"].astype(bool)].reset_index(drop=True)
        return frame

    source = pd.concat([read(path) for path in args.source], ignore_index=True)
    target = read(args.target)
    outputs = []
    for name, frame in (("source", source), ("target", target)):
        for score_name, values in score_map(frame).items():
            outputs.append(risk_curve(frame, name, score_name, values))
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(args.output_dir / "risk_coverage.csv", index=False)
    per_pair = []
    for dataset, frame in (("source", source), ("target", target)):
        for pair, pair_frame in frame.groupby("pair", sort=True):
            for score_name, values in score_map(pair_frame).items():
                row = risk_curve(pair_frame, pair, score_name, values).iloc[2].to_dict()
                row["dataset"] = dataset
                row["pair"] = pair
                per_pair.append(row)
    pd.DataFrame(per_pair).to_csv(args.output_dir / "per_pair_at_20pct.csv", index=False)
    source_summary = result[result["dataset"] == "source"].groupby("score", as_index=False).apply(
        lambda group: group.loc[group["coverage"].sub(0.2).abs().idxmin(), ["semantic_precision", "instance_precision"]],
        include_groups=False,
    )
    source_summary.to_csv(args.output_dir / "source_at_20pct.csv", index=False)
    print(result[result["coverage"].between(0.19, 0.21)].to_string(index=False))


if __name__ == "__main__":
    main()
