"""Compute scene-level bootstrap confidence intervals for the paper tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, statistic, seed: int = 17, samples: int = 20000) -> tuple[float, float, float]:
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = np.asarray([statistic(values[index]) for index in indices], dtype=float)
    return float(statistic(values)), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def read_consensus(path: Path) -> dict[str, object]:
    pair = re.sub(r"^\d+_3rscan_consensus_", "", path.parent.name).replace("_vitg_diag", "")
    row = pd.read_csv(path).iloc[0]
    return {
        "pair": pair,
        "queries": float(row.queries),
        "inliers": float(row.inliers),
        "coverage": float(row.inlier_coverage),
        "semantic": float(row.inlier_semantic_accuracy),
        "rotation": float(row.rotation_error_deg),
        "translation": float(row.translation_error_m),
        "pose_success": float(row.rotation_error_deg < 5.0 and row.translation_error_m < 0.25),
    }


def add_ci(rows: list[dict[str, object]], name: str, values: np.ndarray, statistic, seed: int) -> None:
    estimate, lower, upper = bootstrap_ci(values, statistic, seed=seed)
    rows.append({"metric": name, "estimate": estimate, "ci95_lower": lower, "ci95_upper": upper, "units": "proportion"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consensus-source", type=Path, nargs="+", required=True)
    parser.add_argument("--transfer-per-pair", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    consensus = pd.DataFrame([read_consensus(path) for path in args.consensus_source]).sort_values("pair")
    consensus.to_csv(args.output_dir / "source_pair_table.csv", index=False)
    rows: list[dict[str, object]] = []
    for metric, column in (("macro_coverage", "coverage"), ("macro_semantic_precision", "semantic"), ("pose_success_rate", "pose_success")):
        add_ci(rows, metric, consensus[column].to_numpy(float), np.mean, 17 + len(rows))
    add_ci(rows, "weighted_coverage", consensus[["queries", "inliers"]].to_numpy(float), lambda x: x[:, 1].sum() / x[:, 0].sum(), 31)
    add_ci(
        rows,
        "weighted_semantic_precision",
        consensus[["inliers", "semantic"]].to_numpy(float),
        lambda x: (x[:, 0] * x[:, 1]).sum() / x[:, 0].sum(),
        32,
    )

    transfer = pd.read_csv(args.transfer_per_pair)
    transfer = transfer[transfer.dataset == "source"]
    for selector, frame in transfer.groupby("selector", sort=True):
        for metric in ("coverage", "semantic_mIoU", "instance_mIoU"):
            add_ci(rows, f"{selector}_{metric}", frame[metric].to_numpy(float), np.mean, 100 + len(rows))
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "confidence_intervals.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
