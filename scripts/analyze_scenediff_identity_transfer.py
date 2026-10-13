#!/usr/bin/env python3
"""Summarize saved SceneDiff frozen-descriptor predictions with pair bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/reports/20261009_scenediff_identity_transfer/query_rows.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/20261009_scenediff_identity_transfer/subset_bootstrap.json"),
    )
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()

    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with args.input.open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[row["subset"]][row["pair"]].append(row)

    rng = np.random.default_rng(args.seed)
    output: dict[str, object] = {
        "input": str(args.input),
        "bootstrap_unit": "video pair; resample pairs with replacement within subset",
        "bootstrap_caveat": (
            "The released SceneDiff split lists pair IDs but no physical-scene cluster IDs; "
            "intervals do not model dependence among pairs from one physical scene."
        ),
        "draws": args.draws,
        "seed": args.seed,
        "subsets": {},
        "same_label_queries": {},
    }
    subsets: dict[str, object] = {}
    for subset, pairs in sorted(grouped.items()):
        pair_rows = list(pairs.values())
        success = {
            metric: np.asarray(
                [np.mean([float(row[f"{metric}_correct"]) for row in rows]) for rows in pair_rows]
            )
            for metric in ("maskpool", "cls", "rgbhist")
        }
        indices = rng.integers(0, len(pair_rows), size=(args.draws, len(pair_rows)))
        metric_summary = {}
        for metric, values in success.items():
            boot = values[indices].mean(axis=1)
            metric_summary[metric] = {
                "pair_macro_top1": float(values.mean()),
                "pair_cluster_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            }
        differences = {}
        for metric in ("maskpool", "cls"):
            delta = success[metric] - success["rgbhist"]
            boot = delta[indices].mean(axis=1)
            differences[f"{metric}_minus_rgbhist"] = {
                "pair_macro_difference": float(delta.mean()),
                "pair_cluster_bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            }
        subsets[subset] = {
            "queries": sum(map(len, pair_rows)),
            "pairs": len(pair_rows),
            "metrics": metric_summary,
            "paired_differences": differences,
        }
    output["subsets"] = subsets

    with args.input.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    hard = [row for row in rows if row["maskpool_same_label_correct"] != ""]
    output["same_label_queries"] = {
        "count": len(hard),
        "fraction_of_scored_queries": len(hard) / len(rows) if rows else 0.0,
        "top1": {
            metric: {
                "correct": int(sum(float(row[f"{metric}_same_label_correct"]) for row in hard)),
                "total": len(hard),
                "rate": float(np.mean([float(row[f"{metric}_same_label_correct"]) for row in hard])),
            }
            for metric in ("maskpool", "cls", "rgbhist")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
