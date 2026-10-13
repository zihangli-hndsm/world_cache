#!/usr/bin/env python3
"""Descriptive leave-one-reference-group-out audit for segment comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = {
    "linked_semantic_precision": ("linked_semantic_correct", "linked_assigned"),
    "linked_precision": ("linked_correct", "linked_assigned"),
    "linked_coverage": ("linked_assigned", "raw_queries"),
    "semantic_precision": ("semantic_correct", "semantic_assigned"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", default="dino_mutual")
    parser.add_argument("--reference", default="segmast3r")
    args = parser.parse_args()

    groups: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with args.input.open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["environment_id"]][row["arm"]] = row
    if len(groups) != 47 or any(args.arm not in x or args.reference not in x for x in groups.values()):
        raise SystemExit(f"Expected 47 complete groups; found {len(groups)}")

    report = {
        "groups": len(groups),
        "comparison": f"{args.arm} minus {args.reference}",
        "interpretation": "Descriptive stability over released reference-group IDs; physical-site independence is unverified.",
        "metrics": {},
    }
    for metric, (numerator, denominator) in METRICS.items():
        group_differences = []
        pooled = {args.arm: [0, 0], args.reference: [0, 0]}
        for arms in groups.values():
            group_differences.append(
                float(arms[args.arm][metric]) - float(arms[args.reference][metric])
            )
            for arm in pooled:
                pooled[arm][0] += int(arms[arm][numerator])
                pooled[arm][1] += int(arms[arm][denominator])
        loo = [
            sum(d for j, d in enumerate(group_differences) if j != i) / (len(groups) - 1)
            for i in range(len(groups))
        ]
        pooled_rates = {arm: n / d for arm, (n, d) in pooled.items()}
        report["metrics"][metric] = {
            "group_macro_difference": sum(group_differences) / len(group_differences),
            "positive_groups": sum(d > 0 for d in group_differences),
            "negative_groups": sum(d < 0 for d in group_differences),
            "zero_groups": sum(d == 0 for d in group_differences),
            "leave_one_group_out_min": min(loo),
            "leave_one_group_out_max": max(loo),
            "pooled_rates": pooled_rates,
            "pooled_difference": pooled_rates[args.arm] - pooled_rates[args.reference],
            "pooled_counts": pooled,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
