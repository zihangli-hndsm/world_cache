#!/usr/bin/env python3
"""Audit physical-instance errors among category-correct UW object predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=Path("outputs/reports/20261009_rgbd_object_eval_multiview/outcomes.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/20261009_rgbd_object_eval_multiview/same_category_summary.json"),
    )
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()

    by_object: dict[str, list[dict[str, object]]] = defaultdict(list)
    object_categories: dict[str, str] = {}
    with args.outcomes.open() as stream:
        for line in stream:
            row = json.loads(line)
            oid = str(row["object_id"])
            by_object[oid].append(row)
            object_categories[oid] = str(row["category"])

    category_objects: dict[str, set[str]] = defaultdict(set)
    for oid, category in object_categories.items():
        category_objects[category].add(oid)
    if not by_object or any(len(objects) < 2 for objects in category_objects.values()):
        raise ValueError("expected multiple physical instances in every evaluated category")

    conditional_by_object = []
    category_correct_queries = 0
    within_category_identity_errors = 0
    for rows in by_object.values():
        category_correct = [row for row in rows if bool(row["dino_category_correct"])]
        if not category_correct:
            raise ValueError("object has no category-correct predictions")
        category_correct_queries += len(category_correct)
        within_category_identity_errors += sum(
            not bool(row["dino_instance_correct"]) for row in category_correct
        )
        conditional_by_object.append(
            np.mean([bool(row["dino_instance_correct"]) for row in category_correct])
        )

    values = np.asarray(conditional_by_object, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, len(values), size=(args.draws, len(values)))
    draws = values[sample_indices].mean(axis=1)
    result = {
        "physical_objects": len(by_object),
        "categories": len(category_objects),
        "physical_objects_per_category_range": [
            min(len(objects) for objects in category_objects.values()),
            max(len(objects) for objects in category_objects.values()),
        ],
        "query_predictions": sum(map(len, by_object.values())),
        "category_correct_query_predictions": category_correct_queries,
        "wrong_instance_but_correct_category_predictions": within_category_identity_errors,
        "within_category_identity_error_fraction_of_all_queries": (
            within_category_identity_errors / sum(map(len, by_object.values()))
        ),
        "identity_accuracy_conditional_on_category_correct_micro": (
            1.0 - within_category_identity_errors / category_correct_queries
        ),
        "identity_accuracy_conditional_on_category_correct_object_macro": float(values.mean()),
        "conditional_object_macro_bootstrap_95": [
            float(x) for x in np.quantile(draws, [0.025, 0.975])
        ],
        "bootstrap": {"draws": args.draws, "seed": args.seed, "unit": "physical object ID"},
        "interpretation": "Among predictions that choose the correct category, this measures exact physical-object identification; the benchmark has same-category distractors in every category.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
