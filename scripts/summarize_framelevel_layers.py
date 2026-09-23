"""Summarize frame-level layer/matching selection with source-only LOSO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(paths: list[Path]) -> pd.DataFrame:
    tables = []
    for path in paths:
        table = pd.read_csv(path / "summary.csv")
        table["scene"] = path.name
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def validate_compatible(source: pd.DataFrame, target: pd.DataFrame) -> None:
    """Fail early when source and target were produced with different grids."""
    keys = ["feature", "method"]
    source_keys = set(map(tuple, source[keys].drop_duplicates().to_numpy()))
    target_keys = set(map(tuple, target[keys].drop_duplicates().to_numpy()))
    missing_in_target = sorted(source_keys - target_keys)
    missing_in_source = sorted(target_keys - source_keys)
    if missing_in_target or missing_in_source:
        raise ValueError(
            "source/target frame-level grids differ; "
            f"missing in target={missing_in_target[:5]}, "
            f"missing in source={missing_in_source[:5]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load(args.source_dir)
    target = load([args.target_dir])
    validate_compatible(source, target)
    keys = ["feature", "method"]
    metric_columns = [
        "coverage", "instance_accuracy", "semantic_accuracy", "ransac_inlier_rate",
        "rotation_error_deg", "translation_error_m", "matches",
    ]
    aggregate = source.groupby(keys, as_index=False)[metric_columns].mean()
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)

    loso_rows = []
    for held_out in sorted(source["scene"].unique()):
        train = source[source["scene"] != held_out]
        validation = train.groupby(keys).mean(numeric_only=True)
        eligible = validation[validation["coverage"] >= args.min_coverage]
        selected = eligible["semantic_accuracy"].idxmax()
        held = source[(source["scene"] == held_out) & (source["feature"] == selected[0]) & (source["method"] == selected[1])].iloc[0]
        loso_rows.append({
            "held_out_scene": held_out,
            "selected_feature": selected[0],
            "selected_method": selected[1],
            "validation_semantic_accuracy": float(eligible.loc[selected, "semantic_accuracy"]),
            "held_out_semantic_accuracy": float(held["semantic_accuracy"]),
            "held_out_coverage": float(held["coverage"]),
            "held_out_ransac_inlier_rate": float(held["ransac_inlier_rate"]),
        })
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(args.output_dir / "loso.csv", index=False)

    source_best = aggregate[aggregate["coverage"] >= args.min_coverage].sort_values("semantic_accuracy", ascending=False).iloc[0]
    target_best = target[
        (target["feature"] == source_best["feature"]) & (target["method"] == source_best["method"])
    ].iloc[0]
    result = {
        "status": "complete",
        "source_scenes": sorted(source["scene"].unique()),
        "min_coverage": args.min_coverage,
        "source_best_feature": source_best["feature"],
        "source_best_method": source_best["method"],
        "source_best_semantic_accuracy": float(source_best["semantic_accuracy"]),
        "target_semantic_accuracy": float(target_best["semantic_accuracy"]),
        "target_coverage": float(target_best["coverage"]),
        "loso_selected": {
            f"{feature}/{method}": int(count)
            for (feature, method), count in loso[["selected_feature", "selected_method"]].value_counts().items()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(aggregate.sort_values("semantic_accuracy", ascending=False).head(12).to_string(index=False))
    print(loso.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
