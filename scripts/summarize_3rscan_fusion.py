"""Select a geometry/descriptor fusion weight with scene-level CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_fusion_table(paths: list[Path]) -> pd.DataFrame:
    tables = []
    for path in paths:
        table = pd.read_csv(path / "summary.csv")
        table = table[(table["field"] == "raw") & table["method"].str.startswith("fused_a")].copy()
        table["alpha"] = table["method"].str.removeprefix("fused_a").astype(float)
        table["scene"] = path.name
        tables.append(table)
    if not tables:
        raise ValueError("no evaluation directories supplied")
    return pd.concat(tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = load_fusion_table(args.source_dir)
    aggregate = source.groupby("alpha", as_index=False).apply(
        lambda group: pd.Series({
            "scene_balanced_accuracy": group["instance_accuracy"].mean(),
            "query_weighted_accuracy": (group["instance_accuracy"] * group["queries"]).sum() / group["queries"].sum(),
            "scenes": group["scene"].nunique(),
            "queries": group["queries"].sum(),
        }),
        include_groups=False,
    ).reset_index(drop=True)
    aggregate.to_csv(args.output_dir / "aggregate.csv", index=False)

    loso_rows = []
    for held_out in sorted(source["scene"].unique()):
        train = source[source["scene"] != held_out]
        validation = train.groupby("alpha")["instance_accuracy"].mean()
        alpha = float(validation.idxmax())
        held = source[(source["scene"] == held_out) & (source["alpha"] == alpha)]
        loso_rows.append({
            "held_out_scene": held_out,
            "selected_alpha": alpha,
            "validation_scene_balanced_accuracy": float(validation.loc[alpha]),
            "held_out_accuracy": float(held["instance_accuracy"].iloc[0]),
            "held_out_queries": int(held["queries"].iloc[0]),
        })
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(args.output_dir / "loso.csv", index=False)

    result = {
        "status": "complete",
        "source_scenes": sorted(source["scene"].unique()),
        "scene_balanced_best_alpha": float(aggregate.loc[aggregate["scene_balanced_accuracy"].idxmax(), "alpha"]),
        "query_weighted_best_alpha": float(aggregate.loc[aggregate["query_weighted_accuracy"].idxmax(), "alpha"]),
        "loso_selected_alphas": loso["selected_alpha"].value_counts().sort_index().to_dict(),
    }
    if args.target_dir is not None:
        target = load_fusion_table([args.target_dir])
        target = target[["alpha", "instance_accuracy", "semantic_accuracy", "queries"]]
        target.to_csv(args.output_dir / "target.csv", index=False)
        result["target_best_alpha"] = float(target.loc[target["instance_accuracy"].idxmax(), "alpha"])
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(aggregate.sort_values("scene_balanced_accuracy", ascending=False).to_string(index=False))
    print(loso.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
