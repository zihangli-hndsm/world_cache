"""Summarize raw versus world-receptive-field descriptor consistency."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", default="world_r0.3")
    parser.add_argument("--mix-method", default="query_cosine_mix_0.1")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for eval_dir in args.eval_dir:
        table = pd.read_csv(eval_dir / "query_cosine_summary.csv")
        raw = table[(table["field"] == "raw") & (table["method"] == "query_cosine_geometry")].iloc[0]
        field = table[(table["field"] == args.field) & (table["method"] == "query_cosine_geometry")].iloc[0]
        mixture = table[(table["field"] == args.field) & (table["method"] == args.mix_method)].iloc[0]
        rows.append({
            "evaluation_dir": str(eval_dir),
            "raw_geometry_median": float(raw["cosine_median"]),
            "field_geometry_median": float(field["cosine_median"]),
            "field_mix_median": float(mixture["cosine_median"]),
            "raw_geometry_p10": float(raw["cosine_p10"]),
            "field_geometry_p10": float(field["cosine_p10"]),
            "field_mix_p10": float(mixture["cosine_p10"]),
            "median_gain": float(mixture["cosine_median"] - raw["cosine_median"]),
            "p10_gain": float(mixture["cosine_p10"] - raw["cosine_p10"]),
        })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"median_gain_min={result['median_gain'].min():.6f}")
    print(f"median_gain_max={result['median_gain'].max():.6f}")
    print(f"median_gain_mean={result['median_gain'].mean():.6f}")


if __name__ == "__main__":
    main()
