"""Aggregate 3RScan keyframe interpolation evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for eval_dir in args.eval_dir:
        summary = pd.read_csv(eval_dir / "summary.csv")
        selected = summary[summary["stride"] == args.stride]
        if len(selected) != 1:
            raise ValueError(f"expected one stride={args.stride} row in {eval_dir}")
        row = selected.iloc[0].to_dict()
        row["evaluation_dir"] = str(eval_dir)
        status = eval_dir / "status.json"
        if status.exists():
            import json

            metadata = json.loads(status.read_text(encoding="utf-8"))
            row["sequence"] = metadata.get("sequence", "unknown")
        else:
            row["sequence"] = "unknown"
        rows.append(row)

    table = pd.DataFrame(rows)
    preferred = [
        "evaluation_dir", "sequence", "frames", "keyframes", "recompute_fraction",
        "full_cosine_median", "full_cosine_p10", "full_cosine_min",
        "cls_cosine_p10", "patch_cosine_p10",
    ]
    table = table[[column for column in preferred if column in table.columns]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"full_p10_min={table['full_cosine_p10'].min():.6f}")
    print(f"full_p10_max={table['full_cosine_p10'].max():.6f}")
    print(f"full_p10_mean={table['full_cosine_p10'].mean():.6f}")


if __name__ == "__main__":
    main()
