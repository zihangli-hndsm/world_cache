"""Aggregate keyframe-interpolation scale-up evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="two_sided_linear_interpolation_stride_5")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for eval_dir in args.eval_dir:
        summary = pd.read_csv(eval_dir / "summary.csv")
        selected = summary[summary["policy"] == args.policy]
        if len(selected) != 1:
            raise ValueError(f"expected one {args.policy!r} row in {eval_dir}")
        row = selected.iloc[0].to_dict()
        row["evaluation_dir"] = str(eval_dir)
        row["pass"] = bool(row["full_cosine_p10"] >= 0.95 and row["recompute_fraction"] <= 0.30)
        rows.append(row)

    table = pd.DataFrame(rows)
    preferred = [
        "evaluation_dir", "frames", "keyframes", "recompute_fraction", "reuse_fraction",
        "full_cosine_median", "full_cosine_p10", "full_cosine_min",
        "cls_cosine_p10", "patch_cosine_p10", "pass",
    ]
    table = table[[column for column in preferred if column in table.columns]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"pass_count={int(table['pass'].sum())}/{len(table)}")


if __name__ == "__main__":
    main()
