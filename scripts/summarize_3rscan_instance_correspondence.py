"""Aggregate annotated 3RScan instance/semantic correspondence summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tables = []
    for eval_dir in args.eval_dir:
        table = pd.read_csv(eval_dir / "summary.csv")
        table.insert(0, "evaluation_dir", str(eval_dir))
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
