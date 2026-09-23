"""Aggregate one-row spatial-consensus summaries into a reproducible table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PAIR_RE = re.compile(
    r"(0cac|20c993|4aca|5630|6bde|751a|95be|c670|chairs|pair0[2-8]|target)"
)


def pair_name(path: Path) -> str:
    match = PAIR_RE.search(path.parent.name)
    if match is None:
        raise ValueError(f"cannot infer pair from {path}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.source:
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise ValueError(f"expected one summary row in {path}, found {len(frame)}")
        row = frame.iloc[0].to_dict()
        row["pair"] = pair_name(path)
        rows.append(row)
    result = pd.DataFrame(rows)
    result["_order"] = result["pair"].map(
        {name: index for index, name in enumerate(
            ["0cac", "20c993", "4aca", "5630", "6bde", "751a", "95be", "c670",
             "chairs", "pair02", "pair03", "pair04", "pair05", "pair06", "pair07", "pair08", "target"]
        )}
    )
    if result["_order"].isna().any():
        raise ValueError(f"unrecognized pairs: {result.loc[result['_order'].isna(), 'pair'].tolist()}")
    result = result.sort_values("_order").drop(columns="_order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
