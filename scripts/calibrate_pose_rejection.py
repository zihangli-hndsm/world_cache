"""Calibrate a source-only pose rejection rule from consensus summaries.

The calibration deliberately uses only inference-visible confidence features.
Ground-truth pose labels are used for source-pair selection, never for the
held-out target. This script is a diagnostic for whether a simple selective
pose policy can separate semantic correspondence from globally correct pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ("inlier_coverage", "bidirectional_geometry_overlap", "queries")


def load_summaries(paths: list[Path], target_token: str) -> pd.DataFrame:
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[frame["top_k"] == 20]
        if len(frame) != 1:
            raise ValueError(f"expected exactly one K=20 row in {path}, found {len(frame)}")
        row = frame.iloc[0].to_dict()
        row["run"] = path.parent.name
        row["is_target"] = target_token in path.parent.name
        rows.append(row)
    result = pd.DataFrame(rows)
    result["pose_success"] = (result["rotation_error_deg"] < 5.0) & (result["translation_error_m"] < 0.5)
    return result


def candidate_rules(source: pd.DataFrame) -> list[dict[str, float]]:
    thresholds = {
        "inlier_coverage": sorted({0.0, *source["inlier_coverage"].astype(float).tolist()}),
        "bidirectional_geometry_overlap": sorted({0.0, *source["bidirectional_geometry_overlap"].astype(float).tolist()}),
        "queries": sorted({0.0, *source["queries"].astype(float).tolist()}),
    }
    return [
        {feature: value for feature, value in zip(FEATURES, values)}
        for values in __import__("itertools").product(*(thresholds[feature] for feature in FEATURES))
    ]


def accepts(frame: pd.DataFrame, rule: dict[str, float]) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for feature, threshold in rule.items():
        mask &= frame[feature].to_numpy(float) >= threshold
    return pd.Series(mask, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-token", default="target")
    parser.add_argument("--precision-levels", type=float, nargs="+", default=[0.75, 0.90, 1.0])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_summaries(args.summary, args.target_token)
    source = data[~data["is_target"]].copy()
    target = data[data["is_target"]].copy()
    if len(target) != 1:
        raise ValueError(f"expected one target row, found {len(target)}")

    rows = []
    for rule in candidate_rules(source):
        selected = accepts(source, rule)
        count = int(selected.sum())
        if not count:
            continue
        rows.append({
            **rule,
            "source_selected": count,
            "source_precision": float(source.loc[selected, "pose_success"].mean()),
            "source_recall": float(source.loc[selected, "pose_success"].sum() / max(1, source["pose_success"].sum())),
            "target_accepted": bool(accepts(target, rule).iloc[0]),
            "target_pose_success_hidden_for_selection": bool(target["pose_success"].iloc[0]),
        })
    rules = pd.DataFrame(rows)
    selected_rows = []
    for level in args.precision_levels:
        eligible = rules[rules["source_precision"] >= level]
        if len(eligible):
            best = eligible.sort_values(
                ["source_selected", "source_precision", "source_recall"],
                ascending=[False, False, False],
            ).iloc[0]
            selected_rows.append({"requested_precision": level, "status": "rule_found", **best.to_dict()})
        else:
            selected_rows.append({"requested_precision": level, "status": "no_source_rule"})
    rules.to_csv(args.output_dir / "candidate_rules.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output_dir / "selected_rules.csv", index=False)
    summary = {
        "status": "complete",
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "source_pose_success_count": int(source["pose_success"].sum()),
        "pose_success_definition": "rotation_error_deg < 5 and translation_error_m < 0.5",
        "features": list(FEATURES),
        "precision_levels": args.precision_levels,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(selected_rows).to_string(index=False))


if __name__ == "__main__":
    main()
