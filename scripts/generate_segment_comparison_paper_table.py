#!/usr/bin/env python3
"""Render the complete paired segment-comparator effect table from scored output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_LABELS = {
    "semantic_coverage": "Semantic coverage",
    "semantic_precision": "Semantic precision",
    "linked_coverage": "Linked-identity coverage",
    "linked_precision": "Linked-identity precision",
    "linked_semantic_precision": "Linked semantic precision",
    "linked_present_coverage": "Present linked coverage",
    "linked_with_absence_accuracy": "Linked correct-abstention accuracy",
    "things_present_coverage": "Present non-structure coverage",
}
COMPARATOR_LABELS = {
    "dino_nn": "DINO NN",
    "dino_mutual": "DINO mutual",
    "dino_equal_segment_budget": "DINO raw-budget",
    "dino_equal_budget_one_to_one": "DINO budget 1:1",
}


def pp(value: float) -> str:
    return f"{value * 100:+.2f}"


def interval(values: list[float]) -> str:
    return f"[{pp(values[0])},{pp(values[1])}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    summary_path = (
        root / "outputs/reports/20261008_segmast3r_validation"
        / "segment_comparison_dataset/summary.json"
    )
    summary = json.loads(summary_path.read_text())
    expected = len(METRIC_LABELS) * len(COMPARATOR_LABELS)
    if summary["input_preprocessing"] != "official-dataset":
        raise ValueError("table input must use official dataset preprocessing")
    if summary["frame_pairs"] != 2181 or summary["environments"] != 47:
        raise ValueError("full frozen cohort is required")
    if summary["bonferroni_effect_family"] != 96 or len(summary["effects"]) != expected:
        raise ValueError("complete 96-effect family is required")
    if summary.get("bootstrap_resamples", 0) < 1_000_000:
        raise ValueError("one million bootstrap resamples are required for the adjusted tail interval")

    rows = []
    for effect in summary["effects"]:
        rows.append(
            f"{METRIC_LABELS[effect['metric']]} & "
            f"{COMPARATOR_LABELS[effect['comparator']]} & "
            f"{pp(effect['segmast3r_minus_dino'])} & "
            f"${interval(effect['ci95'])}$ & "
            f"${interval(effect['bonferroni_ci95'])}$ \\\\"
        )
    table = r"""\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrcc}
\toprule
Metric & DINO control & Difference (pp) & Ordinary 95\% CI & Adjusted 95\% CI \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Paired SegMASt3R-minus-DINO differences across all four DINO
controls. Intervals resample the 47 released reference groups. Adjusted
intervals use the declared 96-effect family, including the eight-group pilot
under both preprocessing variants and the full-cohort outcomes. Negative
differences favor DINO.}
\label{tab:segmentation-assisted-all-effects}
\end{table*}
"""
    destination = root / "paper/generated_segment_comparison_effects.tex"
    destination.write_text(table)
    print(f"wrote {len(rows)} paired effects to {destination}")


if __name__ == "__main__":
    main()
