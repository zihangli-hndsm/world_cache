"""Render paper figures for the full selective-correspondence audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "descriptor_margin": "descriptor margin",
    "top1_descriptor_score": "top-1 descriptor score",
    "selected_descriptor_score": "selected descriptor score",
    "low_spatial_residual": "low spatial residual",
    "consensus_inlier": "consensus inlier",
}
COLORS = {
    "descriptor_margin": "#7f8c8d",
    "top1_descriptor_score": "#2980b9",
    "selected_descriptor_score": "#8e44ad",
    "low_spatial_residual": "#d35400",
    "consensus_inlier": "#c0392b",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risk-coverage", type=Path, required=True)
    parser.add_argument("--per-pair", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    risk = pd.read_csv(args.risk_coverage)
    per_pair = pd.read_csv(args.per_pair)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for score in LABELS:
        color = COLORS[score]
        for dataset, linestyle, alpha in (("source", "-", 0.95), ("target", "--", 0.8)):
            frame = risk[(risk.dataset == dataset) & (risk.score == score)]
            axes[0].plot(
                frame.coverage * 100,
                frame.semantic_precision * 100,
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                linestyle=linestyle,
                color=color,
                alpha=alpha,
                label=f"{LABELS[score]} ({dataset})",
            )
    axes[0].axhline(90, color="black", linestyle=":", linewidth=1)
    axes[0].set_xlabel("retained query coverage (%)")
    axes[0].set_ylabel("semantic precision (%)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(50, 102)
    axes[0].set_title("Risk–coverage across all pairs")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2, frameon=False, loc="lower left")

    chosen = per_pair[per_pair.score.isin(("top1_descriptor_score", "low_spatial_residual"))].copy()
    pivot = chosen.pivot(index="pair", columns="score", values="semantic_precision") * 100
    order = list(pivot.sort_values("low_spatial_residual").index)
    x = np.arange(len(order))
    width = 0.38
    axes[1].bar(x - width / 2, pivot.loc[order, "top1_descriptor_score"], width, label="top-1 score", color=COLORS["top1_descriptor_score"])
    axes[1].bar(x + width / 2, pivot.loc[order, "low_spatial_residual"], width, label="low residual", color=COLORS["low_spatial_residual"])
    axes[1].axhline(90, color="black", linestyle=":", linewidth=1)
    axes[1].set_xticks(x, order, rotation=55, ha="right")
    axes[1].set_ylabel("semantic precision at 20% (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("Per-pair failure boundary")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"selective_correspondence_overview.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
