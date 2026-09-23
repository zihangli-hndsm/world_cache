"""Create the first layer-wise geometric-alignment control plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    grouped = frame.groupby(["layer", "control"], as_index=False).cosine.median()
    labels = {"aligned": "geometry-aligned", "same_index": "same 2D index", "random": "random target"}
    colors = {"aligned": "#0072B2", "same_index": "#D55E00", "random": "#999999"}
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for control in ("aligned", "same_index", "random"):
        subset = grouped[grouped.control == control]
        ax.plot(subset.layer, subset.cosine, marker="o", linewidth=2, label=labels[control], color=colors[control])
    ax.set(xlabel="ViT-B/14 transformer block", ylabel="median patch-token cosine similarity", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("ReplicaCAD small-viewpoint sweep (60 pairs, 224 px)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
