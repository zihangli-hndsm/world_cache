"""Plot oracle recompute/quality frontiers excluding identity controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    frame = frame[frame.pair_type != "identity"]
    grouped = frame.groupby(["layer", "requested_recompute_ratio"], as_index=False).agg(
        actual_recompute_ratio=("actual_recompute_ratio", "median"),
        final_cls_cosine=("final_cls_cosine", "median"),
        final_patch_cosine=("final_patch_cosine", "median"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, constrained_layout=True)
    for layer, subset in grouped.groupby("layer"):
        axes[0].plot(subset.actual_recompute_ratio, subset.final_cls_cosine, marker="o", label=f"layer {layer}")
        axes[1].plot(subset.actual_recompute_ratio, subset.final_patch_cosine, marker="o", label=f"layer {layer}")
    for axis, label in zip(axes, ("final CLS cosine", "final patch cosine")):
        axis.set(xlabel="actual recompute ratio", ylabel=label, ylim=(0, 1.02))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
