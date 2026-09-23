"""Plot aligned-token stability by requested viewpoint bins."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def binned(frame: pd.DataFrame, column: str, edges: list[float], labels: list[str]) -> pd.DataFrame:
    result = frame.copy()
    result["bin"] = pd.cut(result[column], bins=edges, labels=labels, include_lowest=True)
    return result


def plot(frame: pd.DataFrame, output: Path, title: str) -> None:
    controls = [("aligned", "geometry-aligned", "#0072B2"), ("same_index", "same 2D index", "#D55E00")]
    fig, axes = plt.subplots(1, len(frame.bin.cat.categories), figsize=(15, 3.8), sharey=True, constrained_layout=True)
    for axis, motion_bin in zip(axes, frame.bin.cat.categories):
        subset = frame[frame.bin == motion_bin]
        for control, label, color in controls:
            data = subset[subset.control == control].groupby("layer", as_index=False).cosine.median()
            axis.plot(data.layer, data.cosine, marker="o", linewidth=2, label=label, color=color)
        axis.set_title(str(motion_bin))
        axis.set_xlabel("block")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("median cosine")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rotation-output", type=Path, required=True)
    parser.add_argument("--translation-output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    # Identity pairs are an explicit calibration control. Excluding them keeps
    # the 0–10°/0–0.25m bins from being dominated by exact self-comparisons.
    frame = frame[frame.pair_type != "identity"].copy()
    rotation = binned(frame, "rotation_degrees", [-1e-6, 10, 30, 60, np.inf], ["0–10°", "10–30°", "30–60°", ">60°"])
    translation = binned(frame, "translation_m", [-1e-6, 0.25, 0.5, 1.0, np.inf], ["0–0.25m", "0.25–0.5m", "0.5–1.0m", ">1.0m"])
    plot(rotation, args.rotation_output, "ReplicaCAD: stability by rotation bin")
    plot(translation, args.translation_output, "ReplicaCAD: stability by translation bin")


if __name__ == "__main__":
    main()
