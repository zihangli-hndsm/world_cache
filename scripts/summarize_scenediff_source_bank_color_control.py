#!/usr/bin/env python3
"""Summarize and plot the paired SceneDiff source-bank color control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BANKS = ["1", "2", "4", "8", "all"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "outputs/reports/20261010_scenediff_source_bank_color_control/pair_run_scores.csv"))
    parser.add_argument("--run-summary", type=Path, default=Path(
        "outputs/reports/20261010_scenediff_source_bank_color_control/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/reports/20261010_scenediff_source_bank_color_control"))
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    summary = json.loads(args.run_summary.read_text())
    pairs = sorted(frame.pair.astype(str).unique())
    runs = sorted(int(x) for x in frame.run.unique())
    if len(pairs) != 104 or len(runs) != 20 or set(frame.bank_views.astype(str).unique()) != set(BANKS):
        raise ValueError("input does not match the frozen 104-pair / 20-run population")
    matrices = {metric: {bank: np.full((len(runs), len(pairs)), np.nan)
                         for bank in BANKS}
                for metric in ("correct", "rgbhist_correct")}
    pair_index = {pair: i for i, pair in enumerate(pairs)}
    run_index = {run: i for i, run in enumerate(runs)}
    for row in frame.itertuples(index=False):
        i, j, bank = run_index[int(row.run)], pair_index[str(row.pair)], str(row.bank_views)
        matrices["correct"][bank][i, j] = float(row.correct)
        matrices["rgbhist_correct"][bank][i, j] = float(row.rgbhist_correct)
    if any(not np.isfinite(m).all() for by_bank in matrices.values() for m in by_bank.values()):
        raise ValueError("missing pair/run/bank cells")

    rng = np.random.default_rng(args.seed)
    boot_curve = {metric: {bank: np.empty(args.draws, dtype=np.float32) for bank in BANKS}
                  for metric in matrices}
    boot_effect = {"dino": np.empty(args.draws, dtype=np.float32),
                   "rgbhist": np.empty(args.draws, dtype=np.float32),
                   "interaction": np.empty(args.draws, dtype=np.float32)}
    for start in range(0, args.draws, 1000):
        count = min(1000, args.draws - start)
        sampled_pairs = rng.integers(0, len(pairs), size=(count, len(pairs)))
        sampled_runs = rng.integers(0, len(runs), size=(count, len(pairs)))
        curve_values = {}
        for metric in matrices:
            for bank in BANKS:
                values = matrices[metric][bank][sampled_runs, sampled_pairs].mean(axis=1)
                boot_curve[metric][bank][start:start + count] = values
                curve_values[(metric, bank)] = values
        dino_gain = curve_values[("correct", "4")] - curve_values[("correct", "1")]
        hist_gain = curve_values[("rgbhist_correct", "4")] - curve_values[("rgbhist_correct", "1")]
        boot_effect["dino"][start:start + count] = dino_gain
        boot_effect["rgbhist"][start:start + count] = hist_gain
        boot_effect["interaction"][start:start + count] = dino_gain - hist_gain

    curves = {}
    for metric, label in (("correct", "dino"), ("rgbhist_correct", "rgbhist")):
        curves[label] = {}
        for bank in BANKS:
            curves[label][bank] = {
                "estimate": float(matrices[metric][bank].mean()),
                "nested_pair_view_interval_95_pointwise": [
                    float(x) for x in np.quantile(boot_curve[metric][bank], [.025, .975])
                ],
            }
    primary = summary["four_vs_one"]
    for key in ("dino", "rgbhist", "interaction"):
        point = (primary["dino"]["difference_vs_one"] if key == "dino" else
                 primary["rgbhist"]["difference_vs_one"] if key == "rgbhist" else
                 primary["dino_minus_rgbhist_gain"])
        primary.setdefault("plot_bootstrap", {})[key] = {
            "estimate": float(point),
            "nested_pair_view_interval_95": [float(x) for x in np.quantile(boot_effect[key], [.025, .975])],
        }
    summary["pointwise_curve_intervals"] = curves
    summary["curve_bootstrap"] = {"draws": args.draws, "seed": args.seed,
                                  "interval_type": "pointwise percentile; pairs and within-pair runs resampled",
                                  "not_familywise_adjusted": True}
    args.run_summary.write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "curve_bootstrap.json").write_text(json.dumps({
        "status": "complete", "pairs": len(pairs), "runs": len(runs),
        "curves": curves, "four_vs_one": primary["plot_bootstrap"],
        "bootstrap": summary["curve_bootstrap"],
    }, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.0, 3.1), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    colors = {"dino": "#3569a8", "rgbhist": "#d07a22"}
    x = np.arange(len(BANKS))
    for metric, label, color in (("dino", "DINOv2", colors["dino"]),
                                 ("rgbhist", "RGB histogram", colors["rgbhist"])):
        means = np.asarray([curves[metric][bank]["estimate"] for bank in BANKS]) * 100
        bounds = np.asarray([curves[metric][bank]["nested_pair_view_interval_95_pointwise"]
                             for bank in BANKS]) * 100
        ax0.errorbar(x, means, yerr=np.vstack([means - bounds[:, 0], bounds[:, 1] - means]),
                     color=color, marker="o", linewidth=1.8, markersize=4,
                     capsize=2.5, label=label)
    ax0.set_xticks(x, ["1", "2", "4", "8", "All"])
    ax0.set_xlabel("Source views per gallery object")
    ax0.set_ylabel("Pair-macro top-1 (%)")
    ax0.set_title("(a) Retrieval across bank sizes")
    ax0.set_ylim(45, 90)
    ax0.grid(axis="y", color="#dddddd", linewidth=.6)
    ax0.legend(frameon=False, loc="lower right")

    effect_keys = ["dino", "rgbhist", "interaction"]
    labels = ["DINO: 4−1", "Color: 4−1", "Difference in gains"]
    centers = np.asarray([primary["plot_bootstrap"][k]["estimate"] for k in effect_keys]) * 100
    intervals = np.asarray([primary["plot_bootstrap"][k]["nested_pair_view_interval_95"]
                            for k in effect_keys]) * 100
    effect_colors = [colors["dino"], colors["rgbhist"], "#555555"]
    for i, (center, bounds, color) in enumerate(zip(centers, intervals, effect_colors)):
        ax1.errorbar(center, i, xerr=[[center - bounds[0]], [bounds[1] - center]],
                     color=color, marker="o", markersize=5, capsize=3, linewidth=1.8)
    ax1.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax1.set_yticks(np.arange(3), labels)
    ax1.set_xlabel("Four minus one view (percentage points)")
    ax1.set_title("(b) Paired view-count effects")
    ax1.grid(axis="x", color="#dddddd", linewidth=.6)
    fig.savefig(args.output_dir / "view_count_color_control.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "view_count_color_control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"status": "complete", "figure": str(args.output_dir / "view_count_color_control.pdf"),
                      "four_vs_one": primary["plot_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
