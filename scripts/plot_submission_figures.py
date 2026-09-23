"""Render compact, paper-ready method and failure-taxonomy diagrams."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd


BLUE = "#1f4e79"
TEAL = "#177e89"
ORANGE = "#d97706"
RED = "#b42318"
GREEN = "#2f855a"
GRAY = "#5b6573"
LIGHT_BLUE = "#e8f1f8"
LIGHT_TEAL = "#e7f5f3"
LIGHT_ORANGE = "#fff4df"
LIGHT_RED = "#fdeaea"


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def rounded_box(ax: plt.Axes, x: float, y: float, width: float, height: float,
                title: str, body: str, face: str, edge: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.018,rounding_size=0.04",
            linewidth=1.3, edgecolor=edge, facecolor=face,
        )
    )
    ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1f2933")
    ax.text(x + width / 2, y + height * 0.32, body, ha="center", va="center",
            fontsize=8.2, color="#374151", linespacing=1.25)


def render_method(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.05))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.05)
    ax.axis("off")

    boxes = [
        (0.15, "RGB-D scan pair", "query + reference\nRGB, depth, 3-D points", LIGHT_BLUE, BLUE),
        (2.15, "Frozen DINOv2", "blocks (2, 20, 40)\nnormalized patch tokens", LIGHT_BLUE, BLUE),
        (4.15, "Top-K candidates", "K = 20\nappearance-only ranking", LIGHT_TEAL, TEAL),
        (6.15, "Spatial consensus", "3-point RANSAC\nτ = 0.10 m", LIGHT_ORANGE, ORANGE),
        (8.15, "Selective output", "residual + inlier\naccept / abstain", LIGHT_RED, RED),
        (10.15, "Two evaluations", "semantic transfer\npose diagnostic", LIGHT_TEAL, TEAL),
    ]
    for x, title, body, face, edge in boxes:
        rounded_box(ax, x, 1.35, 1.7, 1.05, title, body, face, edge)
    for x in (1.86, 3.86, 5.86, 7.86, 9.86):
        ax.add_patch(FancyArrowPatch((x, 1.87), (x + 0.27, 1.87),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.2, color=GRAY))

    ax.text(6, 2.72, "Matching path: no query-to-reference alignment is used",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1f2933")
    ax.add_patch(FancyArrowPatch((10.95, 0.95), (10.95, 1.30), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0,
                                 linestyle="--", color=GRAY))
    ax.text(6, 0.52,
            "Hidden 3RScan alignment is used only after matching for evaluation of pose error and label correctness",
            ha="center", va="center", fontsize=8.7, color=GRAY,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f7f8fa",
                  "edgecolor": "#c9ced6", "linestyle": "--"})
    save(fig, output_dir, "method_pipeline")


def load_source_summaries(root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    clean_aggregate = root / "20261002_3rscan_consensus_vitg_clean_v2/summary_all.csv"
    if clean_aggregate.exists():
        frame = pd.read_csv(clean_aggregate)
        frame = frame[~frame.pair.astype(str).str.contains("target")]
        for raw in frame.to_dict("records"):
            pair = str(raw["pair"])
            match = re.search(r"(0cac|20c993|4aca|5630|6bde|751a|95be|c670|chairs|pair0[2-8])", pair)
            rows.append({
                "pair": match.group(1) if match else pair,
                "semantic": float(raw["inlier_semantic_accuracy"]) * 100,
                "rotation": float(raw["rotation_error_deg"]),
                "translation": float(raw["translation_error_m"]),
                "coverage": float(raw["inlier_coverage"]) * 100,
            })
        return pd.DataFrame(rows)
    for path in sorted(root.glob("*_consensus_*_vitg_diag/summary.csv")):
        pair_match = re.search(r"_consensus_(.+)_vitg_diag$", path.parent.name)
        if pair_match is None or pair_match.group(1) == "target":
            continue
        row = pd.read_csv(path).iloc[0]
        rows.append({
            "pair": pair_match.group(1),
            "semantic": float(row.inlier_semantic_accuracy) * 100,
            "rotation": float(row.rotation_error_deg),
            "translation": float(row.translation_error_m),
            "coverage": float(row.inlier_coverage) * 100,
        })
    return pd.DataFrame(rows)


def render_failures(output_dir: Path, per_pair_path: Path, summary_root: Path) -> None:
    per_pair = pd.read_csv(per_pair_path)
    low = per_pair[per_pair.score == "low_spatial_residual"].set_index("pair")
    summaries = load_source_summaries(summary_root).set_index("pair")
    selected = ["4aca", "95be", "pair06", "pair07", "20c993", "6bde", "pair08"]
    selected = [pair for pair in selected if pair in low.index and pair in summaries.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.25),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    x = np.arange(len(selected))
    values = low.loc[selected, "semantic_precision"].to_numpy() * 100
    colors = [RED if pair in {"4aca", "95be"} else ORANGE if pair in {"pair06", "pair07"} else GREEN
              for pair in selected]
    ax.bar(x, values, color=colors, width=0.68, edgecolor="white", linewidth=0.6)
    ax.axhline(90, color="#222", linestyle=":", linewidth=1)
    ax.set_ylim(0, 105)
    ax.set_ylabel("semantic precision at 20% coverage (%)")
    ax.set_xticks(x, selected, rotation=45, ha="right")
    ax.set_title("Selective failure boundary", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.02, 0.96, "red: candidate/consensus failures\norange: intermediate semantic\ngreen: high semantic",
            transform=ax.transAxes, va="top", fontsize=8.2, color="#374151",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85,
                  "edgecolor": "#d1d5db"})

    ax = axes[1]
    category_colors = {
        "low semantic (<50%)": RED,
        "intermediate semantic (50--90%)": ORANGE,
        "high semantic (>=90%)": GREEN,
    }
    for pair, row in summaries.iterrows():
        if row.semantic < 50:
            category = "low semantic (<50%)"
        elif row.semantic < 90:
            category = "intermediate semantic (50--90%)"
        else:
            category = "high semantic (>=90%)"
        strict_pose = row.rotation < 5 and row.translation < 0.25
        ax.scatter(row.rotation, row.translation, s=58, color=category_colors[category],
                   edgecolor="#111827" if strict_pose else "white",
                   linewidth=1.2 if strict_pose else 0.7, alpha=0.9, zorder=3)
        if pair in {"4aca", "95be", "20c993", "pair07", "pair08", "pair06"}:
            offsets = {
                "4aca": (4, 3), "95be": (4, -12), "20c993": (4, 4),
                "pair07": (-35, -15), "pair08": (5, 4), "pair06": (4, -12),
            }
            dx, dy = offsets[pair]
            ax.annotate(pair, (row.rotation, row.translation), xytext=(dx, dy),
                        textcoords="offset points", fontsize=8)
    ax.axvline(5, color="#222", linestyle=":", linewidth=1)
    ax.axhline(0.25, color="#222", linestyle=":", linewidth=1)
    ax.set_xlim(-3, 180)
    ax.set_ylim(-0.05, max(3.2, float(summaries.translation.max()) * 1.08))
    ax.set_xlabel("rotation error (degrees; hidden evaluation only)")
    ax.set_ylabel("translation error (m; hidden evaluation only)")
    ax.set_title("Semantic quality and joint pose error separate", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=7,
                   label="low semantic (<50%)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ORANGE, markersize=7,
                   label="intermediate semantic (50--90%)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=7,
                   label="high semantic (>=90%)"),
        plt.Line2D([0], [0], marker="o", color="#111827", markerfacecolor="white", markersize=7,
                   label="strict pose success (outlined)"),
    ]
    ax.legend(handles=handles, frameon=True, framealpha=0.88, facecolor="white",
              edgecolor="#d1d5db", fontsize=8, loc="upper left")
    save(fig, output_dir, "failure_taxonomy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-pair", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_method(args.output_dir)
    render_failures(args.output_dir, args.per_pair, args.summary_root)


if __name__ == "__main__":
    main()
