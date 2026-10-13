#!/usr/bin/env python3
"""Build the quota-subset exclusion ledger and summary figure."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent


def main() -> None:
    subset = json.loads((BASE / "quota_subset_manifest.json").read_text())
    chosen = set(subset["selected_scene_ids"])
    with (BASE / "candidate_pairs.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    excluded = [row for row in pairs if row["scene_id"] not in chosen]
    path = BASE / "quota_subset_exclusions.csv"
    with path.open("w", newline="") as stream:
        fields = list(pairs[0]) + ["evaluation_status", "exclusion_reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in excluded:
            writer.writerow({**row, "evaluation_status": "not_downloaded_not_evaluated",
                             "exclusion_reason": "outside the frozen lexicographic room prefix that fits the 40,000,000,000-byte quota budget; excluded before preflight and prediction"})

    boot = json.loads((BASE / "results" / "bootstrap.json").read_text())
    metrics = boot["metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"DINOv2 ViT-G": "#315a9b", "RGB patch mean": "#8d959e"}

    def estimate(key):
        item = metrics[key]
        return 100 * item["estimate"], np.asarray(item["ci95_percentile"]) * 100

    ax = axes[0]
    positions = np.arange(2)
    width = 0.32
    for offset, method, keys in [
        (-width/2, "DINOv2 ViT-G", ("dino_semantic_recall_at20", "dino_physical_instance_recall_at20")),
        (width/2, "RGB patch mean", ("rgb_semantic_recall_at20", "rgb_physical_instance_recall_at20")),
    ]:
        for j, key in enumerate(keys):
            value, interval = estimate(key)
            error = np.asarray([[value-interval[0]], [interval[1]-value]])
            x = positions[j] + offset
            ax.bar(x, value, width=width, color=colors[method], alpha=0.95 if j == 0 else 0.60,
                   hatch="" if j == 0 else "///", edgecolor="white", label=method if j == 0 else None)
            ax.errorbar(x, value, yerr=error, color="black", capsize=3, lw=1)
    ax.set_xticks(positions, ["Semantic", "Physical instance"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall@20 (%)")
    ax.set_title("Same query and candidate pools")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.2)

    ax = axes[1]
    metric_specs = [
        ("accepted_physical_instance_precision", "Accepted precision"),
        ("accepted_query_coverage", "Accepted coverage"),
        ("correct_accepted_query_coverage", "Correct coverage"),
    ]
    x = np.arange(len(metric_specs))
    width = .32
    for offset, method, prefix in [(-width/2, "DINOv2 ViT-G", "dino_"), (width/2, "RGB patch mean", "rgb_")]:
        for j, (name, _) in enumerate(metric_specs):
            key = prefix + name
            value, interval = estimate(key)
            error = np.asarray([[value-interval[0]], [interval[1]-value]])
            xpos = x[j] + offset
            ax.bar(xpos, value, width=width, color=colors[method], label=method if j == 0 else None)
            ax.errorbar(xpos, value, yerr=error, color="black", capsize=3, lw=1)
    ax.set_xticks(x, [name for _, name in metric_specs])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Frozen spatial consensus")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.2)
    figure.suptitle("Quota-limited exploratory subset (17 environments); 95% environment-cluster bootstrap intervals")
    figure.savefig(BASE / "results" / "quota_subset_summary.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"excluded_relation_count": len(excluded),
                      "excluded_environment_count": len({r['scene_id'] for r in excluded}),
                      "plot": str((BASE / 'results' / 'quota_subset_summary.png').relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
