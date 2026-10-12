#!/usr/bin/env python3
"""Compare physical-identity and semantic gains from a source-frozen gate.

The 3RScan q75 threshold is refit only from source relations in each bootstrap
draw. Rescan scene clusters are resampled jointly for both outcomes, preserving
the within-query pairing of semantic and physical top-1 correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rescan_source_frozen_threshold import load_rescan, load_source  # noqa: E402


SEED = 20261010


def summarize(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    keep = frame.score.to_numpy() >= threshold
    physical = frame.physical_top1.to_numpy(dtype=bool)
    semantic = frame.semantic_top1.to_numpy(dtype=bool)
    n = len(frame)
    n_keep = int(keep.sum())
    p0, s0 = float(physical.mean()), float(semantic.mean())
    p1 = float(physical[keep].mean()) if n_keep else float("nan")
    s1 = float(semantic[keep].mean()) if n_keep else float("nan")
    return {
        "queries": n,
        "accepted": n_keep,
        "coverage": n_keep / n,
        "physical_baseline": p0,
        "physical_accepted": p1,
        "physical_gain": p1 - p0,
        "semantic_baseline": s0,
        "semantic_accepted": s1,
        "semantic_gain": s1 - s0,
        "physical_minus_semantic_gain": (p1 - p0) - (s1 - s0),
    }


def bootstrap(
    source_relations: list[np.ndarray],
    sites: dict[str, pd.DataFrame],
    *,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    names = np.asarray(sorted(sites))
    arrays = {
        name: (
            g.score.to_numpy(dtype=np.float64),
            g.physical_top1.to_numpy(dtype=bool),
            g.semantic_top1.to_numpy(dtype=bool),
        )
        for name, g in sites.items()
    }
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(replicates):
        sampled_rel = rng.integers(0, len(source_relations), size=len(source_relations))
        threshold = float(np.quantile(np.concatenate([source_relations[i] for i in sampled_rel]), 0.75))
        sampled_sites = rng.integers(0, len(names), size=len(names))
        metrics = []
        for idx in sampled_sites:
            score, physical, semantic = arrays[names[idx]]
            keep = score >= threshold
            p0, s0 = float(physical.mean()), float(semantic.mean())
            p1 = float(physical[keep].mean()) if keep.any() else np.nan
            s1 = float(semantic[keep].mean()) if keep.any() else np.nan
            metrics.append((p1 - p0, s1 - s0, (p1 - p0) - (s1 - s0)))
        values = np.asarray(metrics, dtype=np.float64)
        rows.append({
            "threshold": threshold,
            "physical_gain": float(np.nanmean(values[:, 0])),
            "semantic_gain": float(np.nanmean(values[:, 1])),
            "physical_minus_semantic_gain": float(np.nanmean(values[:, 2])),
        })
    return pd.DataFrame(rows)


def interval(series: pd.Series) -> dict[str, float]:
    values = series.to_numpy(dtype=np.float64)
    return {
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "q975": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_rescan_selectivity_outcome_gap"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    source_relations, _ = load_source(root)
    source_scores = np.concatenate(source_relations)
    threshold = float(np.quantile(source_scores, 0.75))
    rescan = load_rescan(root / "outputs/reports/20261006_rescan_dinog_scoreconfidence_corrected/query_metrics.csv")
    subsets = {
        "all_13_scenes": rescan,
        "nonpilot_11_scenes": rescan[~rescan.scene.isin(["scene_b", "scene_c"])].copy(),
    }
    per_scene_rows = []
    summaries = {}
    boot_tables = {}
    for name, frame in subsets.items():
        sites = {site: g for site, g in frame.groupby("scene", sort=True)}
        site_summaries = []
        for site, group in sites.items():
            row = {"subset": name, "scene": site, **summarize(group, threshold)}
            per_scene_rows.append(row)
            site_summaries.append(row)
        boot = bootstrap(source_relations, sites, replicates=args.replicates, seed=args.seed)
        boot.to_csv(out / f"bootstrap_{name}.csv", index=False)
        boot_tables[name] = boot
        summaries[name] = {
            "scenes": len(sites),
            "target_scans": int(frame.target_scan.nunique()),
            "queries": len(frame),
            "macro": {key: float(pd.DataFrame(site_summaries)[key].mean()) for key in (
                "coverage", "physical_baseline", "physical_accepted", "physical_gain",
                "semantic_baseline", "semantic_accepted", "semantic_gain",
                "physical_minus_semantic_gain",
            )},
            "positive_scene_gains": {
                "physical": int((pd.DataFrame(site_summaries).physical_gain > 0).sum()),
                "semantic": int((pd.DataFrame(site_summaries).semantic_gain > 0).sum()),
                "physical_gain_exceeds_semantic": int((pd.DataFrame(site_summaries).physical_minus_semantic_gain > 0).sum()),
            },
            "joint_bootstrap_95": {
                metric: interval(boot[metric]) for metric in (
                    "physical_gain", "semantic_gain", "physical_minus_semantic_gain"
                )
            },
        }
    pd.DataFrame(per_scene_rows).to_csv(out / "per_scene_outcome_metrics.csv", index=False)
    summary = {
        "status": "complete",
        "source_threshold": threshold,
        "source_relations": len(source_relations),
        "source_queries": int(len(source_scores)),
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "bootstrap_units": "jointly resample 3RScan source relations to refit q75 and Rescan scenes; semantic and physical outcomes remain paired within each query",
        "subsets": summaries,
        "interpretation": "The physical-minus-semantic gain is descriptive; positive values indicate the gate raises physical top-1 precision more than semantic top-1 precision. It is not a causal decomposition.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Physical identity vs semantic selectivity under the source-frozen Rescan gate",
        "",
        f"The source-only 3RScan q75 threshold is {threshold:.6f}. Semantic and physical top-1 correctness are evaluated on the same Rescan queries. Bootstrap draws jointly resample source relations (refitting q75) and Rescan scenes, preserving paired outcomes.",
        "",
        "| Rescan subset | Scenes | Coverage | Baseline physical | Accepted physical | Physical gain | Baseline semantic | Accepted semantic | Semantic gain | Physical minus semantic gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summaries.items():
        m = result["macro"]
        lines.append(
            f"| {name.replace('_', ' ')} | {result['scenes']} | {m['coverage']:.1%} | {m['physical_baseline']:.1%} | {m['physical_accepted']:.1%} | {m['physical_gain']*100:+.1f} pp | {m['semantic_baseline']:.1%} | {m['semantic_accepted']:.1%} | {m['semantic_gain']*100:+.1f} pp | {m['physical_minus_semantic_gain']*100:+.1f} pp |"
        )
    lines += ["", "## Joint source-relation / Rescan-scene bootstrap, 95% percentile intervals", ""]
    for name, result in summaries.items():
        ci = result["joint_bootstrap_95"]
        lines.append(
            f"- {name.replace('_', ' ')}: physical gain {ci['physical_gain']['median']*100:+.1f} pp [{ci['physical_gain']['q025']*100:+.1f},{ci['physical_gain']['q975']*100:+.1f}], semantic gain {ci['semantic_gain']['median']*100:+.1f} pp [{ci['semantic_gain']['q025']*100:+.1f},{ci['semantic_gain']['q975']*100:+.1f}], paired gain contrast {ci['physical_minus_semantic_gain']['median']*100:+.1f} pp [{ci['physical_minus_semantic_gain']['q025']*100:+.1f},{ci['physical_minus_semantic_gain']['q975']*100:+.1f}]."
        )
    lines += [
        "",
        "The contrast is the difference between two precision lifts computed on the same scene resamples. It describes which correctness outcome is enriched more by the gate; it does not identify a causal mechanism. The Rescan query population is annotation-defined and synthetic-view based, so this remains an exploratory cross-dataset/protocol stress test.",
        "",
        "Per-scene rows, paired bootstrap draws, and the machine-readable summary are stored beside this report. Reproduce with `scripts/analyze_rescan_selectivity_outcome_gap.py`.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
