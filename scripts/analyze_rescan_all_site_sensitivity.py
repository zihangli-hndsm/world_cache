#!/usr/bin/env python3
"""Sensitivity of Rescan semantic/instance retrieval gaps to pilot-site inclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_N = 20_000
SEED = 20261006


def interval(values: np.ndarray) -> list[float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return [float(lo), float(hi)]


def summarize(group: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    n = len(group)
    # Resample whole physical spaces; target scans and point queries are nested
    # within these site rows and are never treated as independent clusters.
    indices = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    result: dict[str, object] = {"sites": n, "site_names": group.scene.tolist()}
    for method in ("dino_g", "rgb"):
        gap = group[f"{method}_semantic"].to_numpy() - group[f"{method}_physical"].to_numpy()
        samples = gap[indices].mean(axis=1)
        result[f"{method}_semantic_minus_physical_gap"] = {
            "estimate": float(gap.mean()),
            "ci95": interval(samples),
            "positive_sites": int((gap > 0).sum()),
            "site_gaps": gap.tolist(),
        }
    for metric in ("physical", "semantic"):
        effect = group[f"dino_g_{metric}"].to_numpy() - group[f"rgb_{metric}"].to_numpy()
        samples = effect[indices].mean(axis=1)
        result[f"dino_g_minus_rgb_{metric}"] = {
            "estimate": float(effect.mean()),
            "ci95": interval(samples),
            "dino_g_wins": int((effect > 0).sum()),
            "rgb_wins": int((effect < 0).sum()),
            "ties": int((effect == 0).sum()),
            "site_effects": effect.tolist(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_rescan_all_site_sensitivity"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    bm = pd.read_csv(root / "outputs/reports/20261006_rescan_dinog_external_full/scene_metrics.csv")
    bm = bm.rename(columns={
        "dino_synthetic_views_physical_recall": "dino_g_physical",
        "dino_synthetic_views_semantic_recall": "dino_g_semantic",
        "rgb_color_physical_recall": "rgb_physical",
        "rgb_color_semantic_recall": "rgb_semantic",
    })[["scene", "target_scans", "dino_g_physical", "dino_g_semantic", "rgb_physical", "rgb_semantic"]]
    if set(bm.scene) != {f"scene_{c}" for c in "bcdefghijklm"}:
        raise ValueError(f"unexpected B--M sites in saved aggregate: {sorted(bm.scene)}")

    a = json.loads((root / "outputs/reports/20261006_rescan_scene_a_external/scene_a_summary.json").read_text(encoding="utf-8"))
    added = {
        "scene": "scene_a",
        "target_scans": int(a["target_scans"]),
        "dino_g_physical": float(a["scores"]["dino_g"]["physical_recall"]),
        "dino_g_semantic": float(a["scores"]["dino_g"]["semantic_recall"]),
        "rgb_physical": float(a["scores"]["rgb_color"]["physical_recall"]),
        "rgb_semantic": float(a["scores"]["rgb_color"]["semantic_recall"]),
    }
    all_sites = pd.concat([bm, pd.DataFrame([added])], ignore_index=True).sort_values("scene").reset_index(drop=True)
    if len(all_sites) != 13 or all_sites.scene.nunique() != 13:
        raise ValueError("expected 13 unique spaces including separately evaluated scene A")
    all_sites.to_csv(output / "site_metrics.csv", index=False)

    # Use separate deterministic RNG streams so subset summaries are invariant
    # to the order in which additional sensitivity groups are added.
    results = {
        "method": "frozen DINOv2-G top-20 retrieval on rendered colored PLYs; no pose recovery",
        "statistical_unit": "physical Rescan site; repeated target scans are aggregated within each site",
        "query_population": "annotation-defined positive-instance/class points sampled by the original protocol; outcomes are post-hoc",
        "pilot_sites": ["scene_b", "scene_c"],
        "posthoc_added_site": "scene_a",
        "bootstrap_replicates": BOOTSTRAP_N,
        "seed": SEED,
        "subsets": {},
    }
    for offset, (name, data) in enumerate((
        ("all_13_including_pilots_and_scene_a", all_sites),
        ("11_nonpilot_d_m_plus_a", all_sites[~all_sites.scene.isin(["scene_b", "scene_c"])]),
        ("12_b_to_m_including_pilots", bm),
    )):
        results["subsets"][name] = summarize(data.reset_index(drop=True), np.random.default_rng(SEED + offset))
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    subsets = results["subsets"]
    lines = [
        "# Rescan all-site sensitivity including pilot spaces",
        "",
        "This sensitivity recomputes site-macro semantic Recall@20 minus physical-instance Recall@20 for every available Rescan space. It combines the saved B--M DINO-G/RGB aggregate with the separately evaluated scene A result. Scenes B/C were render pilots and are excluded from the primary 11-site analysis; scene A was added post hoc. Including these spaces is a robustness check, not independent or confirmatory evidence. The input remains rendered colored PLYs, and the query population is annotation-defined.",
        "",
        "| Site set | Sites | DINO-G semantic-minus-physical gap (95% site-bootstrap CI) | Positive sites | RGB gap (95% CI) | DINO-G minus RGB physical recall (95% CI) | DINO-G minus RGB semantic recall (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "all_13_including_pilots_and_scene_a": "All available spaces (B/C pilots + A)",
        "11_nonpilot_d_m_plus_a": "Pilot-excluded subset (D--M + A)",
        "12_b_to_m_including_pilots": "B--M only (includes both pilots)",
    }
    for key, label in labels.items():
        s = subsets[key]
        dg = s["dino_g_semantic_minus_physical_gap"]
        rgb = s["rgb_semantic_minus_physical_gap"]
        p = s["dino_g_minus_rgb_physical"]
        sem = s["dino_g_minus_rgb_semantic"]
        lines.append(
            f"| {label} | {s['sites']} | {dg['estimate']*100:.1f} [{dg['ci95'][0]*100:.1f},{dg['ci95'][1]*100:.1f}] pp | {dg['positive_sites']}/{s['sites']} | {rgb['estimate']*100:.1f} [{rgb['ci95'][0]*100:.1f},{rgb['ci95'][1]*100:.1f}] pp | {p['estimate']*100:+.1f} [{p['ci95'][0]*100:+.1f},{p['ci95'][1]*100:+.1f}] pp | {sem['estimate']*100:+.1f} [{sem['ci95'][0]*100:+.1f},{sem['ci95'][1]*100:+.1f}] pp |"
        )
    lines += [
        "",
        "The semantic-over-physical gap remains positive at every included site, including the two pilot spaces and scene A. DINO-G's advantage over RGB is less uniform: individual scenes can favor color, as scene A does. Site-bootstrap intervals use the physical space as the resampling unit; they do not remove selection after earlier exploration.",
        "",
        "Reproduce with `scripts/analyze_rescan_all_site_sensitivity.py`. Machine-readable summary and site values are saved alongside this report.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
