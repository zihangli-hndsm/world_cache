#!/usr/bin/env python3
"""Aggregate the Rescan synthetic-view DINO point-retrieval audit by scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("dino_synthetic_views", "rgb_color")
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_SAMPLES = 20_000


def scene_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scene, group in frame.groupby("scene", sort=True):
        row: dict[str, object] = {"scene": scene, "target_scans": int(group.target_scan.nunique())}
        for method in METHODS:
            selected = group[group.method == method]
            denominator = int(selected.target_present_queries.sum())
            if denominator <= 0:
                raise ValueError(f"no target-present queries for {scene}/{method}")
            physical = int(selected.physical_correct.sum())
            semantic = int(selected.semantic_correct_present.sum())
            row[f"{method}_queries_present"] = denominator
            row[f"{method}_physical_recall"] = physical / denominator
            row[f"{method}_semantic_recall"] = semantic / denominator
            row[f"{method}_semantic_instance_gap"] = (semantic - physical) / denominator
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, seed: int) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "clusters": int(len(values)),
    }


def summarize(frame: pd.DataFrame, label: str) -> tuple[dict[str, object], pd.DataFrame]:
    sites = scene_rates(frame)
    delta_physical = sites.dino_synthetic_views_physical_recall - sites.rgb_color_physical_recall
    delta_semantic = sites.dino_synthetic_views_semantic_recall - sites.rgb_color_semantic_recall
    result: dict[str, object] = {
        "subset": label,
        "scenes": int(len(sites)),
        "target_scan_pairs": int(frame.target_scan.nunique()),
        "query_counts_present_total": int(frame[frame.method == METHODS[0]].target_present_queries.sum()),
        "site_macro": {},
        "paired_method_delta_dino_minus_rgb": {
            "physical_recall": bootstrap_mean(delta_physical.to_numpy(), BOOTSTRAP_SEED + len(sites)),
            "semantic_recall": bootstrap_mean(delta_semantic.to_numpy(), BOOTSTRAP_SEED + 10 + len(sites)),
        },
        "site_macro_semantic_minus_physical_gap": {
            "dino_synthetic_views": bootstrap_mean(sites.dino_synthetic_views_semantic_instance_gap.to_numpy(), BOOTSTRAP_SEED + 20 + len(sites)),
            "rgb_color": bootstrap_mean(sites.rgb_color_semantic_instance_gap.to_numpy(), BOOTSTRAP_SEED + 30 + len(sites)),
        },
        "site_wins_dino_vs_rgb": {
            "physical_recall": int((delta_physical > 0).sum()),
            "semantic_recall": int((delta_semantic > 0).sum()),
        },
        "site_rows": sites.to_dict(orient="records"),
    }
    for method in METHODS:
        result["site_macro"][method] = {
            "physical_recall": bootstrap_mean(sites[f"{method}_physical_recall"].to_numpy(), BOOTSTRAP_SEED + 40 + len(sites)),
            "semantic_recall": bootstrap_mean(sites[f"{method}_semantic_recall"].to_numpy(), BOOTSTRAP_SEED + 50 + len(sites)),
        }
    return result, sites


def fmt(metric: dict[str, object]) -> str:
    return f"{metric['estimate'] * 100:.1f}% [{metric['ci95'][0] * 100:.1f}, {metric['ci95'][1] * 100:.1f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/reports/20261006_rescan_dino_external_full/pair_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_rescan_dino_external_full"))
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    probe_summary_path = args.input.parent / "summary.json"
    probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    model_size = probe_summary.get("model_size", "small")
    layers = probe_summary.get("layers", [2, 6, 12])
    model_label = f"DINOv2-{model_size}"
    full = raw[raw.k == 20].copy()
    expected_pairs = full.target_scan.nunique()
    expected_scenes = full.scene.nunique()
    if expected_pairs != 29 or expected_scenes != 12:
        raise ValueError(f"expected 29 target scans across 12 scenes; got {expected_pairs}/{expected_scenes}")
    all_result, all_sites = summarize(full, "all_12_rescan_scenes")
    nonpilot = full[~full.scene.isin(["scene_b", "scene_c"])].copy()
    heldout_result, heldout_sites = summarize(nonpilot, "10_scenes_excluding_render_pilot_b_c")
    summary = {
        "status": "complete",
        "input": str(args.input),
        "model": f"pretrained {model_label}; normalized concatenation of blocks {layers}",
        "candidate_pool": "5 cm voxel sample from source colored PLY; top-20 cosine retrieval",
        "query_sample": "query points have positive instance and class IDs; up to 96 per target instance, capped at 3000 per target scan",
        "input_protocol": "eight deterministic wide-FOV RGB views rendered from each colored point-cloud reconstruction; not raw RGB-D frames",
        "labels_used_for_features_or_ranking": False,
        "label_use": "balanced query sampling, full-source target-presence denominator, and post-hoc scoring only",
        "reference_presence_uses_full_source_annotation": True,
        "statistical_unit": "Rescan scene cluster; target scans within each scene are aggregated before macro averaging/bootstrap",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "all_scenes": all_result,
        "nonpilot_scenes": heldout_result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    all_sites.to_csv(args.output_dir / "scene_metrics.csv", index=False)
    heldout_sites.to_csv(args.output_dir / "nonpilot_scene_metrics.csv", index=False)

    lines = [
        "# Independent Rescan candidate-recall audit (synthetic-view DINOv2)", "",
        "The [official Rescan dataset page](https://rescan.cs.princeton.edu/) describes stable object IDs over repeated captures in 13 distinct spaces. This audit uses the locally available 12 spaces B--M (29 target scans); `scene_a` was not present in the extracted annotation subset. Labels define a balanced query sample and the full-source target-presence denominator, then score predictions; labels never enter feature extraction or candidate ranking. An object omitted by the 5 cm candidate downsampling remains a candidate miss. See also the [ICCV 2019 Rescan paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Halber_Rescan_Inductive_Instance_Segmentation_for_Indoor_RGBD_Scans_ICCV_2019_paper.pdf).", "",
        f"For each scene, scan `_0` is the reference and every later scan is a target. We render eight deterministic wide-FOV views from the colored point cloud, run pretrained {model_label} at 224px, concatenate normalized block {', '.join(map(str, layers))} patch features, map visible patch descriptors to a 5 cm voxel sample, and retrieve top-20 source points. The baseline ranks the same candidate points by RGB Euclidean distance. Queries require positive instance and class IDs and are capped at 96 points per target instance and 3,000 per target scan. This is a synthetic-render point-cloud test, not the project's original raw-frame RGB-D protocol.", "",
        "Metrics use the same target-present query denominator: semantic hit@20 is any candidate with the query class, and physical hit@20 is any candidate with the stable query instance ID. We average target scans within each scene and then average/bootstrap over scenes. The 10-scene result excludes B/C, which were used for the initial rendering feasibility pilot.", "",
        "| Scene subset | Scenes / target scans | DINO physical Recall@20 | RGB physical Recall@20 | DINO semantic Recall@20 | RGB semantic Recall@20 | DINO semantic-physical gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in (all_result, heldout_result):
        dino = result["site_macro"]["dino_synthetic_views"]
        rgb = result["site_macro"]["rgb_color"]
        gap = result["site_macro_semantic_minus_physical_gap"]["dino_synthetic_views"]
        lines.append(
            f"| {result['subset']} | {result['scenes']} / {result['target_scan_pairs']} | {fmt(dino['physical_recall'])} | {fmt(rgb['physical_recall'])} | {fmt(dino['semantic_recall'])} | {fmt(rgb['semantic_recall'])} | {fmt(gap)} |"
        )
    delta = heldout_result["paired_method_delta_dino_minus_rgb"]
    lines.extend([
        "",
        f"On the 10 non-pilot scenes, DINO-minus-RGB physical Recall@20 is {delta['physical_recall']['estimate']*100:+.1f} pp (scene-cluster 95% bootstrap CI [{delta['physical_recall']['ci95'][0]*100:+.1f},{delta['physical_recall']['ci95'][1]*100:+.1f}]); DINO wins in {heldout_result['site_wins_dino_vs_rgb']['physical_recall']}/10 scenes. The semantic difference is {delta['semantic_recall']['estimate']*100:+.1f} pp ([{delta['semantic_recall']['ci95'][0]*100:+.1f},{delta['semantic_recall']['ci95'][1]*100:+.1f}]); DINO wins in {heldout_result['site_wins_dino_vs_rgb']['semantic_recall']}/10 scenes.",
        "",
        "Interpretation: this supplies external evidence that semantic candidate retrieval exceeds tracked-identity retrieval in an independently collected rescan dataset, while the DINO-vs-color effect is only a point-cloud synthetic-render comparison. The B--M annotations have also appeared in prior Rescan geometry diagnostics, and only two scenes informed the initial rendering pilot, so report the 10-scene subset as exploratory replication rather than a preregistered test. Do not transfer these values to raw RGB-D frame retrieval or global pose accuracy.",
        "",
        "Reproduce with `scripts/probe_rescan_dino_point_candidates.py` followed by `scripts/aggregate_rescan_dino_point_candidates.py`. Pair-level retrieval counts are in `pair_metrics.csv`; scene-cluster summaries are in `aggregate_summary.json`.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote scene-cluster summary to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
