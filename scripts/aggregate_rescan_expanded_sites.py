#!/usr/bin/env python3
"""Aggregate 10 non-pilot Rescan sites plus the newly inspected scene A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/reports")
NAMES = {
    "dino_s": "dino_synthetic_views",
    "dino_g": "dino_synthetic_views",
    "rgb_color": "rgb_color",
}
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20261020


def read_metrics(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame.k == 20) & (frame.method == method)].copy()
    if frame.empty:
        raise ValueError(f"no top-20 {method} rows in {path}")
    return frame


def site_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for scene, group in frame.groupby("scene", sort=True):
        row: dict[str, object] = {"scene": scene, "target_scans": int(group.target_scan.nunique())}
        for scale in ("dino_s", "dino_g", "rgb_color"):
            rows = group[group.scale == scale]
            denominator = int(rows.target_present_queries.sum())
            row[f"{scale}_physical"] = int(rows.physical_correct.sum()) / denominator
            row[f"{scale}_semantic"] = int(rows.semantic_correct_present.sum()) / denominator
            row[f"{scale}_gap"] = row[f"{scale}_semantic"] - row[f"{scale}_physical"]
            row.setdefault("queries_present", denominator)
        output.append(row)
    return pd.DataFrame(output)


def boot(values: np.ndarray, seed_offset: int) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "clusters": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_rescan_expanded_sites"))
    args = parser.parse_args()
    b_m = {
        "dino_s": args.root / "20261006_rescan_dino_external_full/pair_metrics.csv",
        "dino_g": args.root / "20261006_rescan_dinog_external_full/pair_metrics.csv",
        "rgb_color": args.root / "20261006_rescan_dinog_external_full/pair_metrics.csv",
    }
    scene_a = {
        "dino_s": args.root / "20261006_rescan_scene_a_external/dino_s/pair_metrics.csv",
        "dino_g": args.root / "20261006_rescan_scene_a_external/dino_g/pair_metrics.csv",
        "rgb_color": args.root / "20261006_rescan_scene_a_external/dino_g/pair_metrics.csv",
    }
    frames = []
    for scale in ("dino_s", "dino_g", "rgb_color"):
        method = NAMES[scale]
        old = read_metrics(b_m[scale], method)
        old = old[old.scene.isin([f"scene_{letter}" for letter in "defghijklm"])]
        new = read_metrics(scene_a[scale], method)
        for frame in (old, new):
            frame = frame.copy()
            frame["scale"] = scale
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    # Same scan keys and query denominators across S/G. Color rankings are
    # identical regardless of which backbone run carried the RGB rows.
    s = raw[raw.scale == "dino_s"]
    g = raw[raw.scale == "dino_g"]
    keys = ["scene", "source_scan", "target_scan"]
    sg = s.merge(g, on=keys, suffixes=("_s", "_g"), validate="one_to_one")
    if len(sg) != 25 or not (sg.queries_s == sg.queries_g).all() or not (sg.target_present_queries_s == sg.target_present_queries_g).all():
        raise ValueError("S/G rows are not scan-count matched over the expected 25 target pairs")
    color_runs = raw[raw.scale == "rgb_color"]
    site = site_metrics(raw)
    if len(site) != 11 or set(site.scene) != {f"scene_{letter}" for letter in "adefghijklm"}:
        raise ValueError(f"expected the 10 non-pilot B-M sites plus A, got {site.scene.tolist()}")

    metrics = {}
    for scale in ("dino_s", "dino_g", "rgb_color"):
        metrics[scale] = {}
        for metric, offset in (("physical", 1), ("semantic", 2), ("gap", 3)):
            metrics[scale][metric] = boot(site[f"{scale}_{metric}"].to_numpy(), offset)
        metrics[scale]["positive_gap_sites"] = int((site[f"{scale}_gap"] > 0).sum())
        metrics[scale]["minimum_site_gap"] = float(site[f"{scale}_gap"].min())
    deltas = {}
    for scale, scale_offset in (("dino_s", 10), ("dino_g", 20)):
        deltas[scale] = {}
        for metric, metric_offset in (("physical", 1), ("semantic", 2)):
            delta = site[f"{scale}_{metric}"] - site[f"rgb_color_{metric}"]
            deltas[scale][metric] = {
                **boot(delta.to_numpy(), scale_offset + metric_offset),
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
            }

    a = site[site.scene == "scene_a"].iloc[0].to_dict()
    result = {
        "status": "complete",
        "site_subset": "10 prior non-pilot B-M spaces (D-M) plus separately inspected scene A; scenes B/C excluded as render pilots",
        "sites": int(len(site)),
        "target_scan_pairs": int(raw[raw.scale == "dino_s"].target_scan.nunique()),
        "target_present_queries": int(raw[raw.scale == "dino_s"].target_present_queries.sum()),
        "statistical_unit": "space cluster; target scans pooled within space, then sites equally weighted",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "labels_used_for_ranking": False,
        "dino_vs_rgb": deltas,
        "site_macro": metrics,
        "scene_a_site_row": a,
        "scene_a_relative_to_rgb_pp": {
            "dino_g_physical": float((a["dino_g_physical"] - a["rgb_color_physical"]) * 100),
            "dino_g_semantic": float((a["dino_g_semantic"] - a["rgb_color_semantic"]) * 100),
        },
        "query_id_caveat": "DINO-S/G per-scan query and target-presence counts match; exact sampled point IDs are not serialized",
        "raw_rgbd_caveat": "colored annotated PLYs rendered into synthetic views; no raw RGB-D or pose evaluation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    site.to_csv(args.output_dir / "site_metrics.csv", index=False)
    scene_a_rows = raw[raw.scene == "scene_a"]
    scene_a_rows.to_csv(args.output_dir / "scene_a_pair_rows.csv", index=False)

    def ci_text(item: dict[str, object]) -> str:
        ci = item["ci95"]
        return f"{item['estimate']*100:.1f}% [{ci[0]*100:.1f},{ci[1]*100:.1f}]"

    lines = [
        "# Expanded Rescan site check: D-M plus scene A", "",
        "This analysis adds one previously absent Rescan space (A) to the 10 B-M non-pilot spaces D-M. B/C remain excluded because they were used for the rendering feasibility pilot. Scene A's labeled PLYs were fetched by byte range after the B-M results existed; this is therefore an exploratory fixed-protocol extension, not a preregistered confirmation. Only scene A's four PLYs were extracted; no raw video was downloaded.", "",
        f"The 11 spaces contribute {result['target_scan_pairs']} target scans and {result['target_present_queries']:,} target-present queries. Site rates pool scans within each space and weight the 11 spaces equally; confidence intervals bootstrap sites.", "",
        "| Descriptor | Physical Recall@20 | Semantic Recall@20 | Semantic minus physical |",
        "|---|---:|---:|---:|",
    ]
    for scale, label in (("dino_s", "DINOv2-S"), ("dino_g", "DINOv2-G"), ("rgb_color", "RGB distance")):
        lines.append(f"| {label} | {ci_text(metrics[scale]['physical'])} | {ci_text(metrics[scale]['semantic'])} | {ci_text(metrics[scale]['gap'])} |")
    lines.extend(["", "| DINO descriptor minus RGB | Physical Recall@20 | Semantic Recall@20 | Sites won |", "|---|---:|---:|---:|"])
    for scale, label in (("dino_s", "DINOv2-S"), ("dino_g", "DINOv2-G")):
        physical, semantic = deltas[scale]["physical"], deltas[scale]["semantic"]
        lines.append(f"| {label} | {physical['estimate']*100:+.1f} pp [{physical['ci95'][0]*100:+.1f},{physical['ci95'][1]*100:+.1f}] | {semantic['estimate']*100:+.1f} pp [{semantic['ci95'][0]*100:+.1f},{semantic['ci95'][1]*100:+.1f}] | {physical['wins']}/11, {semantic['wins']}/11 |")
    lines.extend([
        "",
        f"On the new scene A alone, DINO-G is {result['scene_a_relative_to_rgb_pp']['dino_g_physical']:+.1f} pp below RGB physical recall and {result['scene_a_relative_to_rgb_pp']['dino_g_semantic']:+.1f} pp below RGB semantic recall. The semantic-over-identity gap remains positive there, but this one space is not an independent estimate of across-site variance.",
        "",
        "Semantic Recall@20 exceeds physical-instance Recall@20 in all 11 spaces for DINO-S, DINO-G, and RGB distance. This is a descriptive site-wise consistency check after exploratory analysis, not a preregistered sign test.",
        "",
        "Interpretation: adding A attenuates the apparent DINO-over-color advantage. The G-vs-RGB means remain positive across the 11 space clusters, but one site's failure and the exploratory selection history counsel against a broad descriptor-superiority claim. The semantic/physical gap is more consistent across this check. The protocol still uses synthetic views of labeled point clouds, not raw RGB-D frames, and measures candidate recall only.",
        "",
        "Reproduce the scene-A feature runs with `scripts/probe_rescan_dino_point_candidates.py`; the added PLY inventory is in `outputs/reports/20261006_rescan_scene_a_external/scan_manifest.csv`; reproduce this aggregation with `scripts/aggregate_rescan_expanded_sites.py`.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote 11-site summary to {args.output_dir}")


if __name__ == "__main__":
    main()
