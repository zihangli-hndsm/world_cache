#!/usr/bin/env python3
"""Compare matched Rescan DINO-S and DINO-G point-retrieval audits by scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20261019


def bootstrap(values: np.ndarray, seed: int) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    draws = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "clusters": int(len(values)),
    }


def scene_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scene, group in frame.groupby("scene", sort=True):
        denominator = int(group.target_present_queries_s.sum())
        row: dict[str, object] = {"scene": scene, "target_scans": int(group.target_scan.nunique())}
        for scale in ("s", "g"):
            row[f"physical_{scale}"] = float(group[f"physical_correct_{scale}"].sum() / denominator)
            row[f"semantic_{scale}"] = float(group[f"semantic_correct_present_{scale}"].sum() / denominator)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, label: str) -> dict[str, object]:
    sites = scene_rates(frame)
    result: dict[str, object] = {
        "subset": label,
        "scenes": int(len(sites)),
        "target_scan_pairs": int(frame.target_scan.nunique()),
        "target_present_queries": int(frame.target_present_queries_s.sum()),
        "scene_macro": {},
    }
    for metric in ("physical", "semantic"):
        delta = (sites[f"{metric}_g"] - sites[f"{metric}_s"]).to_numpy()
        result["scene_macro"][metric] = {
            "dino_s": bootstrap(sites[f"{metric}_s"].to_numpy(), BOOTSTRAP_SEED + len(sites)),
            "dino_g": bootstrap(sites[f"{metric}_g"].to_numpy(), BOOTSTRAP_SEED + 10 + len(sites)),
            "dino_g_minus_s": bootstrap(delta, BOOTSTRAP_SEED + 20 + len(sites)),
            "dino_g_wins": int((delta > 0).sum()),
            "dino_s_wins": int((delta < 0).sum()),
        }
    result["site_rows"] = sites.to_dict(orient="records")
    return result


def fmt(metric: dict[str, object]) -> str:
    ci = metric["ci95"]
    return f"{metric['estimate'] * 100:+.1f} pp [{ci[0] * 100:+.1f},{ci[1] * 100:+.1f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, default=Path("outputs/reports/20261006_rescan_dino_external_full/pair_metrics.csv"))
    parser.add_argument("--giant", type=Path, default=Path("outputs/reports/20261006_rescan_dinog_external_full/pair_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_rescan_dinog_external_full"))
    args = parser.parse_args()

    small = pd.read_csv(args.small)
    giant = pd.read_csv(args.giant)
    small = small[small.k == 20].copy()
    giant = giant[giant.k == 20].copy()
    keys = ["scene", "source_scan", "target_scan", "method"]
    merged = small.merge(giant, on=keys, suffixes=("_s", "_g"), validate="one_to_one")
    if len(merged) != 58 or merged.target_scan.nunique() != 29:
        raise ValueError(f"expected 58 method rows / 29 target scans; got {len(merged)} / {merged.target_scan.nunique()}")
    for column in ("queries", "target_present_queries"):
        if not (merged[f"{column}_s"] == merged[f"{column}_g"]).all():
            raise ValueError(f"per-pair DINO-S/G {column} differs")

    dino = merged[merged.method == "dino_synthetic_views"].copy()
    all_result = summarize(dino, "all_12_rescan_scenes")
    nonpilot = dino[~dino.scene.isin(["scene_b", "scene_c"])].copy()
    heldout_result = summarize(nonpilot, "10_scenes_excluding_render_pilot_b_c")
    output = {
        "status": "complete",
        "small_input": str(args.small),
        "giant_input": str(args.giant),
        "pairing": "same scene/scan/method keys; per-pair query and target-present counts match; scene-cluster bootstrap; exact query IDs were not serialized",
        "statistical_unit": "Rescan scene; target-scan counts are pooled within each scene before macro averaging/bootstrap",
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "all_scenes": all_result,
        "nonpilot_scenes": heldout_result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scale_comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.DataFrame(all_result["site_rows"]).to_csv(args.output_dir / "scale_comparison_all_sites.csv", index=False)
    pd.DataFrame(heldout_result["site_rows"]).to_csv(args.output_dir / "scale_comparison_nonpilot_sites.csv", index=False)

    lines = [
        "# Rescan DINOv2 scale comparison", "",
        "The DINO-S and DINO-G probes use the same scenes, scan pairs, candidate sampling, and deterministic query-sampling rule. Per-pair query and target-present counts match for all 29 target scans. Query point IDs were not serialized, so exact query-ID equality is not independently verified. Scene-level confidence intervals bootstrap spaces, not scans or points.", "",
        "| Scene subset | DINO-S physical | DINO-G physical | G minus S physical | DINO-S semantic | DINO-G semantic | G minus S semantic |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in (all_result, heldout_result):
        p, s = result["scene_macro"]["physical"], result["scene_macro"]["semantic"]
        lines.append(f"| {result['subset']} | {p['dino_s']['estimate']*100:.1f}% | {p['dino_g']['estimate']*100:.1f}% | {fmt(p['dino_g_minus_s'])} | {s['dino_s']['estimate']*100:.1f}% | {s['dino_g']['estimate']*100:.1f}% | {fmt(s['dino_g_minus_s'])} |")
    lines += ["", "G wins S in the following number of scene clusters on the 10 non-pilot spaces:"]
    for metric in ("physical", "semantic"):
        value = heldout_result["scene_macro"][metric]
        lines.append(f"- {metric}: {value['dino_g_wins']}/10 (S wins {value['dino_s_wins']}/10); G-minus-S {fmt(value['dino_g_minus_s'])}.")
    lines += ["", "This comparison tests backbone scale only within the synthetic point-cloud rendering protocol; it does not validate raw RGB-D transfer or pose accuracy."]
    (args.output_dir / "scale_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote scale comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
