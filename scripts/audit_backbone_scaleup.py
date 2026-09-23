"""Paired scene-bootstrap audit for the ViT-L to ViT-G scale comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def weighted(frame: pd.DataFrame, correct_column: str = "inlier_semantic_accuracy") -> float:
    return float((frame.inliers * frame[correct_column]).sum() / frame.inliers.sum())


def bootstrap_delta(
    vitl: pd.DataFrame,
    vitg: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, object]:
    if list(vitl.pair) != list(vitg.pair):
        raise ValueError("paired scale audit requires identical pair ordering")
    n_pairs = len(vitl)
    if n_pairs == 0:
        raise ValueError("paired scale audit requires at least one pair")
    l_inliers = vitl.inliers.to_numpy(float)
    g_inliers = vitg.inliers.to_numpy(float)
    l_semantic = vitl.inlier_semantic_accuracy.to_numpy(float)
    g_semantic = vitg.inlier_semantic_accuracy.to_numpy(float)
    l_coverage = vitl.inlier_coverage.to_numpy(float)
    g_coverage = vitg.inlier_coverage.to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_pairs, size=(samples, n_pairs))
    l_boot = (l_inliers[indices] * l_semantic[indices]).sum(axis=1) / l_inliers[indices].sum(axis=1)
    g_boot = (g_inliers[indices] * g_semantic[indices]).sum(axis=1) / g_inliers[indices].sum(axis=1)
    delta_boot = g_boot - l_boot
    l_cov_boot = l_coverage[indices].mean(axis=1)
    g_cov_boot = g_coverage[indices].mean(axis=1)
    semantic_delta = weighted(vitg) - weighted(vitl)
    coverage_delta = float(g_coverage.mean() - l_coverage.mean())
    pair_delta = g_semantic - l_semantic
    return {
        "pairs": int(n_pairs),
        "pair_names": list(vitl.pair),
        "vitl_weighted_semantic": weighted(vitl),
        "vitg_weighted_semantic": weighted(vitg),
        "semantic_delta": semantic_delta,
        "semantic_delta_ci95_lower": float(np.quantile(delta_boot, 0.025)),
        "semantic_delta_ci95_upper": float(np.quantile(delta_boot, 0.975)),
        "vitl_macro_coverage": float(l_coverage.mean()),
        "vitg_macro_coverage": float(g_coverage.mean()),
        "coverage_delta": coverage_delta,
        "coverage_delta_ci95_lower": float(np.quantile(g_cov_boot - l_cov_boot, 0.025)),
        "coverage_delta_ci95_upper": float(np.quantile(g_cov_boot - l_cov_boot, 0.975)),
        "semantic_pair_positive": int((pair_delta > 0).sum()),
        "semantic_pair_negative": int((pair_delta < 0).sum()),
        "semantic_pair_ties": int((pair_delta == 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitl", type=Path, required=True)
    parser.add_argument("--vitg", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vitl_all = pd.read_csv(args.vitl).sort_values("pair").reset_index(drop=True)
    vitg_all = pd.read_csv(args.vitg).sort_values("pair").reset_index(drop=True)
    common = sorted(set(vitl_all.pair) & set(vitg_all.pair))
    vitl_all = vitl_all[vitl_all.pair.isin(common)].sort_values("pair").reset_index(drop=True)
    vitg_all = vitg_all[vitg_all.pair.isin(common)].sort_values("pair").reset_index(drop=True)
    rows = []
    for split, mask in (
        ("matched_14_including_target", np.ones(len(common), dtype=bool)),
        ("matched_13_source_only", ~vitl_all.pair.astype(str).str.contains("target").to_numpy()),
    ):
        rows.append({"split": split, **bootstrap_delta(
            vitl_all[mask].reset_index(drop=True),
            vitg_all[mask].reset_index(drop=True),
            args.samples,
            seed=17 if split.endswith("target") else 31,
        )})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    per_pair = vitl_all[["pair", "queries", "inliers", "inlier_coverage", "inlier_semantic_accuracy"]].rename(
        columns={
            "inliers": "vitl_inliers",
            "inlier_coverage": "vitl_coverage",
            "inlier_semantic_accuracy": "vitl_semantic",
        }
    )
    per_pair = per_pair.merge(
        vitg_all[["pair", "inliers", "inlier_coverage", "inlier_semantic_accuracy"]].rename(
            columns={
                "inliers": "vitg_inliers",
                "inlier_coverage": "vitg_coverage",
                "inlier_semantic_accuracy": "vitg_semantic",
            }
        ),
        on="pair",
    )
    per_pair["semantic_delta"] = per_pair.vitg_semantic - per_pair.vitl_semantic
    per_pair["coverage_delta"] = per_pair.vitg_coverage - per_pair.vitl_coverage
    per_pair.to_csv(args.output_dir / "per_pair.csv", index=False)
    status = {
        "status": "complete",
        "protocol": {
            "bootstrap_unit": "pair",
            "samples": args.samples,
            "seeds": {"matched_14_including_target": 17, "matched_13_source_only": 31},
            "target_labels_used_for_selection": False,
            "known_alignment_used_for_selection": False,
        },
        "vitl": str(args.vitl),
        "vitg": str(args.vitg),
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
