#!/usr/bin/env python3
"""Summarize post-hoc dense MASt3R consensus seed sensitivity on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OVERLAP_TARGETS = {
    "4fbad31e-465b-2a5d-84b7-c0ddea978db4",
    "c7895f7c-339c-2d13-819f-3bb0b26c91f6",
}


def bootstrap_delta(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


def compare(env: pd.DataFrame, seed: int, samples: int, boot_seed: int) -> dict:
    delta = (env[f"seed_{seed}_semantic_coverage"] - env.vitg_semantic_coverage).to_numpy()
    lo, hi = bootstrap_delta(delta, samples, boot_seed)
    return {
        "environments": int(len(env)),
        "mast3r_environment_macro_coverage": float(env[f"seed_{seed}_semantic_coverage"].mean()),
        "vitg_environment_macro_coverage": float(env.vitg_semantic_coverage.mean()),
        "difference": float(delta.mean()),
        "difference_ci95": [lo, hi],
        "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run-dir", type=Path, required=True)
    parser.add_argument("--dense-run-dir", type=Path, default=Path("outputs/reports/20261004_mast3r_dense_official_validation"))
    parser.add_argument("--vitg-relation-csv", type=Path, default=Path("outputs/reports/20261003_vitg_validation/aggregate/per_relation.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    args = parser.parse_args()

    runs = pd.read_csv(args.seed_run_dir / "per_relation_seed.csv")
    original = pd.read_csv(args.dense_run_dir / "aggregate/per_relation.csv")
    manifest = pd.read_csv(args.dense_run_dir / "validation_run_manifest.csv")
    vitg = pd.read_csv(args.vitg_relation_csv)
    vitg = vitg[vitg.arm == "ViT-G"].copy()
    vitg["match_key"] = (
        vitg.environment_id.astype(str) + "|" + vitg.reference_scan.astype(str)
        + "|" + vitg.rescan_scan.astype(str)
    )
    if len(original) != 110 or len(manifest) != 110 or len(vitg) != 110:
        raise ValueError("expected 110 original dense, manifest, and ViT-G relations")
    if len(runs) != 110 * len(args.seeds):
        raise ValueError(f"expected {110 * len(args.seeds)} seed rows, found {len(runs)}")
    if set(runs.ransac_seed.astype(int)) != set(args.seeds):
        raise ValueError("seed set mismatch")
    if runs.duplicated(["pair_run", "ransac_seed"]).any():
        raise ValueError("duplicate relation/seed rows")
    if set(runs.candidate_top_k.astype(int)) != {20} or set(runs.ransac_iterations.astype(int)) != {1500}:
        raise ValueError("candidate budget or iteration count changed")
    if set(runs.linked_physical_instance_queries.astype(int)) != set(original.MASt3R_physical_queries.astype(int)):
        raise ValueError("physical query denominators changed")

    run_to_env = dict(zip(manifest.run_id.astype(str), manifest.environment_id.astype(str)))
    dense = original.set_index("run_id")
    vitg = vitg.set_index("match_key")
    records = []
    for row in runs.itertuples(index=False):
        run_id = str(row.pair_run)
        if run_id not in run_to_env or run_id not in dense.index:
            raise ValueError(f"unknown relation in seed output: {run_id}")
        old = dense.loc[run_id]
        match_key = f"{old.environment_id}|{old.reference_scan}|{old.rescan_scan}"
        if match_key not in vitg.index:
            raise ValueError(f"ViT-G relation missing for {run_id}")
        ref = vitg.loc[match_key]
        if int(row.semantic_evaluable_queries) != int(old.MASt3R_semantic_queries):
            raise ValueError(f"semantic denominator changed: {run_id}")
        if int(row.linked_physical_instance_queries) != int(old.MASt3R_physical_queries):
            raise ValueError(f"physical denominator changed: {run_id}")
        records.append({
            "run_id": run_id,
            "environment_id": run_to_env[run_id],
            "seed": int(row.ransac_seed),
            "semantic_queries": int(row.semantic_evaluable_queries),
            "semantic_correct_inliers": int(row.semantic_correct_inliers),
            "physical_queries": int(row.linked_physical_instance_queries),
            "physical_correct_transfers": int(row.physical_correct_transfers),
            "physical_candidate_recall": float(row.physical_instance_candidate_recall_at_k),
            "strict_pose_success": bool(row.strict_scene_pose_success),
            "rotation_error_deg": float(row.scene_rotation_error_deg),
            "translation_error_m": float(row.scene_translation_error_m),
            "original_semantic_correct_inliers": int(old.MASt3R_semantic_correct_inliers),
            "original_physical_correct_transfers": int(old.MASt3R_physical_correct_inliers),
            "vitg_semantic_correct_inliers": int(ref.semantic_correct_inliers),
            "vitg_semantic_queries": int(ref.queries),
        })
    rel = pd.DataFrame(records)
    if rel.groupby("seed").run_id.nunique().ne(110).any():
        raise ValueError("a seed does not cover all 110 unique relations")
    if rel.groupby("seed").environment_id.nunique().ne(47).any():
        raise ValueError("a seed does not cover all 47 environments")
    seed17 = rel[rel.seed == 17].set_index("run_id")
    for row in seed17.itertuples(index=False):
        if row.semantic_correct_inliers != row.original_semantic_correct_inliers:
            raise ValueError(f"seed-17 semantic result mismatch: {row.run_id}")
        if row.physical_correct_transfers != row.original_physical_correct_transfers:
            raise ValueError(f"seed-17 physical result mismatch: {row.run_id}")

    env_records = []
    for seed in args.seeds:
        subset = rel[rel.seed == seed]
        for env_id, group in subset.groupby("environment_id", sort=True):
            queries = int(group.semantic_queries.sum())
            physical_queries = int(group.physical_queries.sum())
            env_records.append({
                "environment_id": env_id,
                "seed": seed,
                "semantic_queries": queries,
                "semantic_correct_inliers": int(group.semantic_correct_inliers.sum()),
                "semantic_coverage": float(group.semantic_correct_inliers.sum() / queries),
                "physical_queries": physical_queries,
                "physical_correct_transfers": int(group.physical_correct_transfers.sum()),
                "physical_transfer_coverage": float(group.physical_correct_transfers.sum() / physical_queries),
                "strict_pose_successes": int(group.strict_pose_success.sum()),
                "relations": len(group),
            })
    env_long = pd.DataFrame(env_records)
    env_wide = env_long.pivot(index="environment_id", columns="seed")
    env = pd.DataFrame({"environment_id": env_wide.index})
    for seed in args.seeds:
        env[f"seed_{seed}_semantic_coverage"] = env_wide["semantic_coverage"][seed].to_numpy()
        env[f"seed_{seed}_physical_coverage"] = env_wide["physical_transfer_coverage"][seed].to_numpy()
        env[f"seed_{seed}_strict_pose_successes"] = env_wide["strict_pose_successes"][seed].to_numpy()
    vitg_env = original.groupby("environment_id").agg(
        vitg_correct=("ViT-G_semantic_correct_inliers", "sum"),
        vitg_queries=("ViT-G_semantic_queries", "sum"),
    )
    env["vitg_semantic_coverage"] = [
        float(vitg_env.loc[e, "vitg_correct"] / vitg_env.loc[e, "vitg_queries"])
        for e in env.environment_id
    ]
    env["overlap_flagged"] = env.environment_id.isin(OVERLAP_TARGETS)

    comparisons = {}
    summary_by_seed = []
    for seed in args.seeds:
        full = compare(env, seed, args.bootstrap_samples, args.bootstrap_seed + seed)
        clean = compare(env[~env.overlap_flagged].reset_index(drop=True), seed,
                        args.bootstrap_samples, args.bootstrap_seed + seed + 100)
        comparisons[str(seed)] = {"all_environments": full, "excluding_flagged_overlap_targets": clean}
        group = rel[rel.seed == seed]
        nsem = int(group.semantic_queries.sum())
        nphys = int(group.physical_queries.sum())
        summary_by_seed.append({
            "seed": seed,
            "relations": int(group.run_id.nunique()),
            "environments": int(group.environment_id.nunique()),
            "environment_macro_semantic_coverage": full["mast3r_environment_macro_coverage"],
            "environment_macro_semantic_difference_vs_vitg": full["difference"],
            "ci_lower": full["difference_ci95"][0],
            "ci_upper": full["difference_ci95"][1],
            "pooled_physical_transfer_coverage": float(group.physical_correct_transfers.sum() / nphys),
            "environment_macro_physical_transfer_coverage": float(env[f"seed_{seed}_physical_coverage"].mean()),
            "strict_pose_successes": int(group.strict_pose_success.sum()),
            "mean_candidate_recall_weighted": float((group.physical_candidate_recall * group.physical_queries).sum() / nphys),
        })
    by_seed = pd.DataFrame(summary_by_seed)
    summary = {
        "status": "complete",
        "analysis_timing": "post-hoc after observing the complete dense MASt3R validation result",
        "protocol": {
            "relations": 110, "environments": 47, "seeds": args.seeds,
            "fixed_candidates": "saved unique dense MASt3R top-20 caches",
            "threshold_m": 0.1, "iterations": 1500,
            "seed_17_exact_relation_reproduction": True,
        },
        "comparisons": comparisons,
        "caveats": [
            "The sensitivity analysis changes only the RANSAC random seed on one fixed candidate set.",
            "Seeds and relations are not independent replications; the environment bootstrap describes variation among these official environment IDs.",
            "Excluding two flagged environments is post-hoc and does not establish site-level independence.",
            "The dense MASt3R validation procedure itself was exploratory and initiated after the sparse validation result was observed.",
        ],
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out / "per_relation_seed.csv", index=False)
    env.to_csv(out / "per_environment_seed.csv", index=False)
    by_seed.to_csv(out / "summary_by_seed.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Dense MASt3R validation: post-hoc RANSAC seed sensitivity", "",
        "The dense MASt3R validation result was already observed before this sensitivity analysis was initiated. It is exploratory, not an independent replication. Candidate caches, query sets, top-20 budget, 10 cm threshold, and 1,500 iterations are fixed; only the RANSAC seed varies. Seed 17 exactly reproduces all 110 original per-relation semantic and physical transfer counts.", "",
        "## Environment-macro semantic correct-inlier coverage", "",
        "| Seed | MASt3R | ViT-G | Difference (95% environment-bootstrap CI) | Wins/ties/losses | Excluding flagged environments |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in args.seeds:
        full = comparisons[str(seed)]["all_environments"]
        clean = comparisons[str(seed)]["excluding_flagged_overlap_targets"]
        lines.append(
            f"| {seed} | {full['mast3r_environment_macro_coverage']:.1%} | {full['vitg_environment_macro_coverage']:.1%} | "
            f"{full['difference']*100:+.1f} pp [{full['difference_ci95'][0]*100:+.1f}, {full['difference_ci95'][1]*100:+.1f}] | "
            f"{full['wins_ties_losses'][0]}/{full['wins_ties_losses'][1]}/{full['wins_ties_losses'][2]} | "
            f"{clean['difference']*100:+.1f} pp [{clean['difference_ci95'][0]*100:+.1f}, {clean['difference_ci95'][1]*100:+.1f}] ({clean['environments']} env) |"
        )
    lines.extend(["", "## Other outcomes", "", "| Seed | Physical transfer pooled | Physical transfer environment-macro | Strict pose successes | Weighted physical candidate recall |", "|---:|---:|---:|---:|---:|"])
    for row in by_seed.itertuples(index=False):
        lines.append(f"| {row.seed} | {row.pooled_physical_transfer_coverage:.1%} | {row.environment_macro_physical_transfer_coverage:.1%} | {row.strict_pose_successes}/110 | {row.mean_candidate_recall_weighted:.1%} |")
    lines.extend([
        "", "The additional seeds quantify stochastic sensitivity conditional on the same retrieved candidates. They do not create independent model runs, datasets, or sites. The two-environment exclusion remains a post-hoc robustness check.",
        "", "Reproduce the consensus replay with:", "", "```bash",
        "conda run -n worldcache python scripts/evaluate_mast3r_consensus_seeds.py \\",
        "  --cache-root outputs/reports/20261004_mast3r_dense_official_validation/relations \\",
        "  --output-root outputs/runs/20261004_mast3r_dense_validation_seed_sweep \\",
        "  --expected-relations 110 --seeds 17 29 41 --threshold 0.10 --iterations 1500",
        "```",
    ])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": summary, "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
