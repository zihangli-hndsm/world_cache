#!/usr/bin/env python3
"""Aggregate seed-matched ViT-G and dense MASt3R validation sensitivity."""
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
IDENTITY_FIELDS = (
    "query_index", "frame_id", "query_global_id", "query_semantic_id", "query_object_id",
)


def paired_bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    draw = values[indexes].mean(axis=1)
    lower, upper = np.quantile(draw, [0.025, 0.975])
    return float(lower), float(upper)


def read_query_metrics(path: Path) -> tuple[dict[str, int | float], pd.DataFrame]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty query diagnostics: {path}")
    semantic_eval = frame.query_evaluable.astype(bool)
    inlier = frame.consensus_inlier.astype(bool)
    semantic_correct = frame.semantic_correct.astype(bool)
    physical_eval = frame.physical_instance_evaluable.astype(bool)
    physical_hit = frame.physical_instance_candidate_hit.astype(bool)
    physical_correct = frame.physical_instance_correct.astype(bool)
    semantic_correct_inlier = semantic_eval & inlier & semantic_correct
    physical_correct_inlier = physical_eval & inlier & physical_correct
    physical_assigned = physical_eval & inlier
    metrics = {
        "query_points": int(len(frame)),
        "semantic_queries": int(semantic_eval.sum()),
        "semantic_correct_inliers": int(semantic_correct_inlier.sum()),
        "semantic_inliers": int((semantic_eval & inlier).sum()),
        "physical_queries": int(physical_eval.sum()),
        "physical_candidate_hits": int((physical_eval & physical_hit).sum()),
        "physical_correct_transfers": int(physical_correct_inlier.sum()),
        "physical_inliers": int(physical_assigned.sum()),
        "semantic_coverage": float(semantic_correct_inlier.sum() / semantic_eval.sum()) if semantic_eval.any() else float("nan"),
        "physical_candidate_recall": float((physical_eval & physical_hit).sum() / physical_eval.sum()) if physical_eval.any() else float("nan"),
        "physical_transfer_coverage": float(physical_correct_inlier.sum() / physical_eval.sum()) if physical_eval.any() else float("nan"),
        "physical_precision": float(physical_correct_inlier.sum() / physical_assigned.sum()) if physical_assigned.any() else float("nan"),
    }
    return metrics, frame


def compare(env: pd.DataFrame, seed: int, samples: int, boot_seed: int) -> dict:
    delta = (env[f"mast3r_{seed}_semantic_coverage"] - env[f"vitg_{seed}_semantic_coverage"]).to_numpy()
    lo, hi = paired_bootstrap(delta, samples, boot_seed)
    return {
        "environments": int(len(env)),
        "mast3r_environment_macro_coverage": float(env[f"mast3r_{seed}_semantic_coverage"].mean()),
        "vitg_environment_macro_coverage": float(env[f"vitg_{seed}_semantic_coverage"].mean()),
        "difference": float(delta.mean()),
        "difference_ci95": [lo, hi],
        "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-run-dir", type=Path, required=True)
    parser.add_argument("--mast3r-seed-run-dir", type=Path, default=Path("outputs/runs/20261004_mast3r_dense_validation_seed_sweep"))
    parser.add_argument("--vitg-original-dir", type=Path, default=Path("outputs/reports/20261003_vitg_validation"))
    parser.add_argument("--mast3r-original-dir", type=Path, default=Path("outputs/reports/20261004_mast3r_dense_official_validation"))
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    args = parser.parse_args()

    manifest = pd.read_csv(args.pair_manifest)
    vitg_run = pd.read_csv(args.seed_run_dir / "seed_sweep_manifest.csv")
    mast_runs = pd.read_csv(args.mast3r_seed_run_dir / "per_relation_seed.csv")
    if len(manifest) != 110 or manifest.environment_id.nunique() != 47:
        raise ValueError("expected the fixed 110-relation / 47-environment manifest")
    manifest = manifest.copy()
    manifest["run_id"] = [
        f"{i:03d}_{str(env)[:8]}_{str(scan)[:8]}"
        for i, (env, scan) in enumerate(zip(manifest.environment_id, manifest.rescan_scan))
    ]
    run_id_by_key = {
        (str(row.environment_id), str(row.reference_scan), str(row.rescan_scan)): str(row.run_id)
        for row in manifest.itertuples(index=False)
    }
    if len(vitg_run) != 110 or vitg_run.run_id.nunique() != 110:
        raise ValueError("ViT-G seed sweep does not contain 110 unique relations")
    if len(mast_runs) != 110 * len(args.seeds):
        raise ValueError("MASt3R seed run is incomplete")
    if set(mast_runs.ransac_seed.astype(int)) != set(args.seeds):
        raise ValueError("MASt3R seed set mismatch")

    all_rows: list[dict[str, object]] = []
    original_vitg = args.vitg_original_dir / "aggregate/per_relation.csv"
    original_vitg_frame = pd.read_csv(original_vitg)
    original_vitg_frame = original_vitg_frame[original_vitg_frame.arm == "ViT-G"].copy()
    original_vitg_frame["run_id"] = [
        run_id_by_key[(str(env), str(ref), str(scan))]
        for env, ref, scan in zip(original_vitg_frame.environment_id, original_vitg_frame.reference_scan, original_vitg_frame.rescan_scan)
    ]
    original_vitg_frame = original_vitg_frame.set_index("run_id")
    original_mast = pd.read_csv(args.mast3r_original_dir / "aggregate/per_relation.csv").set_index("run_id")
    pair_by_run = manifest.set_index("run_id")

    for row in vitg_run.itertuples(index=False):
        run_id = str(row.run_id)
        if run_id not in pair_by_run.index or run_id not in original_vitg_frame.index or run_id not in original_mast.index:
            raise ValueError(f"unexpected run ID: {run_id}")
        pair = pair_by_run.loc[run_id]
        vitg_root = Path(row.output_dir) / "vitg"
        seed_summary = pd.read_csv(vitg_root / "summary.csv")
        if set(seed_summary.ransac_seed.astype(int)) != set(args.seeds) or len(seed_summary) != len(args.seeds):
            raise ValueError(f"incomplete ViT-G seed summary: {run_id}")
        seed_diag = pd.read_csv(vitg_root / "query_diagnostics.csv")
        if not set(seed_diag.ransac_seed.astype(int)) == set(args.seeds):
            raise ValueError(f"incomplete ViT-G query diagnostics: {run_id}")
        base_path = args.vitg_original_dir / "relations" / run_id / "vitg/query_diagnostics.csv"
        base = pd.read_csv(base_path)
        old = original_vitg_frame.loc[run_id]
        old_mast = original_mast.loc[run_id]
        if len(base) != len(seed_diag[seed_diag.ransac_seed == 17]):
            raise ValueError(f"seed-17 query count changed: {run_id}")
        seed17_diag = seed_diag[seed_diag.ransac_seed == 17].reset_index(drop=True)
        base = base.reset_index(drop=True)
        reproduction_fields = IDENTITY_FIELDS + (
            "query_evaluable", "semantic_correct", "consensus_inlier",
            "selected_reference_global_id", "selected_reference_object_id",
            "physical_instance_evaluable", "physical_instance_candidate_hit",
            "physical_instance_correct",
        )
        for field in reproduction_fields:
            if field not in base or field not in seed17_diag:
                raise ValueError(f"missing identity/evaluation field {field}: {run_id}")
            if not np.array_equal(base[field].to_numpy(), seed17_diag[field].to_numpy()):
                raise ValueError(f"seed-17 query/label identity changed for {run_id}: {field}")
        base_metrics, _ = read_query_metrics(base_path)
        if base_metrics["semantic_correct_inliers"] != int(old.semantic_correct_inliers):
            raise ValueError(f"saved original ViT-G semantic aggregate mismatch: {run_id}")
        if base_metrics["physical_correct_transfers"] != int(old.physical_correct_inliers):
            raise ValueError(f"saved original ViT-G physical aggregate mismatch: {run_id}")

        for seed in args.seeds:
            group = seed_diag[seed_diag.ransac_seed == seed].reset_index(drop=True)
            metrics = read_query_metrics_from_frame(group)
            mast_row = mast_runs[(mast_runs.pair_run == run_id) & (mast_runs.ransac_seed == seed)]
            if len(mast_row) != 1:
                raise ValueError(f"MASt3R relation/seed row missing: {run_id}, {seed}")
            mast_row = mast_row.iloc[0]
            # MASt3R and ViT-G must expose identical query and evaluation populations.
            if int(metrics["semantic_queries"]) != int(mast_row.semantic_evaluable_queries):
                raise ValueError(f"semantic query denominator mismatch: {run_id}, {seed}")
            if int(metrics["physical_queries"]) != int(mast_row.linked_physical_instance_queries):
                raise ValueError(f"physical query denominator mismatch: {run_id}, {seed}")
            mast_diag_path = args.mast3r_seed_run_dir / run_id / f"seed_{seed}/query_diagnostics.csv"
            mast_diag = pd.read_csv(mast_diag_path)
            mast_phys_eval = mast_diag.physical_instance_evaluable.astype(bool)
            mast_phys_hit = mast_diag.physical_instance_candidate_hit.astype(bool)
            mast_phys_inlier = mast_diag.consensus_inlier.astype(bool)
            mast_physical_hits = int((mast_phys_eval & mast_phys_hit).sum())
            mast_physical_inliers = int((mast_phys_eval & mast_phys_inlier).sum())
            if mast_physical_hits != int(round(float(mast_row.physical_instance_candidate_recall_at_k) * int(mast_row.linked_physical_instance_queries))):
                raise ValueError(f"MASt3R candidate hit count mismatch: {run_id}, {seed}")
            summary_row = seed_summary[seed_summary.ransac_seed == seed].iloc[0]
            all_rows.append({
                "run_id": run_id,
                "environment_id": str(pair.environment_id),
                "reference_scan": str(pair.reference_scan),
                "rescan_scan": str(pair.rescan_scan),
                "seed": int(seed),
                **{f"vitg_{key}": value for key, value in metrics.items()},
                "vitg_rotation_error_deg": float(summary_row.rotation_error_deg),
                "vitg_translation_error_m": float(summary_row.translation_error_m),
                "vitg_strict_pose_success": bool(float(summary_row.rotation_error_deg) < 5 and float(summary_row.translation_error_m) < 0.25),
                "mast3r_semantic_queries": int(mast_row.semantic_evaluable_queries),
                "mast3r_semantic_correct_inliers": int(mast_row.semantic_correct_inliers),
                "mast3r_physical_queries": int(mast_row.linked_physical_instance_queries),
                "mast3r_physical_candidate_hits": mast_physical_hits,
                "mast3r_physical_correct_transfers": int(mast_row.physical_correct_transfers),
                "mast3r_physical_inliers": mast_physical_inliers,
                "mast3r_semantic_coverage": float(mast_row.semantic_correct_inliers / mast_row.semantic_evaluable_queries),
                "mast3r_physical_candidate_recall": float(mast_row.physical_instance_candidate_recall_at_k),
                "mast3r_physical_transfer_coverage": float(mast_row.physical_correct_transfers / mast_row.linked_physical_instance_queries),
                "mast3r_physical_precision": float(mast_row.physical_assigned_precision),
                "mast3r_rotation_error_deg": float(mast_row.scene_rotation_error_deg),
                "mast3r_translation_error_m": float(mast_row.scene_translation_error_m),
                "mast3r_strict_pose_success": bool(mast_row.strict_scene_pose_success),
                "original_vitg_semantic_correct_inliers": int(old.semantic_correct_inliers),
                "original_vitg_physical_correct_transfers": int(old.physical_correct_inliers),
                "original_mast3r_semantic_correct_inliers": int(old_mast.MASt3R_semantic_correct_inliers),
                "original_mast3r_physical_correct_transfers": int(old_mast.MASt3R_physical_correct_inliers),
            })

    rel = pd.DataFrame(all_rows)
    if len(rel) != 110 * len(args.seeds):
        raise ValueError("not all relation/seed combinations were aggregated")
    seed17 = rel[rel.seed == 17]
    for row in seed17.itertuples(index=False):
        if row.vitg_semantic_correct_inliers != row.original_vitg_semantic_correct_inliers:
            raise ValueError(f"ViT-G seed-17 semantic count mismatch: {row.run_id}")
        if row.vitg_physical_correct_transfers != row.original_vitg_physical_correct_transfers:
            raise ValueError(f"ViT-G seed-17 physical transfer mismatch: {row.run_id}")
        if row.mast3r_semantic_correct_inliers != row.original_mast3r_semantic_correct_inliers:
            raise ValueError(f"MASt3R seed-17 semantic count mismatch: {row.run_id}")
        if row.mast3r_physical_correct_transfers != row.original_mast3r_physical_correct_transfers:
            raise ValueError(f"MASt3R seed-17 physical transfer mismatch: {row.run_id}")

    env_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        subset = rel[rel.seed == seed]
        for env_id, group in subset.groupby("environment_id", sort=True):
            sem_n = int(group.vitg_semantic_queries.sum())
            phys_n = int(group.vitg_physical_queries.sum())
            env_rows.append({
                "environment_id": env_id,
                "seed": seed,
                "relations": int(group.run_id.nunique()),
                "vitg_semantic_queries": sem_n,
                "vitg_semantic_correct_inliers": int(group.vitg_semantic_correct_inliers.sum()),
                "vitg_semantic_coverage": float(group.vitg_semantic_correct_inliers.sum() / sem_n),
                "mast3r_semantic_queries": int(group.mast3r_semantic_queries.sum()),
                "mast3r_semantic_correct_inliers": int(group.mast3r_semantic_correct_inliers.sum()),
                "mast3r_semantic_coverage": float(group.mast3r_semantic_correct_inliers.sum() / group.mast3r_semantic_queries.sum()),
                "vitg_physical_queries": phys_n,
                "vitg_physical_candidate_hits": int(group.vitg_physical_candidate_hits.sum()),
                "vitg_physical_correct_transfers": int(group.vitg_physical_correct_transfers.sum()),
                "vitg_physical_inliers": int(group.vitg_physical_inliers.sum()),
                "mast3r_physical_queries": int(group.mast3r_physical_queries.sum()),
                "mast3r_physical_candidate_hits": int(group.mast3r_physical_candidate_hits.sum()),
                "mast3r_physical_correct_transfers": int(group.mast3r_physical_correct_transfers.sum()),
                "mast3r_physical_inliers": int(group.mast3r_physical_inliers.sum()),
                "vitg_strict_pose_successes": int(group.vitg_strict_pose_success.sum()),
                "mast3r_strict_pose_successes": int(group.mast3r_strict_pose_success.sum()),
                "overlap_flagged": env_id in OVERLAP_TARGETS,
            })
    env_long = pd.DataFrame(env_rows)
    env = env_long.pivot(index="environment_id", columns="seed")
    wide = pd.DataFrame({"environment_id": env.index})
    for seed in args.seeds:
        for method in ("vitg", "mast3r"):
            for metric in ("semantic_coverage",):
                wide[f"{method}_{seed}_{metric}"] = env[f"{method}_{metric}"][seed].to_numpy()
    wide["overlap_flagged"] = wide.environment_id.isin(OVERLAP_TARGETS)

    comparisons = {}
    seed_summaries = []
    for seed in args.seeds:
        all_env = compare(wide, seed, args.bootstrap_samples, args.bootstrap_seed + seed)
        clean_env = compare(wide[~wide.overlap_flagged].reset_index(drop=True), seed,
                            args.bootstrap_samples, args.bootstrap_seed + seed + 100)
        comparisons[str(seed)] = {
            "all_environments": all_env,
            "excluding_flagged_overlap_targets": clean_env,
        }
        group = rel[rel.seed == seed]
        sem_n = int(group.vitg_semantic_queries.sum())
        phys_n = int(group.vitg_physical_queries.sum())
        seed_summaries.append({
            "seed": seed,
            "relations": int(group.run_id.nunique()),
            "environments": int(group.environment_id.nunique()),
            "vitg_environment_macro_semantic_coverage": float(wide[f"vitg_{seed}_semantic_coverage"].mean()),
            "mast3r_environment_macro_semantic_coverage": float(wide[f"mast3r_{seed}_semantic_coverage"].mean()),
            "difference": all_env["difference"],
            "ci_lower": all_env["difference_ci95"][0],
            "ci_upper": all_env["difference_ci95"][1],
            "pooled_vitg_physical_candidate_recall": float(group.vitg_physical_candidate_hits.sum() / phys_n),
            "pooled_mast3r_physical_candidate_recall": float(group.mast3r_physical_candidate_hits.sum() / phys_n),
            "pooled_vitg_physical_transfer_coverage": float(group.vitg_physical_correct_transfers.sum() / phys_n),
            "pooled_mast3r_physical_transfer_coverage": float(group.mast3r_physical_correct_transfers.sum() / phys_n),
            "vitg_strict_pose_successes": int(group.vitg_strict_pose_success.sum()),
            "mast3r_strict_pose_successes": int(group.mast3r_strict_pose_success.sum()),
        })
    by_seed = pd.DataFrame(seed_summaries)
    summary = {
        "status": "complete",
        "analysis_timing": "post-hoc after observing dense MASt3R validation and sensitivity results",
        "protocol": {
            "relations": 110, "environments": 47, "seeds": args.seeds,
            "candidate_budget": "top-20 for each method",
            "threshold_m": 0.10, "iterations": 1500,
            "vitg_seed_17_exact_original_reproduction": True,
            "mast3r_seed_17_exact_original_reproduction": True,
            "query_identity_and_denominator_checks": "passed for every relation and seed",
        },
        "comparisons": comparisons,
        "caveats": [
            "The comparison is between complete descriptor-plus-retrieval-plus-consensus procedures, not descriptor quality alone.",
            "Dense MASt3R validation and this seed-matched follow-up were post-hoc; seed runs are sensitivity settings, not independent replications.",
            "The environment bootstrap does not establish independent physical sites; the split-content audit remains a limitation.",
            "Each method has a different descriptor ranking and candidate set, even though both use a top-20 cap and the same query population.",
        ],
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out / "per_relation_seed_matched.csv", index=False)
    env_long.to_csv(out / "per_environment_seed_matched.csv", index=False)
    wide.to_csv(out / "per_environment_seed_matched_wide.csv", index=False)
    by_seed.to_csv(out / "summary_by_seed.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Seed-matched dense MASt3R / ViT-G validation comparison", "",
        "This is a post-hoc sensitivity analysis after the dense MASt3R validation result was observed. ViT-G was rerun on all 110 validation relations because its original top-20 candidates were not saved. At each seed, both complete procedures use the same query population, relation set, top-20 candidate cap, 10 cm consensus threshold, and 1,500 iterations; their ranked candidate sets differ. Seed 17 exactly reproduces the existing ViT-G and MASt3R relation-level semantic and physical-transfer counts.", "",
        "## Seed-matched semantic coverage", "",
        "| Seed | Dense MASt3R macro | ViT-G macro | Difference (95% environment-bootstrap CI) | Wins/ties/losses | Excluding flagged environments |",
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
    lines.extend(["", "## Physical transfer and pose", "", "| Seed | Dense/ViT-G candidate Recall@20 | Dense/ViT-G physical transfer | Dense/ViT-G strict pose successes |", "|---:|---:|---:|---:|"])
    for row in by_seed.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.pooled_mast3r_physical_candidate_recall:.1%} / {row.pooled_vitg_physical_candidate_recall:.1%} | "
            f"{row.pooled_mast3r_physical_transfer_coverage:.1%} / {row.pooled_vitg_physical_transfer_coverage:.1%} | "
            f"{row.mast3r_strict_pose_successes}/110 / {row.vitg_strict_pose_successes}/110 |"
        )
    lines.extend([
        "", "This is a top-20 procedure-level comparison. Seed-matching does not make the descriptor rankings or candidate sets identical, and the shared validation environments are not independent-site replications.",
        "", "Reproduction starts with `scripts/run_vitg_validation_seed_sensitivity.py`, followed by `scripts/aggregate_vitg_validation_seed_sensitivity.py`.",
    ])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": summary, "output_dir": str(out)}, indent=2))


def read_query_metrics_from_frame(frame: pd.DataFrame) -> dict[str, int | float]:
    semantic_eval = frame.query_evaluable.astype(bool)
    inlier = frame.consensus_inlier.astype(bool)
    semantic_correct = frame.semantic_correct.astype(bool)
    physical_eval = frame.physical_instance_evaluable.astype(bool)
    physical_hit = frame.physical_instance_candidate_hit.astype(bool)
    physical_correct = frame.physical_instance_correct.astype(bool)
    sem_correct_inliers = semantic_eval & inlier & semantic_correct
    phys_correct_inliers = physical_eval & inlier & physical_correct
    phys_inliers = physical_eval & inlier
    return {
        "query_points": int(len(frame)),
        "semantic_queries": int(semantic_eval.sum()),
        "semantic_correct_inliers": int(sem_correct_inliers.sum()),
        "semantic_inliers": int((semantic_eval & inlier).sum()),
        "physical_queries": int(physical_eval.sum()),
        "physical_candidate_hits": int((physical_eval & physical_hit).sum()),
        "physical_correct_transfers": int(phys_correct_inliers.sum()),
        "physical_inliers": int(phys_inliers.sum()),
        "semantic_coverage": float(sem_correct_inliers.sum() / semantic_eval.sum()) if semantic_eval.any() else float("nan"),
        "physical_candidate_recall": float((physical_eval & physical_hit).sum() / physical_eval.sum()) if physical_eval.any() else float("nan"),
        "physical_transfer_coverage": float(phys_correct_inliers.sum() / physical_eval.sum()) if physical_eval.any() else float("nan"),
        "physical_precision": float(phys_correct_inliers.sum() / phys_inliers.sum()) if phys_inliers.any() else float("nan"),
    }


if __name__ == "__main__":
    main()
