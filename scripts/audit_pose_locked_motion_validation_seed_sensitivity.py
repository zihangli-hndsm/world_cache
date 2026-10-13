#!/usr/bin/env python3
"""Seed sensitivity for the frozen pose-locked motion rule on 3RScan val.

Reuses only the saved top-20 ViT-G geometry caches and annotation/label fields
from the completed fixed-split audit. The train-selected mode, support, and
switch thresholds remain unchanged. Seed 17 must exactly reproduce the saved
validation output before additional seeds are accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_3rscan_instance_correspondence import read_annotated_mesh
from evaluate_3rscan_multimotion import assign_modes, fit_fixed_consensus, fit_modes, load_feature_cache
from threadpoolctl import threadpool_limits


DEFAULT_ROOT = Path("outputs/reports/20261006_vitg_pose_locked_motion_validation")
DEFAULT_OUTPUT = Path("outputs/reports/20261004_pose_locked_motion_seed_sensitivity")
SELECTION = {"max_modes": 4, "minimum_independent_support": 32, "switch_margin_m": 0.15}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def fit_seed(features, seed: int, min_independent_support: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    limits = np.full(len(features.query_points), 20, dtype=np.int64)
    baseline = fit_fixed_consensus(
        features.query_points, features.reference_points,
        features.candidate_indices[:, :20], threshold=0.10,
        iterations=1500, seed=seed,
    )
    modes = fit_modes(
        features, limits, threshold=0.10, iterations=1500,
        secondary_iterations=1000, seed=seed, max_modes=4,
        min_support=8, fixed_baseline=baseline,
    )
    motion = assign_modes(features, limits, modes, threshold=0.10)
    mode_selected_matrix, mode_residual_matrix = motion[4], motion[5]

    independent_support: list[int] = []
    unexplained = baseline.residuals > 0.10
    for mode_index in range(1, len(modes)):
        inlier = mode_residual_matrix[:, mode_index] <= 0.10
        independent_support.append(int((inlier & unexplained).sum()))
        unexplained &= ~inlier
    valid_modes = [
        i for i, support in enumerate(independent_support, start=1)
        if i < 4 and support >= min_independent_support
    ]
    selected = baseline.selected.copy()
    assigned = baseline.inliers.copy()
    if valid_modes:
        residuals = mode_residual_matrix[:, valid_modes]
        rows = np.arange(len(features.query_points))
        best_local = np.argmin(residuals, axis=1)
        best_residual = residuals[rows, best_local]
        best_mode = np.asarray(valid_modes, dtype=np.int64)[best_local]
        switch = (
            np.isfinite(best_residual)
            & (best_residual <= 0.10)
            & ((baseline.residuals > 0.10)
               | ((baseline.residuals - best_residual) >= SELECTION["switch_margin_m"]))
        )
        selected[switch] = mode_selected_matrix[rows[switch], best_mode[switch]]
        assigned[switch] = True
    return baseline.selected, baseline.inliers, selected, assigned


def bootstrap_delta(rows: pd.DataFrame, metric: str, samples: int, seed: int) -> dict[str, object]:
    # First pool all relations from each environment, then resample environments.
    env = rows.groupby("environment_id", sort=True).agg(
        moved=("moved_queries", "sum"),
        base_correct=("baseline_correct", "sum"),
        method_correct=("method_correct", "sum"),
        base_assigned=("baseline_assigned", "sum"),
        method_assigned=("method_assigned", "sum"),
    )
    numerator_denominator = {
        "coverage": ("method_correct", "base_correct", "moved"),
        "precision": ("method_correct", "base_correct", "method_assigned", "base_assigned"),
    }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(env), size=(samples, len(env)))
    values = env.to_numpy(dtype=np.float64)
    if metric == "coverage":
        moved = values[:, 0]
        base_correct, method_correct = values[:, 1], values[:, 2]
        delta_by_env = np.divide(method_correct, moved, out=np.full_like(moved, np.nan), where=moved > 0) - np.divide(base_correct, moved, out=np.full_like(moved, np.nan), where=moved > 0)
        macro = float(np.nanmean(delta_by_env))
        draw = np.nanmean(delta_by_env[idx], axis=1)
    else:
        method_correct, base_correct = values[:, 2], values[:, 1]
        method_assigned, base_assigned = values[:, 4], values[:, 3]
        delta_by_env = np.divide(method_correct, method_assigned, out=np.full_like(method_correct, np.nan), where=method_assigned > 0) - np.divide(base_correct, base_assigned, out=np.full_like(base_correct, np.nan), where=base_assigned > 0)
        macro = float(np.nanmean(delta_by_env))
        draw = np.nanmean(delta_by_env[idx], axis=1)
    return {"estimate": macro, "ci95": [float(x) for x in np.nanquantile(draw, [0.025, 0.975])], "eligible_environments": int(np.isfinite(delta_by_env).sum())}


def process_relation_task(task: tuple) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    relation_index, pair, root, input_dir, annotation_root, seeds = task
    root, input_dir, annotation_root = Path(root), Path(input_dir), Path(annotation_root)
    run_id = f"{relation_index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
    relation_dir = input_dir / "relations" / run_id
    saved = pd.read_csv(relation_dir / "query_diagnostics.csv")
    saved_base = saved[saved.arm == "global_ransac"].sort_values("query_index").reset_index(drop=True)
    saved_method = saved[saved.arm == "combined"].sort_values("query_index").reset_index(drop=True)
    if not np.array_equal(saved_base.query_index, saved_method.query_index):
        raise ValueError(f"saved arm query mismatch: {run_id}")
    cache_path = input_dir / "descriptor_candidates" / f"{run_id}.npz"
    feature = load_feature_cache(
        cache_path, run_id, pair["reference_scan"], pair["rescan_scan"],
        relation_dir, "giant", (2, 20, 40), 224,
    )
    if feature is None or len(feature.query_points) != len(saved_base):
        raise ValueError(f"feature cache missing or misaligned: {run_id}")
    tree, reference_object_ids, _, _, _ = read_annotated_mesh(annotation_root, pair["reference_scan"])
    _, nearest = tree.query(feature.reference_points.astype(np.float32), k=1)
    reference_patch_ids = reference_object_ids[nearest]
    moved = saved_base.moved_rigid.astype(bool).to_numpy()
    expected_id = saved_base.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
    baseline_rows = saved_base.baseline_correct.astype(bool).to_numpy()
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            base_selected, base_assigned, method_selected, method_assigned = fit_seed(feature, seed)
            base_correct = moved & base_assigned & (reference_patch_ids[base_selected] == expected_id)
            method_correct = moved & method_assigned & (reference_patch_ids[method_selected] == expected_id)
            if seed == 17:
                saved_selected = saved_method.selected_reference_index.to_numpy(dtype=np.int64)
                check = {
                    "run_id": run_id,
                    "selected_match": bool(np.array_equal(method_selected, saved_selected)),
                    "assigned_match": bool(np.array_equal(method_assigned, saved_method.consensus_inlier.astype(bool).to_numpy())),
                    "correct_match": bool(np.array_equal(method_correct, saved_method.method_correct.astype(bool).to_numpy())),
                    "baseline_match": bool(np.array_equal(base_correct, baseline_rows)),
                }
                checks.append(check)
                if not all(v for k, v in check.items() if k != "run_id"):
                    raise ValueError(f"seed-17 reproduction mismatch for {run_id}: {check}")
            rows.append({
                "seed": seed, "run_id": run_id, "environment_id": pair["environment_id"],
                "moved_queries": int(moved.sum()), "baseline_correct": int(base_correct.sum()),
                "method_correct": int(method_correct.sum()), "baseline_assigned": int((moved & base_assigned).sum()),
                "method_assigned": int((moved & method_assigned).sum()),
                "rescued": int((method_correct & ~base_correct).sum()),
                "lost": int((base_correct & ~method_correct).sum()),
            })
    return rows, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotation-root", type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=10_042_027)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--aggregate-only", action="store_true", help="regenerate summary/report from existing per_relation_seed.csv")
    args = parser.parse_args()
    existing_rows = args.output_dir / "per_relation_seed.csv"
    seeds = sorted(pd.read_csv(existing_rows).seed.astype(int).unique().tolist()) if args.aggregate_only and existing_rows.is_file() else [int(s) for s in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)) or 17 not in seeds:
        raise ValueError("include unique seeds and retain seed 17 for reproduction validation")
    pairs = read_csv(args.root / args.pair_manifest)
    if len(pairs) != 110 or len({p["environment_id"] for p in pairs}) != 47:
        raise ValueError("expected 110 relations and 47 environments")
    expected_selection = json.loads((args.root / args.input_dir / "frozen_train_selection.json").read_text())
    if (expected_selection["modes"], expected_selection["margin_m"], expected_selection["min_independent_support"]) != (4, 0.15, 32):
        raise ValueError("saved train-only selector differs from the frozen paper configuration")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reproduction_checks: list[dict[str, object]] = []
    reproduction_relation_count: int | None = None
    if args.aggregate_only:
        relation_frame = pd.read_csv(existing_rows)
        prior_summary = json.loads((args.output_dir / "summary.json").read_text())
        reproduction_checks = [{"seed17_reproduction": prior_summary["seed17_reproduction"]["all_exact"]}]
        reproduction_relation_count = int((relation_frame[relation_frame.seed == 17].run_id.nunique()))
        if not prior_summary["seed17_reproduction"]["all_exact"]:
            raise ValueError("existing seed-17 reproduction check failed")
    else:
        tasks = [
            (i, pair, str(args.root), str(args.input_dir), str(args.annotation_root), seeds)
            for i, pair in enumerate(pairs)
        ]
        relation_rows: list[dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_relation_task, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                rows, checks = future.result()
                relation_rows.extend(rows)
                reproduction_checks.extend(checks)
                if done % 10 == 0 or done == len(futures):
                    print(f"processed {done}/110 relations", flush=True)
        relation_frame = pd.DataFrame(relation_rows)
        relation_frame.to_csv(existing_rows, index=False)
    per_seed: dict[str, object] = {}
    for seed in seeds:
        sub = relation_frame[relation_frame.seed == seed]
        pooled = {
            "moved_queries": int(sub.moved_queries.sum()),
            "baseline_correct": int(sub.baseline_correct.sum()),
            "method_correct": int(sub.method_correct.sum()),
            "baseline_assigned": int(sub.baseline_assigned.sum()),
            "method_assigned": int(sub.method_assigned.sum()),
            "rescued": int(sub.rescued.sum()), "lost": int(sub.lost.sum()),
        }
        pooled["baseline_coverage"] = pooled["baseline_correct"] / pooled["moved_queries"]
        pooled["method_coverage"] = pooled["method_correct"] / pooled["moved_queries"]
        pooled["baseline_precision"] = pooled["baseline_correct"] / pooled["baseline_assigned"]
        pooled["method_precision"] = pooled["method_correct"] / pooled["method_assigned"]
        coverage_delta = bootstrap_delta(sub, "coverage", args.bootstrap_samples, args.bootstrap_seed + seed)
        precision_delta = bootstrap_delta(sub, "precision", args.bootstrap_samples, args.bootstrap_seed + seed)
        env_added = sub.assign(
            added_assigned=sub.method_assigned - sub.baseline_assigned,
            added_correct=sub.rescued - sub.lost,
        ).groupby("environment_id", sort=True)[["added_assigned", "added_correct"]].sum()
        env_added = env_added[env_added.added_assigned > 0]
        added_values = (env_added.added_correct / env_added.added_assigned).to_numpy(dtype=float)
        rng = np.random.default_rng(args.bootstrap_seed + 1_000 + seed)
        added_draws = rng.integers(0, len(added_values), size=(args.bootstrap_samples, len(added_values)))
        added_precision = {
            "pooled": float(pooled["rescued"] / (pooled["method_assigned"] - pooled["baseline_assigned"])),
            "environment_macro": float(np.mean(added_values)),
            "environment_macro_ci95": [float(x) for x in np.quantile(added_values[added_draws].mean(axis=1), [0.025, 0.975])],
            "eligible_environments": int(len(added_values)),
        }
        per_seed[str(seed)] = {"pooled": pooled, "environment_macro_delta_coverage": coverage_delta, "environment_macro_delta_precision": precision_delta, "new_assignment_precision": added_precision}
    summary = {
        "status": "complete", "seeds": seeds, "relations": 110, "environments": 47,
        "frozen_selection": expected_selection,
        "validation_labels_used_for_hypothesis_or_gate_selection": False,
        "seed17_reproduction": {"relations_checked": reproduction_relation_count if reproduction_relation_count is not None else len(reproduction_checks), "all_exact": all(all(v for k, v in row.items() if k != "run_id") for row in reproduction_checks)},
        "paired_bootstrap": {"unit": "environment; pool relations within environment", "samples": args.bootstrap_samples, "seed_base": args.bootstrap_seed},
        "per_seed": per_seed,
        "interpretation": "Repeated seeds on the same fixed split are stochastic sensitivity, not independent replication. The selector remains frozen from train-only evidence; seed 17 exactly reproduces its saved validation diagnostics. No lost baseline-correct inliers is guaranteed by the asymmetric switch rule.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Frozen pose-locked motion: validation seed sensitivity", "", "The 0.15 m switch margin, four-mode cap, 32-query independent-support floor, 0.10 m residual threshold, top-20 candidates, and iteration budgets remain frozen from source-only selection. Only the RANSAC random seed changes. Seed 17 must reproduce all 110 saved relations exactly; no validation labels select modes or assignments.", "", "| Seed | Moved queries | Baseline coverage | Motion coverage | Coverage Δ, env macro (95% CI) | Baseline precision | Motion precision | Precision Δ, env macro (95% CI) | New correct / all assigned (pooled) | Rescued / lost |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        item = per_seed[str(seed)]
        p, c, q, added = item["pooled"], item["environment_macro_delta_coverage"], item["environment_macro_delta_precision"], item["new_assignment_precision"]
        lines.append(f"| {seed} | {p['moved_queries']:,} | {p['baseline_coverage']:.1%} | {p['method_coverage']:.1%} | {100*c['estimate']:+.1f} pp [{100*c['ci95'][0]:+.1f}, {100*c['ci95'][1]:+.1f}] | {p['baseline_precision']:.1%} | {p['method_precision']:.1%} | {100*q['estimate']:+.1f} pp [{100*q['ci95'][0]:+.1f}, {100*q['ci95'][1]:+.1f}] | {added['pooled']:.1%} | {p['rescued']:,} / {p['lost']:,} |")
    lines.extend(["", "Seed 17 exact-reproduction checks: all 110 relations matched selected reference indices, assigned masks, correct-transfer masks, and baseline outcomes. Different seeds reuse the same images, queries, annotations, and environments; their intervals are within-seed environment bootstrap intervals and must not be pooled as independent samples. The added-assignment precision is the pooled correctness among method assignments beyond the corresponding seed's global baseline; its environment-macro interval is in `summary.json`.", "", "Reproduction: `scripts/audit_pose_locked_motion_validation_seed_sensitivity.py`. To regenerate aggregate tables from the saved relation rows, add `--aggregate-only`. Machine outputs: `per_relation_seed.csv` and `summary.json`."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote seed sensitivity to {args.output_dir}")


if __name__ == "__main__":
    main()
