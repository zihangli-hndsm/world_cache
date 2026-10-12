#!/usr/bin/env python3
"""Apply the frozen train-selected pose-locked motion rule to 3RScan val.

The secondary-motion mode count, residual margin, and support floor are read
from the predesignated target row of the original 16-relation train-only
source fit. Validation annotations are used only after inference for scoring.
Only one relation's RGB-D preparation is held in scratch at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_3rscan_multimotion import (
    assign_modes,
    extract_pair_features,
    fit_fixed_consensus,
    fit_modes,
    save_feature_cache,
    score_outputs,
)
from worldcache.backbone.dinov2 import load_dinov2


PAIR_COLUMNS = ["environment_id", "reference_scan", "rescan_scan", "relation_index"]
TRAIN_SELECTION = Path("outputs/reports/20260929_multimotion_seed_robustness/seed_17/source_lopo_selection.csv")
BASELINE_PROTOCOL = Path("outputs/reports/20261003_vitg_validation/relations")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def frozen_selection(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    selected = [row for row in rows if row.get("pair") == "target_once"]
    if len(selected) != 1:
        raise ValueError(f"expected one predesignated train-only selector row in {path}")
    row = selected[0]
    result = {
        "modes": int(row["modes"]),
        "margin_m": float(row["margin_m"]),
        "min_independent_support": int(row["min_independent_support"]),
        "training_relations": int(row["training_relations"]),
        "seed": 17,
        "source_artifact": str(path),
    }
    if result != {
        "modes": 4,
        "margin_m": 0.15,
        "min_independent_support": 32,
        "training_relations": 16,
        "seed": 17,
        "source_artifact": str(path),
    }:
        raise ValueError(f"frozen training selector differs from audited configuration: {result}")
    return result


def pack_base(base: Any, query_count: int) -> dict[str, Any]:
    selected_matrix = base.selected[:, None]
    residual_matrix = base.residuals[:, None]
    return {
        "modes": [base],
        "scene_matrix": base.matrix,
        "mode_assignment": np.where(base.inliers, 0, -1).astype(np.int64),
        "selected": base.selected,
        "residuals": base.residuals,
        "assigned": base.inliers,
        "limits": np.full(query_count, 20, dtype=np.int64),
        "mode_selected_matrix": selected_matrix,
        "mode_residual_matrix": residual_matrix,
        "adaptive_threshold": 0.0,
        "k50_fraction": 0.0,
        "threshold": 0.10,
    }


def relation_is_complete(output: Path, cache: Path) -> bool:
    status_path = output / "status.json"
    summary_path = output / "summary.csv"
    diagnostics_path = output / "query_diagnostics.csv"
    if not all(path.is_file() for path in (status_path, summary_path, diagnostics_path, cache)):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary = pd.read_csv(summary_path)
        diagnostics = pd.read_csv(diagnostics_path, nrows=1)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    required = {"moved_rigid", "baseline_correct", "method_correct"}
    return (
        status.get("status") == "complete"
        and set(summary.arm) == {"global_ransac", "combined"}
        and required.issubset(diagnostics.columns)
    )


def write_relation(
    *,
    pair: str,
    reference_id: str,
    rescan_id: str,
    pair_run: Path,
    cache_path: Path,
    output: Path,
    backbone: torch.nn.Module,
    device: torch.device,
    annotation_root: Path,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    features = extract_pair_features(
        pair,
        reference_id,
        rescan_id,
        pair_run,
        backbone,
        (2, 20, 40),
        device,
        # The frozen validation runner computes the full query-by-reference
        # score matrix in one matmul. Keep that shape here as well: splitting
        # the same matrix into chunks can perturb near-tied top-K ordering and
        # change a sensitive RANSAC consensus.
        descriptor_chunk=100_000,
        preserve_top_k=20,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_name(cache_path.stem + ".partial.npz")
    save_feature_cache(temporary_cache, features, "giant", (2, 20, 40), 224)
    temporary_cache.replace(cache_path)

    limits = np.full(len(features.query_points), 20, dtype=np.int64)
    baseline = fit_fixed_consensus(
        features.query_points,
        features.reference_points,
        features.candidate_indices[:, :20],
        threshold=0.10,
        iterations=1500,
        seed=17,
    )
    modes = fit_modes(
        features,
        limits,
        threshold=0.10,
        iterations=1500,
        secondary_iterations=1000,
        seed=17,
        max_modes=4,
        min_support=8,
        fixed_baseline=baseline,
    )
    motion = assign_modes(features, limits, modes, threshold=0.10)
    mode_assignment, mode_selected, mode_residuals, _, mode_selected_matrix, mode_residual_matrix = motion

    # Apply the exact train-selected geometric switch rule. Independent support
    # is counted only among queries unexplained by the global mode and earlier
    # secondary modes. No annotation IDs enter this decision.
    secondary_support: list[int] = []
    unexplained = baseline.residuals > 0.10
    for mode_index in range(1, len(modes)):
        inlier = mode_residual_matrix[:, mode_index] <= 0.10
        secondary_support.append(int((inlier & unexplained).sum()))
        unexplained &= ~inlier

    valid_secondary = [
        mode_index
        for mode_index, support in enumerate(secondary_support, start=1)
        if mode_index < 4 and support >= 32
    ]
    selected = baseline.selected.copy()
    residuals = baseline.residuals.copy()
    assigned = baseline.inliers.copy()
    gated_assignment = np.where(baseline.inliers, 0, -1).astype(np.int64)
    if valid_secondary:
        candidate_residuals = mode_residual_matrix[:, valid_secondary]
        best_local = np.argmin(candidate_residuals, axis=1)
        rows = np.arange(len(features.query_points))
        best_residual = candidate_residuals[rows, best_local]
        best_mode = np.asarray(valid_secondary, dtype=np.int64)[best_local]
        switch = (
            np.isfinite(best_residual)
            & (best_residual <= 0.10)
            & ((baseline.residuals > 0.10) | ((baseline.residuals - best_residual) >= 0.15))
        )
        selected[switch] = mode_selected_matrix[rows[switch], best_mode[switch]]
        residuals[switch] = best_residual[switch]
        assigned[switch] = True
        gated_assignment[switch] = best_mode[switch]
    outputs = {
        "global_ransac": pack_base(baseline, len(features.query_points)),
        "combined": {
            "modes": modes,
            "scene_matrix": baseline.matrix,
            "mode_assignment": gated_assignment,
            "selected": selected,
            "residuals": residuals,
            "assigned": assigned,
            "limits": limits,
            "mode_selected_matrix": mode_selected_matrix,
            "mode_residual_matrix": mode_residual_matrix,
            "adaptive_threshold": 0.0,
            "k50_fraction": 0.0,
            "threshold": 0.10,
            "selector": {
                "margin_m": 0.15,
                "min_independent_support": 32,
                "secondary_independent_support": secondary_support,
                "valid_secondary_modes": valid_secondary,
            },
        },
    }
    config = json.loads((pair_run / "config.json").read_text(encoding="utf-8"))
    summary_rows, diagnostic_frames = score_outputs(
        features, outputs, annotation_root, metadata, config
    )
    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    diagnostics["moved_rigid"] = (
        diagnostics.query_evaluable.astype(bool)
        & (diagnostics.official_rigid_reference_instance_id.astype(np.int64) > 0)
    )
    diagnostics["baseline_correct"] = (
        diagnostics.moved_rigid
        & diagnostics.consensus_inlier.astype(bool)
        & diagnostics.official_rigid_instance_correct.astype(bool)
    )
    diagnostics["method_correct"] = diagnostics.baseline_correct
    combined_rows = diagnostics.arm == "combined"
    diagnostics.loc[combined_rows, "method_correct"] = diagnostics.loc[
        combined_rows, "baseline_correct"
    ]
    diagnostics.loc[~combined_rows, "method_correct"] = False

    baseline_path = BASELINE_PROTOCOL / f"{pair[:3]}_{reference_id[:8]}_{rescan_id[:8]}" / "vitg" / "summary.csv"
    # The run ID is passed in pair; relation directory lookup is handled by the
    # caller and recorded separately. Baseline equivalence is checked there.
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    diagnostics.to_csv(output / "query_diagnostics.csv", index=False)
    status = {
        "status": "inference_complete_pending_baseline_check",
        "pair": pair,
        "reference_scan": reference_id,
        "rescan_scan": rescan_id,
        "protocol": {
            "layers": [2, 20, 40],
            "top_k": 20,
            "candidate_ordering": "torch.topk(k=20) order preserved before cached top-50 tail",
            "threshold_m": 0.10,
            "ransac_iterations": 1500,
            "secondary_iterations": 1000,
            "seed": 17,
            "frozen_train_selector": {"modes": 4, "margin_m": 0.15, "min_independent_support": 32},
        },
        "selector_diagnostics": outputs["combined"]["selector"],
        "baseline_path_hint": str(baseline_path),
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "diagnostics": diagnostics}


def aggregate(
    manifest: list[dict[str, str]],
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    for index, pair in enumerate(manifest):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        summary = pd.read_csv(output_dir / "relations" / run_id / "summary.csv")
        diagnostics = pd.read_csv(output_dir / "relations" / run_id / "query_diagnostics.csv")
        base = summary[summary.arm == "global_ransac"].iloc[0]
        method = summary[summary.arm == "combined"].iloc[0]
        base_queries = diagnostics[(diagnostics.arm == "global_ransac") & diagnostics.moved_rigid.astype(bool)]
        method_queries = diagnostics[(diagnostics.arm == "combined") & diagnostics.moved_rigid.astype(bool)]
        if not np.array_equal(base_queries.query_index.to_numpy(), method_queries.query_index.to_numpy()):
            raise ValueError(f"moved-rigid query mismatch for {run_id}")
        base_correct = base_queries.baseline_correct.astype(bool).to_numpy()
        method_correct = method_queries.method_correct.astype(bool).to_numpy()
        base_assigned = base_queries.consensus_inlier.astype(bool).to_numpy()
        method_assigned = method_queries.consensus_inlier.astype(bool).to_numpy()
        outcomes.append({
            "run_id": run_id,
            "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"],
            "rescan_scan": pair["rescan_scan"],
            "relation_index": int(pair["relation_index"]),
            "moved_rigid_queries": int(len(base_queries)),
            "baseline_correct": int(base_correct.sum()),
            "method_correct": int(method_correct.sum()),
            "baseline_assigned": int(base_assigned.sum()),
            "method_assigned": int(method_assigned.sum()),
            "baseline_false_positive": int((base_assigned & ~base_correct).sum()),
            "method_false_positive": int((method_assigned & ~method_correct).sum()),
            "baseline_precision": float(base_correct.sum() / base_assigned.sum()) if base_assigned.any() else float("nan"),
            "method_precision": float(method_correct.sum() / method_assigned.sum()) if method_assigned.any() else float("nan"),
            "rescued": int((method_correct & ~base_correct).sum()),
            "lost": int((base_correct & ~method_correct).sum()),
            "baseline_coverage": float(base_correct.mean()) if len(base_correct) else float("nan"),
            "method_coverage": float(method_correct.mean()) if len(method_correct) else float("nan"),
            "coverage_delta": float(method_correct.mean() - base_correct.mean()) if len(base_correct) else float("nan"),
            "strict_pose_success": bool(base.strict_scene_pose_success),
            "method_strict_pose_success": bool(method.strict_scene_pose_success),
            "rotation_error_deg": float(base.scene_rotation_error_deg),
            "translation_error_m": float(base.scene_translation_error_m),
        })
    frame = pd.DataFrame(outcomes)
    if len(frame) != len(manifest) or frame.environment_id.nunique() != 47:
        raise ValueError("expected all 110 official validation relations in 47 environments")
    if not (frame.strict_pose_success == frame.method_strict_pose_success).all():
        raise ValueError("pose-locked branch changed the global scene pose")

    env_rows: list[dict[str, Any]] = []
    for env, subset in frame.groupby("environment_id", sort=True):
        n = int(subset.moved_rigid_queries.sum())
        base = int(subset.baseline_correct.sum())
        method = int(subset.method_correct.sum())
        base_assigned = int(subset.baseline_assigned.sum())
        method_assigned = int(subset.method_assigned.sum())
        env_rows.append({
            "environment_id": env,
            "relations": int(subset.rescan_scan.nunique()),
            "moved_rigid_queries": n,
            "baseline_correct": base,
            "method_correct": method,
            "baseline_assigned": base_assigned,
            "method_assigned": method_assigned,
            "baseline_false_positive": int(subset.baseline_false_positive.sum()),
            "method_false_positive": int(subset.method_false_positive.sum()),
            "baseline_precision": base / base_assigned if base_assigned else float("nan"),
            "method_precision": method / method_assigned if method_assigned else float("nan"),
            "precision_delta": (
                method / method_assigned - base / base_assigned
                if base_assigned and method_assigned else float("nan")
            ),
            "baseline_coverage": base / n if n else float("nan"),
            "method_coverage": method / n if n else float("nan"),
            "coverage_delta": (method - base) / n if n else float("nan"),
            "rescued": int(subset.rescued.sum()),
            "lost": int(subset.lost.sum()),
        })
    env_frame = pd.DataFrame(env_rows)
    eligible_env = env_frame[env_frame.moved_rigid_queries > 0].reset_index(drop=True)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(eligible_env), size=(bootstrap_samples, len(eligible_env)))
    macro_delta = eligible_env.coverage_delta.to_numpy(dtype=np.float64)
    boot_macro = macro_delta[indices].mean(axis=1)
    boot_pooled = (
        (eligible_env.method_correct.to_numpy()[indices].sum(axis=1)
         - eligible_env.baseline_correct.to_numpy()[indices].sum(axis=1))
        / eligible_env.moved_rigid_queries.to_numpy()[indices].sum(axis=1)
    )
    precision_rng = np.random.default_rng(bootstrap_seed + 1)
    precision_indices = precision_rng.integers(
        0, len(env_frame), size=(bootstrap_samples, len(env_frame))
    )
    base_precision_boot = (
        env_frame.baseline_correct.to_numpy()[precision_indices].sum(axis=1)
        / env_frame.baseline_assigned.to_numpy()[precision_indices].sum(axis=1)
    )
    method_precision_boot = (
        env_frame.method_correct.to_numpy()[precision_indices].sum(axis=1)
        / env_frame.method_assigned.to_numpy()[precision_indices].sum(axis=1)
    )
    precision_delta_boot = method_precision_boot - base_precision_boot
    common_precision_env = env_frame[
        (env_frame.baseline_assigned > 0) & (env_frame.method_assigned > 0)
    ].reset_index(drop=True)
    common_precision_delta = common_precision_env.precision_delta.to_numpy(dtype=np.float64)
    common_precision_indices = precision_rng.integers(
        0, len(common_precision_env),
        size=(bootstrap_samples, len(common_precision_env)),
    )
    precision_macro_boot = common_precision_delta[common_precision_indices].mean(axis=1)
    base_precision = int(frame.baseline_correct.sum()) / int(frame.baseline_assigned.sum())
    method_precision = int(frame.method_correct.sum()) / int(frame.method_assigned.sum())
    summary = {
        "status": "complete",
        "relations": int(len(frame)),
        "environments": int(frame.environment_id.nunique()),
        "eligible_environments": int(len(eligible_env)),
        "moved_rigid_queries": int(frame.moved_rigid_queries.sum()),
        "baseline_correct": int(frame.baseline_correct.sum()),
        "method_correct": int(frame.method_correct.sum()),
        "baseline_assigned": int(frame.baseline_assigned.sum()),
        "method_assigned": int(frame.method_assigned.sum()),
        "baseline_precision": base_precision,
        "method_precision": method_precision,
        "pooled_precision_delta": method_precision - base_precision,
        "environment_bootstrap_pooled_precision_delta_ci95": [
            float(x) for x in np.quantile(precision_delta_boot, [0.025, 0.975])
        ],
        "common_precision_environments": int(len(common_precision_env)),
        "common_environment_macro_precision_delta": float(common_precision_delta.mean()),
        "common_environment_macro_precision_delta_ci95": [
            float(x) for x in np.quantile(precision_macro_boot, [0.025, 0.975])
        ],
        "precision_bootstrap_samples": bootstrap_samples,
        "precision_bootstrap_seed": bootstrap_seed + 1,
        "rescued": int(frame.rescued.sum()),
        "lost": int(frame.lost.sum()),
        "pooled_baseline_coverage": float(frame.baseline_correct.sum() / frame.moved_rigid_queries.sum()),
        "pooled_method_coverage": float(frame.method_correct.sum() / frame.moved_rigid_queries.sum()),
        "pooled_delta": float((frame.method_correct.sum() - frame.baseline_correct.sum()) / frame.moved_rigid_queries.sum()),
        "environment_macro_delta": float(macro_delta.mean()),
        "environment_macro_delta_ci95": [float(x) for x in np.quantile(boot_macro, [0.025, 0.975])],
        "paired_environment_pooled_delta_ci95": [float(x) for x in np.quantile(boot_pooled, [0.025, 0.975])],
        "environments_method_better": int((macro_delta > 0).sum()),
        "environments_tied": int((macro_delta == 0).sum()),
        "environments_method_worse": int((macro_delta < 0).sum()),
        "strict_pose_successes_locked": int(frame.strict_pose_success.sum()),
        "strict_pose_successes_after": int(frame.method_strict_pose_success.sum()),
        "bootstrap_unit": "official validation environment; all relations from an environment stay together",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
    frame.to_csv(output_dir / "per_relation.csv", index=False)
    env_frame.to_csv(output_dir / "per_environment.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = f"""# Pose-locked secondary-motion validation

The validation manifest contains {len(frame)} official relations across {frame.environment_id.nunique()} environment IDs. Hyperparameters were selected on the 16 source relations only and frozen before this validation analysis: four candidate modes, a 0.15 m residual improvement margin, at least 32 independent secondary-mode inliers, top-20 ViT-G candidates, a 0.10 m residual threshold, 1,500 global RANSAC iterations, 1,000 secondary-mode iterations, and seed 17. The official rigid-instance annotations are used for scoring after inference.

## Transfer coverage

Among {summary['moved_rigid_queries']:,} annotated moved-rigid query patches, the global-pose baseline correctly transfers {summary['baseline_correct']:,} ({summary['pooled_baseline_coverage']:.1%}); the pose-locked motion branch transfers {summary['method_correct']:,} ({summary['pooled_method_coverage']:.1%}). This is {summary['rescued']:,} rescues and {summary['lost']:,} lost baseline-correct queries. The pooled coverage difference is {summary['pooled_delta']:+.1%}; the environment-macro difference is {summary['environment_macro_delta']:+.1%} (95% environment-bootstrap CI [{summary['environment_macro_delta_ci95'][0]:+.1%}, {summary['environment_macro_delta_ci95'][1]:+.1%}]). The environment-bootstrap CI for the pooled difference is [{summary['paired_environment_pooled_delta_ci95'][0]:+.1%}, {summary['paired_environment_pooled_delta_ci95'][1]:+.1%}]. Coverage intervals use {summary['bootstrap_samples']:,} environment resamples with seed {summary['bootstrap_seed']}. The method improves coverage in {summary['environments_method_better']} environments, ties in {summary['environments_tied']}, and is worse in {summary['environments_method_worse']}; five environments have no annotated moved-rigid queries.

## Assignment precision and scene pose

The branch assigns {summary['method_assigned']:,} moved-rigid queries, compared with {summary['baseline_assigned']:,} for global RANSAC. Pooled physical-instance precision changes from {summary['baseline_precision']:.1%} to {summary['method_precision']:.1%} ({summary['pooled_precision_delta']:+.1%}; 95% environment-bootstrap CI [{summary['environment_bootstrap_pooled_precision_delta_ci95'][0]:+.1%}, {summary['environment_bootstrap_pooled_precision_delta_ci95'][1]:+.1%}]; {summary['precision_bootstrap_samples']:,} resamples, seed {summary['precision_bootstrap_seed']}). Thus the frozen rule expands coverage but accepts lower-precision assignments. Strict global-pose successes remain {summary['strict_pose_successes_locked']}/{summary['relations']}; each method arm uses the same baseline transform by construction.

## Scope

This is a frozen train-selected branch evaluated on the official validation relations; no validation labels select modes or assignments. The validation split is reused from the correspondence audit, and its environment IDs have not been verified as independent physical sites. Treat the result as split-level evidence for a coverage/precision trade-off in pose-locked local transfer, not a claim of reliable global registration or independent-site generalization.

Machine-readable outputs: `summary.json`, `per_relation.csv`, `per_environment.csv`, and `run_manifest.csv`. Per-relation diagnostics and baseline-reproduction deltas are under `relations/`.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "protocol_status.json").write_text(
        json.dumps({"status": "complete", **summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--validation-root", type=Path, default=Path("outputs/reports/20261003_vitg_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261006_vitg_pose_locked_motion_validation"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=10_062_026)
    parser.add_argument("--max-relations", type=int, help="run an initial relation subset only")
    args = parser.parse_args()
    pairs = read_rows(args.pair_manifest)
    if len(pairs) != 110 or len({p["environment_id"] for p in pairs}) != 47:
        raise ValueError("expected the audited official validation manifest")
    selection = frozen_selection(TRAIN_SELECTION)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frozen_train_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    plan = {
        "official_validation_relations": len(pairs),
        "official_validation_environments": len({p["environment_id"] for p in pairs}),
        "method": "global top-20 ViT-G pose locked; train-selected secondary rigid modes",
        "train_only_selection": selection,
        "protocol": {"layers": [2, 20, 40], "top_k": 20, "threshold_m": 0.10, "iterations": 1500, "secondary_iterations": 1000, "seed": 17},
        "validation_labels_used_for_selection": False,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_unit": "environment",
        "execution_requested": True,
    }
    (args.output_dir / "protocol_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2("giant", 224, device)
    generate_script = Path(__file__).with_name("generate_3rscan_pairs.py")
    manifest_path = args.output_dir / "run_manifest.csv"
    completed_rows: list[dict[str, Any]] = []
    run_pairs = pairs[: args.max_relations] if args.max_relations else pairs

    def save_manifest() -> None:
        if not completed_rows:
            return
        tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
        with tmp_manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(completed_rows[0]))
            writer.writeheader()
            writer.writerows(completed_rows)
        tmp_manifest.replace(manifest_path)

    for index, pair in enumerate(run_pairs):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        output = args.output_dir / "relations" / run_id
        cache_path = args.output_dir / "descriptor_candidates" / f"{run_id}.npz"
        row: dict[str, Any] = {key: pair[key] for key in PAIR_COLUMNS}
        row.update({"run_id": run_id, "output_dir": str(output)})
        if relation_is_complete(output, cache_path):
            row["status"] = "complete"
            completed_rows.append(row)
            save_manifest()
            continue
        with tempfile.TemporaryDirectory(prefix=f"wc_motion_val_{index:03d}_", dir=args.scratch_root) as scratch_name:
            pair_run = Path(scratch_name) / "pair"
            generation = [
                sys.executable, str(generate_script),
                "--root", str(args.data_root), "--metadata", str(args.metadata),
                "--reference", pair["reference_scan"], "--rescan", pair["rescan_scan"],
                "--output-dir", str(pair_run), "--resolution", "224",
                "--reference-frames", "20", "--rescan-frames", "20",
            ]
            log_path = output / "runner.log"
            output.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(generation, check=True, stdout=log, stderr=subprocess.STDOUT)
                result = write_relation(
                    pair=run_id,
                    reference_id=pair["reference_scan"],
                    rescan_id=pair["rescan_scan"],
                    pair_run=pair_run,
                    cache_path=cache_path,
                    output=output,
                    backbone=backbone,
                    device=device,
                    annotation_root=args.data_root,
                    metadata=metadata,
                )
            # Confirm the global baseline reproduces the already-completed
            # ViT-G validation pose before accepting any motion result.
            baseline = result["summary"].query("arm == 'global_ransac'").iloc[0]
            expected_run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
            prior_path = args.validation_root / "relations" / expected_run_id / "vitg" / "summary.csv"
            prior = pd.read_csv(prior_path).iloc[0]
            prior_diagnostics = pd.read_csv(
                args.validation_root / "relations" / expected_run_id / "vitg" / "query_diagnostics.csv"
            ).sort_values("query_index").reset_index(drop=True)
            current_diagnostics = result["diagnostics"].query(
                "arm == 'global_ransac'"
            ).sort_values("query_index").reset_index(drop=True)
            if len(current_diagnostics) != len(prior_diagnostics):
                raise ValueError(f"query count changed while reproducing {run_id}")
            residual_delta = np.abs(
                current_diagnostics.spatial_residual_m.to_numpy(dtype=np.float64)
                - prior_diagnostics.spatial_residual_m.to_numpy(dtype=np.float64)
            )
            inliers_match = np.array_equal(
                current_diagnostics.consensus_inlier.astype(bool).to_numpy(),
                prior_diagnostics.consensus_inlier.astype(bool).to_numpy(),
            )
            if (
                not np.array_equal(current_diagnostics.frame_id, prior_diagnostics.frame_id)
                or not inliers_match
                or not np.allclose(residual_delta, 0.0, rtol=0.0, atol=1e-8)
                or abs(float(baseline.scene_rotation_error_deg) - float(prior.rotation_error_deg)) > 1e-5
                or abs(float(baseline.scene_translation_error_m) - float(prior.translation_error_m)) > 1e-5
            ):
                raise ValueError(f"pose-locked baseline failed ViT-G validation reproduction for {run_id}")
            status_path = output / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["status"] = "complete"
            status["baseline_reproduction"] = {
                "rotation_error_abs_delta_deg": abs(
                    float(baseline.scene_rotation_error_deg) - float(prior.rotation_error_deg)
                ),
                "translation_error_abs_delta_m": abs(
                    float(baseline.scene_translation_error_m) - float(prior.translation_error_m)
                ),
                "max_query_residual_abs_delta_m": float(residual_delta.max(initial=0.0)),
                "query_inliers_exact": inliers_match,
            }
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            row["status"] = "complete"
        completed_rows.append(row)
        save_manifest()
        print(f"completed {index + 1}/{len(run_pairs)}: {run_id}", flush=True)

    if len(run_pairs) == 110:
        summary = aggregate(pairs, args.output_dir, args.bootstrap_samples, args.bootstrap_seed)
        (args.output_dir / "protocol_status.json").write_text(json.dumps({"status": "complete", **summary}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    else:
        print(f"pilot subset complete: {len(run_pairs)} relations; no full validation aggregate written")


if __name__ == "__main__":
    main()
