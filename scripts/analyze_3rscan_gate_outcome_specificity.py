#!/usr/bin/env python3
"""Compare physical and semantic selectivity of the frozen 3RScan score gate.

The gate is the source-only 75th percentile of top-1 DINO similarity. Both
outcomes are scored on the same physically evaluable validation queries, so
the paired contrast asks whether the gate enriches physical identity more
than semantic correctness. Bootstrap draws resample source relations and
validation environments independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261009
BOOTSTRAP_DRAWS = 10_000


def load_source(root: Path) -> tuple[list[np.ndarray], list[str]]:
    paths = sorted(
        root.glob("outputs/runs/20260922_3rscan_consensus_*_vitg_diag_clean/query_diagnostics.csv")
    )
    paths = [p for p in paths if p.parent.name != "20260922_3rscan_consensus_target_vitg_diag_clean"]
    if len(paths) != 16:
        raise ValueError(f"expected 16 source relations; found {len(paths)}")
    scores: list[np.ndarray] = []
    names: list[str] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[
            frame.geometry_weight.eq(0.0)
            & frame.top_k.eq(20)
            & frame.query_evaluable.astype(bool)
        ]
        if frame.empty:
            raise ValueError(f"no eligible calibration queries in {path}")
        scores.append(frame.top1_descriptor_score.to_numpy(dtype=np.float64))
        names.append(path.parent.name)
    return scores, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_3rscan_gate_outcome_specificity"))
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    source_scores, source_names = load_source(root)
    pred_path = root / "outputs/reports/20261006_3rscan_score_selectivity_lopo/crossfit_predictions.csv.gz"
    predictions = pd.read_csv(pred_path)
    q = predictions[
        predictions.calibration_protocol.eq("source_train_fixed")
        & predictions.signal.eq("DINO top-1 similarity")
        & predictions.train_acceptance_target.eq(0.25)
        & predictions.physical_eval.astype(bool)
    ].copy()
    if q.empty or q.environment_id.nunique() != 47:
        raise ValueError("expected source-frozen q75 rows for 47 validation environments")
    q = q.sort_values(["environment_id", "run_id", "query_index"]).reset_index(drop=True)

    env_names = np.array(sorted(q.environment_id.unique()))
    env_to_idx = {name: i for i, name in enumerate(env_names)}
    env_idx = q.environment_id.map(env_to_idx).to_numpy(dtype=np.int32)
    score = q.top1_descriptor_score.to_numpy(dtype=np.float64)
    physical = q.physical_correct.astype(bool).to_numpy()
    semantic = q.semantic_correct_eval.astype(bool).to_numpy()
    counts = np.bincount(env_idx, minlength=len(env_names)).astype(np.float64)
    phys_total = np.bincount(env_idx, weights=physical, minlength=len(env_names))
    sem_total = np.bincount(env_idx, weights=semantic, minlength=len(env_names))
    baseline_phys = phys_total / counts
    baseline_sem = sem_total / counts
    point_threshold = float(np.quantile(np.concatenate(source_scores), 0.75))

    def env_deltas(threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        accepted = score >= threshold
        accepted_n = np.bincount(env_idx, weights=accepted, minlength=len(env_names))
        accepted_phys = np.bincount(env_idx, weights=accepted & physical, minlength=len(env_names))
        accepted_sem = np.bincount(env_idx, weights=accepted & semantic, minlength=len(env_names))
        valid = accepted_n > 0
        phys_precision = np.divide(accepted_phys, accepted_n, out=np.full_like(accepted_n, np.nan), where=valid)
        sem_precision = np.divide(accepted_sem, accepted_n, out=np.full_like(accepted_n, np.nan), where=valid)
        return phys_precision - baseline_phys, sem_precision - baseline_sem, valid

    point_phys, point_sem, point_valid = env_deltas(point_threshold)
    if not point_valid.all():
        raise ValueError("point threshold produced an environment with no accepted queries")
    point_contrast = point_phys - point_sem

    rng = np.random.default_rng(SEED)
    draws = np.empty((args.draws, 3), dtype=np.float64)
    n_env = len(env_names)
    n_rel = len(source_scores)
    for b in range(args.draws):
        rel_draw = rng.integers(0, n_rel, size=n_rel)
        threshold = float(np.quantile(np.concatenate([source_scores[i] for i in rel_draw]), 0.75))
        d_phys, d_sem, valid = env_deltas(threshold)
        env_draw = rng.integers(0, n_env, size=n_env)
        selected = env_draw[valid[env_draw]]
        if selected.size == 0:
            raise ValueError(f"bootstrap draw {b} has no environments with accepted queries")
        mean_phys = float(d_phys[selected].mean())
        mean_sem = float(d_sem[selected].mean())
        draws[b] = mean_phys, mean_sem, mean_phys - mean_sem

    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    summary = {
        "calibration": "source-only 75th percentile of DINO top-1 similarity",
        "source_relations": n_rel,
        "source_query_count": int(sum(len(x) for x in source_scores)),
        "validation_environments": n_env,
        "validation_queries_on_paired_physical_eval_population": int(len(q)),
        "threshold": point_threshold,
        "paired_population": "queries with unambiguous physical-instance evaluation; semantic and physical correctness scored on exactly these rows",
        "macro_baseline_physical_precision": float(baseline_phys.mean()),
        "macro_baseline_semantic_precision": float(baseline_sem.mean()),
        "macro_accepted_physical_precision": float((baseline_phys + point_phys).mean()),
        "macro_accepted_semantic_precision": float((baseline_sem + point_sem).mean()),
        "physical_precision_gain": float(point_phys.mean()),
        "semantic_precision_gain": float(point_sem.mean()),
        "physical_minus_semantic_gain": float(point_contrast.mean()),
        "bootstrap_draws": int(args.draws),
        "bootstrap_seed": SEED,
        "bootstrap_interval": "percentile 95%; independently resample 16 source relations and 47 validation environments; refit q75 in every draw",
        "physical_gain_ci": [float(intervals[0, 0]), float(intervals[1, 0])],
        "semantic_gain_ci": [float(intervals[0, 1]), float(intervals[1, 1])],
        "paired_contrast_ci": [float(intervals[0, 2]), float(intervals[1, 2])],
        "environments_with_positive_physical_gain": int((point_phys > 0).sum()),
        "environments_with_positive_semantic_gain": int((point_sem > 0).sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame({
        "environment_id": env_names,
        "query_count": counts.astype(int),
        "baseline_physical_precision": baseline_phys,
        "accepted_physical_precision": baseline_phys + point_phys,
        "physical_gain": point_phys,
        "baseline_semantic_precision": baseline_sem,
        "accepted_semantic_precision": baseline_sem + point_sem,
        "semantic_gain": point_sem,
        "physical_minus_semantic_gain": point_contrast,
    }).to_csv(output / "environment_metrics.csv", index=False)
    pd.DataFrame(draws, columns=["physical_gain", "semantic_gain", "physical_minus_semantic_gain"]).to_csv(
        output / "bootstrap_draws.csv.gz", index=False, compression="gzip"
    )
    report = f"""# Does the source-frozen 3RScan gate select physical identity beyond semantics?\n\nThe source-only DINO top-1 similarity threshold is the 75th percentile across {n_rel} 3RScan source relations ({summary['source_query_count']:,} queries), refit inside each bootstrap draw. We apply it unchanged to 110 official validation relations in {n_env} environments. To compare outcomes fairly, both semantic and physical correctness are scored on the same {len(q):,} queries with an unambiguous physical-instance evaluation.\n\n| Outcome | Baseline precision | Accepted precision | Gain | Joint 95% interval |\n|---|---:|---:|---:|---:|\n| Physical identity | {summary['macro_baseline_physical_precision']:.1%} | {summary['macro_accepted_physical_precision']:.1%} | {summary['physical_precision_gain']*100:+.1f} pp | [{intervals[0,0]*100:+.1f}, {intervals[1,0]*100:+.1f}] pp |\n| Semantic correctness | {summary['macro_baseline_semantic_precision']:.1%} | {summary['macro_accepted_semantic_precision']:.1%} | {summary['semantic_precision_gain']*100:+.1f} pp | [{intervals[0,1]*100:+.1f}, {intervals[1,1]*100:+.1f}] pp |\n| Physical minus semantic gain | — | — | {summary['physical_minus_semantic_gain']*100:+.1f} pp | [{intervals[0,2]*100:+.1f}, {intervals[1,2]*100:+.1f}] pp |\n\nBootstrap draws independently resample source relations and validation environments ({args.draws:,} draws; seed {SEED}); each draw refits the source q75. The final row is a paired contrast: both gains use the same environment resamples and query population. A positive physical gain alone does not show identity-specific selectivity; that requires the contrast interval to exclude zero.\n\nThis is an exploratory reanalysis of a previously inspected 3RScan split, not independent confirmation or a preregistered test. The query population is annotation/depth-valid and is restricted to uniquely physically evaluable queries. Per-environment results, bootstrap draws, and the exact machine-readable summary are stored alongside this report. Reproduce with `python scripts/analyze_3rscan_gate_outcome_specificity.py`.\n"""
    (output / "report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
