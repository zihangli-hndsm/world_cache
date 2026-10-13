#!/usr/bin/env python3
"""Evaluate source-fit and cross-fit score abstention on official 3RScan RGB-D runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


SIGNALS = {
    "DINO top-1 similarity": ("top1_descriptor_score", "high"),
    "DINO top-1/top-2 margin": ("top1_top2_margin", "high"),
    "selected-match similarity": ("selected_descriptor_score", "high"),
    "spatial residual": ("spatial_residual_m", "low"),
}
TARGETS = (0.50, 0.25, 0.10)
BOOTSTRAP_N = 20_000
SEED = 20261009


def auc_score(score: np.ndarray, label: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    label = np.asarray(label, dtype=bool)
    positives = int(label.sum())
    negatives = len(label) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def load_source_scores(root: Path) -> tuple[pd.DataFrame, int]:
    base = root / "outputs/runs"
    paths = sorted(base.glob("20260922_3rscan_consensus_*_vitg_diag_clean/query_diagnostics.csv"))
    paths = [path for path in paths if path.parent.name != "20260922_3rscan_consensus_target_vitg_diag_clean"]
    if len(paths) != 16:
        raise ValueError(f"expected 16 frozen source relations for score calibration, found {len(paths)}")
    chunks = []
    for path in paths:
        data = pd.read_csv(path)
        data = data[data.geometry_weight.eq(0.0) & data.top_k.eq(20) & data.query_evaluable.astype(bool)].copy()
        data["source_relation"] = path.parent.name
        chunks.append(data)
    scores = pd.concat(chunks, ignore_index=True)
    if scores.source_relation.nunique() != 16:
        raise ValueError("source score calibration did not include all 16 relations")
    return scores, len(paths)


def load_validation(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = root / "outputs/reports/20261003_vitg_validation"
    manifest = pd.read_csv(report / "validation_run_manifest.csv")
    if len(manifest) != 110 or manifest.status.ne("complete").any():
        raise ValueError("expected all 110 official validation relations to be complete")
    chunks: list[pd.DataFrame] = []
    for relation in manifest.itertuples(index=False):
        output_dir = Path(relation.output_dir)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        path = output_dir / "vitg/query_diagnostics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        data = pd.read_csv(path)
        data = data[data.geometry_weight.eq(0.0) & data.top_k.eq(20)].copy()
        if data.empty:
            raise ValueError(f"missing canonical top-20 rows in {path}")
        data["environment_id"] = relation.environment_id
        data["run_id"] = relation.run_id
        chunks.append(data)
    queries = pd.concat(chunks, ignore_index=True)
    if queries.run_id.nunique() != 110 or queries.environment_id.nunique() != 47:
        raise ValueError("expected 110 relation runs across 47 environments")
    if queries.duplicated(["run_id", "query_index"]).any():
        raise ValueError("duplicate query rows in canonical validation diagnostics")
    return manifest, queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/20261006_3rscan_score_selectivity_lopo"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    source_scores, source_relations = load_source_scores(root)
    manifest, all_queries = load_validation(root)
    queries = all_queries[all_queries.query_evaluable.astype(bool)].copy()
    queries["physical_eval"] = queries.physical_instance_evaluable.astype(bool)
    queries["physical_correct"] = queries.physical_instance_correct.astype(bool)
    queries["semantic_correct_eval"] = queries.semantic_correct.astype(bool)
    queries["consensus_inlier"] = queries.consensus_inlier.astype(bool)
    environments = sorted(queries.environment_id.unique())

    site_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    prediction_chunks: list[pd.DataFrame] = []
    protocols = ("source_train_fixed", "leave_one_environment_out")
    for protocol in protocols:
      for signal_name, (column, direction) in SIGNALS.items():
        for target in TARGETS:
            for environment in environments:
                train = source_scores if protocol == "source_train_fixed" else queries[queries.environment_id.ne(environment)]
                test = queries[queries.environment_id.eq(environment)].copy()
                quantile = target if direction == "low" else 1.0 - target
                threshold = float(train[column].quantile(quantile))
                accepted = test[column].le(threshold) if direction == "low" else test[column].ge(threshold)
                test["calibration_protocol"] = protocol
                test["signal"] = signal_name
                test["train_acceptance_target"] = target
                test["threshold"] = threshold
                test["accepted"] = accepted.to_numpy()
                prediction_chunks.append(
                    test[[
                        "run_id", "environment_id", "query_index", "frame_id", "calibration_protocol", "signal",
                        "train_acceptance_target", "threshold", "accepted", column,
                        "physical_eval", "physical_correct", "semantic_correct_eval",
                        "consensus_inlier", "physical_instance_candidate_hit",
                    ]]
                )

                physical = test.physical_eval
                correct = test.physical_correct
                accepted_physical = accepted & physical
                inlier_physical = physical & test.consensus_inlier
                accepted_inlier = accepted & inlier_physical
                auc_score_values = test[column].to_numpy(dtype=float)
                if direction == "low":
                    auc_score_values = -auc_score_values
                site_rows.append(
                    {
                        "signal": signal_name,
                        "calibration_protocol": protocol,
                        "score_column": column,
                        "direction": direction,
                        "train_acceptance_target": target,
                        "environment_id": environment,
                        "relations": int(test.run_id.nunique()),
                        "query_evaluable": int(len(test)),
                        "physical_queries": int(physical.sum()),
                        "accepted_query_fraction": float(accepted.mean()),
                        "physical_coverage": float(accepted_physical.sum() / physical.sum()),
                        "physical_baseline_precision": float(correct[physical].mean()),
                        "physical_accepted_precision": float(correct[accepted_physical].mean()) if accepted_physical.any() else np.nan,
                        "physical_precision_delta": float(correct[accepted_physical].mean() - correct[physical].mean()) if accepted_physical.any() else np.nan,
                        "semantic_baseline_precision": float(test.semantic_correct_eval.mean()),
                        "semantic_accepted_precision": float(test.loc[accepted, "semantic_correct_eval"].mean()) if accepted.any() else np.nan,
                        "inlier_fraction_of_physical_queries": float(inlier_physical.sum() / physical.sum()),
                        "accepted_inlier_fraction_of_physical_queries": float(accepted_inlier.sum() / physical.sum()),
                        "inlier_baseline_precision": float(correct[inlier_physical].mean()) if inlier_physical.any() else np.nan,
                        "inlier_accepted_precision": float(correct[accepted_inlier].mean()) if accepted_inlier.any() else np.nan,
                        "inlier_precision_delta": float(correct[accepted_inlier].mean() - correct[inlier_physical].mean()) if accepted_inlier.any() and inlier_physical.any() else np.nan,
                        "physical_auc": auc_score(auc_score_values[physical.to_numpy()], correct[physical].to_numpy()),
                    }
                )
                physical_candidate_hit = test.physical_instance_candidate_hit.astype(bool)
                for candidate_hit in (False, True):
                    stratum = physical & physical_candidate_hit.eq(candidate_hit)
                    accepted_stratum = accepted & stratum
                    availability_rows.append(
                        {
                            "signal": signal_name,
                            "calibration_protocol": protocol,
                            "train_acceptance_target": target,
                            "environment_id": environment,
                            "candidate_available_in_top20": candidate_hit,
                            "queries": int(stratum.sum()),
                            "accepted_queries": int(accepted_stratum.sum()),
                            "coverage": float(accepted_stratum.sum() / stratum.sum()) if stratum.any() else np.nan,
                            "baseline_precision": float(correct[stratum].mean()) if stratum.any() else np.nan,
                            "accepted_precision": float(correct[accepted_stratum].mean()) if accepted_stratum.any() else np.nan,
                            "precision_delta": float(correct[accepted_stratum].mean() - correct[stratum].mean()) if accepted_stratum.any() else np.nan,
                        }
                    )

    sites = pd.DataFrame(site_rows)
    availability = pd.DataFrame(availability_rows)
    predictions = pd.concat(prediction_chunks, ignore_index=True)
    sites.to_csv(output / "environment_metrics.csv", index=False)
    availability.to_csv(output / "candidate_availability_metrics.csv", index=False)
    predictions.to_csv(output / "crossfit_predictions.csv.gz", index=False, compression="gzip")

    summary_rows: list[dict[str, object]] = []
    for index, ((protocol, signal, target), group) in enumerate(
        sites.groupby(["calibration_protocol", "signal", "train_acceptance_target"], sort=False)
    ):
        physical_delta = group.physical_precision_delta.dropna().to_numpy(dtype=float)
        inlier_delta = group.inlier_precision_delta.dropna().to_numpy(dtype=float)
        phys_delta = bootstrap_mean(physical_delta, SEED + 3 * index)
        inlier_delta_summary = bootstrap_mean(inlier_delta, SEED + 3 * index + 1)
        auc_values = group.physical_auc.dropna().to_numpy(dtype=float)
        auc_summary = bootstrap_mean(auc_values, SEED + 3 * index + 2)
        summary_rows.append(
            {
                "signal": signal,
                "calibration_protocol": protocol,
                "train_acceptance_target": float(target),
                "environments": int(group.environment_id.nunique()),
                "mean_environment_query_acceptance": float(group.accepted_query_fraction.mean()),
                "mean_environment_physical_coverage": float(group.physical_coverage.mean()),
                "mean_environment_physical_precision_baseline": float(group.physical_baseline_precision.mean()),
                "mean_environment_physical_precision_accepted": float(group.physical_accepted_precision.mean()),
                "physical_precision_delta": phys_delta[0],
                "physical_delta_ci_low": phys_delta[1],
                "physical_delta_ci_high": phys_delta[2],
                "environments_with_positive_physical_delta": int((physical_delta > 0).sum()),
                "defined_physical_deltas": len(physical_delta),
                "mean_environment_semantic_precision_baseline": float(group.semantic_baseline_precision.mean()),
                "mean_environment_semantic_precision_accepted": float(group.semantic_accepted_precision.mean()),
                "mean_environment_inlier_coverage": float(group.inlier_fraction_of_physical_queries.mean()),
                "mean_environment_inlier_coverage_after_gate": float(group.accepted_inlier_fraction_of_physical_queries.mean()),
                "mean_environment_inlier_precision_baseline": float(group.inlier_baseline_precision.mean()),
                "mean_environment_inlier_precision_accepted": float(group.inlier_accepted_precision.mean()),
                "inlier_precision_delta": inlier_delta_summary[0],
                "inlier_delta_ci_low": inlier_delta_summary[1],
                "inlier_delta_ci_high": inlier_delta_summary[2],
                "environments_with_positive_inlier_delta": int((inlier_delta > 0).sum()),
                "defined_inlier_deltas": len(inlier_delta),
                "mean_environment_physical_auc": auc_summary[0],
                "physical_auc_ci_low": auc_summary[1],
                "physical_auc_ci_high": auc_summary[2],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)

    availability_summary_rows: list[dict[str, object]] = []
    for index, ((protocol, signal, target, candidate_hit), group) in enumerate(
        availability.groupby(
            ["calibration_protocol", "signal", "train_acceptance_target", "candidate_available_in_top20"],
            sort=False,
        )
    ):
        deltas = group.precision_delta.dropna().to_numpy(dtype=float)
        delta_summary = bootstrap_mean(deltas, SEED + 100 + index) if len(deltas) else (np.nan, np.nan, np.nan)
        availability_summary_rows.append(
            {
                "calibration_protocol": protocol,
                "signal": signal,
                "train_acceptance_target": float(target),
                "candidate_available_in_top20": bool(candidate_hit),
                "environments": int(group.environment_id.nunique()),
                "environments_with_accepted_queries": int(group.accepted_queries.gt(0).sum()),
                "mean_environment_coverage": float(group.coverage.mean()),
                "mean_environment_baseline_precision": float(group.baseline_precision.mean()),
                "mean_environment_accepted_precision": float(group.accepted_precision.mean()),
                "precision_delta": delta_summary[0],
                "precision_delta_ci_low": delta_summary[1],
                "precision_delta_ci_high": delta_summary[2],
                "environments_with_positive_delta": int((deltas > 0).sum()),
                "defined_precision_deltas": len(deltas),
                "query_count": int(group.queries.sum()),
                "accepted_query_count": int(group.accepted_queries.sum()),
            }
        )
    availability_summary = pd.DataFrame(availability_summary_rows)
    availability_summary.to_csv(output / "candidate_availability_summary.csv", index=False)
    payload = {
        "status": "complete",
        "dataset_protocol": "official 3RScan validation, frozen RGB-D ViT-G pipeline, geometry_weight=0, top_k=20, seed 17",
        "relations": int(manifest.run_id.nunique()),
        "environments": int(queries.environment_id.nunique()),
        "query_evaluable": int(len(queries)),
        "physical_instance_evaluable_queries": int(queries.physical_eval.sum()),
        "physical_instance_correct_queries": int((queries.physical_eval & queries.physical_correct).sum()),
        "source_training_relations": source_relations,
        "calibration_protocols": {
            "source_train_fixed": "fixed numerical score quantiles fit on the 16 source/train relations; no validation scores or labels set the threshold",
            "leave_one_environment_out": "score quantile estimated from the other 46 official validation environments; held-out outcome labels never select threshold",
        },
        "threshold_targets": list(TARGETS),
        "signals": {name: {"column": col, "direction": direction} for name, (col, direction) in SIGNALS.items()},
        "uncertainty": f"environment-cluster bootstrap over up to 47 held-out environment metrics; {BOOTSTRAP_N} draws, seed {SEED}; exploratory post-hoc analysis",
        "limitation": "query evaluation population is annotation/depth-valid; official validation environments have been explored and cross-split repeated-content signals are documented, so this is not independent-dataset validation",
        "summaries": summary.to_dict(orient="records"),
        "candidate_availability_summaries": availability_summary.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Label-free similarity abstention on official 3RScan validation",
        "",
        "This post-hoc analysis reuses the frozen RGB-D ViT-G run for all 110 official validation relations across 47 environments, using geometry weight 0 and top-20 candidates. It compares two label-free calibration rules: a fixed numeric threshold calibrated from the 16 source/train relations, and a leave-one-environment-out threshold calibrated from the other 46 validation environments. The query-score calibration pools are annotation/depth-valid. The 3RScan validation split and repeated-content signals have been explored, so neither analysis is independent-dataset validation.",
        "",
        "Physical identity is the final RANSAC-selected reference object identity, evaluated only where the official cross-scan object link is defined (98,221 queries). The primary comparison applies the score gate to this final assignment without changing candidate ranking or RANSAC. A second column asks whether the score adds selectivity after the existing consensus-inlier decision. Residual is included as a geometry-based comparison; it is not a learned confidence score.",
        "",
        "| Calibration | Signal | Train acceptance | Held-out physical coverage | Physical precision baseline → accepted | Delta pp (95% env bootstrap CI) | Env wins | Consensus-inlier precision baseline → accepted | Inlier delta pp (95% CI) | Physical AUC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        if row["train_acceptance_target"] != 0.25:
            continue
        lines.append(
            f"| {row['calibration_protocol']} | {row['signal']} | 25% | {100*row['mean_environment_physical_coverage']:.1f}% | "
            f"{100*row['mean_environment_physical_precision_baseline']:.1f}% → {100*row['mean_environment_physical_precision_accepted']:.1f}% | "
            f"{100*row['physical_precision_delta']:+.1f} [{100*row['physical_delta_ci_low']:+.1f},{100*row['physical_delta_ci_high']:+.1f}] | "
            f"{row['environments_with_positive_physical_delta']}/{row['defined_physical_deltas']} | "
            f"{100*row['mean_environment_inlier_precision_baseline']:.1f}% → {100*row['mean_environment_inlier_precision_accepted']:.1f}% | "
            f"{100*row['inlier_precision_delta']:+.1f} [{100*row['inlier_delta_ci_low']:+.1f},{100*row['inlier_delta_ci_high']:+.1f}] | "
            f"{row['mean_environment_physical_auc']:.3f} [{row['physical_auc_ci_low']:.3f},{row['physical_auc_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "For `source_train_fixed`, the 25% target uses the 75th percentile of scores in the 16 training/source relations, and that numerical threshold is then fixed for every validation environment; no validation score or label calibrates it. For `leave_one_environment_out`, the same target quantile is recomputed from the other 46 environments. Realized physical-query coverage differs from 25% due distribution shift and the physically evaluable subset. Environment-bootstrap intervals are descriptive; this is not a preregistered confirmation.",
            "",
            "Full risk/coverage operating points: `summary.csv`. Environment details: `environment_metrics.csv`. Cross-fitted query decisions: `crossfit_predictions.csv.gz`.",
            "",
            "## Candidate availability stratification",
            "",
            "This annotation-defined post-hoc split checks whether the source-fitted top-score gate only detects that a correct object exists in the candidate set. It does not change threshold selection.",
            "",
            "| Candidate in top-20 | Queries | Accepted | Mean site coverage | Baseline precision | Accepted precision | Delta pp (95% env bootstrap CI) | Site wins |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in availability_summary.to_dict(orient="records"):
        if row["calibration_protocol"] != "source_train_fixed" or row["signal"] != "DINO top-1 similarity" or row["train_acceptance_target"] != 0.25:
            continue
        lines.append(
            f"| {'yes' if row['candidate_available_in_top20'] else 'no'} | {row['query_count']:,} | {row['accepted_query_count']:,} | "
            f"{100*row['mean_environment_coverage']:.1f}% | {100*row['mean_environment_baseline_precision']:.1f}% | "
            f"{100*row['mean_environment_accepted_precision']:.1f}% | "
            f"{100*row['precision_delta']:+.1f} [{100*row['precision_delta_ci_low']:+.1f},{100*row['precision_delta_ci_high']:+.1f}] | "
            f"{row['environments_with_positive_delta']}/{row['defined_precision_deltas']} |"
        )
    lines.extend(
        [
            "",
            "The gate still accepts physically incorrect assignments when the correct object has no top-20 candidate. Candidate availability is annotation-defined and serves as a mechanism diagnostic only.",
            "",
            "Candidate availability environment metrics: `candidate_availability_metrics.csv`; summaries: `candidate_availability_summary.csv`.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
