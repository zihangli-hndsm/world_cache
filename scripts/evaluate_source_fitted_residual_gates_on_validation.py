#!/usr/bin/env python3
"""Apply frozen source-fitted residual thresholds to all official val queries.

This is an offline transfer audit: candidate selection, spatial consensus,
annotations, and query populations are read from completed run artifacts. The
source thresholds are recovered from the saved 16-relation calibration data;
no validation labels are used to choose a threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_risk_controlled_calibration import fit_threshold, read as read_source


DEFAULT_INPUT = Path("outputs/reports/20261004_vitg_validation_candidate_audit")
DEFAULT_OUTPUT = Path("outputs/reports/20261004_vitg_source_gate_validation")
SOURCE_GLOB = "20260922_3rscan_consensus_*_vitg_diag_clean/query_diagnostics.csv"
SOURCE_TARGET_NAME = "20260922_3rscan_consensus_target_vitg_diag_clean"
EMPIRICAL_CALIBRATION = Path("outputs/runs/20260923_selective_calibration_vitg_clean_v2/leave_one_pair_out.csv")
RISK_THRESHOLD_EXPECTED = 0.0330402718601783
FLAGGED_ENVIRONMENTS = {"4fbad31e-465b-2a5d-84b7-c0ddea978db4", "c7895f7c-339c-2d13-819f-3bb0b26c91f6"}


def load_source_thresholds(root: Path) -> dict[str, dict[str, object]]:
    source_paths = sorted(
        p for p in (root / "outputs/runs").glob(SOURCE_GLOB)
        if p.parent.name != SOURCE_TARGET_NAME
    )
    if len(source_paths) != 16:
        raise ValueError(f"expected the 16 frozen source relations, found {len(source_paths)}")
    source = pd.concat([read_source(path) for path in source_paths], ignore_index=True)
    risk_fit = fit_threshold(source, precision_target=0.90, confidence_level=0.95, min_queries=100)
    if not np.isclose(float(risk_fit["threshold"]), RISK_THRESHOLD_EXPECTED, atol=1e-12):
        raise ValueError(f"source risk threshold changed: {risk_fit['threshold']}")

    calibration = pd.read_csv(root / EMPIRICAL_CALIBRATION)
    row = calibration[
        (calibration.held_pair == "target_source_only_fit")
        & (calibration.precision_target == 0.9)
        & (calibration.score == "low_spatial_residual")
    ]
    if len(row) != 1 or row.iloc[0].status != "calibrated":
        raise ValueError("could not uniquely recover the published empirical source-only gate")
    empirical_threshold = abs(float(row.iloc[0].threshold))
    return {
        "empirical90": {
            "threshold_m": empirical_threshold,
            "fit_artifact": str(EMPIRICAL_CALIBRATION),
            "fit_rule": "pooled source-only residual gate targeting 90% empirical semantic precision",
            "source_train_queries": int(row.iloc[0].train_queries),
            "source_train_coverage": float(row.iloc[0].train_coverage),
            "source_train_semantic_precision": float(row.iloc[0].train_semantic_precision),
        },
        "risk90": {
            "threshold_m": float(risk_fit["threshold"]),
            "fit_artifact": f"outputs/runs/{SOURCE_GLOB}",
            "fit_rule": "maximum source residual prefix with a raw one-sided exact-binomial 95% lower bound >=90%; post-selection bound is descriptive, not family-wise valid",
            "source_train_queries": int(risk_fit["queries"]),
            "source_train_coverage": float(risk_fit["coverage"]),
            "source_train_semantic_precision": float(risk_fit["semantic_precision"]),
            "source_train_raw_lower_bound": float(risk_fit["semantic_lower_bound"]),
        },
        "consensus10cm": {
            "threshold_m": 0.10,
            "fit_artifact": "frozen official validation protocol",
            "fit_rule": "original consensus inlier threshold; comparison baseline",
        },
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def metric_counts(rows: list[dict[str, str]], threshold: float) -> dict[str, int]:
    eligible = [r for r in rows if r["query_evaluable"].lower() == "true"]
    accepted = [r for r in eligible if float(r["spatial_residual_m"]) <= threshold]
    sem_correct = sum(r["semantic_correct"].lower() == "true" for r in accepted)
    phys_eligible = [r for r in eligible if r["physical_instance_evaluable"].lower() == "true"]
    phys_accepted = [r for r in phys_eligible if float(r["spatial_residual_m"]) <= threshold]
    phys_correct = sum(r["physical_instance_correct"].lower() == "true" for r in phys_accepted)
    return {
        "queries": len(eligible),
        "semantic_accepted": len(accepted),
        "semantic_correct_accepted": sem_correct,
        "physical_queries": len(phys_eligible),
        "physical_accepted": len(phys_accepted),
        "physical_correct_accepted": phys_correct,
    }


def ratios(counts: dict[str, int]) -> dict[str, float]:
    return {
        "semantic_coverage": counts["semantic_accepted"] / counts["queries"] if counts["queries"] else float("nan"),
        "semantic_correct_coverage": counts["semantic_correct_accepted"] / counts["queries"] if counts["queries"] else float("nan"),
        "semantic_precision": counts["semantic_correct_accepted"] / counts["semantic_accepted"] if counts["semantic_accepted"] else float("nan"),
        "physical_coverage": counts["physical_correct_accepted"] / counts["physical_queries"] if counts["physical_queries"] else float("nan"),
        "physical_precision": counts["physical_correct_accepted"] / counts["physical_accepted"] if counts["physical_accepted"] else float("nan"),
    }


def aggregate_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    return {key: sum(c[key] for c in counts) for key in counts[0]}


def macro_ratios(env_counts: dict[str, dict[str, int]]) -> dict[str, float]:
    per_env = [ratios(c) for c in env_counts.values()]
    return {
        key: float(np.nanmean([row[key] for row in per_env]))
        for key in per_env[0]
    }


def bootstrap_pooled_ratios(
    env_counts: dict[str, dict[str, int]],
    metrics: list[str],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    envs = sorted(env_counts)
    generator = np.random.default_rng(seed)
    output: dict[str, list[float]] = {metric: [] for metric in metrics}
    draws = generator.integers(0, len(envs), size=(samples, len(envs)))
    for draw in draws:
        totals = aggregate_counts([env_counts[envs[index]] for index in draw])
        current = ratios(totals)
        for metric in metrics:
            output[metric].append(current[metric])
    return {metric: [float(v) for v in np.quantile(values, [0.025, 0.975])] for metric, values in output.items()}


def bootstrap_macro_ratios(
    env_counts: dict[str, dict[str, int]],
    metrics: list[str],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    envs = sorted(env_counts)
    env_values = {env: ratios(env_counts[env]) for env in envs}
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(envs), size=(samples, len(envs)))
    result: dict[str, list[float]] = {metric: [] for metric in metrics}
    for draw in draws:
        for metric in metrics:
            result[metric].append(float(np.nanmean([env_values[envs[index]][metric] for index in draw])))
    return {metric: [float(v) for v in np.quantile(values, [0.025, 0.975])] for metric, values in result.items()}


def paired_macro_delta(
    left: dict[str, dict[str, int]],
    right: dict[str, dict[str, int]],
    metrics: list[str],
    samples: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    envs = sorted(set(left) & set(right))
    left_values = {env: ratios(left[env]) for env in envs}
    right_values = {env: ratios(right[env]) for env in envs}
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(envs), size=(samples, len(envs)))
    result: dict[str, dict[str, object]] = {}
    for metric in metrics:
        differences = np.asarray([left_values[env][metric] - right_values[env][metric] for env in envs], dtype=float)
        boot = np.asarray([np.nanmean(differences[draw]) for draw in draws])
        result[metric] = {
            "estimate": float(np.nanmean(differences)),
            "95ci": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_042_027)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    thresholds = load_source_thresholds(args.root)
    manifest = read_rows(args.input_dir / "run_manifest.csv")
    if len(manifest) != 110 or len({r["environment_id"] for r in manifest}) != 47:
        raise ValueError("expected 110 official validation relations across 47 environments")

    counts_by_gate_relation: dict[str, list[dict[str, object]]] = {name: [] for name in thresholds}
    counts_by_gate_env: dict[str, dict[str, dict[str, int]]] = {name: {} for name in thresholds}
    relation_deltas: list[dict[str, object]] = []
    for relation in manifest:
        if relation["status"] != "complete":
            raise ValueError(f"incomplete relation: {relation['run_id']}")
        rows = read_rows(Path(relation["output_dir"]) / "query_diagnostics.csv")
        environment = relation["environment_id"]
        relation_counts: dict[str, dict[str, int]] = {}
        for name, spec in thresholds.items():
            count = metric_counts(rows, float(spec["threshold_m"]))
            relation_counts[name] = count
            counts_by_gate_relation[name].append({
                "environment_id": environment,
                "run_id": relation["run_id"],
                **count,
                **ratios(count),
            })
            env_count = counts_by_gate_env[name].setdefault(environment, {key: 0 for key in count})
            for key, value in count.items():
                env_count[key] += value
        for gate in ("empirical90", "risk90"):
            relation_deltas.append({
                "environment_id": environment,
                "run_id": relation["run_id"],
                "gate": gate,
                "semantic_coverage_delta_vs_consensus": ratios(relation_counts[gate])["semantic_coverage"] - ratios(relation_counts["consensus10cm"])["semantic_coverage"],
                "semantic_precision_delta_vs_consensus": ratios(relation_counts[gate])["semantic_precision"] - ratios(relation_counts["consensus10cm"])["semantic_precision"],
                "physical_coverage_delta_vs_consensus": ratios(relation_counts[gate])["physical_coverage"] - ratios(relation_counts["consensus10cm"])["physical_coverage"],
                "physical_precision_delta_vs_consensus": ratios(relation_counts[gate])["physical_precision"] - ratios(relation_counts["consensus10cm"])["physical_precision"],
            })

    # Aggregate each environment across all rescans before computing rates.
    summary: dict[str, object] = {
        "status": "complete",
        "posthoc_application": True,
        "thresholds": thresholds,
        "relations": len(manifest),
        "environments": len(counts_by_gate_env["consensus10cm"]),
        "bootstrap": {
            "unit": "environment; resample 47 environment clusters with replacement",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "interval": "percentile 95% CI for pooled ratios after environment-cluster resampling",
        },
        "gates": {},
        "paired_environment_macro_changes_vs_consensus": {},
        "excluding_two_previously_flagged_environments": {},
        "limitations": [
            "threshold fitting used the original 16 source relations; validation annotations did not select the threshold",
            "the analysis itself is post-hoc on a validation split whose aggregate outcomes were already inspected",
            "residual gating does not validate physical-site independence or ensure a 90% risk guarantee",
            "physical metrics are conditional on the 98,221 unambiguous tracked-object queries where applicable",
        ],
    }
    relation_rows: list[dict[str, object]] = []
    environment_rows: list[dict[str, object]] = []
    for gate, env_counts in counts_by_gate_env.items():
        all_counts = aggregate_counts(list(env_counts.values()))
        all_rates = ratios(all_counts)
        ci = bootstrap_pooled_ratios(env_counts, list(all_rates), args.bootstrap_samples, args.seed)
        macro = macro_ratios(env_counts)
        macro_ci = bootstrap_macro_ratios(env_counts, list(macro), args.bootstrap_samples, args.seed + 2)
        summary["gates"][gate] = {
            **all_counts,
            **all_rates,
            "cluster_bootstrap_95ci": ci,
            "environment_macro": macro,
            "environment_macro_cluster_bootstrap_95ci": macro_ci,
        }
        env_ids = [env for env in sorted(env_counts) if env not in FLAGGED_ENVIRONMENTS]
        clean_counts = aggregate_counts([env_counts[env] for env in env_ids])
        clean_ci = bootstrap_pooled_ratios({env: env_counts[env] for env in env_ids}, list(ratios(clean_counts)), args.bootstrap_samples, args.seed)
        summary["excluding_two_previously_flagged_environments"][gate] = {
            "excluded_environment_ids": sorted(FLAGGED_ENVIRONMENTS),
            "environments": len(env_ids),
            **clean_counts,
            **ratios(clean_counts),
            "cluster_bootstrap_95ci": clean_ci,
        }
        for env, count in sorted(env_counts.items()):
            environment_rows.append({"gate": gate, "environment_id": env, **count, **ratios(count)})
        for row in counts_by_gate_relation[gate]:
            relation_rows.append({"gate": gate, **row})
    delta_metrics = ["semantic_coverage", "semantic_correct_coverage", "semantic_precision", "physical_coverage", "physical_precision"]
    for gate, offset in (("empirical90", 3), ("risk90", 4)):
        summary["paired_environment_macro_changes_vs_consensus"][gate] = paired_macro_delta(
            counts_by_gate_env[gate], counts_by_gate_env["consensus10cm"], delta_metrics,
            args.bootstrap_samples, args.seed + offset,
        )
    write_csv_path = args.output_dir
    write_csv_path.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("per_relation.csv", relation_rows),
        ("per_environment.csv", environment_rows),
        ("paired_relation_deltas.csv", relation_deltas),
    ):
        with (write_csv_path / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (write_csv_path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# Source-fitted residual gates on the complete 3RScan validation split",
        "",
        "The residual thresholds below were fit using only the original 16 source relations, then applied unchanged to saved ViT-G diagnostics for 110 validation relations / 47 environments. No validation labels or query outcomes changed either threshold. This offline transfer audit follows prior inspection of aggregate validation results, so it is a post-hoc application rather than a pre-registered test.",
        "",
        "| Gate | Threshold (m) | Accepted / queries | Semantic coverage pooled / macro | Semantic precision pooled / macro (macro 95% CI) | Physical coverage pooled / macro* | Physical precision pooled / macro* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for gate, metrics in summary["gates"].items():
        spec = thresholds[gate]
        lines.append(
            f"| {gate} | {spec['threshold_m']:.8f} | {metrics['semantic_accepted']:,} / {metrics['queries']:,} | "
            f"{metrics['semantic_coverage']:.1%} / {metrics['environment_macro']['semantic_coverage']:.1%} | "
            f"{metrics['semantic_precision']:.1%} / {metrics['environment_macro']['semantic_precision']:.1%} "
            f"({metrics['environment_macro_cluster_bootstrap_95ci']['semantic_precision'][0]:.1%}–{metrics['environment_macro_cluster_bootstrap_95ci']['semantic_precision'][1]:.1%}) | "
            f"{metrics['physical_coverage']:.1%} / {metrics['environment_macro']['physical_coverage']:.1%} | "
            f"{metrics['physical_precision']:.1%} / {metrics['environment_macro']['physical_precision']:.1%}"
        )
    lines += [
        "",
        "*Physical-instance coverage and precision are conditional on queries with an unambiguous tracked-object mapping; all candidate and consensus choices were label-free, and object labels are only used for scoring.",
        "",
        "The consensus gate is the original 0.10 m inlier threshold. The empirical90 gate targets 90% point precision on source data; the risk90 gate's source threshold was selected to meet a raw one-sided 95% exact-binomial lower bound, but the prefix search is post-selected and not family-wise valid. Neither is a finite-sample deployment guarantee.",
        "",
        "Paired environment-macro changes versus the 0.10 m consensus gate are in `summary.json`; their intervals resample the same 47 environments for both rules.",
        "",
        "Removing the two previously flagged target environments leaves 45 environments; sensitivity results are recorded in `summary.json`. These exclusions are post-hoc. The full results include pooled and environment-cluster bootstrap intervals, per-environment/per-relation counts, and paired changes versus consensus.",
    ]
    (write_csv_path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["gates"], indent=2))
if __name__ == "__main__":
    main()
