#!/usr/bin/env python3
"""Aggregate the frozen official 3RScan validation run by environment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes"}


def selected_row(path: Path, *, method_column: str | None = None, method: str | None = None) -> dict[str, str]:
    candidates = rows(path)
    if method_column is not None:
        candidates = [row for row in candidates if row.get(method_column) == method]
    else:
        candidates = [row for row in candidates if int(float(row.get("top_k", -1))) == 20]
    if len(candidates) != 1:
        raise ValueError(f"expected one frozen top-20 result in {path}, found {len(candidates)}")
    return candidates[0]


def score_diagnostics(path: Path, *, physical: bool) -> dict[str, int]:
    diagnostic_rows = rows(path)
    if not diagnostic_rows:
        raise ValueError(f"empty query diagnostics: {path}")
    eval_mask = np.asarray([truth(row["query_evaluable"]) for row in diagnostic_rows], dtype=bool)
    inlier = np.asarray([truth(row["consensus_inlier"]) for row in diagnostic_rows], dtype=bool)
    semantic_correct = np.asarray([
        int(float(row["query_semantic_id"])) == int(float(row["selected_reference_semantic_id"]))
        for row in diagnostic_rows
    ], dtype=bool)
    result = {
        "queries": int(eval_mask.sum()),
        "inliers": int((eval_mask & inlier).sum()),
        "semantic_correct_inliers": int((eval_mask & inlier & semantic_correct).sum()),
    }
    if physical:
        physical_eval = np.asarray([
            truth(row.get("physical_instance_evaluable", "false")) for row in diagnostic_rows
        ], dtype=bool)
        physical_hit = np.asarray([
            truth(row.get("physical_instance_candidate_hit", "false")) for row in diagnostic_rows
        ], dtype=bool)
        physical_correct = np.asarray([
            truth(row.get("physical_instance_correct", "false")) for row in diagnostic_rows
        ], dtype=bool)
        result.update({
            "physical_queries": int(physical_eval.sum()),
            "physical_candidate_hits": int((physical_eval & physical_hit).sum()),
            "physical_inliers": int((physical_eval & inlier).sum()),
            "physical_correct_inliers": int((physical_eval & inlier & physical_correct).sum()),
        })
    return result


def make_relation_rows(output_dir: Path, manifest: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, pair in enumerate(manifest):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        root = output_dir / "relations" / run_id
        relation_status = json.loads((root / "protocol_relation.json").read_text(encoding="utf-8"))
        if any(relation_status.get(key) != pair[key] for key in ("environment_id", "reference_scan", "rescan_scan")):
            raise ValueError(f"relation identity mismatch for {run_id}")

        dino_summary = selected_row(root / "vitg" / "summary.csv")
        dino_diag = score_diagnostics(root / "vitg" / "query_diagnostics.csv", physical=True)
        arms = [("ViT-G", dino_summary, dino_diag)]

        for method in ("fpfh_one_way", "fpfh_mutual"):
            summary = selected_row(root / "fpfh" / "summary.csv", method_column="method", method=method)
            diagnostics = score_diagnostics(root / "fpfh" / f"query_diagnostics_{method}.csv", physical=False)
            # The mutual FPFH control may reject a query before RANSAC. Keep
            # its fixed source-evaluable population as the coverage denominator.
            diagnostics["queries"] = int(float(summary["evaluation_queries"]))
            arms.append(("FPFH one-way" if method == "fpfh_one_way" else "FPFH mutual", summary, diagnostics))

        fcgf_summary = selected_row(root / "fcgf" / "summary.csv")
        fcgf_diag = score_diagnostics(root / "fcgf" / "query_diagnostics_k20.csv", physical=False)
        arms.append(("FCGF", fcgf_summary, fcgf_diag))

        for arm, summary, diagnostics in arms:
            queries = int(diagnostics["queries"])
            inliers = int(diagnostics["inliers"])
            correct = int(diagnostics["semantic_correct_inliers"])
            rotation = float(summary["rotation_error_deg"])
            translation = float(summary["translation_error_m"])
            row: dict[str, Any] = {
                "environment_id": pair["environment_id"],
                "reference_scan": pair["reference_scan"],
                "rescan_scan": pair["rescan_scan"],
                "relation_index": int(pair["relation_index"]),
                "has_rigid_transfer": truth(pair["has_rigid_transfer"]),
                "arm": arm,
                "queries": queries,
                "inliers": inliers,
                "semantic_correct_inliers": correct,
                "semantic_inlier_coverage": correct / queries if queries else float("nan"),
                "semantic_precision_among_inliers": correct / inliers if inliers else float("nan"),
                "inlier_coverage": inliers / queries if queries else float("nan"),
                "rotation_error_deg": rotation,
                "translation_error_m": translation,
                "strict_pose_success": rotation < 5.0 and translation < 0.25,
            }
            if arm == "ViT-G":
                physical_queries = int(diagnostics["physical_queries"])
                physical_hits = int(diagnostics["physical_candidate_hits"])
                physical_inliers = int(diagnostics["physical_inliers"])
                physical_correct = int(diagnostics["physical_correct_inliers"])
                row.update({
                    "physical_queries": physical_queries,
                    "physical_candidate_hits": physical_hits,
                    "physical_candidate_recall": physical_hits / physical_queries if physical_queries else float("nan"),
                    "physical_inliers": physical_inliers,
                    "physical_correct_inliers": physical_correct,
                    "physical_inlier_coverage": physical_correct / physical_queries if physical_queries else float("nan"),
                    "physical_precision_among_inliers": physical_correct / physical_inliers if physical_inliers else float("nan"),
                })
            result.append(row)
    return result


def safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def json_safe(value: Any) -> Any:
    """Convert undefined CSV-side rates to JSON nulls, which JSON permits."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def percent(value: Any) -> str:
    """Render undefined rates explicitly instead of failing report creation."""
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.1%}"


def summarize(
    rows_by_env: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arms = sorted({row["arm"] for row in rows_by_env})
    envs = sorted({row["environment_id"] for row in rows_by_env})
    fields = ["semantic_inlier_coverage", "semantic_precision_among_inliers", "inlier_coverage", "strict_pose_success"]
    for key in ("physical_candidate_recall", "physical_inlier_coverage", "physical_precision_among_inliers"):
        if any(key in row for row in rows_by_env):
            fields.append(key)

    env_rows: list[dict[str, Any]] = []
    for env in envs:
        relations = [row for row in rows_by_env if row["environment_id"] == env]
        output: dict[str, Any] = {"environment_id": env, "relations": len({r["rescan_scan"] for r in relations})}
        for arm in arms:
            subset = [row for row in relations if row["arm"] == arm]
            query_count = sum(row["queries"] for row in subset)
            inlier_count = sum(row["inliers"] for row in subset)
            correct_count = sum(row["semantic_correct_inliers"] for row in subset)
            pose_count = sum(bool(row["strict_pose_success"]) for row in subset)
            output[f"{arm}_queries"] = query_count
            output[f"{arm}_semantic_correct_inliers"] = correct_count
            output[f"{arm}_inliers"] = inlier_count
            output[f"{arm}_strict_pose_successes"] = pose_count
            output[f"{arm}_semantic_inlier_coverage_env"] = correct_count / query_count if query_count else float("nan")
            output[f"{arm}_semantic_precision_among_inliers_env"] = correct_count / inlier_count if inlier_count else float("nan")
            output[f"{arm}_inlier_coverage_env"] = inlier_count / query_count if query_count else float("nan")
            output[f"{arm}_strict_pose_success_env"] = pose_count / len(subset) if subset else float("nan")
            if arm == "ViT-G":
                for count_key in ("physical_queries", "physical_candidate_hits", "physical_correct_inliers", "physical_inliers"):
                    output[f"{arm}_{count_key}"] = sum(int(row.get(count_key, 0)) for row in subset)
                physical_queries = output[f"{arm}_physical_queries"]
                physical_hits = output[f"{arm}_physical_candidate_hits"]
                physical_correct = output[f"{arm}_physical_correct_inliers"]
                physical_inliers = output[f"{arm}_physical_inliers"]
                output[f"{arm}_physical_candidate_recall_env"] = physical_hits / physical_queries if physical_queries else float("nan")
                output[f"{arm}_physical_inlier_coverage_env"] = physical_correct / physical_queries if physical_queries else float("nan")
                output[f"{arm}_physical_precision_among_inliers_env"] = physical_correct / physical_inliers if physical_inliers else float("nan")
        env_rows.append(output)

    summary: dict[str, Any] = {
        "environments": len(envs),
        "relations": len({row["rescan_scan"] for row in rows_by_env}),
        "bootstrap_unit": "official 3RScan environment; all rescans and relations from a sampled environment stay together",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "arms": {},
        "paired_environment_macro_deltas": {},
    }
    rng = np.random.default_rng(seed)
    by_env = {row["environment_id"]: row for row in env_rows}
    for arm in arms:
        arm_relations = [row for row in rows_by_env if row["arm"] == arm]
        counts = {
            "queries": sum(row["queries"] for row in arm_relations),
            "inliers": sum(row["inliers"] for row in arm_relations),
            "semantic_correct_inliers": sum(row["semantic_correct_inliers"] for row in arm_relations),
        }
        rates = {
            "semantic_inlier_coverage": counts["semantic_correct_inliers"] / counts["queries"] if counts["queries"] else float("nan"),
            "semantic_precision_among_inliers": counts["semantic_correct_inliers"] / counts["inliers"] if counts["inliers"] else float("nan"),
            "inlier_coverage": counts["inliers"] / counts["queries"] if counts["queries"] else float("nan"),
        }
        if arm == "ViT-G":
            physical_queries = sum(int(row.get("physical_queries", 0)) for row in arm_relations)
            physical_hits = sum(int(row.get("physical_candidate_hits", 0)) for row in arm_relations)
            physical_correct = sum(int(row.get("physical_correct_inliers", 0)) for row in arm_relations)
            physical_inliers = sum(int(row.get("physical_inliers", 0)) for row in arm_relations)
            rates.update({
                "physical_candidate_recall": physical_hits / physical_queries if physical_queries else float("nan"),
                "physical_inlier_coverage": physical_correct / physical_queries if physical_queries else float("nan"),
                "physical_precision_among_inliers": physical_correct / physical_inliers if physical_inliers else float("nan"),
            })
        pose_env_values = np.asarray([
            float(by_env[env][f"{arm}_strict_pose_success_env"])
            for env in envs if np.isfinite(by_env[env][f"{arm}_strict_pose_success_env"])
        ])
        rates["strict_pose_success_environment_macro"] = float(np.mean(pose_env_values)) if len(pose_env_values) else float("nan")
        environment_rates: dict[str, Any] = {}
        metric_to_env_field = {
            "semantic_inlier_coverage": f"{arm}_semantic_inlier_coverage_env",
            "semantic_precision_among_inliers": f"{arm}_semantic_precision_among_inliers_env",
            "inlier_coverage": f"{arm}_inlier_coverage_env",
            "strict_pose_success_environment_macro": f"{arm}_strict_pose_success_env",
        }
        if arm == "ViT-G":
            metric_to_env_field.update({
                "physical_candidate_recall": "ViT-G_physical_candidate_recall_env",
                "physical_inlier_coverage": "ViT-G_physical_inlier_coverage_env",
                "physical_precision_among_inliers": "ViT-G_physical_precision_among_inliers_env",
            })
        for metric, env_field in metric_to_env_field.items():
            values = np.asarray([
                float(by_env[env][env_field])
                for env in envs if np.isfinite(by_env[env].get(env_field, float("nan")))
            ], dtype=np.float64)
            if not len(values):
                environment_rates[metric] = {"estimate": None, "ci95": [None, None], "environments": 0}
                continue
            boot_indices = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
            boot_values = values[boot_indices].mean(axis=1)
            lo, hi = np.quantile(boot_values, [0.025, 0.975])
            environment_rates[metric] = {
                "estimate": float(values.mean()),
                "ci95": [float(lo), float(hi)],
                "environments": len(values),
            }
        summary["arms"][arm] = {
            "counts": counts,
            "pooled_rates": {key: safe_float(value) for key, value in rates.items()},
            "environment_macro_rates_ci95": environment_rates,
        }

    # Compare environmental means so every official environment, rather than
    # every query point, is the unit of inference. Method-specific query
    # denominators remain explicit for geometry-only controls.
    for baseline in ("FPFH one-way", "FPFH mutual", "FCGF"):
        for field in ("semantic_inlier_coverage", "semantic_precision_among_inliers", "inlier_coverage", "strict_pose_success"):
            deltas = []
            for env in envs:
                left = by_env[env].get(f"ViT-G_{field}_env", float("nan"))
                right = by_env[env].get(f"{baseline}_{field}_env", float("nan"))
                if np.isfinite(float(left)) and np.isfinite(float(right)):
                    deltas.append((env, float(left) - float(right)))
            if not deltas:
                continue
            delta_values = np.asarray([value for _, value in deltas], dtype=np.float64)
            delta_envs = [env for env, _ in deltas]
            draws = rng.integers(0, len(delta_values), size=(bootstrap_samples, len(delta_values)))
            boot = delta_values[draws].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            key = f"ViT-G_minus_{baseline}:{field}"
            summary["paired_environment_macro_deltas"][key] = {
                "estimate": float(delta_values.mean()),
                "ci95": [float(lo), float(hi)],
                "environments": len(delta_values),
            }
    summary["inference_note"] = (
        "Per-method query denominators differ for geometry baselines because their sampled point sets differ. "
        "Paired comparisons are environment-macro differences, not same-query paired effects."
    )
    return summary, env_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_022_026)
    args = parser.parse_args()
    manifest = rows(args.pair_manifest)
    if len(manifest) != 110 or len({row["environment_id"] for row in manifest}) != 47:
        raise ValueError("expected the complete official 3RScan validation manifest")
    relation_rows = make_relation_rows(args.run_dir, manifest)
    summary, environment_rows = summarize(relation_rows, args.bootstrap_samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_relation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(relation_rows[0]))
        writer.writeheader()
        writer.writerows(relation_rows)
    with (args.output_dir / "per_environment.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(environment_rows[0]))
        writer.writeheader()
        writer.writerows(environment_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Official 3RScan validation: frozen ViT-G and geometry controls",
        "",
        f"Scored {summary['relations']} reference/rescan relations in {summary['environments']} official validation environments. The bootstrap unit is the environment; all rescans from a sampled environment remain grouped.",
        "",
        "| Arm | Query denominator | Correct semantic inliers / query (pooled) | Semantic precision (pooled) | Coverage, environment macro (95% CI) | Strict pose successes by relation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, values in summary["arms"].items():
        rates = values["pooled_rates"]
        macro = values["environment_macro_rates_ci95"]["semantic_inlier_coverage"]
        successes = sum(bool(row["strict_pose_success"]) for row in relation_rows if row["arm"] == arm)
        denominator = sum(row["arm"] == arm for row in relation_rows)
        report.append(
            f"| {arm} | {values['counts']['queries']} | {percent(rates['semantic_inlier_coverage'])} | "
            f"{percent(rates['semantic_precision_among_inliers'])} | {percent(macro['estimate'])} "
            f"[{percent(macro['ci95'][0])}, {percent(macro['ci95'][1])}] | {successes}/{denominator} |"
        )
    report.extend(["", "## Paired environment-macro differences", "", "| Comparison | Difference | 95% environment bootstrap CI | Environments |", "|---|---:|---:|---:|"])
    for key, value in summary["paired_environment_macro_deltas"].items():
        report.append(f"| {key} | {percent(value['estimate'])} | [{percent(value['ci95'][0])}, {percent(value['ci95'][1])}] | {value['environments']} |")
    report.extend(["", summary["inference_note"], "", "## Physical-instance results (ViT-G)", ""])
    dino = summary["arms"]["ViT-G"]
    for metric, value in dino["pooled_rates"].items():
        if metric.startswith("physical_"):
            report.append(f"- {metric}: {percent(value)}")
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
