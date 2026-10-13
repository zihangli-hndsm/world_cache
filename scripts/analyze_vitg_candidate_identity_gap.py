#!/usr/bin/env python3
"""Decompose semantic-hit / physical-instance-hit gaps in cached top-20 sets.

This post-hoc audit reads the saved 110-relation query diagnostics. It cannot
recover candidate ranks or identities beyond the saved annotation counts, so
it characterizes same-class candidate-set ambiguity, not descriptor ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("outputs/reports/20261004_vitg_validation_candidate_audit")
DEFAULT_OUTPUT = Path("outputs/reports/20261004_vitg_candidate_identity_gap")
OVERLAP_FLAGGED = {"4fbad31e-465b-2a5d-84b7-c0ddea978db4", "c7895f7c-339c-2d13-819f-3bb0b26c91f6"}
BIN_EDGES = (0, 1, 5, 10, 15, 21)
BIN_LABELS = ("0", "1-4", "5-9", "10-14", "15-20")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[draws].mean(axis=1)
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_042_026)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")

    manifest = read_csv(args.input_dir / "run_manifest.csv")
    if len(manifest) != 110 or len({row["environment_id"] for row in manifest}) != 47:
        raise ValueError("expected the frozen 110-relation, 47-environment validation manifest")
    rows: list[dict[str, object]] = []
    for rel in manifest:
        if rel["status"] != "complete":
            raise ValueError(f"incomplete relation: {rel['run_id']}")
        path = Path(rel["output_dir"]) / "query_diagnostics.csv"
        diagnostics = read_csv(path)
        for item in diagnostics:
            if item["physical_instance_evaluable"].lower() != "true":
                continue
            semantic_count = int(item["candidate_semantic_count"])
            physical_count = int(item["physical_instance_candidate_count"])
            semantic_hit = item["candidate_semantic_hit"].lower() == "true"
            physical_hit = item["physical_instance_candidate_hit"].lower() == "true"
            if not 0 <= physical_count <= semantic_count <= 20:
                raise ValueError(f"candidate count invariant failed in {rel['run_id']}")
            if physical_hit != (physical_count > 0) or semantic_hit != (semantic_count > 0):
                raise ValueError(f"candidate hit/count mismatch in {rel['run_id']}")
            rows.append({
                "environment_id": rel["environment_id"],
                "semantic_count": semantic_count,
                "physical_count": physical_count,
                "semantic_hit": semantic_hit,
                "physical_hit": physical_hit,
                "semantic_only": semantic_hit and not physical_hit,
            })
    if len(rows) != 98_221:
        raise ValueError(f"expected 98,221 mapped query rows; got {len(rows)}")

    environments = sorted({str(row["environment_id"]) for row in rows})
    environment_rows: list[dict[str, object]] = []
    by_environment: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_environment[str(row["environment_id"])].append(row)
    for environment in environments:
        group = by_environment[environment]
        n = len(group)
        semantic = sum(bool(r["semantic_hit"]) for r in group)
        physical = sum(bool(r["physical_hit"]) for r in group)
        semantic_only = sum(bool(r["semantic_only"]) for r in group)
        environment_rows.append({
            "environment_id": environment,
            "relations": sum(1 for r in manifest if r["environment_id"] == environment),
            "mapped_queries": n,
            "semantic_hits": semantic,
            "physical_hits": physical,
            "semantic_only_hits": semantic_only,
            "semantic_recall": semantic / n,
            "physical_recall": physical / n,
            "physical_recall_given_semantic_hit": physical / semantic if semantic else float("nan"),
            "semantic_only_given_semantic_hit": semantic_only / semantic if semantic else float("nan"),
        })

    strata: list[dict[str, object]] = []
    for label, low, high in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:]):
        group = [r for r in rows if low <= int(r["semantic_count"]) < high]
        n = len(group)
        ph = sum(bool(r["physical_hit"]) for r in group)
        sh = sum(bool(r["semantic_hit"]) for r in group)
        so = sum(bool(r["semantic_only"]) for r in group)
        strata.append({
            "semantic_candidate_count_bin": label,
            "queries": n,
            "share_of_queries": n / len(rows),
            "semantic_hit_rate": sh / n if n else float("nan"),
            "physical_hit_rate": ph / n if n else float("nan"),
            "semantic_only_rate": so / n if n else float("nan"),
            "physical_rate_given_semantic_hit": ph / sh if sh else None,
            "mean_physical_candidate_count": float(np.mean([int(r["physical_count"]) for r in group])) if n else float("nan"),
        })

    def aggregate(envs: list[str]) -> dict[str, float | int]:
        chosen = [r for r in rows if r["environment_id"] in envs]
        n = len(chosen)
        semantic = sum(bool(r["semantic_hit"]) for r in chosen)
        physical = sum(bool(r["physical_hit"]) for r in chosen)
        semantic_only = sum(bool(r["semantic_only"]) for r in chosen)
        env_values = [r for r in environment_rows if r["environment_id"] in envs]
        cond_values = np.asarray([float(r["physical_recall_given_semantic_hit"]) for r in env_values if np.isfinite(float(r["physical_recall_given_semantic_hit"]))])
        loss_values = np.asarray([float(r["semantic_only_given_semantic_hit"]) for r in env_values if np.isfinite(float(r["semantic_only_given_semantic_hit"]))])
        return {
            "environments": len(env_values),
            "mapped_queries": n,
            "pooled_semantic_recall": semantic / n,
            "pooled_physical_recall": physical / n,
            "pooled_physical_recall_given_semantic_hit": physical / semantic,
            "pooled_semantic_only_given_semantic_hit": semantic_only / semantic,
            "environment_macro_physical_recall_given_semantic_hit": float(cond_values.mean()),
            "environment_macro_semantic_only_given_semantic_hit": float(loss_values.mean()),
            "environment_bootstrap_95ci_physical_recall_given_semantic_hit": bootstrap_mean(cond_values, args.bootstrap_samples, args.seed),
            "environment_bootstrap_95ci_semantic_only_given_semantic_hit": bootstrap_mean(loss_values, args.bootstrap_samples, args.seed + 1),
        }

    all_summary = aggregate(environments)
    clean_environments = [env for env in environments if env not in OVERLAP_FLAGGED]
    clean_summary = aggregate(clean_environments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_environment.csv", environment_rows)
    write_csv(args.output_dir / "semantic_candidate_count_strata.csv", strata)
    summary = {
        "status": "complete",
        "posthoc": True,
        "protocol": "saved top-20 candidate counts; identical 98,221-query mapped physical-instance denominator",
        "top_k": 20,
        "relations": len(manifest),
        "environments": len(environments),
        "query_pooled": all_summary,
        "excluding_two_previously_flagged_environments": {
            "excluded_environment_ids": sorted(OVERLAP_FLAGGED),
            **clean_summary,
        },
        "strata": strata,
        "bootstrap_unit": "environment, resampling environment-level conditional rates equally",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "limitations": [
            "post-hoc on the official validation split",
            "diagnostics retain candidate annotation counts but not candidate IDs or their full rank order",
            "candidate counts are patch counts, not distinct physical objects",
            "does not establish independent-site generalization or a causal descriptor effect",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Same-class candidate availability and physical-instance retrieval",
        "",
        "Post-hoc decomposition of the frozen top-20 ViT-G candidate sets on all 110 official 3RScan validation relations. The denominator is the same 98,221 queries with unambiguous mapped object IDs used in the semantic-versus-physical candidate audit.",
        "",
        f"- Semantic-category candidate hit: {all_summary['pooled_semantic_recall']:.1%} of queries.",
        f"- Linked physical-instance candidate hit: {all_summary['pooled_physical_recall']:.1%}.",
        f"- Among semantic-hit queries, physical-instance availability is {all_summary['pooled_physical_recall_given_semantic_hit']:.1%} pooled and {all_summary['environment_macro_physical_recall_given_semantic_hit']:.1%} environment-macro (95% environment-bootstrap CI {all_summary['environment_bootstrap_95ci_physical_recall_given_semantic_hit'][0]:.1%}–{all_summary['environment_bootstrap_95ci_physical_recall_given_semantic_hit'][1]:.1%}).",
        f"- Thus {all_summary['pooled_semantic_only_given_semantic_hit']:.1%} of semantic-hit query sets contain category-matching candidates but no patch from the linked physical instance; the environment-macro conditional rate is {all_summary['environment_macro_semantic_only_given_semantic_hit']:.1%} (95% CI {all_summary['environment_bootstrap_95ci_semantic_only_given_semantic_hit'][0]:.1%}–{all_summary['environment_bootstrap_95ci_semantic_only_given_semantic_hit'][1]:.1%}).",
        "",
        "## Candidate-count strata",
        "",
        "| Same-category candidate patches | Queries | Physical-instance hit | Physical hit given semantic hit |",
        "|---:|---:|---:|---:|",
    ]
    for row in strata:
        conditional = row["physical_rate_given_semantic_hit"]
        conditional_text = f"{conditional:.1%}" if conditional is not None else "—"
        lines.append(f"| {row['semantic_candidate_count_bin']} | {row['queries']} | {row['physical_hit_rate']:.1%} | {conditional_text} |")
    lines += [
        "",
        f"Removing the two previously flagged environments leaves {clean_summary['environments']} environments and {clean_summary['mapped_queries']} queries; physical-instance availability conditional on semantic hit is {clean_summary['pooled_physical_recall_given_semantic_hit']:.1%} pooled / {clean_summary['environment_macro_physical_recall_given_semantic_hit']:.1%} environment-macro.",
        "",
        "The saved diagnostics retain hit and candidate counts only. Candidate ranks and distinct candidate object IDs were not saved, so this audit cannot distinguish descriptor misranking from limited target-view visibility or repeated patches. Patch counts are not counts of distinct objects. All results remain post-hoc same-split characterization, not independent-site validation.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
