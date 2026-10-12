#!/usr/bin/env python3
"""Aggregate fixed-seed FCGF / ViT-G official validation comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_3rscan_validation import json_safe, rows as read_rows, score_diagnostics, summarize, truth


def one_row(path: Path, seed: int | None = None) -> dict[str, str]:
    candidates = read_rows(path)
    candidates = [r for r in candidates if int(float(r.get("top_k", -1))) == 20]
    if seed is not None:
        candidates = [r for r in candidates if int(r["ransac_seed"]) == seed]
    if len(candidates) != 1:
        raise ValueError(f"expected one top-20 result for seed {seed} at {path}; found {len(candidates)}")
    return candidates[0]


def seed_rows(run_dir: Path, vitg_dir: Path, manifest: list[dict[str, str]], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    exact17_checked = 0
    fcgf_query_counts: dict[str, int] = {}
    vitg_query_counts: dict[str, int] = {}
    for index, pair in enumerate(manifest):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        relation = run_dir / "relations" / run_id
        relation_status = json.loads((relation / "relation_status.json").read_text(encoding="utf-8"))
        for key in ("environment_id", "reference_scan", "rescan_scan"):
            if relation_status.get(key) != pair[key]:
                raise ValueError(f"FCGF relation identity mismatch for {run_id}: {key}")
        fcgf_path = relation / "fcgf"
        fcgf_summary = one_row(fcgf_path / "summary.csv", seed)
        fcgf_diag_path = fcgf_path / f"query_diagnostics_k20_seed{seed}.csv"
        fcgf_diag_rows = read_rows(fcgf_diag_path)
        fcgf_diag = score_diagnostics(fcgf_diag_path, physical=False)
        if len(fcgf_diag_rows) != int(fcgf_summary["matching_query_points"]):
            raise ValueError(f"FCGF query diagnostic length disagrees with summary for {run_id}, seed {seed}")
        fcgf_query_counts[run_id] = fcgf_diag["queries"]

        vitg_relation = vitg_dir / "relations" / run_id
        vitg_status = json.loads((vitg_relation / "vitg" / "status.json").read_text(encoding="utf-8"))
        if vitg_status.get("status") != "complete" or seed not in vitg_status.get("ransac_seeds", []):
            raise ValueError(f"missing ViT-G seed {seed} for {run_id}")
        vitg_summary = one_row(vitg_relation / "vitg" / "summary.csv", seed)
        vitg_diag_path = vitg_relation / "vitg" / "query_diagnostics.csv"
        vitg_diag_rows = [r for r in read_rows(vitg_diag_path) if int(r["ransac_seed"]) == seed]
        vitg_diag = score_diagnostics_from_rows(vitg_diag_rows)
        vitg_query_counts[run_id] = vitg_diag["queries"]
        if len(vitg_diag_rows) != int(vitg_summary["matching_query_points"]):
            raise ValueError(f"ViT-G query diagnostic length disagrees with summary for {run_id}, seed {seed}")
        if fcgf_diag["queries"] != int(fcgf_summary["query_queries"]):
            raise ValueError(f"FCGF query denominator disagrees with summary for {run_id}, seed {seed}")
        if vitg_diag["queries"] != int(vitg_summary["queries"]):
            raise ValueError(f"ViT-G query denominator disagrees with summary for {run_id}, seed {seed}")

        if seed == 17:
            old_dir = Path("outputs/reports/20261003_vitg_validation/relations") / run_id / "fcgf"
            old_summary = one_row(old_dir / "summary.csv")
            old_diag_rows = read_rows(old_dir / "query_diagnostics_k20.csv")
            fields = ("query_queries", "matching_query_points", "matching_inliers", "inliers")
            for field in fields:
                if int(float(fcgf_summary[field])) != int(float(old_summary[field])):
                    raise ValueError(f"seed-17 FCGF mismatch for {run_id} field {field}")
            for field in ("rotation_error_deg", "translation_error_m"):
                if not np.isclose(float(fcgf_summary[field]), float(old_summary[field]), rtol=0, atol=1e-10):
                    raise ValueError(f"seed-17 FCGF mismatch for {run_id} field {field}")
            if fcgf_summary["pose_success_lt5deg_lt025m"] != old_summary["pose_success_lt5deg_lt025m"]:
                raise ValueError(f"seed-17 FCGF pose-pass mismatch for {run_id}")
            columns = (
                "query_semantic_id", "selected_reference_semantic_id", "query_instance_id",
                "selected_reference_instance_id", "query_evaluable", "consensus_inlier",
            )
            old_identity = [tuple(r[k] for k in columns) for r in old_diag_rows]
            new_identity = [tuple(r[k] for k in columns) for r in fcgf_diag_rows]
            if new_identity != old_identity:
                raise ValueError(f"seed-17 ordered FCGF query/selection diagnostics changed for {run_id}")
            exact17_checked += 1

        for arm, summary_row, diagnostics in (
            ("FCGF", fcgf_summary, fcgf_diag),
            ("ViT-G", vitg_summary, vitg_diag),
        ):
            queries = int(diagnostics["queries"])
            inliers = int(diagnostics["inliers"])
            correct = int(diagnostics["semantic_correct_inliers"])
            paired.append({
                "environment_id": pair["environment_id"], "reference_scan": pair["reference_scan"],
                "rescan_scan": pair["rescan_scan"], "relation_index": int(pair["relation_index"]),
                "arm": arm, "seed": seed, "queries": queries, "inliers": inliers,
                "semantic_correct_inliers": correct,
                "semantic_inlier_coverage": correct / queries if queries else float("nan"),
                "semantic_precision_among_inliers": correct / inliers if inliers else float("nan"),
                "inlier_coverage": inliers / queries if queries else float("nan"),
                "strict_pose_success": (
                    float(summary_row["rotation_error_deg"]) < 5.0
                    and float(summary_row["translation_error_m"]) < 0.25
                ),
                "rotation_error_deg": float(summary_row["rotation_error_deg"]),
                "translation_error_m": float(summary_row["translation_error_m"]),
            })

    if len(set(fcgf_query_counts.values())) < 1 or len(set(vitg_query_counts.values())) < 1:
        raise ValueError("empty per-relation query counts")
    if exact17_checked not in (0, 110):
        raise ValueError(f"seed-17 integrity checks covered {exact17_checked} relations, expected all 110")
    return paired, {
        "relations": len(manifest), "environments": len({r["environment_id"] for r in manifest}),
        "seed17_exact_relations_checked": exact17_checked,
        "fcgf_total_queries": sum(fcgf_query_counts.values()),
        "vitg_total_queries": sum(vitg_query_counts.values()),
        "fcgf_query_denominator_variants_across_relations": len(set(fcgf_query_counts.values())),
        "vitg_query_denominator_variants_across_relations": len(set(vitg_query_counts.values())),
        "fcgf_query_counts_by_relation": fcgf_query_counts,
        "vitg_query_counts_by_relation": vitg_query_counts,
    }


def score_diagnostics_from_rows(rows: list[dict[str, str]]) -> dict[str, int]:
    if not rows:
        raise ValueError("empty ViT-G per-seed query diagnostic")
    mask = np.asarray([truth(r["query_evaluable"]) for r in rows], dtype=bool)
    inlier = np.asarray([truth(r["consensus_inlier"]) for r in rows], dtype=bool)
    correct = np.asarray([
        int(float(r["query_semantic_id"])) == int(float(r["selected_reference_semantic_id"]))
        for r in rows
    ], dtype=bool)
    return {
        "queries": int(mask.sum()), "inliers": int((mask & inlier).sum()),
        "semantic_correct_inliers": int((mask & inlier & correct).sum()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vitg-run-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    manifest = read_rows(args.pair_manifest)
    if len(manifest) != 110 or len({r["environment_id"] for r in manifest}) != 47:
        raise ValueError("expected all 110 official validation relations and 47 environments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_seed: list[dict[str, Any]] = []
    report = [
        "# Post-hoc FCGF / ViT-G validation seed sensitivity", "",
        "This reruns the frozen FCGF top-20 procedure at seeds 17, 29, and 41 and compares each seed with the existing ViT-G run at the same seed.",
        "The comparison is post-hoc and procedure-level: query populations and candidate rankings differ, and seed matching does not create shared random draws.",
        "FCGF retains its frozen 3,000 iterations; ViT-G retains 1,500.", "",
        "| Seed | FCGF macro coverage | ViT-G macro coverage | ViT-G minus FCGF (95% environment-bootstrap CI) | Wins/ties/losses | FCGF / ViT-G strict pose |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    integrity: dict[str, Any] = {}
    reference_query_counts: dict[str, dict[str, int] | None] = {"FCGF": None, "ViT-G": None}
    for seed in args.seeds:
        relation_rows, checks = seed_rows(args.run_dir, args.vitg_run_dir, manifest, seed)
        for method, key in (
            ("FCGF", "fcgf_query_counts_by_relation"),
            ("ViT-G", "vitg_query_counts_by_relation"),
        ):
            counts = checks.pop(key)
            if reference_query_counts[method] is None:
                reference_query_counts[method] = counts
            elif reference_query_counts[method] != counts:
                raise ValueError(f"{method} query denominators changed across RANSAC seeds")
        checks["query_denominators_constant_across_seeds"] = True
        integrity[str(seed)] = checks
        # Reuse the same environment-resampling stream for all three seed
        # settings so their interval widths are not confounded by RNG changes.
        summary, env_rows = summarize(relation_rows, args.bootstrap_samples, 10_042_026)
        delta = summary["paired_environment_macro_deltas"]["ViT-G_minus_FCGF:semantic_inlier_coverage"]
        arm = summary["arms"]
        vitg_by_env = {r["environment_id"]: r for r in env_rows}
        wins = ties = losses = 0
        for row in env_rows:
            d = row["ViT-G_semantic_inlier_coverage_env"] - row["FCGF_semantic_inlier_coverage_env"]
            wins += d > 1e-12
            losses += d < -1e-12
            ties += abs(d) <= 1e-12
        f_cov = arm["FCGF"]["environment_macro_rates_ci95"]["semantic_inlier_coverage"]["estimate"]
        v_cov = arm["ViT-G"]["environment_macro_rates_ci95"]["semantic_inlier_coverage"]["estimate"]
        f_pose = sum(r["strict_pose_success"] for r in relation_rows if r["arm"] == "FCGF")
        v_pose = sum(r["strict_pose_success"] for r in relation_rows if r["arm"] == "ViT-G")
        by_seed.append({
            "seed": seed, "relations": 110, "environments": 47,
            "fcgf_environment_macro_semantic_coverage": f_cov,
            "vitg_environment_macro_semantic_coverage": v_cov,
            "vitg_minus_fcgf_difference": delta["estimate"],
            "ci_lower": delta["ci95"][0], "ci_upper": delta["ci95"][1],
            "fcgf_strict_pose_successes": f_pose, "vitg_strict_pose_successes": v_pose,
            "fcgf_pooled_query_count": arm["FCGF"]["counts"]["queries"],
            "vitg_pooled_query_count": arm["ViT-G"]["counts"]["queries"],
            "wins": wins, "ties": ties, "losses": losses,
        })
        report.append(
            f"| {seed} | {f_cov:.1%} | {v_cov:.1%} | {delta['estimate']*100:+.1f} pp "
            f"[{delta['ci95'][0]*100:+.1f}, {delta['ci95'][1]*100:+.1f}] | {wins}/{ties}/{losses} | {f_pose}/110 / {v_pose}/110 |"
        )
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_csv(seed_dir / "per_relation.csv", relation_rows)
        write_csv(seed_dir / "per_environment.csv", env_rows)
        (seed_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary_by_seed.csv", by_seed)
    (args.output_dir / "integrity_checks.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    report.extend([
        "", "Seed 17 must reproduce all original FCGF relation-level counts, pose outputs, and ordered query/selection diagnostics; the machine-readable integrity summary records the check.",
        "Bootstrap intervals resample the 47 official environments, keeping each environment's relations together. They quantify variation over this fixed split, not independent-site generalization.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
