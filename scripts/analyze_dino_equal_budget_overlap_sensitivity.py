#!/usr/bin/env python3
"""Reaggregate the exact-budget DINO/REMIND audit after a frozen overlap screen.

This excludes the ten validation reference UUIDs identified by the prior
geometry-only cross-split content screen. It reads the already frozen,
label-blind exact-budget per-group outcomes and does not alter either matcher.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXACT = ROOT / "outputs/reports/20261009_dino_equal_budget_identity_availability/summary.json"
DEFAULT_SCREEN = ROOT / "outputs/reports/20261009_dino_source_gate_overlap_sensitivity/summary.json"
DEFAULT_OUTPUT = ROOT / "outputs/reports/20261009_dino_equal_budget_overlap_sensitivity"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-summary", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--screen-summary", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()

    exact = json.loads(args.exact_summary.read_text())
    screen = json.loads(args.screen_summary.read_text())
    excluded = set(screen["cross_split_geometry_screen"]["candidate_validation_reference_uuids"])
    rows = exact["reference_group_rows"]
    present = {r["environment_id"] for r in rows}
    missing = excluded - present
    if missing:
        raise ValueError(f"screen UUIDs absent from exact-budget cohort: {sorted(missing)}")
    retained = [r for r in rows if r["environment_id"] not in excluded]
    if len(retained) != len(rows) - len(excluded):
        raise AssertionError("overlap exclusion count mismatch")

    metrics = {
        "precision_equal_budget": ("dino_precision", "remind_precision"),
        "coverage_equal_budget": ("dino_coverage", "remind_coverage"),
        "precision_when_identity_available": ("dino_precision_available", "remind_precision_available"),
        "coverage_when_identity_available": ("dino_coverage_available", "remind_coverage_available"),
        "false_accept_rate_when_identity_absent": ("dino_absent_false_accept_rate", "remind_absent_false_accept_rate"),
    }
    rng = np.random.default_rng(args.seed)
    effects = {}
    for name, (dk, rk) in metrics.items():
        pairs = [(r[dk], r[rk]) for r in retained if r[dk] is not None and r[rk] is not None]
        d = np.asarray([x[0] for x in pairs], dtype=float)
        t = np.asarray([x[1] for x in pairs], dtype=float)
        delta = d - t
        sample = rng.integers(0, len(delta), size=(args.draws, len(delta)))
        draws = delta[sample].mean(axis=1)
        effects[name] = {
            "dino_macro": float(d.mean()),
            "remind_macro": float(t.mean()),
            "difference_dino_minus_remind": float(delta.mean()),
            "groups": len(delta),
            "groups_dino_higher": int(np.sum(delta > 1e-12)),
            "groups_remind_higher": int(np.sum(delta < -1e-12)),
            "groups_tied": int(np.sum(np.abs(delta) <= 1e-12)),
            "paired_group_bootstrap_95": [float(x) for x in np.quantile(draws, [.025, .975])],
            "bonferroni_5_outcomes_percentile_95": [float(x) for x in np.quantile(draws, [.005, .995])],
        }

    decomposition_by_id = {
        r["environment_id"]: r for r in exact["equal_budget_correct_coverage_decomposition_group_rows"]
    }
    decomposition = {}
    components = [
        ("one_sided_acceptance_set_effect", "different accepted-query sets"),
        ("shared_accepted_assignment_effect", "different assignments on shared accepts"),
    ]
    for key, label in components:
        vals = np.asarray([decomposition_by_id[r["environment_id"]][key] for r in retained], dtype=float)
        sample = rng.integers(0, len(vals), size=(args.draws, len(vals)))
        draws = vals[sample].mean(axis=1)
        decomposition[key] = {
            "description": label,
            "mean": float(vals.mean()),
            "groups": len(vals),
            "bonferroni_2_components_percentile_95": [float(x) for x in np.quantile(draws, [.0125, .9875])],
        }
    if not np.isclose(sum(x["mean"] for x in decomposition.values()),
                      effects["coverage_equal_budget"]["difference_dino_minus_remind"], atol=1e-12):
        raise AssertionError("component decomposition does not sum to coverage effect")

    totals = {key: sum(r[key] for r in retained) for key in (
        "query_masks", "available_queries", "absent_queries", "dino_accepted", "remind_accepted",
        "dino_correct", "remind_correct", "dino_available_accepts", "remind_available_accepts",
        "dino_absent_accepts", "remind_absent_accepts")}

    def portable(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(ROOT).as_posix()
        except ValueError:
            return str(resolved)

    result = {
        "status": "complete",
        "scope": "post-hoc sensitivity of exact per-relation DINO/REMIND acceptance-budget outcomes to prior geometry-based cross-split overlap candidates",
        "input_exact_budget_summary": portable(args.exact_summary),
        "input_overlap_screen_summary": portable(args.screen_summary),
        "excluded_reference_uuids": sorted(excluded),
        "excluded_count": len(excluded),
        "retained_count": len(retained),
        "retained_reference_uuids": sorted(r["environment_id"] for r in retained),
        "pooled_retained_counts": totals,
        "effects": effects,
        "coverage_decomposition": decomposition,
        "bootstrap": {"draws": args.draws, "seed": args.seed, "unit": "paired released reference-scan UUID",
                      "family_correction": "Bonferroni across five metric outcomes; two components for decomposition"},
        "limits": [
            "The overlap candidates were selected post hoc by geometry-only FPFH/FGR/ICP from source references; they are not certified physical-site duplicates.",
            "This reaggregates saved exact-budget predictions; it does not refit or rerun DINO or the REMIND-style tracker.",
            "Excluding candidates does not prove that retained validation groups are independent of training content.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    names = {
        "precision_equal_budget": "Macro accepted-link precision",
        "coverage_equal_budget": "Macro correct-link coverage",
        "precision_when_identity_available": "Precision where target ID is source-available",
        "coverage_when_identity_available": "Correct coverage where target ID is source-available",
        "false_accept_rate_when_identity_absent": "False-accept rate where target ID is absent",
    }
    lines = [
        "# Exact-budget DINO/REMIND overlap sensitivity", "",
        f"Excluded the {len(excluded)} validation reference groups selected by the prior geometry-only cross-split content screen; {len(retained)} groups remain. The exact label-blind per-relation acceptance counts are unchanged within retained relations.",
        "", "| Outcome | DINO | REMIND | DINO - REMIND (5-outcome-adjusted 95% interval) |",
        "|---|---:|---:|---:|",
    ]
    for key, label in names.items():
        e = effects[key]
        ci = e["bonferroni_5_outcomes_percentile_95"]
        lines.append(f"| {label} | {e['dino_macro']:.1%} | {e['remind_macro']:.1%} | {100*e['difference_dino_minus_remind']:+.1f} pp [{100*ci[0]:+.1f},{100*ci[1]:+.1f}] |")
    lines += ["", "The correct-coverage difference decomposes into:"]
    for x in decomposition.values():
        ci = x["bonferroni_2_components_percentile_95"]
        lines.append(f"- {x['description']}: {100*x['mean']:+.2f} pp [{100*ci[0]:+.2f},{100*ci[1]:+.2f}].")
    lines += ["", "This is a post-hoc sensitivity analysis, not independent-site validation. The geometry screen can miss related content and can flag false matches; the remaining groups are not certified independent.", ""]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"groups": len(retained), "effects": {k: v["difference_dino_minus_remind"] for k,v in effects.items()}, "decomposition": {k:v['mean'] for k,v in decomposition.items()}}, indent=2))


if __name__ == "__main__":
    main()
