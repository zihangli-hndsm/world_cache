#!/usr/bin/env python3
"""Sensitivity of the source-frozen DINO/REMIND contrast to train-val overlap screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, fit_threshold, load_split


def summarize(dino_relations: list[dict], remind_groups: dict[str, dict], excluded: set[str], threshold: float,
              draws: int, seed: int) -> dict:
    per_group: dict[str, dict] = {}
    for rel in dino_relations:
        env = rel["environment_id"]
        if env in excluded:
            continue
        accepted = rel["valid"] & (rel["scores"] >= threshold)
        correct = accepted & rel["correct"]
        g = per_group.setdefault(env, {"query_masks": 0, "dino_accepted": 0, "dino_correct": 0})
        g["query_masks"] += len(accepted)
        g["dino_accepted"] += int(accepted.sum())
        g["dino_correct"] += int(correct.sum())
    envs = sorted(per_group)
    if not envs:
        raise ValueError("no reference groups remain after the exclusion")
    if any(env not in remind_groups for env in envs):
        raise ValueError("missing REMIND group rows for DINO reference UUIDs")
    rows = []
    for env in envs:
        g = per_group[env]
        b = remind_groups[env]
        rows.append({"environment_id": env, **g,
                     "dino_precision": g["dino_correct"] / g["dino_accepted"] if g["dino_accepted"] else None,
                     "dino_coverage": g["dino_correct"] / g["query_masks"],
                     "remind_precision": float(b["precision"]),
                     "remind_coverage": float(b["coverage"])})
    if any(r["dino_precision"] is None for r in rows):
        raise ValueError("threshold left a reference group with zero DINO accepts")
    rng = np.random.default_rng(seed)
    n = len(rows)
    sample = rng.integers(0, n, size=(draws, n))
    arms = {}
    effects = {}
    for metric in ("precision", "coverage"):
        d = np.asarray([r[f"dino_{metric}"] for r in rows])
        r = np.asarray([r[f"remind_{metric}"] for r in rows])
        diff = d - r
        boot = diff[sample].mean(axis=1)
        arms[f"dino_{metric}"] = float(d.mean())
        arms[f"remind_{metric}"] = float(r.mean())
        effects[metric] = {"difference": float(diff.mean()),
                           "bonferroni_2_outcomes_percentile_95": [float(x) for x in np.quantile(boot, [.0125, .9875])],
                           "groups_dino_higher": int(np.sum(diff > 0)), "groups_tied": int(np.sum(diff == 0))}
    return {"reference_groups": n, "excluded_reference_groups": sorted(excluded),
            "threshold": threshold, "arms_macro": arms, "paired_effects": effects,
            "reference_group_rows": rows,
            "bootstrap": {"draws": draws, "seed": seed, "unit": "released reference scan UUID"}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-input-root", type=Path, default=ROOT / "outputs/reports/20261009_unseen_oasis_equal_budget")
    p.add_argument("--source-prediction-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank_train8")
    p.add_argument("--validation-input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    p.add_argument("--validation-prediction-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    p.add_argument("--remind-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank/summary.json")
    p.add_argument("--overlap-summary", type=Path, default=ROOT / "outputs/reports/20261003_3rscan_cross_split_overlap/train_full_complete/summary.json")
    p.add_argument("--gate-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_gate/source_frozen_gate.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reports/20261009_dino_source_gate_overlap_sensitivity")
    p.add_argument("--draws", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261013)
    a = p.parse_args()
    source = load_split(a.source_input_root, a.source_prediction_root)
    validation = load_split(a.validation_input_root, a.validation_prediction_root)
    gate = json.loads(a.gate_summary.read_text())
    remind = json.loads(a.remind_summary.read_text())
    remind_groups = {r["environment_id"]: r["remind_one_to_one"] for r in remind["reference_groups"]}
    source_ids = {r["environment_id"] for r in source}
    validation_ids = {r["environment_id"] for r in validation}
    if source_ids & validation_ids:
        raise ValueError("exact reference UUID overlap between source and validation splits")

    overlap = json.loads(a.overlap_summary.read_text())
    geometry_candidates = overlap["screening_candidates"]
    possible_targets = {r["target_id"] for r in geometry_candidates if r["source_id"] in source_ids}
    relevant_candidates = [r for r in geometry_candidates if r["source_id"] in source_ids]
    if not possible_targets.issubset(validation_ids):
        raise ValueError("overlap-screen targets are not all in the frozen validation cohort")

    full_threshold = float(gate["frozen_threshold"])
    strict_source = "77361fbc-d054-2a22-8bd1-20da69ee28dc"
    strict_fit_rows = [r for r in source if r["environment_id"] != strict_source]
    refit_threshold, refit_metrics = fit_threshold(strict_fit_rows, .90)
    if refit_threshold is None:
        raise ValueError("source-only refit without the geometric-overlap source group is infeasible")
    selected_validation_overlap = next((r for r in relevant_candidates
                                        if r["source_id"] == strict_source
                                        and r["target_id"] == "0cac7540-8d6f-2d13-8eee-36ba2a428e3f"), None)

    scenarios = {
        "all_47_full_source_fit": summarize(validation, remind_groups, set(), full_threshold, a.draws, a.seed),
        "exclude_moderate_overlap_target": summarize(validation, remind_groups,
            {"0cac7540-8d6f-2d13-8eee-36ba2a428e3f"}, full_threshold, a.draws, a.seed + 1),
        "exclude_all_source_train_geometry_candidates": summarize(validation, remind_groups,
            possible_targets, full_threshold, a.draws, a.seed + 2),
        "exclude_all_candidates_and_refit_without_linked_source": summarize(validation, remind_groups,
            possible_targets, float(refit_threshold), a.draws, a.seed + 3),
    }
    result = {"status": "complete",
              "scope": "post-hoc sensitivity of the source-frozen all-view DINO threshold comparison to a prior cross-split geometric overlap screen",
              "source_reference_scan_uuids": sorted(source_ids),
              "validation_reference_scan_uuid_count": len(validation_ids),
              "exact_uuid_overlap": sorted(source_ids & validation_ids),
              "source_fit_threshold": full_threshold,
              "source_refit_without_77361": {"excluded_source_reference_uuid": strict_source,
                                              "threshold": refit_threshold,
                                              "remaining_source_groups": len(strict_fit_rows),
                                              "source_macro_precision": refit_metrics["macro_precision"],
                                              "source_macro_coverage": refit_metrics["macro_coverage"]},
              "cross_split_geometry_screen": {"method": overlap["method"],
                  "screening_cutoff": overlap["screening_candidate_cutoff"],
                  "candidate_edges_from_eight_source_refs": len(relevant_candidates),
                  "candidate_validation_reference_uuids": sorted(possible_targets),
                  "candidate_edges": [{"source_id": r["source_id"], "target_id": r["target_id"],
                                       "symmetric_fitness": r["icp_symmetric_fitness"],
                                       "symmetric_rmse_m": r["icp_symmetric_rmse_m"],
                                       "annotation_overlap_2cm": r.get("annotation_overlap_2cm")}
                                      for r in relevant_candidates],
                  "moderate_overlap_pair": selected_validation_overlap},
              "scenarios": scenarios,
              "limits": [
                  "The geometric screen is a prior FPFH/ICP content-similarity screen, not proof of shared physical site; excluding every candidate is deliberately conservative.",
                  "This analysis remains post-hoc on one benchmark, and does not certify independence of the remaining validation references.",
                  "Threshold refitting excludes the source reference UUID that had the measured partial overlap candidate; all other source relations are unchanged.",
              ]}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# DINO source-gate cross-split overlap sensitivity", "",
             "## Split identity audit", "",
             f"Exact source/validation reference UUID overlap: {len(source_ids & validation_ids)} (source {len(source_ids)}, validation {len(validation_ids)}). A prior train-to-validation FPFH/ICP screen finds {len(relevant_candidates)} candidate geometry edges from these eight train references to {len(possible_targets)} validation references at symmetric fitness >= {overlap['screening_candidate_cutoff']['symmetric_fitness_min']:.2f}, RMSE <= {overlap['screening_candidate_cutoff']['symmetric_rmse_m_max']:.2f} m. This is a conservative screen and does not prove shared sites.", "",
             "One source/validation candidate has explicit partial annotation agreement: source 77361fbc to validation 0cac7540 has 6.7%/5.7% surface overlap within 2 cm and 62.9% exact RGB/instance/semantic agreement among overlapping vertices. It is evidence of repeated content, not proof that the whole environments are identical.", "",
             "## Paired operating-point sensitivity", "",
             "| Scenario | Groups | Threshold | DINO precision | REMIND precision | Difference (adjusted interval) | DINO coverage | REMIND coverage | Difference (adjusted interval) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in scenarios.items():
        pe, ce = row["paired_effects"]["precision"], row["paired_effects"]["coverage"]
        pi, ci = pe["bonferroni_2_outcomes_percentile_95"], ce["bonferroni_2_outcomes_percentile_95"]
        m = row["arms_macro"]
        lines.append(f"| {name} | {row['reference_groups']} | {row['threshold']:.6f} | {m['dino_precision']:.1%} | {m['remind_precision']:.1%} | {pe['difference']*100:+.1f} [{pi[0]*100:+.1f},{pi[1]*100:+.1f}] | {m['dino_coverage']:.1%} | {m['remind_coverage']:.1%} | {ce['difference']*100:+.1f} [{ci[0]*100:+.1f},{ci[1]*100:+.1f}] |")
    lines += ["", "The first scenario reproduces the full-cohort contrast. The second drops the validation group with the strongest partial overlap candidate. The third drops every validation reference touched by a geometric candidate from any of the eight threshold-fit source references. The fourth also refits the operating cutoff on the other seven source references, then applies it unchanged to the remaining validation groups.", "", "## Limits", "",
              "This addresses one known overlap screen only. Geometry candidates may be false matches, while missing candidates do not establish physical-site independence. Results remain explored-split evidence with privileged masks and unequal DINO/REMIND native acceptance counts.", ""]
    (a.output / "report.md").write_text("\n".join(lines))
    print(json.dumps({"status": "complete", "source_groups": len(source_ids),
                      "exact_uuid_overlap": len(source_ids & validation_ids),
                      "candidate_validation_groups": len(possible_targets),
                      "refit_threshold": refit_threshold,
                      "scenarios": {k: {"n": v["reference_groups"], "threshold": v["threshold"],
                          "precision_diff": v["paired_effects"]["precision"]["difference"],
                          "precision_interval": v["paired_effects"]["precision"]["bonferroni_2_outcomes_percentile_95"],
                          "coverage_diff": v["paired_effects"]["coverage"]["difference"],
                          "coverage_interval": v["paired_effects"]["coverage"]["bonferroni_2_outcomes_percentile_95"]}
                          for k, v in scenarios.items()}}, indent=2))


if __name__ == "__main__":
    main()
