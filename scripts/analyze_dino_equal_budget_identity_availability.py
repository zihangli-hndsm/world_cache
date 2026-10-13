#!/usr/bin/env python3
"""Availability-stratified DINO/REMIND comparison at an exact per-relation budget."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, load_split
from analyze_dino_remind_query_overlap import load_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    p.add_argument("--dino-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    p.add_argument("--remind-root", type=Path, default=ROOT / "outputs/reports/20261109_remind_style_3rscan_validation")
    p.add_argument("--dino-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank/summary.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reports/20261009_dino_equal_budget_identity_availability")
    p.add_argument("--draws", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261015)
    a = p.parse_args()

    dino_relations = load_split(a.input_root, a.dino_root)
    dino_by_id = {r["run_id"]: r for r in dino_relations}
    dino_summary = json.loads(a.dino_summary.read_text())
    summary_rows = {r["run_id"]: r for r in dino_summary["relations"]}
    manifest = json.loads((a.dino_root / "run_manifest.json").read_text())
    groups: dict[str, list[dict]] = defaultdict(list)
    relation_checks = []

    for item in manifest["results"]:
        run_id = item["run_id"]
        frozen = json.loads((a.input_root / "uniform20_inputs" / run_id / "manifest.json").read_text())
        dino = json.loads((a.dino_root / f"{run_id}.json").read_text())
        remind = json.loads((a.remind_root / f"{run_id}.json").read_text())
        scored = dino_by_id[run_id]
        if dino.get("status") != "complete" or remind.get("status") != "complete":
            raise ValueError(f"incomplete prediction: {run_id}")
        query_records = frozen["records"]["rescan"]
        expected_frames = [int(r["frame_id"]) for r in query_records]
        if [int(x) for x in dino["query_frames"]] != expected_frames:
            raise ValueError(f"DINO frame order differs from frozen masks: {run_id}")
        if int(remind["query_frames"]) != len(query_records):
            raise ValueError(f"REMIND query frame count differs: {run_id}")

        source_count = int(remind["source_frames"])
        tracker_by_frame: dict[int, list[dict]] = defaultdict(list)
        for row in remind["query_mask_rows"]:
            tracker_by_frame[int(row["frame"])].append(row)
        expected_tracker_frames = {
            source_count + slot for slot, record in enumerate(query_records)
            if int(record["masks"]) > 0
        }
        if set(tracker_by_frame) != expected_tracker_frames:
            raise ValueError(f"REMIND frame index mapping differs: {run_id}")

        # The budget is the number of REMIND source links in this relation.
        budget = sum(int(row["predicted_source_link"]) for row in remind["query_mask_rows"])
        if budget != int(remind["query_links_to_source_tracks"]):
            raise ValueError(f"REMIND saved acceptance count disagrees with query rows: {run_id}")
        remind_correct_total = sum(int(row["correct_source_link"]) for row in remind["query_mask_rows"])
        if remind_correct_total != int(remind["correct_cross_scan_links"]):
            raise ValueError(f"REMIND saved correct-link count disagrees with query rows: {run_id}")
        score_rows = dino["query_rows"]
        scores = scored["scores"]
        correct = scored["correct"]
        available = scored["identity_available"]
        if len(score_rows) != len(scores) or len(remind["query_mask_rows"]) != len(scores):
            raise ValueError(f"prediction row count differs from mask population: {run_id}")

        valid_assignment = np.asarray([
            row["assigned_source_observation"] is not None and row["assigned_similarity"] is not None
            for row in score_rows
        ], dtype=bool)
        if not np.array_equal(valid_assignment, scored["valid"]):
            raise ValueError(f"DINO assignment-valid mask differs from validated score rows: {run_id}")
        candidates = np.flatnonzero(valid_assignment)
        if budget > len(candidates):
            raise ValueError(f"DINO has {len(candidates)} valid assignments, below REMIND budget {budget}: {run_id}")
        ranked = np.lexsort((candidates, -scores[candidates]))
        selected = np.zeros(len(scores), dtype=bool)
        selected[candidates[ranked[:budget]]] = True

        cursor = 0
        accepted_dino = correct_dino = accepted_remind = correct_remind = 0
        accepted_dino_available = correct_dino_available = 0
        accepted_remind_available = correct_remind_available = 0
        correct_dino_absent = correct_remind_absent = 0
        local_queries = 0
        for slot, record in enumerate(query_records):
            query_frame = int(record["frame_id"])
            query_ids = load_ids(record)
            tracker_rows = tracker_by_frame.get(source_count + slot, [])
            if len(tracker_rows) != len(query_ids):
                raise ValueError(f"REMIND mask count mismatch: {run_id}/{query_frame}")
            for index, query_id in enumerate(query_ids.tolist()):
                if cursor >= len(score_rows):
                    raise ValueError(f"DINO rows ended early: {run_id}")
                drow = score_rows[cursor]
                rrow = tracker_rows[index]
                if int(drow["query_frame"]) != query_frame or int(drow["query_index"]) != index:
                    raise ValueError(f"DINO query order mismatch: {run_id}/{query_frame}/{index}")
                if int(rrow["gt_id"]) != int(query_id):
                    raise ValueError(f"REMIND identity/mask order mismatch: {run_id}/{query_frame}/{index}")
                da = bool(selected[cursor])
                dc = bool(da and correct[cursor])
                ra = bool(rrow["predicted_source_link"])
                rc = bool(rrow["correct_source_link"])
                av = bool(available[cursor])
                if not av and (dc or rc):
                    raise ValueError(f"correct identity link reported when target is absent: {run_id}/{query_frame}/{index}")
                row = {"run_id": run_id,
                       "environment_id": Path(frozen["records"]["reference"][0]["input"]).parent.name,
                       "query_frame": query_frame, "query_index": index,
                       "identity_available": av,
                       "dino_accepted": da, "dino_correct": dc,
                       "remind_accepted": ra, "remind_correct": rc}
                groups[row["environment_id"]].append(row)
                local_queries += 1
                accepted_dino += int(da); correct_dino += int(dc)
                accepted_remind += int(ra); correct_remind += int(rc)
                if av:
                    accepted_dino_available += int(da)
                    correct_dino_available += int(dc)
                    accepted_remind_available += int(ra)
                    correct_remind_available += int(rc)
                else:
                    correct_dino_absent += int(dc)
                    correct_remind_absent += int(rc)
                cursor += 1

        if cursor != len(scores):
            raise ValueError(f"DINO query rows remain after join: {run_id}")
        if accepted_dino != budget or accepted_remind != budget:
            raise ValueError(f"accepted counts do not match the exact budget: {run_id}")
        expected = summary_rows[run_id]["multiview_exact_budget"]
        if expected["accepted"] != budget or expected["correct"] != correct_dino:
            raise ValueError(f"reconstructed all-view exact-budget result differs: {run_id}")
        relation_checks.append({"run_id": run_id, "query_masks": local_queries,
                                "budget": budget, "dino_correct": correct_dino,
                                "remind_correct": correct_remind,
                                "dino_available_accepts": accepted_dino_available,
                                "dino_available_correct": correct_dino_available,
                                "remind_available_accepts": accepted_remind_available,
                                "remind_available_correct": correct_remind_available,
                                "dino_absent_false_accepts": budget - accepted_dino_available,
                                "remind_absent_false_accepts": budget - accepted_remind_available})

    envs = sorted(groups)
    group_rows = []
    pooled = {"queries": 0, "available": 0, "absent": 0,
              "dino_accepted": 0, "remind_accepted": 0,
              "dino_correct": 0, "remind_correct": 0,
              "dino_available_accepted": 0, "remind_available_accepted": 0,
              "dino_available_correct": 0, "remind_available_correct": 0,
              "dino_absent_accepted": 0, "remind_absent_accepted": 0}
    for env in envs:
        rows = groups[env]
        available_rows = [r for r in rows if r["identity_available"]]
        absent_rows = [r for r in rows if not r["identity_available"]]
        d_acc = sum(r["dino_accepted"] for r in rows)
        r_acc = sum(r["remind_accepted"] for r in rows)
        d_cor = sum(r["dino_correct"] for r in rows)
        r_cor = sum(r["remind_correct"] for r in rows)
        d_aa = sum(r["dino_accepted"] for r in available_rows)
        r_aa = sum(r["remind_accepted"] for r in available_rows)
        d_ca = sum(r["dino_correct"] for r in available_rows)
        r_ca = sum(r["remind_correct"] for r in available_rows)
        d_ua = sum(r["dino_accepted"] for r in absent_rows)
        r_ua = sum(r["remind_accepted"] for r in absent_rows)
        group_rows.append({
            "environment_id": env, "query_masks": len(rows),
            "available_queries": len(available_rows), "absent_queries": len(absent_rows),
            "dino_accepted": d_acc, "remind_accepted": r_acc,
            "dino_correct": d_cor, "remind_correct": r_cor,
            "dino_precision": d_cor / d_acc, "remind_precision": r_cor / r_acc,
            "dino_coverage": d_cor / len(rows), "remind_coverage": r_cor / len(rows),
            "dino_precision_available": d_ca / d_aa if d_aa else None,
            "remind_precision_available": r_ca / r_aa if r_aa else None,
            "dino_coverage_available": d_ca / len(available_rows) if available_rows else None,
            "remind_coverage_available": r_ca / len(available_rows) if available_rows else None,
            "dino_absent_false_accept_rate": d_ua / len(absent_rows) if absent_rows else None,
            "remind_absent_false_accept_rate": r_ua / len(absent_rows) if absent_rows else None,
            "dino_available_accepts": d_aa, "remind_available_accepts": r_aa,
            "dino_absent_accepts": d_ua, "remind_absent_accepts": r_ua,
        })
        pooled["queries"] += len(rows); pooled["available"] += len(available_rows); pooled["absent"] += len(absent_rows)
        for method, acc, cor, aa, ca, ua in (
            ("dino", d_acc, d_cor, d_aa, d_ca, d_ua),
            ("remind", r_acc, r_cor, r_aa, r_ca, r_ua),
        ):
            pooled[f"{method}_accepted"] += acc
            pooled[f"{method}_correct"] += cor
            pooled[f"{method}_available_accepted"] += aa
            pooled[f"{method}_available_correct"] += ca
            pooled[f"{method}_absent_accepted"] += ua

    overlap_counts = {
        "both_correct": 0, "dino_only_correct": 0, "remind_only_correct": 0,
        "neither_correct": 0,
        "dino_only_correct_remind_rejected": 0,
        "dino_only_correct_remind_wrong": 0,
        "remind_only_correct_dino_rejected": 0,
        "remind_only_correct_dino_wrong": 0,
        "both_accepted": 0, "dino_only_accepted": 0,
        "remind_only_accepted": 0, "neither_accepted": 0,
    }
    availability_overlap = {
        "available": {key: 0 for key in ("both_correct", "dino_only_correct", "remind_only_correct", "neither_correct")},
        "absent": {key: 0 for key in ("both_correct", "dino_only_correct", "remind_only_correct", "neither_correct")},
    }
    for rows in groups.values():
        for row in rows:
            dc, rc = row["dino_correct"], row["remind_correct"]
            da, ra = row["dino_accepted"], row["remind_accepted"]
            if dc and rc:
                key = "both_correct"
            elif dc:
                key = "dino_only_correct"
                overlap_counts["dino_only_correct_remind_rejected" if not ra else "dino_only_correct_remind_wrong"] += 1
            elif rc:
                key = "remind_only_correct"
                overlap_counts["remind_only_correct_dino_rejected" if not da else "remind_only_correct_dino_wrong"] += 1
            else:
                key = "neither_correct"
            overlap_counts[key] += 1
            availability_overlap["available" if row["identity_available"] else "absent"][key] += 1
            if da and ra: overlap_counts["both_accepted"] += 1
            elif da: overlap_counts["dino_only_accepted"] += 1
            elif ra: overlap_counts["remind_only_accepted"] += 1
            else: overlap_counts["neither_accepted"] += 1
    if overlap_counts["dino_only_accepted"] != overlap_counts["remind_only_accepted"]:
        raise ValueError("exact per-relation budgets did not produce balanced one-sided accepts")

    metric_keys = {
        "precision_equal_budget": ("dino_precision", "remind_precision"),
        "coverage_equal_budget": ("dino_coverage", "remind_coverage"),
        "precision_when_identity_available": ("dino_precision_available", "remind_precision_available"),
        "coverage_when_identity_available": ("dino_coverage_available", "remind_coverage_available"),
        "false_accept_rate_when_identity_absent": ("dino_absent_false_accept_rate", "remind_absent_false_accept_rate"),
    }
    rng = np.random.default_rng(a.seed)
    boot_ix = rng.integers(0, len(group_rows), size=(a.draws, len(group_rows)))
    effects = {}
    for name, (dk, rk) in metric_keys.items():
        d = np.asarray([r[dk] if r[dk] is not None else np.nan for r in group_rows])
        r = np.asarray([r[rk] if r[rk] is not None else np.nan for r in group_rows])
        valid_groups = np.isfinite(d) & np.isfinite(r)
        d, r = d[valid_groups], r[valid_groups]
        delta = d - r
        sampled = rng.integers(0, len(delta), size=(a.draws, len(delta)))
        draws = delta[sampled].mean(axis=1)
        effects[name] = {
            "dino_macro": float(d.mean()), "remind_macro": float(r.mean()),
            "difference_dino_minus_remind": float(delta.mean()),
            "groups": int(len(delta)),
            "groups_dino_higher": int(np.sum(delta > 1e-12)),
            "groups_remind_higher": int(np.sum(delta < -1e-12)),
            "groups_tied": int(np.sum(np.abs(delta) <= 1e-12)),
            "paired_group_bootstrap_95": [float(x) for x in np.quantile(draws, [.025, .975])],
            "bonferroni_5_outcomes_percentile_95": [float(x) for x in np.quantile(draws, [.005, .995])],
        }

    decomposition_rows = []
    for env in envs:
        rows = groups[env]
        d_only = sum(r["dino_accepted"] and not r["remind_accepted"] and r["dino_correct"] for r in rows)
        r_only = sum(r["remind_accepted"] and not r["dino_accepted"] and r["remind_correct"] for r in rows)
        d_shared_only = sum(r["dino_accepted"] and r["remind_accepted"] and r["dino_correct"] and not r["remind_correct"] for r in rows)
        r_shared_only = sum(r["dino_accepted"] and r["remind_accepted"] and r["remind_correct"] and not r["dino_correct"] for r in rows)
        denominator = len(rows)
        decomposition_rows.append({
            "environment_id": env,
            "one_sided_acceptance_set_effect": (d_only - r_only) / denominator,
            "shared_accepted_assignment_effect": (d_shared_only - r_shared_only) / denominator,
        })
    set_component = np.asarray([r["one_sided_acceptance_set_effect"] for r in decomposition_rows])
    shared_component = np.asarray([r["shared_accepted_assignment_effect"] for r in decomposition_rows])
    if not np.isclose(set_component.mean() + shared_component.mean(),
                      effects["coverage_equal_budget"]["difference_dino_minus_remind"], atol=1e-12):
        raise AssertionError("equal-budget correct-coverage decomposition does not conserve")
    decomp_boot_ix = rng.integers(0, len(decomposition_rows), size=(a.draws, len(decomposition_rows)))
    decomposition = {}
    for key, values in (("one_sided_acceptance_set_effect", set_component),
                        ("shared_accepted_assignment_effect", shared_component)):
        draws = values[decomp_boot_ix].mean(axis=1)
        decomposition[key] = {
            "estimate": float(values.mean()),
            "paired_group_bootstrap_95": [float(x) for x in np.quantile(draws, [.025, .975])],
            "bonferroni_2_component_interval": [float(x) for x in np.quantile(draws, [.0125, .9875])],
        }

    total_budget = sum(r["budget"] for r in relation_checks)
    expected_total_budget = sum(
        int(row["multiview_exact_budget"]["accepted"])
        for row in summary_rows.values()
    )
    expected_total_correct = sum(
        int(row["multiview_exact_budget"]["correct"])
        for row in summary_rows.values()
    )
    if (total_budget != expected_total_budget
            or sum(r["dino_correct"] for r in relation_checks) != expected_total_correct):
        raise ValueError("pooled DINO exact-budget score differs from the frozen source report")
    result = {
        "status": "complete",
        "scope": "post-hoc identity-availability decomposition of the label-blind per-relation exact REMIND-budget DINO all-view comparison",
        "protocol": "Within each relation, rank assigned DINO all-view cosine matches and retain exactly the REMIND accepted-link count; no query correctness or identity-availability labels affect selection.",
        "population": {"relations": len(relation_checks), "reference_groups": len(group_rows),
                       "query_masks": sum(r["query_masks"] for r in relation_checks),
                       "total_equal_accept_budget": total_budget,
                       "identity_available_queries": pooled["available"],
                       "identity_absent_queries": pooled["absent"],
                       "dino_correct": pooled["dino_correct"], "remind_correct": pooled["remind_correct"]},
        "pooled": pooled,
        "pooled_rates": {
            "dino_precision": pooled["dino_correct"] / pooled["dino_accepted"],
            "remind_precision": pooled["remind_correct"] / pooled["remind_accepted"],
            "dino_precision_available": pooled["dino_available_correct"] / pooled["dino_available_accepted"],
            "remind_precision_available": pooled["remind_available_correct"] / pooled["remind_available_accepted"],
            "dino_coverage_available": pooled["dino_available_correct"] / pooled["available"],
            "remind_coverage_available": pooled["remind_available_correct"] / pooled["available"],
            "dino_absent_false_accept_rate": pooled["dino_absent_accepted"] / pooled["absent"],
            "remind_absent_false_accept_rate": pooled["remind_absent_accepted"] / pooled["absent"],
        },
        "paired_group_macro_effects": effects,
        "same_query_correctness_overlap": overlap_counts,
        "correctness_overlap_by_identity_availability": availability_overlap,
        "equal_budget_correct_coverage_decomposition": decomposition,
        "equal_budget_correct_coverage_decomposition_group_rows": decomposition_rows,
        "bootstrap": {"draws": a.draws, "seed": a.seed, "unit": "paired reference scan UUID",
                      "family_correction": "Bonferroni across five availability-stratified outcomes"},
        "relation_checks": relation_checks,
        "reference_group_rows": group_rows,
        "limits": [
            "The equal budget is borrowed from observed REMIND accepted-link counts after inference; this is a diagnostic operating point, not a deployment policy.",
            "Identity availability is computed from ground-truth IDs after inference and is an oracle stratum, not an inference-time signal.",
            "The evaluated 3RScan cohort is previously explored and uses privileged annotation-derived masks; this does not establish independent-site generalization or a causal memory effect.",
        ],
    }
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = ["# Exact-budget DINO/REMIND comparison by target-identity availability", "",
            f"Within every evaluated relation, the all-view DINO assignments were ranked by cosine and truncated to exactly the REMIND accepted-link count. This reproduces {total_budget:,} accepted links and DINO's {pooled['dino_correct']:,} correct links in the frozen source report; labels do not select the DINO links. The physical target identity is available somewhere in the reference mask bank for {pooled['available']:,}/{pooled['queries']:,} query masks.", "",
            "| Outcome | DINO | REMIND | DINO - REMIND (five-outcome-adjusted 95% interval) |", "|---|---:|---:|---:|"]
    display = [
        ("Exact-budget macro precision", "precision_equal_budget"),
        ("Exact-budget macro correct coverage", "coverage_equal_budget"),
        ("Precision among identity-available queries", "precision_when_identity_available"),
        ("Correct coverage among identity-available queries", "coverage_when_identity_available"),
        ("False-accept rate among identity-absent queries", "false_accept_rate_when_identity_absent"),
    ]
    for label, key in display:
        e = effects[key]
        ci = e["bonferroni_5_outcomes_percentile_95"]
        rows.append(f"| {label} | {e['dino_macro']:.1%} | {e['remind_macro']:.1%} | {e['difference_dino_minus_remind']*100:+.1f} pp [{ci[0]*100:+.1f},{ci[1]*100:+.1f}] |")
    absent_groups = effects["false_accept_rate_when_identity_absent"]["groups"]
    net_correct = pooled["dino_correct"] - pooled["remind_correct"]
    one_sided_correct_net = (overlap_counts["dino_only_correct_remind_rejected"]
                             - overlap_counts["remind_only_correct_dino_rejected"])
    shared_accept_correct_net = (overlap_counts["dino_only_correct_remind_wrong"]
                                 - overlap_counts["remind_only_correct_dino_wrong"])
    rows += ["", f"The paired bootstrap uses {a.draws:,} reference-group draws. Precision and coverage among identity-available queries use all {effects['precision_when_identity_available']['groups']} groups and are computed on the same oracle-defined subset. The identity-absent false-accept rate is defined in {absent_groups} groups containing absent-target queries. Any accepted source link in that stratum is false by construction, so the final row measures abstention behavior. All five outcomes use a Bonferroni family correction.", "",
             f"At the identical per-relation acceptance counts, both methods are correct on {overlap_counts['both_correct']:,} queries; DINO alone is correct on {overlap_counts['dino_only_correct']:,}, and REMIND alone on {overlap_counts['remind_only_correct']:,}. Of DINO-only correct queries, REMIND rejects {overlap_counts['dino_only_correct_remind_rejected']:,} and accepts a wrong link on {overlap_counts['dino_only_correct_remind_wrong']:,}. Of REMIND-only correct queries, DINO rejects {overlap_counts['remind_only_correct_dino_rejected']:,} and accepts a wrong link on {overlap_counts['remind_only_correct_dino_wrong']:,}. One-sided accepted-query counts are exactly balanced ({overlap_counts['dino_only_accepted']:,} each), as required by the per-relation budget match.",
             f"The net gain of {net_correct:,} correct links decomposes into {one_sided_correct_net:+,} from selecting different queries ({overlap_counts['dino_only_correct_remind_rejected']:,} DINO-correct queries are REMIND abstentions versus {overlap_counts['remind_only_correct_dino_rejected']:,} REMIND-correct queries rejected by DINO) and {shared_accept_correct_net:+,} on queries both methods accept. The group-macro correct-coverage decomposition is {decomposition['one_sided_acceptance_set_effect']['estimate']*100:+.2f} pp from one-sided acceptance and {decomposition['shared_accepted_assignment_effect']['estimate']*100:+.2f} pp from different assignments on shared accepts; the two components sum exactly to {effects['coverage_equal_budget']['difference_dino_minus_remind']*100:+.2f} pp.", "",
             "This decomposes the known exact-budget advantage by whether the target exists in the source bank. Availability is determined post hoc from annotation IDs; it is not a usable inference-time filter. The subset was selected by target availability, and the full cohort is previously explored with privileged masks, so this is a diagnostic rather than an independent or causal result.", "",
             "Reproduce with `python scripts/analyze_dino_equal_budget_identity_availability.py`. Per-relation verification and reference-group rows are in `summary.json`.", ""]
    (a.output / "report.md").write_text("\n".join(rows))
    print(json.dumps({"status": "complete", "population": result["population"],
                      "pooled_rates": result["pooled_rates"],
                      "effects": {k: {"difference": v["difference_dino_minus_remind"],
                                      "adjusted_interval": v["bonferroni_5_outcomes_percentile_95"]}
                                 for k, v in effects.items()}}, indent=2))


if __name__ == "__main__":
    main()
