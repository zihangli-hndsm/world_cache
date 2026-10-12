#!/usr/bin/env python3
"""Decompose same-query correct-link overlap for the frozen DINO gate and REMIND."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, load_split


def load_ids(record: dict) -> np.ndarray:
    path = Path(record["identity_targets"])
    if not path.is_absolute():
        path = ROOT / path
    return np.load(path).astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    p.add_argument("--dino-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    p.add_argument("--remind-root", type=Path, default=ROOT / "outputs/reports/20261109_remind_style_3rscan_validation")
    p.add_argument("--gate-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_gate/source_frozen_gate.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reports/20261009_dino_remind_query_overlap")
    p.add_argument("--draws", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261011)
    a = p.parse_args()
    gate = json.loads(a.gate_summary.read_text())
    threshold = float(gate["frozen_threshold"])
    dino_relations = load_split(a.input_root, a.dino_root)
    dino_by_id = {r["run_id"]: r for r in dino_relations}
    manifest = json.loads((a.dino_root / "run_manifest.json").read_text())
    all_queries: list[dict] = []
    relation_summaries = []

    for item in manifest["results"]:
        run_id = item["run_id"]
        frozen = json.loads((a.input_root / "uniform20_inputs" / run_id / "manifest.json").read_text())
        dino = json.loads((a.dino_root / f"{run_id}.json").read_text())
        remind = json.loads((a.remind_root / f"{run_id}.json").read_text())
        scored = dino_by_id[run_id]
        if dino.get("status") != "complete" or remind.get("status") != "complete":
            raise ValueError(f"incomplete prediction: {run_id}")
        query_records = frozen["records"]["rescan"]
        frame_ids = [int(r["frame_id"]) for r in query_records]
        if [int(x) for x in dino["query_frames"]] != frame_ids:
            raise ValueError(f"DINO frame order differs from frozen masks: {run_id}")
        if int(remind["query_frames"]) != len(frame_ids) or int(remind["source_frames"]) != len(frozen["records"]["reference"]):
            raise ValueError(f"REMIND source/query frame counts differ from frozen masks: {run_id}")
        source_frame_count = int(remind["source_frames"])
        tracker_rows: dict[int, list[dict]] = defaultdict(list)
        for row in remind["query_mask_rows"]:
            tracker_rows[int(row["frame"])].append(row)
        expected_tracker_frames = {
            source_frame_count + slot for slot, record in enumerate(query_records)
            if int(record["masks"]) > 0
        }
        if set(tracker_rows) != expected_tracker_frames:
            raise ValueError(f"REMIND frame index mapping differs from source/query split: {run_id}")

        scores = scored["scores"]
        dino_correct = scored["correct"]
        cursor = 0
        local_rows = []
        for slot, record in enumerate(query_records):
            query_frame = int(record["frame_id"])
            ids = load_ids(record)
            rows = tracker_rows.get(source_frame_count + slot, [])
            if len(rows) != len(ids):
                raise ValueError(f"REMIND mask count mismatch: {run_id}/{query_frame}")
            if len(dino["query_rows"]) < cursor + len(ids):
                raise ValueError(f"DINO query rows ended early: {run_id}")
            for index, query_id in enumerate(ids.tolist()):
                tracker_row = rows[index]
                if int(tracker_row["gt_id"]) != int(query_id):
                    raise ValueError(f"REMIND identity/mask ordering mismatch: {run_id}/{query_frame}/{index}")
                pred_row = dino["query_rows"][cursor]
                if int(pred_row["query_frame"]) != query_frame or int(pred_row["query_index"]) != index:
                    raise ValueError(f"DINO query ordering mismatch: {run_id}/{query_frame}/{index}")
                d_accept = bool(scored["valid"][cursor] and scores[cursor] >= threshold)
                d_correct = bool(d_accept and dino_correct[cursor])
                r_accept = bool(tracker_row["predicted_source_link"])
                r_correct = bool(tracker_row["correct_source_link"])
                row = {"run_id": run_id,
                       "environment_id": Path(frozen["records"]["reference"][0]["input"]).parent.name,
                       "query_frame": query_frame, "query_index": index,
                       "query_identity": int(query_id),
                       "identity_available": bool(scored["identity_available"][cursor]),
                       "dino_accepted": d_accept, "dino_correct": d_correct,
                       "remind_accepted": r_accept, "remind_correct": r_correct}
                local_rows.append(row)
                all_queries.append(row)
                cursor += 1
        if cursor != len(scores) or cursor != len(dino["query_rows"]):
            raise ValueError(f"DINO query row count mismatch: {run_id}")
        if len(local_rows) != len(remind["query_mask_rows"]):
            raise ValueError(f"REMIND query row count mismatch: {run_id}")
        relation_summaries.append({"run_id": run_id, "query_masks": len(local_rows),
                                   "dino_accepted": sum(r["dino_accepted"] for r in local_rows),
                                   "dino_correct": sum(r["dino_correct"] for r in local_rows),
                                   "remind_accepted": sum(r["remind_accepted"] for r in local_rows),
                                   "remind_correct": sum(r["remind_correct"] for r in local_rows)})

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in all_queries:
        groups[row["environment_id"]].append(row)
    cells = {"both_correct": 0, "dino_only_correct": 0, "remind_only_correct": 0, "neither_correct": 0,
             "dino_only_correct_remind_rejected": 0, "dino_only_correct_remind_wrong": 0,
             "remind_only_correct_dino_rejected": 0, "remind_only_correct_dino_wrong": 0,
             "both_accepted": 0, "dino_only_accepted": 0, "remind_only_accepted": 0, "neither_accepted": 0}
    group_rows = []
    availability_group_rows = []
    for env, rows in sorted(groups.items()):
        c = dict.fromkeys(cells, 0)
        for r in rows:
            dc, rc = r["dino_correct"], r["remind_correct"]
            da, ra = r["dino_accepted"], r["remind_accepted"]
            if dc and rc: c["both_correct"] += 1
            elif dc:
                c["dino_only_correct"] += 1
                c["dino_only_correct_remind_rejected" if not ra else "dino_only_correct_remind_wrong"] += 1
            elif rc:
                c["remind_only_correct"] += 1
                c["remind_only_correct_dino_rejected" if not da else "remind_only_correct_dino_wrong"] += 1
            else: c["neither_correct"] += 1
            if da and ra: c["both_accepted"] += 1
            elif da: c["dino_only_accepted"] += 1
            elif ra: c["remind_only_accepted"] += 1
            else: c["neither_accepted"] += 1
        cells = {k: cells[k] + c[k] for k in cells}
        n = len(rows)
        group_rows.append({"environment_id": env, "query_masks": n, **c,
                           **{f"{k}_coverage": c[k] / n for k in (
                               "both_correct", "dino_only_correct", "remind_only_correct",
                               "dino_only_correct_remind_rejected", "dino_only_correct_remind_wrong",
                               "remind_only_correct_dino_rejected", "remind_only_correct_dino_wrong")}})
        available_rows = [r for r in rows if r["identity_available"]]
        dino_accepted_available = sum(r["dino_accepted"] for r in available_rows)
        remind_accepted_available = sum(r["remind_accepted"] for r in available_rows)
        dino_correct_available = sum(r["dino_correct"] for r in available_rows)
        remind_correct_available = sum(r["remind_correct"] for r in available_rows)
        availability_group_rows.append({
            "environment_id": env, "identity_available_queries": len(available_rows),
            "dino_precision_available": dino_correct_available / dino_accepted_available if dino_accepted_available else None,
            "remind_precision_available": remind_correct_available / remind_accepted_available if remind_accepted_available else None,
            "dino_coverage_available": dino_correct_available / len(available_rows) if available_rows else None,
            "remind_coverage_available": remind_correct_available / len(available_rows) if available_rows else None,
        })

    n_queries = len(all_queries)
    relations = {"relations": len(relation_summaries), "query_masks": n_queries,
                 "dino_accepted": sum(r["dino_accepted"] for r in relation_summaries),
                 "dino_correct": sum(r["dino_correct"] for r in relation_summaries),
                 "remind_accepted": sum(r["remind_accepted"] for r in relation_summaries),
                 "remind_correct": sum(r["remind_correct"] for r in relation_summaries)}
    availability_comparison = {}
    for method in ("dino", "remind"):
        accepted_key, correct_key = f"{method}_accepted", f"{method}_correct"
        counts = {"identity_available_queries": 0, "identity_absent_queries": 0,
                  "accepted_when_identity_available": 0, "correct_when_identity_available": 0,
                  "wrong_accept_when_identity_available": 0, "accepted_when_identity_absent": 0,
                  "correct_when_identity_absent": 0}
        for row in all_queries:
            available = row["identity_available"]
            counts["identity_available_queries" if available else "identity_absent_queries"] += 1
            if available and row[accepted_key]:
                counts["accepted_when_identity_available"] += 1
                counts["correct_when_identity_available"] += int(row[correct_key])
                counts["wrong_accept_when_identity_available"] += int(not row[correct_key])
            elif not available and row[accepted_key]:
                counts["accepted_when_identity_absent"] += 1
                counts["correct_when_identity_absent"] += int(row[correct_key])
        counts["accepted_precision_when_identity_available"] = (
            counts["correct_when_identity_available"] / counts["accepted_when_identity_available"]
            if counts["accepted_when_identity_available"] else None)
        counts["absent_identity_acceptance_rate"] = (
            counts["accepted_when_identity_absent"] / counts["identity_absent_queries"]
            if counts["identity_absent_queries"] else None)
        availability_comparison[method] = counts
    rng = np.random.default_rng(a.seed)
    env_count = len(group_rows)
    sample = rng.integers(0, env_count, size=(a.draws, env_count))
    bootstrap = {}
    for key in ("dino_only_correct_coverage", "remind_only_correct_coverage",
                "dino_only_correct_remind_rejected_coverage", "dino_only_correct_remind_wrong_coverage",
                "remind_only_correct_dino_rejected_coverage", "remind_only_correct_dino_wrong_coverage"):
        values = np.asarray([r[key] for r in group_rows])
        boot = values[sample].mean(axis=1)
        bootstrap[key] = {"estimate": float(values.mean()), "percentile_95": [float(x) for x in np.quantile(boot, [.025, .975])]}
    conditional_effects = {}
    for metric in ("precision_available", "coverage_available"):
        dkey, rkey = f"dino_{metric}", f"remind_{metric}"
        dvals = np.asarray([r[dkey] if r[dkey] is not None else np.nan for r in availability_group_rows])
        rvals = np.asarray([r[rkey] if r[rkey] is not None else np.nan for r in availability_group_rows])
        valid = np.isfinite(dvals) & np.isfinite(rvals)
        diffs = dvals[valid] - rvals[valid]
        paired_sample = rng.integers(0, len(diffs), size=(a.draws, len(diffs)))
        draws = diffs[paired_sample].mean(axis=1)
        conditional_effects[metric] = {
            "dino_macro": float(dvals[valid].mean()), "remind_macro": float(rvals[valid].mean()),
            "difference": float(diffs.mean()), "groups": int(valid.sum()),
            "paired_percentile_95": [float(x) for x in np.quantile(draws, [.025, .975])],
            "bonferroni_2_outcomes_percentile_95": [float(x) for x in np.quantile(draws, [.0125, .9875])],
        }
    result = {"status": "complete", "scope": "query-level correctness overlap for a source-frozen DINO gate and REMIND-style map on the same 110 3RScan relations",
              "threshold": threshold, "population": relations, "pooled_overlap_cells": cells,
              "identity_availability_comparison": availability_comparison,
              "macro_reference_group_bootstrap": bootstrap,
              "availability_conditioned_paired_effects": conditional_effects,
              "bootstrap_design": {"draws": a.draws, "seed": a.seed, "unit": "released reference scan ID"},
              "reference_groups": group_rows, "availability_reference_groups": availability_group_rows,
              "relations": relation_summaries,
              "limits": ["This compares different association procedures and different acceptance counts; it does not isolate track memory causally.",
                         "The 3RScan validation masks have been explored previously and are annotation-derived.",
                         "Correct means matching the released physical identity label, scored after anonymous inference."]}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = f"""# Same-query DINO / REMIND correctness overlap

## Audited population

The source-frozen DINO gate (cosine >= {threshold:.6f}) and the REMIND-style source map were joined on query frame and mask index for {relations['relations']} relations, {env_count} reference groups, and {n_queries:,} identical masks. Every REMIND row's stored ground-truth identity was checked against the frozen query identity target; DINO frame/mask ordering and all query counts were checked.

## Correct-link overlap

| Query outcome | Count |
|---|---:|
| Both methods correct | {cells['both_correct']:,} |
| DINO correct, REMIND not | {cells['dino_only_correct']:,} |
| REMIND correct, DINO not | {cells['remind_only_correct']:,} |
| Neither correct | {cells['neither_correct']:,} |

Of DINO-only correct links, REMIND rejected {cells['dino_only_correct_remind_rejected']:,} and accepted an incorrect link on {cells['dino_only_correct_remind_wrong']:,}. Of REMIND-only correct links, DINO rejected {cells['remind_only_correct_dino_rejected']:,} and accepted an incorrect link on {cells['remind_only_correct_dino_wrong']:,}. The net correct-query difference is {relations['dino_correct']-relations['remind_correct']:+,} masks ({relations['dino_correct']:,} vs {relations['remind_correct']:,}).

The DINO gate accepts {relations['dino_accepted']:,} masks while REMIND accepts {relations['remind_accepted']:,}; the operating points are not equal-budget. Paired precision/coverage effects against REMIND are reported in the source-gate report. This overlap analysis explains query-level complementarity, not a causal memory effect or independent-site result. All inputs use privileged annotation-derived masks and the same previously explored 3RScan cohort.

Identity availability is also checked post hoc. Among {availability_comparison['dino']['identity_available_queries']:,} queries whose physical ID exists in the reference bank, DINO accepts {availability_comparison['dino']['accepted_when_identity_available']:,} at {availability_comparison['dino']['accepted_precision_when_identity_available']:.1%} precision; it accepts {availability_comparison['dino']['accepted_when_identity_absent']:,}/{availability_comparison['dino']['identity_absent_queries']:,} queries whose ID is absent. REMIND accepts {availability_comparison['remind']['accepted_when_identity_absent']:,}/{availability_comparison['remind']['identity_absent_queries']:,} absent-identity queries and {availability_comparison['remind']['accepted_when_identity_available']:,} available-identity queries at {availability_comparison['remind']['accepted_precision_when_identity_available']:.1%} precision. This availability label is evaluation-only and is not an inference-time filter.

Conditioning the macro comparison on the identity being present in the bank, paired 47-group differences are {conditional_effects['precision_available']['difference']*100:+.1f} pp accepted precision (two-outcome-adjusted interval [{conditional_effects['precision_available']['bonferroni_2_outcomes_percentile_95'][0]*100:+.1f},{conditional_effects['precision_available']['bonferroni_2_outcomes_percentile_95'][1]*100:+.1f}]) and {conditional_effects['coverage_available']['difference']*100:+.1f} pp correct coverage among available-identity queries (adjusted interval [{conditional_effects['coverage_available']['bonferroni_2_outcomes_percentile_95'][0]*100:+.1f},{conditional_effects['coverage_available']['bonferroni_2_outcomes_percentile_95'][1]*100:+.1f}]). This is an oracle-conditioned post-hoc subset analysis and still uses unequal acceptance counts.
"""
    (a.output / "report.md").write_text(report)
    print(json.dumps({"status": "complete", "population": relations,
                      "overlap": cells, "macro_coverage_components": bootstrap,
                      "availability_comparison": availability_comparison,
                      "availability_conditioned_effects": conditional_effects}, indent=2))


if __name__ == "__main__":
    main()
