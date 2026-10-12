#!/usr/bin/env python3
"""Object-level post-hoc analysis of the frozen source cosine gate vs REMIND."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, rows_for
from analyze_dino_remind_query_overlap import load_ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_object_view_budget_curve_extension")
    p.add_argument("--dino-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank_extension")
    p.add_argument("--remind-root", type=Path, default=ROOT / "outputs/reports/20261009_remind_style_view_extension")
    p.add_argument("--gate-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_gate_train8_to_extension8/source_frozen_gate.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reports/20261009_dino_source_gate_object_sensitivity_extension8")
    p.add_argument("--draws", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261010)
    a = p.parse_args()

    threshold = float(json.loads(a.gate_summary.read_text())["frozen_threshold"])
    manifest = json.loads((a.dino_root / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("incomplete DINO extension manifest")
    # Each key is a scan-local object. Numeric object IDs are not compared
    # between query scans.
    clusters: dict[tuple[str, str, int], dict[str, int]] = {}
    group_for: dict[tuple[str, str, int], str] = {}
    scored_by_id = {x["run_id"]: rows_for(x["run_id"], a.input_root, a.dino_root) for x in manifest["results"]}
    query_count = relation_count = gate_accepts = gate_correct = remind_accepts = remind_correct = 0
    budget_accepts = budget_correct = 0
    nonpositive_object_labels = 0
    for item in manifest["results"]:
        run_id = item["run_id"]
        frozen = json.loads((a.input_root / "uniform20_inputs" / run_id / "manifest.json").read_text())
        dino = json.loads((a.dino_root / f"{run_id}.json").read_text())
        remind = json.loads((a.remind_root / f"{run_id}.json").read_text())
        if dino.get("status") != "complete" or remind.get("status") != "complete" or dino.get("labels_read") is not False:
            raise ValueError(f"invalid prediction artifact: {run_id}")
        records = frozen["records"]["rescan"]
        frames = [int(r["frame_id"]) for r in records]
        if [int(x) for x in dino["query_frames"]] != frames:
            raise ValueError(f"DINO frame order mismatch: {run_id}")
        qscan = Path(records[0]["input"]).parent.name if records else run_id
        group = Path(frozen["records"]["reference"][0]["input"]).parent.name
        rows_by_key = {}
        row_index_by_key = {}
        scored = scored_by_id[run_id]
        correct_by_key = {}
        for row_index, row in enumerate(dino["query_rows"]):
            key = (int(row["query_frame"]), int(row["query_index"]))
            if key in rows_by_key:
                raise ValueError(f"duplicate DINO query identity {run_id}/{key}")
            rows_by_key[key] = row
            row_index_by_key[key] = row_index
            correct_by_key[key] = bool(scored["correct"][row_index])
        remind_by_frame: dict[int, list[dict]] = defaultdict(list)
        for row in remind["query_mask_rows"]:
            remind_by_frame[int(row["frame"])].append(row)
        source_count = int(remind["source_frames"])
        relation_budget = sum(int(row["predicted_source_link"]) for row in remind["query_mask_rows"])
        candidates = np.flatnonzero(scored["valid"])
        if relation_budget > len(candidates):
            raise ValueError(f"REMIND budget exceeds valid DINO assignments: {run_id}")
        ranked = np.lexsort((candidates, -scored["scores"][candidates]))
        accepted_budget = set(candidates[ranked[:relation_budget]].tolist())
        expected_keys = set()
        for slot, record in enumerate(records):
            frame = int(record["frame_id"])
            ids = load_ids(record)
            tracker_rows = remind_by_frame.get(source_count + slot, [])
            if len(ids) != int(record["masks"]) or len(tracker_rows) != len(ids):
                raise ValueError(f"mask/label count mismatch: {run_id}/{frame}")
            for index, oid in enumerate(ids.tolist()):
                if int(oid) <= 0:
                    nonpositive_object_labels += 1
                    raise ValueError(f"unidentifiable query object label cannot be clustered: {run_id}/{frame}/{index}/{oid}")
                key = (frame, index)
                expected_keys.add(key)
                drow, rrow = rows_by_key.get(key), tracker_rows[index]
                if drow is None or int(rrow["gt_id"]) != int(oid):
                    raise ValueError(f"query identity ordering mismatch: {run_id}/{frame}/{index}")
                accepted = drow["assigned_source_observation"] is not None and drow["assigned_similarity"] is not None and float(drow["assigned_similarity"]) >= threshold
                budget_accepted = row_index_by_key[key] in accepted_budget
                # The row loader computes correctness from the saved source and
                # query identity arrays without changing the prediction gate.
                dino_correct = correct_by_key[key]
                r_accept = bool(rrow["predicted_source_link"])
                r_correct = bool(rrow["correct_source_link"])
                cluster_key = (group, qscan, int(oid))
                value = clusters.setdefault(cluster_key, {"n": 0, "gate_a": 0, "gate_c": 0,
                                                          "budget_a": 0, "budget_c": 0,
                                                          "remind_a": 0, "remind_c": 0})
                value["n"] += 1
                value["gate_a"] += int(accepted)
                value["remind_a"] += int(r_accept)
                value["remind_c"] += int(r_correct)
                value["gate_c"] += int(accepted and dino_correct)
                value["budget_a"] += int(budget_accepted)
                value["budget_c"] += int(budget_accepted and dino_correct)
                gate_accepts += int(accepted); gate_correct += int(accepted and dino_correct)
                budget_accepts += int(budget_accepted); budget_correct += int(budget_accepted and dino_correct)
                remind_accepts += int(r_accept); remind_correct += int(r_correct)
                group_for[cluster_key] = group
                query_count += 1
        if expected_keys != set(rows_by_key):
            raise ValueError(f"DINO query key set mismatch: {run_id}")
        if query_count > 0 and len(remind["query_mask_rows"]) != sum(len(load_ids(x)) for x in records):
            raise ValueError(f"REMIND row count mismatch: {run_id}")
        relation_count += 1

    groups: dict[str, list[dict[str, int]]] = defaultdict(list)
    for key, value in clusters.items():
        groups[group_for[key]].append(value)
    group_names = sorted(groups)
    expected_gate = json.loads(a.gate_summary.read_text())["validation_transfer"]["metrics"]
    if gate_accepts != int(expected_gate["accepted"]) or gate_correct != int(expected_gate["correct"]):
        raise ValueError(f"gate totals differ from frozen report: {gate_accepts}/{gate_correct}")
    # gate_correct from loader-derived truth, check the REMIND comparison totals too.
    paired = json.loads(a.gate_summary.read_text())["paired_remind_comparison"]
    if remind_accepts != int(paired["remind_total_accepted"]) or remind_correct != int(paired["remind_total_correct"]):
        raise ValueError(f"REMIND totals differ from frozen report: {remind_accepts}/{remind_correct}")
    if budget_accepts != remind_accepts:
        raise ValueError(f"equal-budget DINO accepts differ from REMIND total: {budget_accepts}/{remind_accepts}")

    names = ("object_hit", "object_coverage", "object_precision", "accepted_object_rate")
    methods = ("gate", "budget", "remind")
    arrays = {}
    for group in group_names:
        objects = groups[group]
        arrays[group] = {}
        for method in methods:
            n = np.asarray([x["n"] for x in objects], float)
            acc = np.asarray([x[f"{method}_a"] for x in objects], float)
            cor = np.asarray([x[f"{method}_c"] for x in objects], float)
            arrays[group][method] = {
                "object_hit": (cor > 0).astype(float),
                "object_coverage": cor / n,
                "object_precision": np.divide(cor, acc, out=np.full_like(cor, np.nan), where=acc > 0),
                "accepted_object_rate": (acc > 0).astype(float),
            }
    effects = {}
    contrasts = {"gate_minus_remind": ("gate", "remind"),
                 "equal_budget_dino_minus_remind": ("budget", "remind")}
    for metric in names:
        per_group = {m: [] for m in methods}
        for group in group_names:
            for method in methods:
                vals = arrays[group][method][metric]
                per_group[method].append(float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan)
        g = {m: np.asarray(per_group[m]) for m in methods}
        effects[metric] = {f"{m}_macro": float(np.nanmean(g[m])) for m in methods}
        effects[metric]["contrasts"] = {}
        for label, (left, right) in contrasts.items():
            ok = np.isfinite(g[left]) & np.isfinite(g[right])
            effects[metric]["contrasts"][label] = {
                "difference": float((g[left][ok] - g[right][ok]).mean()),
                "paired_groups": int(ok.sum()),
                "groups_left_higher": int(((g[left] - g[right])[ok] > 0).sum()),
            }

    rng = np.random.default_rng(a.seed)
    boots = {metric: {label: np.full(a.draws, np.nan) for label in contrasts} for metric in names}
    for b in range(a.draws):
        chosen = rng.integers(0, len(group_names), len(group_names))
        for metric in names:
            draw_diffs = {label: [] for label in contrasts}
            for gi in chosen:
                group = group_names[int(gi)]
                nobj = len(groups[group])
                oi = rng.integers(0, nobj, nobj)
                vals = {}
                for method in methods:
                    x = arrays[group][method][metric][oi]
                    vals[method] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
                for label, (left, right) in contrasts.items():
                    if np.isfinite(vals[left]) and np.isfinite(vals[right]):
                        draw_diffs[label].append(vals[left] - vals[right])
            for label in contrasts:
                boots[metric][label][b] = float(np.mean(draw_diffs[label])) if draw_diffs[label] else np.nan
    for metric in names:
        for label in contrasts:
            effects[metric]["contrasts"][label]["nested_group_object_bootstrap_95"] = [
                float(x) for x in np.quantile(boots[metric][label], [.025, .975])]
            effects[metric]["contrasts"][label]["eight_comparison_adjusted_95"] = [
                float(x) for x in np.quantile(boots[metric][label], [.003125, .996875])]

    result = {
        "status": "complete",
        "scope": "post-hoc object-balanced comparison of the source-frozen full-reference DINO cosine gate, per-relation exact-budget DINO cosine ranking, and REMIND on the eight-group extension",
        "threshold": threshold,
        "identity_cluster": "reference group, query scan UUID, and scan-local annotated objectId; object IDs are not matched across scans",
        "groups": len(group_names), "relations": relation_count, "query_masks": query_count,
        "nonpositive_object_identity_labels": nonpositive_object_labels,
        "query_object_clusters": len(clusters),
        "accepted_object_clusters": {m: sum(x[f"{m}_a"] > 0 for x in clusters.values()) for m in methods},
        "totals": {"gate": {"accepted": gate_accepts, "correct": gate_correct},
                   "equal_budget_dino": {"accepted": budget_accepts, "correct": budget_correct},
                   "remind": {"accepted": remind_accepts, "correct": remind_correct}},
        "bootstrap": {"draws": a.draws, "seed": a.seed, "unit": "paired reference group then scan-local query object",
                       "family_adjustment": "percentile intervals use the .003125 and .996875 quantiles across four outcomes and two contrasts"},
        "effects": effects,
        "definitions": {
            "object_hit": "fraction of query objects with at least one correct accepted link, macro-averaged within reference group then equally across groups",
            "object_coverage": "mean per-object fraction of visible mask observations receiving a correct link, macro-averaged within group",
            "object_precision": "mean per-object precision among objects with at least one accepted link, macro-averaged within group",
            "accepted_object_rate": "fraction of query objects receiving at least one accepted link, macro-averaged within group",
        },
        "limits": ["Post-hoc reaggregation of the same benchmark, not independent validation.", "The target contains eight reference groups; bootstrap intervals are descriptive under that small sample.", "Object IDs are clustered only within each query scan; the analysis stops if a nonpositive/unidentifiable ID is present rather than merging unknown masks into one object.", "The exact-budget arm uses each relation's REMIND acceptance count and label-free DINO assigned-cosine ranking; it is a post-hoc operating-point comparison."]
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    labels = {"object_hit": "Objects with >=1 correct link", "object_coverage": "Object-macro correct mask coverage", "object_precision": "Precision among accepted objects", "accepted_object_rate": "Objects with >=1 accepted link"}
    lines = ["# Source-frozen gate object-level sensitivity", "", f"Post-hoc reaggregation over {len(group_names)} reference groups, {relation_count} relations, {query_count} masks, and {len(clusters)} scan-local query-object clusters.", "", "| Outcome | DINO source gate | DINO equal budget | REMIND | Gate − REMIND (95% interval) | Equal-budget DINO − REMIND (8-comparison adjusted interval) |", "|---|---:|---:|---:|---:|---:|"]
    for metric in names:
        x = effects[metric]
        gate = x["contrasts"]["gate_minus_remind"]["nested_group_object_bootstrap_95"]
        equal = x["contrasts"]["equal_budget_dino_minus_remind"]["eight_comparison_adjusted_95"]
        lines.append(f"| {labels[metric]} | {x['gate_macro']:.3f} | {x['budget_macro']:.3f} | {x['remind_macro']:.3f} | {x['contrasts']['gate_minus_remind']['difference']:+.3f} [{gate[0]:+.3f}, {gate[1]:+.3f}] | {x['contrasts']['equal_budget_dino_minus_remind']['difference']:+.3f} [{equal[0]:+.3f}, {equal[1]:+.3f}] |")
    lines += ["", f"The source gate accepted {gate_accepts}/{query_count} masks ({result['accepted_object_clusters']['gate']} objects); equal-budget DINO and REMIND each accepted {budget_accepts}/{remind_accepts} masks ({result['accepted_object_clusters']['budget']} vs {result['accepted_object_clusters']['remind']} objects). For equal-budget DINO, each relation's REMIND accept count sets the budget and the assigned cosine scores select predictions without labels. Intervals for that contrast adjust across four outcomes and two contrasts. All {query_count} query masks have positive identity labels; the analyzer refuses to cluster nonpositive/unidentifiable IDs. Object IDs are scan-local. This is an exploratory reanalysis on the same 3RScan benchmark, not an independent test.", ""]
    (a.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"groups": len(group_names), "relations": relation_count, "query_object_clusters": len(clusters), "totals": result["totals"], "effects": effects}, indent=2))


if __name__ == "__main__":
    main()
