#!/usr/bin/env python3
"""Equal-acceptance DINO/REMIND comparison on the next eight train groups.

This is a post-hoc analysis of saved label-blind assignments. DINO accepts the
same number of valid query assignments as REMIND in each relation, ranked by
assigned cosine without using identity labels.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, rows_for


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_object_view_budget_curve_extension")
    p.add_argument("--dino-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank_extension")
    p.add_argument("--remind-root", type=Path, default=ROOT / "outputs/reports/20261009_remind_style_view_extension")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reports/20261009_dino_extension8_equal_budget_transfer")
    p.add_argument("--draws", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261010)
    a = p.parse_args()

    manifest = json.loads((a.dino_root / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("results", [])) != 8:
        raise ValueError("expected a complete eight-relation extension manifest")
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {
        "queries": 0, "dino_accepted": 0, "dino_correct": 0,
        "remind_accepted": 0, "remind_correct": 0,
    })
    relations = []
    for item in manifest["results"]:
        run_id = item["run_id"]
        frozen = json.loads((a.input_root / "uniform20_inputs" / run_id / "manifest.json").read_text())
        dino = json.loads((a.dino_root / f"{run_id}.json").read_text())
        remind = json.loads((a.remind_root / f"{run_id}.json").read_text())
        if dino.get("status") != "complete" or remind.get("status") != "complete":
            raise ValueError(f"incomplete prediction: {run_id}")
        scored = rows_for(run_id, a.input_root, a.dino_root)
        if len(scored["scores"]) != len(remind["query_mask_rows"]):
            raise ValueError(f"query count mismatch: {run_id}")
        budget = sum(int(r["predicted_source_link"]) for r in remind["query_mask_rows"])
        candidates = np.flatnonzero(scored["valid"])
        if budget > len(candidates):
            raise ValueError(f"REMIND accept budget exceeds DINO valid assignments: {run_id}")
        order = np.lexsort((candidates, -scored["scores"][candidates]))
        accepted = np.zeros(len(scored["scores"]), dtype=bool)
        accepted[candidates[order[:budget]]] = True
        correct = scored["correct"]
        q = len(accepted)
        d_correct = int((accepted & correct).sum())
        r_correct = sum(int(r["correct_source_link"]) for r in remind["query_mask_rows"])
        r_accepted = sum(int(r["predicted_source_link"]) for r in remind["query_mask_rows"])
        if int(accepted.sum()) != r_accepted or r_accepted != budget:
            raise ValueError(f"relation-level exact-budget mismatch: {run_id}")
        group = Path(frozen["records"]["reference"][0]["input"]).parent.name
        g = groups[group]
        g["queries"] += q
        g["dino_accepted"] += int(accepted.sum())
        g["dino_correct"] += d_correct
        g["remind_accepted"] += r_accepted
        g["remind_correct"] += r_correct
        relations.append({"run_id": run_id, "group": group, "queries": q,
                          "accepted": r_accepted, "dino_correct": d_correct,
                          "remind_correct": r_correct})

    if len(groups) != 8:
        raise ValueError(f"expected eight reference groups, found {len(groups)}")
    group_ids = sorted(groups)
    metrics = {}
    rng = np.random.default_rng(a.seed)
    ix = rng.integers(0, len(group_ids), size=(a.draws, len(group_ids)))
    for metric, denominator in (("precision", "accepted"), ("coverage", "queries")):
        diffs = []
        per_group = {}
        for group in group_ids:
            g = groups[group]
            dden = g["dino_accepted"] if denominator == "accepted" else g["queries"]
            rden = g["remind_accepted"] if denominator == "accepted" else g["queries"]
            dvalue = g["dino_correct"] / dden if dden else float("nan")
            rvalue = g["remind_correct"] / rden if rden else float("nan")
            if not np.isfinite(dvalue) or not np.isfinite(rvalue):
                raise ValueError(f"undefined {metric} in reference group {group}")
            per_group[group] = {"dino": dvalue, "remind": rvalue, "difference": dvalue-rvalue}
            diffs.append(dvalue-rvalue)
        diffs = np.asarray(diffs)
        boot = diffs[ix].mean(axis=1)
        metrics[metric] = {
            "dino_macro": float(np.mean([per_group[x]["dino"] for x in group_ids])),
            "remind_macro": float(np.mean([per_group[x]["remind"] for x in group_ids])),
            "difference_dino_minus_remind": float(diffs.mean()),
            "ordinary_percentile_95": [float(x) for x in np.quantile(boot, [.025, .975])],
            "two_outcome_adjusted_percentile_95": [float(x) for x in np.quantile(boot, [.0125, .9875])],
            "groups_dino_higher": int((diffs > 0).sum()),
            "groups_tied": int((diffs == 0).sum()),
            "groups_remind_higher": int((diffs < 0).sum()),
            "per_group": per_group,
        }
    totals = {name: sum(g[name] for g in groups.values()) for name in next(iter(groups.values()))}
    if totals["dino_accepted"] != totals["remind_accepted"]:
        raise ValueError("pooled DINO/REMIND acceptance totals differ")
    result = {
        "status": "complete",
        "scope": "post-hoc equal per-relation acceptance comparison on the second hash-ordered eight-relation train extension",
        "selection": "within each relation, accept the top assigned-cosine DINO matches among valid assignments up to REMIND's predicted-source-link count; identity labels are not used in selection",
        "relations": len(relations), "reference_groups": len(group_ids),
        "totals": totals,
        "bootstrap": {"draws": a.draws, "seed": a.seed, "unit": "paired reference group",
                       "family_adjustment": "Bonferroni percentile intervals across precision and coverage"},
        "effects": metrics,
        "group_rows": {k: groups[k] for k in group_ids},
        "relation_rows": relations,
        "limits": ["The eight-group extension was selected after earlier cohort outcomes were known and has been explored on the same 3RScan benchmark.", "The comparison matches accepted query counts, not the identities of accepted queries.", "This is not independent-site validation."]
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Equal-budget DINO/REMIND on the eight-group extension", "",
             "The DINO arm accepts the top valid assigned-cosine matches up to REMIND's per-relation accepted-link count. Scores, not identity labels, select DINO accepts. Both arms accept exactly 137/999 masks.", "",
             "| Outcome | DINO | REMIND | Difference (two-outcome-adjusted paired 95% interval) |",
             "|---|---:|---:|---:|"]
    for name, title in (("precision", "Accepted-link precision"), ("coverage", "Correct coverage of all query masks")):
        x = metrics[name]
        lo, hi = x["two_outcome_adjusted_percentile_95"]
        lines.append(f"| {title} | {x['dino_macro']:.3f} | {x['remind_macro']:.3f} | {x['difference_dino_minus_remind']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", f"DINO gets {totals['dino_correct']} correct accepted links and REMIND gets {totals['remind_correct']}; DINO is higher in {metrics['precision']['groups_dino_higher']}/8 groups for precision and {metrics['coverage']['groups_dino_higher']}/8 for correct coverage.", "", "This is post-hoc same-dataset evidence on eight groups. It is not an independent replication; the target cohort was selected after earlier outcomes were known.", ""]
    (a.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"groups": len(group_ids), "totals": totals,
                      "effects": {k: {key: v for key, v in metric.items() if key != "per_group"} for k, metric in metrics.items()}}, indent=2))


if __name__ == "__main__":
    main()
