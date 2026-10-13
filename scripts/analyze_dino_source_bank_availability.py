#!/usr/bin/env python3
"""Separate DINO gate errors caused by absent identity from retrieval mistakes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, load_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    p.add_argument("--prediction-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    p.add_argument("--gate-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_gate/source_frozen_gate.json")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reports/20261009_dino_source_bank_availability")
    p.add_argument("--draws", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261012)
    a = p.parse_args()
    threshold = float(json.loads(a.gate_summary.read_text())["frozen_threshold"])
    relations = load_split(a.input_root, a.prediction_root)
    groups: dict[str, dict] = defaultdict(lambda: {
        "query_masks": 0, "identity_available": 0, "identity_absent": 0,
        "accepted_available": 0, "correct_available": 0, "wrong_available": 0,
        "accepted_absent": 0, "correct_absent": 0,
    })
    relation_rows = []
    for rel in relations:
        accepted = rel["valid"] & (rel["scores"] >= threshold)
        available = rel["identity_available"]
        correct = rel["correct"] & accepted
        if np.any(correct & ~available):
            raise ValueError(f"correct assignment cannot have an absent physical identity: {rel['run_id']}")
        vals = {
            "query_masks": len(accepted),
            "identity_available": int(available.sum()),
            "identity_absent": int((~available).sum()),
            "accepted_available": int((accepted & available).sum()),
            "correct_available": int(correct.sum()),
            "wrong_available": int((accepted & available & ~correct).sum()),
            "accepted_absent": int((accepted & ~available).sum()),
            "correct_absent": 0,
        }
        g = groups[rel["environment_id"]]
        for key, value in vals.items(): g[key] += value
        relation_rows.append({"run_id": rel["run_id"], "environment_id": rel["environment_id"], **vals})

    group_rows = []
    for env, g in sorted(groups.items()):
        group_rows.append({"environment_id": env, **g,
                           "available_fraction": g["identity_available"] / g["query_masks"],
                           "accepted_precision_on_available": g["correct_available"] / g["accepted_available"] if g["accepted_available"] else None,
                           "correct_coverage_of_available": g["correct_available"] / g["identity_available"] if g["identity_available"] else None,
                           "absent_identity_acceptance_rate": g["accepted_absent"] / g["identity_absent"] if g["identity_absent"] else None,
                           "available_wrong_share_of_false_accepts": g["wrong_available"] / (g["wrong_available"] + g["accepted_absent"]) if g["wrong_available"] + g["accepted_absent"] else None})
    totals = {key: sum(g[key] for g in groups.values()) for key in next(iter(groups.values()))}
    false_accepts = totals["wrong_available"] + totals["accepted_absent"]
    pool = {**totals,
            "available_fraction": totals["identity_available"] / totals["query_masks"],
            "accepted_precision_on_available": totals["correct_available"] / totals["accepted_available"],
            "correct_coverage_of_available": totals["correct_available"] / totals["identity_available"],
            "absent_identity_acceptance_rate": totals["accepted_absent"] / totals["identity_absent"],
            "false_accepts": false_accepts,
            "false_accepts_when_identity_available": totals["wrong_available"],
            "false_accepts_when_identity_absent": totals["accepted_absent"],
            "false_accept_share_available_identity": totals["wrong_available"] / false_accepts,
            "false_accept_share_absent_identity": totals["accepted_absent"] / false_accepts}
    rng = np.random.default_rng(a.seed)
    n = len(group_rows)
    ix = rng.integers(0, n, size=(a.draws, n))
    bootstrap = {}
    for metric in ("available_fraction", "accepted_precision_on_available", "correct_coverage_of_available",
                   "absent_identity_acceptance_rate", "available_wrong_share_of_false_accepts"):
        values = np.asarray([r[metric] if r[metric] is not None else np.nan for r in group_rows], dtype=float)
        draws = np.nanmean(values[ix], axis=1)
        bootstrap[metric] = {"estimate": float(np.nanmean(values)),
                             "percentile_95": [float(x) for x in np.nanquantile(draws, [.025, .975])],
                             "groups_with_defined_metric": int(np.isfinite(values).sum())}
    result = {"status": "complete", "scope": "source-frozen all-reference-view DINO gate; physical identity availability is scored after anonymous inference",
              "threshold": threshold, "pooled": pool, "macro_reference_group": bootstrap,
              "reference_groups": group_rows, "relations": relation_rows,
              "bootstrap_design": {"draws": a.draws, "seed": a.seed, "unit": "released reference scan ID"},
              "limits": [
                  "Identity availability means the query physical identity occurs in at least one annotated reference mask; it is an evaluation-only oracle, not an inference-time input.",
                  "Availability does not guarantee a usable visual exemplar or correct assignment; an error with an available identity is a retrieval/assignment failure, not a proof that a better descriptor could solve it.",
                  "This remains post-hoc analysis of explored 3RScan validation masks with privileged annotations, not independent-site evidence.",
              ]}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    b = result["macro_reference_group"]
    report = f"""# DINO source-bank identity availability audit

## Definition

For each query mask, identity is available if its released physical instance ID occurs in any annotated reference-bank mask for the same relation. This label is used only after inference. The frozen DINO gate is cosine similarity >= {threshold:.6f}.

## Results

| Query identity status | Query masks | Gate accepts | Correct | Accepted precision | Correct coverage within status |
|---|---:|---:|---:|---:|---:|
| Present in reference bank | {pool['identity_available']:,} | {pool['accepted_available']:,} | {pool['correct_available']:,} | {pool['accepted_precision_on_available']:.1%} | {pool['correct_coverage_of_available']:.1%} |
| Absent from reference bank | {pool['identity_absent']:,} | {pool['accepted_absent']:,} | 0 | 0.0% | 0.0% |

The gate has {pool['false_accepts']:,} incorrect accepted links. Of these, {pool['false_accepts_when_identity_available']:,} ({pool['false_accept_share_available_identity']:.1%}) occur despite the correct identity being represented somewhere in the source bank; {pool['false_accepts_when_identity_absent']:,} ({pool['false_accept_share_absent_identity']:.1%}) concern identities absent from the bank. The latter are candidate-availability failures for any source-only identity matcher; the former indicate that availability alone is insufficient and the system selects a wrong source observation or identity.

Reference-group bootstrap 95% intervals: fraction of query identities available {b['available_fraction']['estimate']:.1%} [{b['available_fraction']['percentile_95'][0]:.1%},{b['available_fraction']['percentile_95'][1]:.1%}]; precision conditional on identity availability {b['accepted_precision_on_available']['estimate']:.1%} [{b['accepted_precision_on_available']['percentile_95'][0]:.1%},{b['accepted_precision_on_available']['percentile_95'][1]:.1%}]; correct coverage among available identities {b['correct_coverage_of_available']['estimate']:.1%} [{b['correct_coverage_of_available']['percentile_95'][0]:.1%},{b['correct_coverage_of_available']['percentile_95'][1]:.1%}]. Intervals resample the 47 released reference IDs and condition on the frozen threshold.

This is an oracle availability audit, not a deployable policy: the query identity is consulted only after inference. The benchmark and masks have been explored previously, so the result is not independent-site confirmation.
"""
    (a.output / "report.md").write_text(report)
    print(json.dumps({"status": "complete", "threshold": threshold, "pooled": pool,
                      "macro_reference_group": bootstrap}, indent=2))


if __name__ == "__main__":
    main()
