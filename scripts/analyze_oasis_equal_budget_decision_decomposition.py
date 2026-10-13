#!/usr/bin/env python3
"""Decompose the OASIS-style versus DINO-mutual coverage gap by decision type."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_segment_matcher_population import identity_links, semantics

SEED = 20261009
ORDERS = ("match_fraction_then_confidence", "confidence_then_match_fraction")
OUTCOMES = ("semantic", "linked_identity")
COMPONENTS = ("link_choice", "acceptance_set", "net")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, default=Path("outputs/reports/20261008_segmast3r_validation"))
    p.add_argument("--oasis-root", type=Path, default=Path("outputs/reports/20261009_oasis_style_dinov2"))
    p.add_argument("--data-root", type=Path, default=Path("data/3rscan"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_oasis_decision_decomposition"))
    p.add_argument("--draws", type=int, default=100_000)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((a.input_root / "frozen_segment_population.json").read_text())
    population = frozen["population"]
    population_sha = hashlib.sha256((a.input_root / "frozen_segment_population.json").read_bytes()).hexdigest()
    oasis_protocol = json.loads((a.oasis_root / "protocol.json").read_text())
    if oasis_protocol.get("input_population_sha256") != population_sha:
        raise ValueError("OASIS inference population hash differs from the DINO input population")
    oasis_manifest = json.loads((a.oasis_root / "anonymous_inference_manifest.json").read_text())
    dino_manifest = json.loads((a.input_root / "dino_inference_manifest.json").read_text())
    run_ids = {r["run_id"] for r in population}
    if len(population) != 110 or len(oasis_manifest) != 110 or len(dino_manifest) != 110:
        raise ValueError("expected complete 110-relation cohort")
    if any(r["status"] != "complete" for r in oasis_manifest + dino_manifest):
        raise ValueError("inference manifest has an incomplete relation")
    oasis_by_run = {r["run_id"]: r for r in oasis_manifest}
    dino_by_run = {r["run_id"]: r for r in dino_manifest}
    if set(oasis_by_run) != run_ids or set(dino_by_run) != run_ids:
        raise ValueError("inference population mismatch")
    metadata = json.loads((a.data_root / "3RScan.json").read_text())

    rows = []
    for relation in population:
        run = relation["run_id"]
        info = oasis_by_run[run]
        ds = json.loads((a.input_root / "dino_segment_predictions" / run / "inference_summary.json").read_text())
        if ds.get("identity_targets_read") or ds.get("semantic_labels_read"):
            raise ValueError("DINO features were not label-free")
        dino_records = {(int(x["reference_frame"]), int(x["query_frame"])): x for x in ds["records"]}
        inputs = json.loads((a.input_root / "uniform20_inputs" / run / "manifest.json").read_text())["records"]
        refs = {int(x["frame_id"]): x for x in inputs["reference"]}
        queries = {int(x["frame_id"]): x for x in inputs["rescan"]}
        # Mesh-derived semantic vocabularies and cross-scan links are fixed per
        # relation; load them once rather than once per frame pair.
        ref_sem, ref_names = semantics(a.data_root, relation["reference_scan"])
        query_sem, query_names = semantics(a.data_root, relation["rescan_scan"])
        links, _ = identity_links(metadata, relation["reference_scan"], relation["rescan_scan"], ref_names, query_names)
        for orec in info["records"]:
            rf, qf = int(orec["reference_frame"]), int(orec["query_frame"])
            drec = dino_records[(rf, qf)]
            for record in (refs[rf], queries[qf]):
                if hashlib.sha256(Path(record["input"]).read_bytes()).hexdigest() != record["input_sha256"]:
                    raise ValueError("anonymous input hash mismatch")
            with np.load(drec["predictions"]) as dp:
                scores = dp["scores"].copy()
                q_to_r = dp["query_to_reference"].copy()
                mutual = dp["mutual_assigned"].copy()
            budget = int(mutual.sum())
            dino_assignment = np.full(len(q_to_r), -1, dtype=np.int64)
            dino_assignment[mutual] = q_to_r[mutual]
            with np.load(orec["prediction"]) as op:
                fractions = op["match_fraction"]
                confidence = op["mean_confidence"]
                rrows = op["assigned_reference"].astype(np.int64)
                qcols = op["assigned_query"].astype(np.int64)
                fs, cs = fractions[rrows, qcols], confidence[rrows, qcols]
            expected_shape = (int(orec["query_masks"]), int(orec["reference_masks"]))
            if scores.shape != expected_shape or fractions.shape != expected_shape[::-1] or confidence.shape != expected_shape[::-1]:
                raise ValueError(f"dimension mismatch for {run} {rf} {qf}")
            if (len(rrows) != len(qcols) or budget > len(qcols)
                    or len(np.unique(qcols)) != len(qcols) or not np.isfinite(scores).all()
                    or not np.isfinite(fs).all() or not np.isfinite(cs).all()):
                raise ValueError("invalid assignment budget/support")
            ref_ids = np.load(refs[rf]["identity_targets"])
            query_ids = np.load(queries[qf]["identity_targets"])
            if len(ref_ids) != int(orec["reference_masks"]) or len(query_ids) != int(orec["query_masks"]):
                raise ValueError("identity target count differs from anonymous masks")

            # Labels are used only after both prediction sets are complete.
            qsem = np.asarray([query_sem.get(int(x), 0) for x in query_ids])
            semantic_target = qsem > 0
            expected = np.asarray([links.get(int(x), -1) for x in query_ids])
            identity_target = semantic_target & (expected > 0)
            semantic_edges = np.asarray([ref_sem.get(int(x), 0) for x in ref_ids])

            for order in ORDERS:
                ranked = (np.lexsort((qcols, -cs, -fs)) if order == ORDERS[0]
                          else np.lexsort((qcols, -fs, -cs)))
                oa = np.full(len(query_ids), -1, dtype=np.int64)
                chosen = ranked[:budget]
                oa[qcols[chosen]] = rrows[chosen]
                if int((oa >= 0).sum()) != budget or int((dino_assignment >= 0).sum()) != budget:
                    raise ValueError("exact acceptance budget mismatch")
                da, oa_mask = dino_assignment >= 0, oa >= 0
                same = da & oa_mask & (dino_assignment == oa)
                both_diff = da & oa_mask & (dino_assignment != oa)
                dino_only = da & ~oa_mask
                oasis_only = oa_mask & ~da
                if int(same.sum() + both_diff.sum() + dino_only.sum()) != budget:
                    raise ValueError("DINO acceptance partition does not sum to budget")
                if int(same.sum() + both_diff.sum() + oasis_only.sum()) != budget:
                    raise ValueError("OASIS acceptance partition does not sum to budget")

                dsem = np.zeros(len(query_ids), bool)
                osem = np.zeros(len(query_ids), bool)
                did = np.zeros(len(query_ids), bool)
                oid = np.zeros(len(query_ids), bool)
                dsem[da] = semantic_edges[dino_assignment[da]] == qsem[da]
                osem[oa_mask] = semantic_edges[oa[oa_mask]] == qsem[oa_mask]
                did[da] = ref_ids[dino_assignment[da]] == expected[da]
                oid[oa_mask] = ref_ids[oa[oa_mask]] == expected[oa_mask]

                base = dict(run_id=run, environment_id=relation["environment_id"],
                            reference_frame=rf, query_frame=qf, ranking=order,
                            budget=budget, query_masks=len(query_ids),
                            both_same_pair=int(same.sum()),
                            both_different_pair=int(both_diff.sum()),
                            dino_only_accept=int(dino_only.sum()),
                            oasis_only_accept=int(oasis_only.sum()))
                for name, target, dcorrect, ocorrect in (
                    ("semantic", semantic_target, dsem, osem),
                    ("linked_identity", identity_target, did, oid),
                ):
                    denominator = int(target.sum())
                    # Exact decomposition of correct-query coverage difference.
                    link_choice = int((dcorrect[both_diff] & target[both_diff]).sum()
                                      - (ocorrect[both_diff] & target[both_diff]).sum())
                    acceptance_set = int((dcorrect[dino_only] & target[dino_only]).sum()
                                         - (ocorrect[oasis_only] & target[oasis_only]).sum())
                    d_correct = int((dcorrect & target).sum())
                    o_correct = int((ocorrect & target).sum())
                    if link_choice + acceptance_set != d_correct - o_correct:
                        raise ValueError(f"coverage decomposition failed for {run} {order} {name}")
                    base[f"{name}_targets"] = denominator
                    base[f"{name}_dino_correct"] = d_correct
                    base[f"{name}_oasis_correct"] = o_correct
                    base[f"{name}_link_choice_delta"] = link_choice
                    base[f"{name}_acceptance_set_delta"] = acceptance_set
                    base[f"{name}_net_delta"] = d_correct - o_correct
                rows.append(base)

    per_frame = pd.DataFrame(rows)
    if len(per_frame) != 2 * 2181 or per_frame.duplicated(["run_id", "reference_frame", "query_frame", "ranking"]).any():
        raise ValueError("expected both rankings for all 2,181 unique frozen frame pairs")
    per_frame.to_csv(a.output_dir / "per_frame_decomposition.csv", index=False)
    count_cols = [c for c in per_frame.columns if c not in {"run_id", "environment_id", "reference_frame", "query_frame", "ranking"}]
    per_env = per_frame.groupby(["environment_id", "ranking"], sort=True)[count_cols].sum().reset_index()
    for outcome in OUTCOMES:
        den = per_env[f"{outcome}_targets"]
        for component in COMPONENTS:
            per_env[f"{outcome}_{component}_coverage_delta"] = per_env[f"{outcome}_{component}_delta"] / den
    per_env.to_csv(a.output_dir / "per_environment_decomposition.csv", index=False)

    groups = sorted(per_env.environment_id.unique())
    if len(groups) != 47:
        raise ValueError(f"expected 47 reference groups; found {len(groups)}")
    rng = np.random.default_rng(SEED)
    sample = rng.integers(0, len(groups), size=(a.draws, len(groups)))
    family = len(ORDERS) * len(OUTCOMES) * len(COMPONENTS)
    alpha = 0.05 / (2 * family)
    results = {}
    draws_out = {}
    for order in ORDERS:
        results[order] = {}
        subset = per_env[per_env.ranking.eq(order)].set_index("environment_id").loc[groups]
        for outcome in OUTCOMES:
            results[order][outcome] = {}
            for component in COMPONENTS:
                col = f"{outcome}_{component}_coverage_delta"
                values = subset[col].to_numpy()
                draws = values[sample].mean(axis=1)
                key = f"{order}__{outcome}__{component}"
                draws_out[key] = draws
                results[order][outcome][component] = {
                    "macro_delta_pp": float(values.mean() * 100),
                    "bonferroni_adjusted_95_ci_pp": [float(x * 100) for x in np.quantile(draws, [alpha, 1-alpha])],
                }
    pd.DataFrame(draws_out).to_csv(a.output_dir / "paired_group_bootstrap.csv.gz", index=False, compression="gzip")

    overlap = {}
    for order, group in per_env.groupby("ranking"):
        frame = per_frame[per_frame.ranking.eq(order)]
        overlap[order] = {
            "pooled_query_acceptance_jaccard": float((frame.both_same_pair + frame.both_different_pair).sum()
                / (frame.budget.sum() * 2 - (frame.both_same_pair + frame.both_different_pair).sum())),
            "shared_query_accepts": int((frame.both_same_pair + frame.both_different_pair).sum()),
            "same_query_same_reference_links": int(frame.both_same_pair.sum()),
            "both_accept_different_reference": int(frame.both_different_pair.sum()),
            "dino_only_accepts": int(frame.dino_only_accept.sum()),
            "oasis_only_accepts": int(frame.oasis_only_accept.sum()),
            "total_accepts_each_arm": int(frame.budget.sum()),
        }
    summary = {
        "cohort": "110 official 3RScan validation relations, 2,181 frame pairs, 47 released reference groups",
        "mechanism": "Exact decomposition of macro semantic/linked-identity coverage difference into different-reference links among queries accepted by both and queries accepted by only one method. Same-link joint accepts contribute zero.",
        "rankings": list(ORDERS),
        "coverage_gap_decomposition": results,
        "acceptance_and_pair_overlap": overlap,
        "bootstrap": {"draws": a.draws, "seed": SEED, "cluster": "47 released reference groups", "family_size": family,
                      "interval": "paired percentile intervals Bonferroni-adjusted over two rankings x two outcomes x three components"},
        "interpretation_limit": "Post-hoc mechanism decomposition on the same explored 3RScan validation cohort; not independent-site replication or a causal intervention.",
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (a.output_dir / "report.md").write_text(make_report(summary))
    print(json.dumps(summary, indent=2))


def make_report(s):
    lines = ["# Where does the equal-budget coverage gap come from?", "",
             "This post-hoc decomposition uses the same 110-relation validation cohort, 2,181 frame pairs, 47 released reference groups, masks, and DINOv2 ViT-G features as the exact-budget OASIS-style comparison. For each query mask, both methods either accept the same reference link, accept different reference links, or only one method accepts. The semantic and linked-identity coverage gap can therefore be split exactly into a **link-choice** component (both accept but choose different references) and an **acceptance-set** component (one accepts and the other rejects). Queries with the same selected link contribute zero. Labels are used only for this post-hoc scoring.", "",
             "## Acceptance overlap", "",
             "| OASIS ranking | Shared query accepts | Same query and reference | Both accept, reference differs | DINO-only accepts | OASIS-only accepts |", "|---|---:|---:|---:|---:|---:|"]
    for order in ORDERS:
        v = s["acceptance_and_pair_overlap"][order]
        lines.append(f"| {order.replace('_',' ')} | {v['shared_query_accepts']} | {v['same_query_same_reference_links']} | {v['both_accept_different_reference']} | {v['dino_only_accepts']} | {v['oasis_only_accepts']} |")
    lines += ["", "## Exact macro coverage-gap decomposition", "", "Values are DINO-mutual minus OASIS-style coverage in percentage points. Positive values mean native DINO mutual has more correct coverage. Components sum to the net effect; intervals use paired 47-group bootstrap with a Bonferroni correction across 12 component effects.", ""]
    for order in ORDERS:
        lines += [f"### {order.replace('_',' ')}", "", "| Outcome | Link-choice component | Acceptance-set component | Net coverage difference |", "|---|---:|---:|---:|"]
        for outcome in OUTCOMES:
            vals = s["coverage_gap_decomposition"][order][outcome]
            cells = []
            for comp in ("link_choice", "acceptance_set", "net"):
                v = vals[comp]
                lo, hi = v["bonferroni_adjusted_95_ci_pp"]
                cells.append(f"{v['macro_delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {outcome.replace('_',' ')} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Scope", "", s["interpretation_limit"], "", "Frame/group decomposition rows and 100,000 paired bootstrap draws accompany this report. Reproduce with `python scripts/analyze_oasis_equal_budget_decision_decomposition.py`."]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    main()
