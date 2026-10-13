#!/usr/bin/env python3
"""Score an already completed, label-free unseen-relation exact-budget audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_segment_matcher_population import identity_links, semantics

SEED = 20261009
METRICS = {
    "semantic_coverage": ("semantic_correct", "semantic_queries"),
    "semantic_precision": ("semantic_correct", "semantic_assigned"),
    "linked_identity_coverage": ("linked_correct", "linked_queries"),
    "linked_identity_precision": ("linked_correct", "linked_assigned"),
    "linked_semantic_precision": ("linked_semantic_correct", "linked_assigned"),
}
RANKINGS = ("match_fraction_then_confidence", "confidence_then_match_fraction")


def counts(qids, rids, qsem, rsem, links, selected):
    semantic_eval = qsem > 0
    expected = np.asarray([links.get(int(x), -1) for x in qids], dtype=np.int64)
    linked = semantic_eval & (expected > 0)
    assigned = selected >= 0
    mapped = np.full(len(qids), -1, dtype=np.int64)
    predicted_semantic = np.zeros(len(qids), dtype=np.int64)
    mapped[assigned] = rids[selected[assigned]]
    predicted_semantic[assigned] = rsem[selected[assigned]]
    identity_ok = assigned & linked & (mapped == expected)
    semantic_ok = assigned & semantic_eval & (predicted_semantic == qsem)
    return {
        "raw_queries": len(qids), "raw_assigned": int(assigned.sum()),
        "semantic_queries": int(semantic_eval.sum()),
        "semantic_assigned": int((assigned & semantic_eval).sum()),
        "semantic_correct": int(semantic_ok.sum()),
        "linked_queries": int(linked.sum()),
        "linked_assigned": int((assigned & linked).sum()),
        "linked_correct": int(identity_ok.sum()),
        "linked_semantic_correct": int((linked & semantic_ok).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--oasis-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/3rscan"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--draws", type=int, default=100_000)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((a.input_root / "frozen_segment_population.json").read_text())
    population = frozen["population"]
    oasis = json.loads((a.oasis_root / "anonymous_inference_manifest.json").read_text())
    dino = json.loads((a.input_root / "dino_inference_manifest.json").read_text())
    expected = {r["run_id"] for r in population}
    for manifest in (oasis, dino):
        if {r["run_id"] for r in manifest} != expected or any(r["status"] != "complete" for r in manifest):
            raise ValueError("both complete anonymous inference manifests must cover the frozen population")

    # Validate every anonymous prediction and record per-frame budgets before any labels are read.
    dino_by_run = {}
    for row in dino:
        summary = json.loads((a.input_root / "dino_segment_predictions" / row["run_id"] / "inference_summary.json").read_text())
        if summary.get("identity_targets_read") or summary.get("semantic_labels_read"):
            raise ValueError("DINO inference read labels")
        dino_by_run[row["run_id"]] = {(int(x["reference_frame"]), int(x["query_frame"])): x for x in summary["records"]}
    oasis_by_run = {row["run_id"]: row for row in oasis}
    records = []
    budgets = {}
    for relation in population:
        run = relation["run_id"]
        src = json.loads((a.input_root / "uniform20_inputs" / run / "manifest.json").read_text())["records"]
        refs = {int(x["frame_id"]): x for x in src["reference"]}
        queries = {int(x["frame_id"]): x for x in src["rescan"]}
        o_records = oasis_by_run[run]["records"]
        if len(o_records) != min(len(refs), len(queries)):
            raise ValueError(f"incomplete OASIS frames: {run}")
        d_records = dino_by_run[run]
        for o in o_records:
            rf, qf = int(o["reference_frame"]), int(o["query_frame"])
            if (rf, qf) not in d_records:
                raise ValueError(f"missing DINO prediction for {run} {rf} {qf}")
            for frame in (refs[rf], queries[qf]):
                import hashlib
                if hashlib.sha256(Path(frame["input"]).read_bytes()).hexdigest() != frame["input_sha256"]:
                    raise ValueError("anonymous input hash mismatch")
            dp = np.load(d_records[(rf, qf)]["predictions"])
            budget = int(dp["mutual_assigned"].sum())
            dp.close()
            with np.load(o["prediction"]) as op:
                frac, conf = op["match_fraction"], op["mean_confidence"]
                rr, qc = op["assigned_reference"], op["assigned_query"]
                if len(rr) != len(qc) or len(np.unique(qc)) != len(qc) or budget > len(qc):
                    raise ValueError("invalid OASIS assignment support or impossible exact budget")
                fs, cs = frac[rr, qc], conf[rr, qc]
                if not np.isfinite(fs).all() or not np.isfinite(cs).all():
                    raise ValueError("non-finite OASIS score")
            records.append((relation, o, refs[rf], queries[qf], rf, qf, budget, rr.copy(), qc.copy(), fs.copy(), cs.copy()))
            budgets[(run, rf, qf)] = budget

    # All anonymous inference is now complete and validated; open annotation targets only below.
    metadata = json.loads((a.data_root / "3RScan.json").read_text())
    rows = []
    for relation, o, ref, query, rf, qf, budget, rr, qc, fs, cs in records:
        ref_sem, ref_names = semantics(a.data_root, relation["reference_scan"])
        query_sem, query_names = semantics(a.data_root, relation["rescan_scan"])
        links, _ = identity_links(metadata, relation["reference_scan"], relation["rescan_scan"], ref_names, query_names)
        if len(set(links.values())) != len(links):
            raise ValueError(f"identity link collision: {relation['run_id']}")
        qids = np.load(query["identity_targets"])
        rids = np.load(ref["identity_targets"])
        qsem = np.asarray([query_sem.get(int(x), 0) for x in qids])
        rsem = np.asarray([ref_sem.get(int(x), 0) for x in rids])
        with np.load(dino_by_run[relation["run_id"]][(rf, qf)]["predictions"]) as dp:
            scores = dp["scores"].copy()
            selected = dp["query_to_reference"].copy()
            mutual = dp["mutual_assigned"].copy()
        if scores.shape != (len(qids), len(rids)) or int(mutual.sum()) != budget:
            raise ValueError("DINO prediction dimensions or budget do not match held-out targets")
        for ranking in RANKINGS:
            order = (np.lexsort((qc, -cs, -fs)) if ranking == RANKINGS[0]
                     else np.lexsort((qc, -fs, -cs)))
            chosen = order[:budget]
            pred = np.full(len(qids), -1, dtype=np.int64)
            pred[qc[chosen]] = rr[chosen]
            c = counts(qids, rids, qsem, rsem, links, pred)
            rows.append(dict(run_id=relation["run_id"], environment_id=relation["environment_id"],
                             reference_frame=rf, query_frame=qf, arm=ranking, **c))
        pred = np.full(len(qids), -1, dtype=np.int64)
        pred[mutual] = selected[mutual]
        rows.append(dict(run_id=relation["run_id"], environment_id=relation["environment_id"],
                         reference_frame=rf, query_frame=qf, arm="dino_mutual",
                         **counts(qids, rids, qsem, rsem, links, pred)))

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_frame_equal_budget.csv", index=False)
    groups = frame.groupby(["environment_id", "arm"], sort=True).sum(numeric_only=True).reset_index()
    summary = {}
    for arm, group in groups.groupby("arm", sort=False):
        vals = {}
        for metric, (num, den) in METRICS.items():
            values = group[num] / group[den].replace(0, np.nan)
            if values.isna().any():
                raise ValueError(f"undefined {metric}")
            groups.loc[group.index, metric] = values.to_numpy()
            vals[metric] = float(values.mean())
        summary[arm] = vals
    groups.to_csv(out / "per_environment_equal_budget.csv", index=False)
    rng = np.random.default_rng(SEED)
    envs = sorted(groups.environment_id.unique())
    if len(envs) != len(population):
        raise ValueError("expected one distinct environment per frozen relation")
    sampled = rng.integers(0, len(envs), size=(a.draws, len(envs)))
    alpha = 0.05 / (2 * len(RANKINGS) * len(METRICS))
    intervals, boot = {}, {}
    indexed = {arm: groups[groups.arm.eq(arm)].set_index("environment_id").loc[envs] for arm in summary}
    for arm in RANKINGS:
        intervals[arm], boot[arm] = {}, {}
        for metric in METRICS:
            diff = indexed[arm][metric].to_numpy() - indexed["dino_mutual"][metric].to_numpy()
            draws = diff[sampled].mean(axis=1)
            intervals[arm][metric] = [float(x) for x in np.quantile(draws, [alpha, 1-alpha])]
            boot[arm][metric] = draws
    pd.DataFrame({f"{arm}__{m}": v for arm, ms in boot.items() for m, v in ms.items()}).to_csv(
        out / "paired_group_bootstrap.csv.gz", index=False, compression="gzip")
    result = {
        "cohort": "8 hash-selected, previously unused 3RScan train relations; 160 frame pairs; same dataset, site independence unknown",
        "comparison": "OASIS-style DINOv2 patch-to-mask/Hungarian assignment truncated to native DINO mutual raw acceptance count for each frame",
        "total_raw_accepts_dino_mutual": int(frame[frame.arm.eq("dino_mutual")].raw_assigned.sum()),
        "total_raw_accepts_each_oasis_arm": {arm: int(frame[frame.arm.eq(arm)].raw_assigned.sum()) for arm in RANKINGS},
        "per_frame_acceptance_budget_exact": True,
        "bootstrap": {"cluster": "8 frozen relations", "draws": a.draws, "seed": SEED,
                      "family_size": len(RANKINGS)*len(METRICS), "interval": "Bonferroni-adjusted paired percentile interval"},
        "macro_metrics": summary,
        "oasis_minus_dino_mutual_macro_effects": {arm: {m: summary[arm][m]-summary["dino_mutual"][m] for m in METRICS} for arm in RANKINGS},
        "oasis_minus_dino_mutual_adjusted_intervals": intervals,
        "limitations": ["Within-dataset held-out relation confirmation only; only eight groups.",
                        "Disjoint scan IDs do not establish independent physical sites; prior all-train mesh screens exist.",
                        "Not an OASIS-Map system reproduction; uses DINOv2, privileged annotation-derived masks, and fixed chronological pairs.",
                        "This relation cohort was selected for a prior source-fixed RGB-D experiment; comparison outcomes here were not used for selection."],
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    protocol = {
        "status": "complete",
        "population_sha256": hashlib.sha256((a.input_root / "frozen_segment_population.json").read_bytes()).hexdigest(),
        "population_source": "outputs/reports/20261008_unseen_train_replication/frozen_population.json",
        "selection": "Eight hash-selected 3RScan train environment/rescan relations frozen for the prior source-fixed replication; no selection by OASIS/DINO segment outcomes.",
        "frame_pairs": len(records),
        "mask_input": "uniform 20 frames per scan, mesh-rendered annotation masks, 224x224; identity target files held separate from inference",
        "backbone": "DINOv2 ViT-Giant blocks 2, 20, 40; identical features shared by both matchers",
        "budget": "Per-frame number of native DINO mutual assignments; OASIS-style Hungarian assignment truncated by each of two declared lexicographic rankings; labels not used in inference or ranking.",
        "bootstrap": result["bootstrap"],
        "label_access": "Started only after both complete anonymous inference manifests were persisted and input hashes verified.",
        "limitations": result["limitations"],
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    table = ["| Arm | Semantic coverage | Semantic precision | Linked identity coverage | Linked identity precision | Linked semantic precision |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ("dino_mutual", *RANKINGS):
        table.append("| " + arm.replace("_", " ") + " | " + " | ".join(f"{summary[arm][m]*100:.2f}%" for m in METRICS) + " |")
    effect = ["", "| OASIS ranking | Outcome | Difference vs DINO mutual | Family-adjusted 95% interval |", "|---|---|---:|---:|"]
    for arm in RANKINGS:
        for metric in METRICS:
            lo, hi = intervals[arm][metric]
            effect.append(f"| {arm.replace('_', ' ')} | {metric.replace('_', ' ')} | {result['oasis_minus_dino_mutual_macro_effects'][arm][metric]*100:+.2f} pp | [{lo*100:+.2f}, {hi*100:+.2f}] pp |")
    report = f"""# Held-out relation OASIS-style exact-budget check

The cohort contains eight hash-selected, previously unused 3RScan train relations and 160 frozen frame pairs. These relations were selected for an earlier source-fixed RGB-D replication, before this segment-matcher comparison; the OASIS/DINO outcomes did not determine selection. We use the same DINOv2 ViT-G features and mesh-rendered anonymous masks for both matchers. For each frame, OASIS-style assignments are truncated to exactly the number of native DINO-mutual accepts. Labels were opened only after all anonymous predictions were complete and input hashes were checked.

## Eight-relation macro results

{chr(10).join(table)}

Paired bootstrap intervals resample the eight frozen relation groups and use a Bonferroni correction across two rankings by five metrics.

{chr(10).join(effect)}

## Scope

The OASIS-style arm is lower on all five macro outcomes under both label-free orderings. Nine of ten adjusted intervals are entirely below zero; confidence-first linked-semantic precision is the one interval crossing zero. This is within-dataset held-out relation confirmation, not independent-site replication: physical-site independence is unverified, the cohort has only eight groups, and prior train-split mesh screening exists. This remains a DINOv2/privileged-mask formulation comparison, not an OASIS-Map system reproduction.

The three arms each accept {result['total_raw_accepts_dino_mutual']} query masks in total. Detailed frame/group rows, 100,000 paired bootstrap draws, exact summary, protocol and anonymous predictions are stored beside this report. Reproduce scoring with `python scripts/analyze_unseen_oasis_equal_budget.py --input-root outputs/reports/20261009_unseen_oasis_equal_budget --oasis-root outputs/reports/20261009_unseen_oasis_equal_budget/oasis_style --output-dir outputs/reports/20261009_unseen_oasis_equal_budget` after anonymous inference artifacts are present.
"""
    (out / "report.md").write_text(report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
