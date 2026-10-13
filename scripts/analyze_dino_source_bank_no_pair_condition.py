#!/usr/bin/env python3
"""Re-score DINO source-view subsets without query-to-reference frame pairing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "outputs/reports/20261009_dino_multiview_source_bank"
INPUT_ROOT = ROOT / "outputs/reports/20261008_segmast3r_validation"
REMIND_ROOT = ROOT / "outputs/reports/20261109_remind_style_3rscan_validation"
OUT = ROOT / "outputs/reports/20261009_dino_source_bank_no_pair_condition"
BANK_SIZES = (1, 2, 4, 8, 12, 20)
STRATEGIES = ("random", "temporal_uniform", "feature_diverse")


def load_ids(records):
    by_frame = {}
    for record in records:
        path = Path(record["identity_targets"])
        if not path.is_absolute(): path = ROOT / path
        ids = np.load(path).astype(np.int64)
        if len(ids) != int(record["masks"]): raise ValueError(f"mask/ID mismatch: {path}")
        by_frame[int(record["frame_id"])] = ids
    return by_frame


def diverse_order(frame_means):
    """Greedy feature-space farthest-first order; no labels or query data."""
    x = np.stack([frame_means[f] for f in sorted(frame_means)])
    frames = np.asarray(sorted(frame_means), dtype=np.int32)
    # Start at the medoid so the first exemplar represents the reference bank.
    center = x.mean(axis=0)
    d = np.sum((x - center) ** 2, axis=1)
    first = int(np.argmin(d))
    chosen = [first]
    min_d = np.sum((x - x[first]) ** 2, axis=1)
    while len(chosen) < len(frames):
        min_d[chosen] = -1
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.sum((x - x[nxt]) ** 2, axis=1))
    return frames[chosen]


def fixed_subset(frame_ids, k, strategy, frame_means, rng):
    ordered = np.asarray(sorted(frame_ids), dtype=np.int32)
    if k == len(ordered): return ordered
    if strategy == "random": return np.sort(rng.choice(ordered, size=k, replace=False))
    if strategy == "temporal_uniform":
        return ordered[np.rint(np.linspace(0, len(ordered)-1, k)).astype(int)]
    if strategy == "feature_diverse": return diverse_order(frame_means)[:k]
    raise ValueError(strategy)


def score_relation(item, repeats, seed):
    run_id = item["run_id"]
    frozen = json.loads((INPUT_ROOT / "uniform20_inputs" / run_id / "manifest.json").read_text())
    remind = json.loads((REMIND_ROOT / f"{run_id}.json").read_text())
    if frozen.get("status") not in (None, "complete") and frozen.get("status") != "complete":
        raise ValueError(f"bad frozen manifest: {run_id}")
    if remind.get("status") != "complete": raise ValueError(f"bad REMIND result: {run_id}")
    with np.load(FEATURE_ROOT / f"{run_id}_features.npz") as z:
        sf = z["source_embeddings"].astype(np.float32)
        sframe = z["source_frame_ids"].astype(np.int32)
        qf = z["query_embeddings"].astype(np.float32)
        qframe = z["query_frame_ids"].astype(np.int32)
    sf /= np.maximum(np.linalg.norm(sf, axis=1, keepdims=True), 1e-8)
    qf /= np.maximum(np.linalg.norm(qf, axis=1, keepdims=True), 1e-8)
    refs, queries = frozen["records"]["reference"], frozen["records"]["rescan"]
    ref_frames = [int(r["frame_id"]) for r in refs]
    query_frames = [int(r["frame_id"]) for r in queries]
    src_idx = {f: np.flatnonzero(sframe == f) for f in ref_frames}
    qry_idx = {f: np.flatnonzero(qframe == f) for f in query_frames}
    valid_ref_frames = [f for f in ref_frames if len(src_idx[f])]
    frame_means = {}
    for f in valid_ref_frames:
        emb = sf[src_idx[f]]
        mean = emb.mean(axis=0)
        frame_means[f] = mean / max(float(np.linalg.norm(mean)), 1e-8)
    budget = int(remind["query_links_to_source_tracks"])
    query_count = len(qf)
    configs = []
    for k in BANK_SIZES:
        n_rep = repeats if k < len(ref_frames) else 1
        for strategy in STRATEGIES:
            for rep in range(n_rep if strategy == "random" else 1):
                rng = np.random.default_rng(seed + sum(run_id.encode()) + 100003*k + 7919*rep)
                pool = ref_frames if k == len(ref_frames) else valid_ref_frames
                selected_frames = fixed_subset(pool, k, strategy, frame_means, rng)
                source_idx = np.concatenate([src_idx[int(f)] for f in selected_frames])
                if not len(source_idx): raise ValueError(f"empty selected bank: {run_id}, {k}")
                candidate_rows = []
                for f in query_frames:
                    qi = qry_idx[f]
                    if not len(qi): continue
                    sims = qf[qi] @ sf[source_idx].T
                    rr, cc = linear_sum_assignment(-sims)
                    for q_local, s_local in zip(rr.tolist(), cc.tolist()):
                        q_global, s_global = int(qi[q_local]), int(source_idx[s_local])
                        candidate_rows.append((q_global, s_global, float(sims[q_local, s_local])))
                configs.append({"bank_size": k, "strategy": strategy, "repeat": rep,
                                "selected_frames": selected_frames, "candidates": candidate_rows})
    # Identity targets are loaded only after every feature-only assignment has
    # been generated, preserving the label-blind retrieval path.
    ref_ids = load_ids(refs); query_ids = load_ids(queries)
    if sum(len(v) for v in ref_ids.values()) != len(sf):
        raise ValueError(f"source mask/feature count mismatch after label-blind inference: {run_id}")
    if sum(len(v) for v in query_ids.values()) != len(qf):
        raise ValueError(f"query mask/feature count mismatch after label-blind inference: {run_id}")
    source_id_by_index = np.concatenate([ref_ids[f] for f in ref_frames])
    query_id_by_index = np.empty(len(qf), dtype=np.int64)
    for f in query_frames:
        idx = qry_idx[f]; ids = query_ids[f]
        if len(idx) != len(ids): raise ValueError(f"query mask/feature mismatch {run_id}:{f}")
        query_id_by_index[idx] = ids
    rows = []
    # At small bank sizes, some subsets cannot generate as many one-to-one
    # assignments as REMIND accepts. Use the largest per-relation budget
    # supported by every randomized and deterministic subset at that K.
    for k in BANK_SIZES:
        this_k = [c for c in configs if c["bank_size"] == k]
        common_budget = min(budget, min(len(c["candidates"]) for c in this_k))
        for config in this_k:
            candidates = sorted(config["candidates"], key=lambda x: (-x[2], x[0]))
            accepted = candidates[:common_budget]
            correct = sum(int(query_id_by_index[qidx] > 0 and query_id_by_index[qidx] == source_id_by_index[sidx]) for qidx, sidx, _ in accepted)
            rows.append({
                "run_id": run_id, "environment_id": Path(refs[0]["input"]).parent.name,
                "bank_size": k, "strategy": config["strategy"], "repeat": config["repeat"],
                "query_masks": query_count, "accepted": common_budget, "correct": correct,
                "remind_budget": budget, "source_descriptors": sum(len(src_idx[int(f)]) for f in config["selected_frames"]),
                "full_source_descriptors": len(sf),
                "selected_reference_frames": ",".join(str(int(f)) for f in config["selected_frames"]),
            })
    return rows


def summarize(rows, draws, seed):
    df = pd.DataFrame(rows)
    groups = sorted(df.environment_id.unique())
    # Aggregate relation-level exact budgets within environment before macro comparison.
    env = df.groupby(["bank_size", "strategy", "repeat", "environment_id"], as_index=False).agg(
        queries=("query_masks", "sum"), accepted=("accepted", "sum"), correct=("correct", "sum"))
    env["precision"] = env.correct / env.accepted
    env["coverage"] = env.correct / env.queries
    # Random repeats are averaged within environment; deterministic strategies have repeat 0.
    macro = env.groupby(["bank_size", "strategy", "environment_id"], as_index=False).agg(
        precision=("precision", "mean"), coverage=("coverage", "mean"))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(groups), size=(draws, len(groups)))
    out = {}
    for k in BANK_SIZES:
        out[str(k)] = {}
        for strat in STRATEGIES:
            sub = macro[(macro.bank_size == k) & (macro.strategy == strat)].set_index("environment_id").reindex(groups)
            if sub.precision.isna().any(): continue
            rec = {}
            for metric in ("precision", "coverage"):
                vals = sub[metric].to_numpy(float)
                boot = vals[ix].mean(axis=1)
                rec[metric] = {"estimate": float(vals.mean()), "ci95": [float(x) for x in np.quantile(boot, [.025,.975])],
                               "reference_groups_positive": int((vals > 0).sum())}
            out[str(k)][strat] = rec
    pooled = {}
    for (k, strat), sub in env.groupby(["bank_size", "strategy"]):
        pooled[f"{k}:{strat}"] = {"precision": float(sub.correct.sum()/sub.accepted.sum()),
                                   "coverage": float(sub.correct.sum()/sub.queries.sum()),
                                   "accepted": int(sub.accepted.sum()), "correct": int(sub.correct.sum())}
    # Descriptive paired difference against REMIND's reference-group rates.
    remind_json = json.loads((FEATURE_ROOT / "summary.json").read_text())
    remind_by_group = {x["environment_id"]: x["remind_one_to_one"] for x in remind_json["reference_groups"]}
    differences = {}
    for k in BANK_SIZES:
        for strat in STRATEGIES:
            sub = macro[(macro.bank_size == k) & (macro.strategy == strat)].set_index("environment_id").reindex(groups)
            if sub.precision.isna().any(): continue
            differences[f"{k}:{strat}"] = {}
            for metric in ("precision", "coverage"):
                base = np.asarray([remind_by_group[g][metric] for g in groups], dtype=float)
                d = sub[metric].to_numpy(float) - base
                boot = d[ix].mean(axis=1)
                differences[f"{k}:{strat}"][metric] = {"estimate": float(d.mean()), "ci95": [float(x) for x in np.quantile(boot,[.025,.975])],
                                                       "groups_positive": int((d>0).sum())}
    return df, {"groups": len(groups), "macro": out, "pooled": pooled, "difference_vs_remind": differences,
                "bootstrap": {"unit":"47 released reference-scan groups", "draws":draws, "seed":seed}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--draws", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20261010)
    args = ap.parse_args()
    manifest = json.loads((FEATURE_ROOT / "run_manifest.json").read_text())
    results = [x for x in manifest["results"] if x.get("status") == "complete"]
    rows=[]
    for i,item in enumerate(results,1):
        rows.extend(score_relation(item,args.repeats,args.seed))
        if i%10==0: print(f"scored {i}/{len(results)} relations",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    df, summary = summarize(rows,args.draws,args.seed)
    storage = df.groupby(["bank_size","strategy"],as_index=False).agg(
        mean_selected_descriptors=("source_descriptors","mean"),
        mean_full_descriptors=("full_source_descriptors","mean"),
        descriptor_fraction=("source_descriptors",lambda x: float(np.mean(x.to_numpy()/df.loc[x.index,"full_source_descriptors"].to_numpy()))))
    summary["feature_storage"] = {f"{int(r.bank_size)}:{r.strategy}": {
        "mean_selected_descriptors_per_relation": float(r.mean_selected_descriptors),
        "mean_full_descriptors_per_relation": float(r.mean_full_descriptors),
        "mean_descriptor_fraction": float(r.descriptor_fraction),
        "mean_storage_reduction": float(1-r.descriptor_fraction),
    } for r in storage.itertuples()}
    df.to_csv(OUT/"per_relation.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps({"status":"complete","protocol":{
        "dataset":"3RScan official validation, 110 explored relations / 47 reference groups",
        "source_bank_selection":"fixed per relation and shared by all query frames; no paired reference frame is privileged; subsets sample reference frames with at least one mask embedding, full bank includes all frames",
        "strategies":["random fixed subset","temporal-uniform source-only subset","farthest-first cosine diversity over unlabeled source-frame mean DINO mask embeddings"],
        "bank_sizes":BANK_SIZES,"random_repeats":args.repeats,
        "matching":"per-query-frame Hungarian assignment from query mask embeddings to selected reference-mask embeddings",
        "acceptance":"within each relation and bank size, use the largest per-relation count supportable by every selection strategy/repeat, capped by REMIND count, ranked by DINO cosine; diagnostic and borrowed after inference",
        "labels":"identity IDs used only after assignment for scoring; source instance masks remain privileged input",
    },"summary":summary},indent=2)+"\n")
    print(json.dumps({"status":"complete","relations":len(results),"groups":summary["groups"],"outputs":str(OUT),"differences":summary["difference_vs_remind"]},indent=2))

if __name__=="__main__": main()
