#!/usr/bin/env python3
"""Object-balanced sensitivity analysis for exact-budget DINO/REMIND outcomes.

Object clusters are scan-local: (query scan UUID, annotated objectId). The
analysis reuses frozen per-relation exact acceptance budgets and excludes the
same post-hoc geometry-screen candidates as the companion sensitivity report.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import ROOT, load_split
from analyze_dino_remind_query_overlap import load_ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    p.add_argument("--dino-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    p.add_argument("--remind-root", type=Path, default=ROOT / "outputs/reports/20261109_remind_style_3rscan_validation")
    p.add_argument("--screen-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_source_gate_overlap_sensitivity/summary.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reports/20261009_dino_equal_budget_object_sensitivity")
    p.add_argument("--draws", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261020)
    a = p.parse_args()

    dino_by_id = {r["run_id"]: r for r in load_split(a.input_root, a.dino_root)}
    manifest = json.loads((a.dino_root / "run_manifest.json").read_text())
    excluded = set(json.loads(a.screen_summary.read_text())["cross_split_geometry_screen"]["candidate_validation_reference_uuids"])
    clusters: dict[tuple[str, str, int], dict] = {}
    env_by_cluster: dict[tuple[str, str, int], str] = {}
    relation_count = query_count = 0
    for item in manifest["results"]:
        run_id = item["run_id"]
        frozen = json.loads((a.input_root / "uniform20_inputs" / run_id / "manifest.json").read_text())
        env = Path(frozen["records"]["reference"][0]["input"]).parent.name
        if env in excluded:
            continue
        dino = json.loads((a.dino_root / f"{run_id}.json").read_text())
        remind = json.loads((a.remind_root / f"{run_id}.json").read_text())
        scored = dino_by_id[run_id]
        records = frozen["records"]["rescan"]
        if [int(x) for x in dino["query_frames"]] != [int(r["frame_id"]) for r in records]:
            raise ValueError(f"DINO frame order mismatch: {run_id}")
        src_count = int(remind["source_frames"])
        r_by_frame: dict[int, list[dict]] = defaultdict(list)
        for row in remind["query_mask_rows"]:
            r_by_frame[int(row["frame"])].append(row)
        budget = sum(int(r["predicted_source_link"]) for r in remind["query_mask_rows"])
        rows = dino["query_rows"]
        valid = np.asarray([r["assigned_source_observation"] is not None and r["assigned_similarity"] is not None for r in rows])
        candidates = np.flatnonzero(valid)
        if budget > len(candidates):
            raise ValueError(f"budget exceeds DINO valid assignments: {run_id}")
        ranked = np.lexsort((candidates, -scored["scores"][candidates]))
        accepted = np.zeros(len(rows), dtype=bool)
        accepted[candidates[ranked[:budget]]] = True
        qscan = Path(records[0]["input"]).parent.name if records else run_id
        cursor = 0
        for slot, record in enumerate(records):
            frame = int(record["frame_id"])
            ids = load_ids(record)
            rrows = r_by_frame.get(src_count + slot, [])
            if len(ids) != len(rrows):
                raise ValueError(f"REMIND mask/label mismatch: {run_id}/{frame}")
            for index, oid in enumerate(ids.tolist()):
                if int(rrows[index]["gt_id"]) != int(oid):
                    raise ValueError(f"query identity ordering mismatch: {run_id}/{frame}/{index}")
                dacc = bool(accepted[cursor])
                dcor = bool(dacc and scored["correct"][cursor])
                racc = bool(rrows[index]["predicted_source_link"])
                rcor = bool(rrows[index]["correct_source_link"])
                key = (env, qscan, int(oid))
                if key not in clusters:
                    clusters[key] = {"dino_n": 0, "remind_n": 0, "dino_a": 0, "remind_a": 0,
                                     "dino_c": 0, "remind_c": 0}
                    env_by_cluster[key] = env
                v = clusters[key]
                v["dino_n"] += 1; v["remind_n"] += 1
                v["dino_a"] += int(dacc); v["remind_a"] += int(racc)
                v["dino_c"] += int(dcor); v["remind_c"] += int(rcor)
                cursor += 1
                query_count += 1
        if cursor != len(rows) or cursor != len(remind["query_mask_rows"]):
            raise ValueError(f"prediction/label counts differ: {run_id}")
        if int(accepted.sum()) != budget or sum(int(x["predicted_source_link"]) for x in remind["query_mask_rows"]) != budget:
            raise ValueError(f"exact-budget acceptance mismatch: {run_id}")
        relation_count += 1

    grouped: dict[str, list[dict]] = defaultdict(list)
    for key, v in clusters.items():
        grouped[env_by_cluster[key]].append(v)
    envs = sorted(grouped)
    if len(envs) != 37:
        raise ValueError(f"expected 37 geometry-screen-retained reference groups, found {len(envs)}")

    # Per-cluster quantities; precision is defined only for objects with an
    # accepted link, while coverage and hit rate use all annotated instances.
    per_env = {}
    for env, objects in grouped.items():
        vals = {}
        for method in ("dino", "remind"):
            n = np.asarray([o[f"{method}_n"] for o in objects], dtype=float)
            acc = np.asarray([o[f"{method}_a"] for o in objects], dtype=float)
            cor = np.asarray([o[f"{method}_c"] for o in objects], dtype=float)
            vals[method] = {
                "object_coverage": cor / n,
                "object_hit": (cor > 0).astype(float),
                "object_precision": np.divide(cor, acc, out=np.full_like(cor, np.nan), where=acc > 0),
            }
        per_env[env] = vals

    names = ("object_coverage", "object_hit", "object_precision")
    estimates = {}
    for name in names:
        dvals, rvals = [], []
        for env in envs:
            dv = per_env[env]["dino"][name]
            rv = per_env[env]["remind"][name]
            dvals.append(float(np.nanmean(dv)) if np.isfinite(dv).any() else np.nan)
            rvals.append(float(np.nanmean(rv)) if np.isfinite(rv).any() else np.nan)
        dvals, rvals = np.asarray(dvals), np.asarray(rvals)
        ok = np.isfinite(dvals) & np.isfinite(rvals)
        estimates[name] = {"dino_macro": float(np.mean(dvals[ok])), "remind_macro": float(np.mean(rvals[ok])),
                           "difference_dino_minus_remind": float(np.mean((dvals-rvals)[ok])),
                           "groups": int(ok.sum()), "groups_dino_higher": int(np.sum((dvals-rvals)[ok] > 1e-12)),
                           "groups_remind_higher": int(np.sum((dvals-rvals)[ok] < -1e-12))}

    rng = np.random.default_rng(a.seed)
    # Nested bootstrap: sample reference groups, then object clusters within
    # each sampled group. For precision, recompute each sampled object's mean
    # among objects with any accepted observations for that method.
    boots = {name: np.empty(a.draws, dtype=float) for name in names}
    group_obj = {env: per_env[env] for env in envs}
    for b in range(a.draws):
        chosen_envs = rng.integers(0, len(envs), size=len(envs))
        per_method = {name: {"dino": [], "remind": []} for name in names}
        for ei in chosen_envs:
            env = envs[int(ei)]
            nobj = len(grouped[env])
            oi = rng.integers(0, nobj, size=nobj)
            for name in names:
                for method in ("dino", "remind"):
                    x = group_obj[env][method][name][oi]
                    if np.isfinite(x).any():
                        per_method[name][method].append(float(np.nanmean(x)))
                    else:
                        per_method[name][method].append(np.nan)
        for name in names:
            dv = np.asarray(per_method[name]["dino"]); rv = np.asarray(per_method[name]["remind"])
            valid_groups = np.isfinite(dv) & np.isfinite(rv)
            boots[name][b] = np.mean((dv-rv)[valid_groups])
    for name in names:
        estimates[name]["nested_group_object_bootstrap_95"] = [float(x) for x in np.quantile(boots[name], [.025, .975])]
        estimates[name]["bonferroni_3_outcome_interval_95"] = [float(x) for x in np.quantile(boots[name], [0.0083333333, 0.9916666667])]

    result = {
        "status": "complete",
        "scope": "post-hoc object-balanced sensitivity of frozen exact per-relation-budget DINO/REMIND predictions after the same geometry-only overlap screen",
        "identity_cluster": "query scan UUID plus scan-local annotated 3RScan objectId; observations from the same object within a query scan are clustered",
        "excluded_reference_uuids": sorted(excluded), "retained_reference_groups": len(envs),
        "relations": relation_count, "query_mask_observations": query_count,
        "query_object_clusters": len(clusters),
        "objects_with_any_dino_accept": sum(1 for o in clusters.values() if o["dino_a"] > 0),
        "bootstrap": {"draws": a.draws, "seed": a.seed, "unit": "paired reference-group then object-cluster resampling"},
        "effects": estimates,
        "definitions": {"object_coverage": "mean per-object fraction of observed masks receiving a correct link, then macro-average objects within reference group and groups equally",
                        "object_hit": "mean per-object indicator of at least one correct link across its observed masks, then macro-average objects within reference group and groups equally",
                        "object_precision": "mean per-object precision among objects with at least one accepted link, then macro-average objects within reference group and groups equally"},
        "limits": ["This is a post-hoc reweighting of saved predictions, not a new independent benchmark.",
                   "The cluster is scan-local; numeric objectIds are not matched across distinct query scans.",
                   "A correct link is defined by the existing annotation identity target and source availability; this does not independently certify annotation geometry or physical identity."]
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    labels = {"object_coverage": "Object-macro correct coverage", "object_hit": "Object hit coverage", "object_precision": "Object-macro precision among accepted objects"}
    lines = ["# Exact-budget DINO/REMIND object-balanced sensitivity", "",
             f"Reweighted the saved exact-budget predictions over {len(envs)} retained reference groups, {relation_count} relations, and {len(clusters)} scan-local query-object clusters. The same ten geometry-screen candidate groups were excluded.",
             "", "| Outcome | DINO | REMIND | Difference (Bonferroni-adjusted 95% interval) |",
             "|---|---:|---:|---:|"]
    for name in names:
        x = estimates[name]
        lines.append(f"| {labels[name]} | {x['dino_macro']:.3f} | {x['remind_macro']:.3f} | {x['difference_dino_minus_remind']:+.3f} [{x['bonferroni_3_outcome_interval_95'][0]:+.3f}, {x['bonferroni_3_outcome_interval_95'][1]:+.3f}] |")
    lines += ["", "The object-hit outcome asks whether each annotated query object received at least one correct link across the sampled views. Object coverage averages the fraction of that object's visible mask observations linked correctly. Precision gives equal weight to each object that received one or more accepted links.",
              "", "This remains a post-hoc sensitivity on the same 3RScan benchmark. Object clusters are scan-local and do not make the validation set independent of training content.", ""]
    (a.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"groups": len(envs), "relations": relation_count, "query_object_clusters": len(clusters), "effects": estimates}, indent=2))


if __name__ == "__main__":
    main()
