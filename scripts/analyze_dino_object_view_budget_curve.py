#!/usr/bin/env python3
"""Object-balanced view-count curve for the frozen exact-budget DINO matcher.

Replays the label-blind reference-frame subsets from
analyze_dino_multiview_budget_curve.py, then evaluates outcomes by scan-local
query object. The REMIND-style comparator keeps its saved per-relation budget
and predictions. No matching threshold or model is fitted here.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze_dino_multiview_budget_curve import BANK_SIZES, load_ids
from calibrate_dino_multiview_source_gate import ROOT


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    p.add_argument("--feature-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    p.add_argument("--remind-root", type=Path, default=ROOT / "outputs/reports/20261109_remind_style_3rscan_validation")
    p.add_argument("--overlap-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_source_gate_overlap_sensitivity/summary.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reports/20261009_dino_object_view_budget_curve")
    p.add_argument("--no-overlap-exclusion", action="store_true", help="use for a separate cohort with no linked geometry-screen exclusions")
    p.add_argument("--expected-groups", type=int, default=37)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--draws", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261009)
    a = p.parse_args()

    excluded = set() if a.no_overlap_exclusion else set(json.loads(a.overlap_summary.read_text())["cross_split_geometry_screen"]["candidate_validation_reference_uuids"])
    manifest = json.loads((a.feature_root / "run_manifest.json").read_text())
    baseline_curve = json.loads((a.feature_root / "view_budget_curve.json").read_text())
    baseline_all_view = {
        r["run_id"]: r for r in baseline_curve["relation_repeat_rows"]
        if int(r["bank_size"]) == 20 and int(r["repeat"]) == 0
    }
    metrics: dict[int, dict[str, dict[tuple[str, str, int], dict[str, list[int]]]]] = {
        k: {"dino": {}, "remind": {}} for k in BANK_SIZES
    }
    retained_envs: set[str] = set()
    retained_relations = 0

    for rel_index, item in enumerate(manifest["results"], start=1):
        run_id = item["run_id"]
        frozen = json.loads((a.input_root / "uniform20_inputs" / run_id / "manifest.json").read_text())
        refs, queries = frozen["records"]["reference"], frozen["records"]["rescan"]
        env = Path(refs[0]["input"]).parent.name
        if env in excluded:
            continue
        retained_envs.add(env); retained_relations += 1
        qscan = Path(queries[0]["input"]).parent.name
        pair = json.loads((a.input_root / "dino_segment_predictions" / run_id / "inference_summary.json").read_text())
        pair_ref = {int(r["query_frame"]): int(r["reference_frame"]) for r in pair["records"]}
        rdata = json.loads((a.remind_root / f"{run_id}.json").read_text())
        if rdata.get("status") != "complete":
            raise ValueError(f"incomplete REMIND artifact: {run_id}")
        src_count = int(rdata["source_frames"])
        remind_by_frame: dict[int, list[dict]] = defaultdict(list)
        for row in rdata["query_mask_rows"]:
            remind_by_frame[int(row["frame"])].append(row)
        remind_budget = int(rdata["query_links_to_source_tracks"])
        if sum(int(r["predicted_source_link"]) for r in rdata["query_mask_rows"]) != remind_budget:
            raise ValueError(f"REMIND budget row mismatch: {run_id}")

        with np.load(a.feature_root / f"{run_id}_features.npz") as f:
            sf = f["source_embeddings"].astype(np.float32)
            sf_frames = f["source_frame_ids"].astype(np.int32)
            qf = f["query_embeddings"].astype(np.float32)
            qf_frames = f["query_frame_ids"].astype(np.int32)
        sf /= np.linalg.norm(sf, axis=1, keepdims=True).clip(1e-8)
        qf /= np.linalg.norm(qf, axis=1, keepdims=True).clip(1e-8)
        ref_ids_by_frame = load_ids(refs)
        query_ids_by_frame = load_ids(queries)
        source_ids = np.concatenate([ref_ids_by_frame[int(r["frame_id"])] for r in refs])
        if len(source_ids) != len(sf):
            raise ValueError(f"source feature/identity mismatch: {run_id}")
        src_ix = {int(r["frame_id"]): np.flatnonzero(sf_frames == int(r["frame_id"])) for r in refs}
        qry_ix = {int(r["frame_id"]): np.flatnonzero(qf_frames == int(r["frame_id"])) for r in queries}
        if any(len(qry_ix[int(r["frame_id"])]) != int(r["masks"]) for r in queries):
            raise ValueError(f"query features/masks mismatch: {run_id}")

        # Save the REMIND outcomes once per scan-local object cluster. Each
        # entry is [number of visible masks, accepted masks, correct masks].
        r_local: dict[tuple[str, str, int], np.ndarray] = {}
        for slot, record in enumerate(queries):
            frame = int(record["frame_id"])
            ids = query_ids_by_frame[frame]
            rrows = remind_by_frame.get(src_count + slot, [])
            if len(ids) != len(rrows):
                raise ValueError(f"REMIND mask count mismatch: {run_id}/{frame}")
            for oid, rr in zip(ids.tolist(), rrows):
                if int(rr["gt_id"]) != int(oid):
                    raise ValueError(f"REMIND ID order mismatch: {run_id}/{frame}")
                key = (env, qscan, int(oid))
                vals = r_local.setdefault(key, np.zeros(3, dtype=np.int64))
                vals += (1, int(bool(rr["predicted_source_link"])), int(bool(rr["correct_source_link"])))
        for key, vals in r_local.items():
            metrics[20]["remind"][key] = {"n": [int(vals[0])], "a": [int(vals[1])], "c": [int(vals[2])]}

        # Exact same label-blind subset seed and assignment/ranking as the
        # established mask-level view-count experiment.
        for bank_size in BANK_SIZES:
            n_repeats = 1 if bank_size == len(refs) else a.repeats
            local_per_object = defaultdict(lambda: {"n": [], "a": [], "c": []})
            for repeat in range(n_repeats):
                rng = np.random.default_rng(a.seed + bank_size * 1_000_003 + repeat * 9_176 + sum(run_id.encode()))
                candidate_rows = []
                for qrecord in queries:
                    qframe = int(qrecord["frame_id"])
                    paired_ref = pair_ref[qframe]
                    other = [int(r["frame_id"]) for r in refs if int(r["frame_id"]) != paired_ref]
                    additional = rng.choice(other, size=bank_size - 1, replace=False).tolist() if bank_size > 1 else []
                    chosen = {paired_ref, *(int(x) for x in additional)}
                    sidx = np.concatenate([src_ix[x] for x in chosen])
                    qidx = qry_ix[qframe]
                    if not len(sidx) or not len(qidx):
                        continue
                    sim = qf[qidx] @ sf[sidx].T
                    qr, sc = linear_sum_assignment(-sim)
                    qids = query_ids_by_frame[qframe]
                    for qi, si in zip(qr.tolist(), sc.tolist()):
                        candidate_rows.append((int(qidx[qi]), int(qids[qi]), int(source_ids[sidx[si]]), float(sim[qi, si])))
                if len(candidate_rows) < remind_budget:
                    raise ValueError(f"bank {bank_size} assignments below exact budget for {run_id}")
                selected = sorted(candidate_rows, key=lambda x: (-x[3], x[0]))[:remind_budget]
                if len(selected) != remind_budget:
                    raise ValueError(f"selected DINO accept count differs from REMIND budget: {run_id}/{bank_size}")
                # Convert selected global query embedding indices into local
                # mask-index arrays, preserving all visible masks in n.
                selected_by_global = {row[0]: row for row in selected}
                for qrecord in queries:
                    qframe = int(qrecord["frame_id"])
                    qidx = qry_ix[qframe]
                    qids = query_ids_by_frame[qframe]
                    for mask_index, (global_idx, oid) in enumerate(zip(qidx.tolist(), qids.tolist())):
                        row = selected_by_global.get(int(global_idx))
                        accepted = row is not None
                        correct = bool(accepted and int(oid) > 0 and int(oid) == row[2])
                        key = (env, qscan, int(oid))
                        vals = local_per_object[key]
                        if len(vals["n"]) <= repeat:
                            vals["n"].append(0); vals["a"].append(0); vals["c"].append(0)
                        vals["n"][repeat] += 1
                        vals["a"][repeat] += int(accepted)
                        vals["c"][repeat] += int(correct)
            if bank_size == 20:
                reconstructed = np.sum(np.asarray([
                    [sum(v["n"]), sum(v["a"]), sum(v["c"])]
                    for v in local_per_object.values()
                ], dtype=np.int64), axis=0)
                expected = baseline_all_view[run_id]
                if reconstructed.tolist() != [int(expected["query_masks"]), int(expected["accepted"]), int(expected["correct"])]:
                    raise ValueError(f"object-level replay does not reproduce saved 20-view mask outcomes: {run_id}")
            metrics[bank_size]["dino"].update(local_per_object)
        if rel_index % 10 == 0:
            print(f"processed {rel_index}/{len(manifest['results'])} relations", flush=True)

    if len(retained_envs) != a.expected_groups:
        raise ValueError(f"expected {a.expected_groups} retained reference groups, got {len(retained_envs)}")
    # Convert each cluster into one metric value, averaging random source-view
    # draws within object. The fixed 20-view condition has one draw.
    values: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for k in BANK_SIZES:
        values[k] = {"dino": {}, "remind": {}}
        for method in ("dino", "remind"):
            for outcome in ("hit", "coverage", "precision"):
                values[k][method][outcome] = {}
            source = metrics[k][method]
            for key, obj in source.items():
                n = np.asarray(obj["n"], dtype=float)
                a_count = np.asarray(obj["a"], dtype=float)
                c_count = np.asarray(obj["c"], dtype=float)
                if not len(n):
                    continue
                values[k][method]["hit"][key] = float(np.mean(c_count > 0))
                values[k][method]["coverage"][key] = float(np.mean(c_count / np.maximum(n, 1)))
                precision = np.divide(c_count, a_count, out=np.full_like(c_count, np.nan), where=a_count > 0)
                values[k][method]["precision"][key] = float(np.nanmean(precision)) if np.isfinite(precision).any() else float("nan")

    groups = sorted(retained_envs)
    estimates = {}
    repeat_effect_ranges = {}
    rng = np.random.default_rng(a.seed + 909)
    n_draws = a.draws
    for k in BANK_SIZES:
        estimates[str(k)] = {}
        for outcome in ("hit", "coverage", "precision"):
            per_group = {}
            for env in groups:
                env_keys = sorted(key for key in values[k]["dino"][outcome] if key[0] == env)
                d = np.asarray([values[k]["dino"][outcome][key] for key in env_keys], dtype=float)
                # REMIND predictions do not vary with bank size; the saved
                # outcome is reused for every DINO condition.
                r = np.asarray([values[20]["remind"][outcome][key] for key in env_keys], dtype=float)
                per_group[env] = (d, r)
            dmacro = np.asarray([np.nanmean(per_group[env][0]) if np.isfinite(per_group[env][0]).any() else np.nan for env in groups])
            rmacro = np.asarray([np.nanmean(per_group[env][1]) if np.isfinite(per_group[env][1]).any() else np.nan for env in groups])
            valid_group = np.isfinite(dmacro) & np.isfinite(rmacro)
            deltas = dmacro[valid_group] - rmacro[valid_group]
            # Report how the group-macro contrast varies over the actual
            # random view-subset draws. This is a range, not an uncertainty CI.
            subset_deltas = []
            n_repeats = 1 if k == 20 else a.repeats
            for repeat in range(n_repeats):
                group_d, group_r = [], []
                for env in groups:
                    dvals, rvals = [], []
                    for key in values[k]["dino"][outcome]:
                        if key[0] != env:
                            continue
                        raw = metrics[k]["dino"][key]
                        n_i = np.asarray(raw["n"], dtype=float)
                        a_i = np.asarray(raw["a"], dtype=float)
                        c_i = np.asarray(raw["c"], dtype=float)
                        if outcome == "hit": value = float(c_i[repeat] > 0)
                        elif outcome == "coverage": value = float(c_i[repeat] / max(n_i[repeat], 1))
                        else: value = float(c_i[repeat] / a_i[repeat]) if a_i[repeat] > 0 else float("nan")
                        dvals.append(value)
                        tracker_value = values[20]["remind"][outcome][key]
                        rvals.append(tracker_value)
                    d_arr, r_arr = np.asarray(dvals), np.asarray(rvals)
                    group_d.append(float(np.nanmean(d_arr)) if np.isfinite(d_arr).any() else np.nan)
                    group_r.append(float(np.nanmean(r_arr)) if np.isfinite(r_arr).any() else np.nan)
                gd, gr = np.asarray(group_d), np.asarray(group_r)
                both = np.isfinite(gd) & np.isfinite(gr)
                subset_deltas.append(float(np.mean(gd[both] - gr[both])))
            repeat_effect_ranges[f"{k}/{outcome}"] = {
                "min_group_macro_difference": float(np.min(subset_deltas)),
                "max_group_macro_difference": float(np.max(subset_deltas)),
                "draws": n_repeats,
                "interpretation": "range across the realized random source-view subsets, not a confidence interval",
            }
            # Paired group bootstrap with object-cluster resampling nested
            # inside each selected reference group.
            valid_envs = [env for env in groups if np.isfinite(per_group[env][0]).any() and np.isfinite(per_group[env][1]).any()]
            # First draw objects independently within each group. Then draw
            # groups and an independent within-group replicate per occurrence
            # (so duplicated groups in one bootstrap draw stay independent).
            group_object_deltas = np.empty((len(valid_envs), n_draws), dtype=float)
            for gi, env in enumerate(valid_envs):
                dv, rv = per_group[env]
                oi = rng.integers(0, len(dv), size=(n_draws, len(dv)))
                sampled_d, sampled_r = dv[oi], rv[oi]
                dcount, rcount = np.isfinite(sampled_d).sum(axis=1), np.isfinite(sampled_r).sum(axis=1)
                dmean = np.nansum(sampled_d, axis=1) / np.maximum(dcount, 1)
                rmean = np.nansum(sampled_r, axis=1) / np.maximum(rcount, 1)
                dmean[dcount == 0] = np.nan
                rmean[rcount == 0] = np.nan
                group_object_deltas[gi] = dmean - rmean
            selected_group = rng.integers(0, len(valid_envs), size=(n_draws, len(valid_envs)))
            selected_object_rep = rng.integers(0, n_draws, size=(n_draws, len(valid_envs)))
            selected_delta = group_object_deltas[selected_group, selected_object_rep]
            draw_count = np.isfinite(selected_delta).sum(axis=1)
            boot = np.nansum(selected_delta, axis=1) / np.maximum(draw_count, 1)
            estimates[str(k)][outcome] = {
                "dino_macro": float(np.nanmean(dmacro)), "remind_macro": float(np.nanmean(rmacro)),
                "difference_dino_minus_remind": float(np.nanmean(deltas)), "groups": int(valid_group.sum()),
                "groups_dino_higher": int(np.sum(deltas > 1e-12)), "groups_remind_higher": int(np.sum(deltas < -1e-12)),
                "nested_group_object_bootstrap_95": [float(x) for x in np.quantile(boot, [.025, .975])],
                "bonferroni_18_outcome_interval_95": [float(x) for x in np.quantile(boot, [.05/36, 1-.05/36])],
            }

    result = {
        "status": "complete", "scope": "object-cluster sensitivity of frozen DINO source-view-count curve at exact per-relation REMIND acceptance budgets",
        "object_cluster": "query scan UUID plus scan-local annotated objectId",
        "retained_reference_groups": len(retained_envs), "relations": retained_relations,
        "excluded_reference_uuids": sorted(excluded), "bank_sizes": list(BANK_SIZES),
        "repeats": {str(k): (1 if k == 20 else a.repeats) for k in BANK_SIZES},
        "bootstrap": {"draws": n_draws, "seed": a.seed + 909, "unit": "paired group then query-object cluster; conditional on frozen view-subset draws", "family": "Bonferroni over 6 sizes x 3 outcomes"},
        "effects": estimates,
        "view_subset_effect_ranges": repeat_effect_ranges,
        "limits": ["This is a post-hoc reweighting of frozen features on the explored 3RScan benchmark, not an independent validation.",
                   "For bank sizes below 20, object metrics average over random source-frame subsets; their intervals condition on the saved subset draws.",
                   "Object IDs are scan-local and are not matched across distinct query scans.",
                   "The exact acceptance budget is borrowed from REMIND and matching is evaluated with annotation identity targets."]
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    names = {"hit": "Object hit coverage", "coverage": "Fractional correct views per object", "precision": "Object-macro precision among accepted objects"}
    exclusion_text = f"The same {len(excluded)} geometry-screen candidate groups are excluded; " if excluded else "No geometry-screen groups are excluded; "
    lines = ["# DINO source-view count under object-cluster weighting", "", "All DINO conditions use the same label-blind exact per-relation acceptance budget as the saved REMIND-style tracker. " + exclusion_text + "estimates average repeated random view subsets within each scan-local object.", "", "| Source views | Outcome | DINO | REMIND | Difference (18-outcome-adjusted 95% interval) |", "|---:|---|---:|---:|---:|"]
    for k in BANK_SIZES:
        for outcome in ("hit", "coverage", "precision"):
            x = estimates[str(k)][outcome]
            ci = x["bonferroni_18_outcome_interval_95"]
            lines.append(f"| {k} | {names[outcome]} | {x['dino_macro']:.3f} | {x['remind_macro']:.3f} | {x['difference_dino_minus_remind']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    range8 = repeat_effect_ranges["8/hit"]
    range12 = repeat_effect_ranges["12/hit"]
    subset_summary = (
        f"Across the {a.repeats} realized random source-view subsets, the group-macro object-hit difference ranges "
        f"{range8['min_group_macro_difference']*100:+.1f} to {range8['max_group_macro_difference']*100:+.1f} points at 8 views and "
        f"{range12['min_group_macro_difference']*100:+.1f} to {range12['max_group_macro_difference']*100:+.1f} at 12 views. "
        "These ranges describe the sampled subsets, not confidence intervals."
    )
    lines += ["", "These metrics give equal weight to each query object within each reference group, then equal weight to groups. Object hit asks whether an object received at least one correct link across its visible masks. Fractional coverage averages its proportion of correctly linked observations. Precision averages per-object precision over objects with accepted links.", "",
              subset_summary, "",
              "The nested group/object bootstrap intervals condition on those view-subset draws. This remains a post-hoc same-benchmark analysis; it does not establish cross-site generalization or causal memory effects.", ""]
    (a.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"groups": len(retained_envs), "relations": retained_relations, "effects": estimates}, indent=2))


if __name__ == "__main__":
    main()
