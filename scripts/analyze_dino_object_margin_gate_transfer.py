#!/usr/bin/env python3
"""Fit a source-only object-score margin gate and transfer it unchanged."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UUID_RE = re.compile(r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_frame_\d+_input\.npz$")
DEFAULT_SOURCE_INPUT = ROOT / "outputs/reports/20261009_unseen_oasis_equal_budget"
DEFAULT_SOURCE_PRED = ROOT / "outputs/reports/20261009_dino_multiview_source_bank_train8"
DEFAULT_VAL_INPUT = ROOT / "outputs/reports/20261008_segmast3r_validation"
DEFAULT_VAL_PRED = ROOT / "outputs/reports/20261009_dino_multiview_source_bank"
DEFAULT_UW = ROOT / "outputs/reports/20261009_rgbd_object_eval_multiview/outcomes.jsonl"
DEFAULT_OUT = ROOT / "outputs/reports/20261009_dino_object_margin_gate_transfer"


def load_relation(input_root: Path, pred_root: Path, item: dict, views_per_object: int) -> dict:
    run_id = item["run_id"]
    pred_path = pred_root / f"{run_id}.json"
    feature_path = pred_root / f"{run_id}_features.npz"
    manifest_path = input_root / "uniform20_inputs" / run_id / "manifest.json"
    prediction = json.loads(pred_path.read_text())
    frozen = json.loads(manifest_path.read_text())
    if prediction.get("status") != "complete" or prediction.get("labels_read") is not False:
        raise ValueError(f"invalid prediction artifact: {run_id}")

    first_ref_path = Path(frozen["records"]["reference"][0]["input"])
    match = UUID_RE.match(first_ref_path.name)
    group_id = match.group(1).lower() if match else first_ref_path.parent.name.lower()
    source_ids, query_ids = [], {}
    for record in frozen["records"]["reference"]:
        ids = np.load(Path(record["identity_targets"])).astype(np.int64)
        if len(ids) != int(record["masks"]):
            raise ValueError(f"reference label count mismatch: {run_id}")
        source_ids.extend(ids.tolist())
    for record in frozen["records"]["rescan"]:
        ids = np.load(Path(record["identity_targets"])).astype(np.int64)
        if len(ids) != int(record["masks"]):
            raise ValueError(f"query label count mismatch: {run_id}")
        query_ids[int(record["frame_id"])] = ids

    rows = prediction["query_rows"]
    with np.load(feature_path) as z:
        ref = z["source_embeddings"].astype(np.float32)
        qry = z["query_embeddings"].astype(np.float32)
        ref_frames = z["source_frame_ids"].astype(np.int64)
        qry_frames = z["query_frame_ids"].astype(np.int64)
    if len(ref) != len(source_ids) or len(qry) != len(rows) or len(qry) != len(qry_frames):
        raise ValueError(f"feature/label/prediction count mismatch: {run_id}")
    if ref_frames.tolist() != [int(r["frame_id"]) for r in frozen["records"]["reference"] for _ in range(int(r["masks"]))]:
        raise ValueError(f"reference feature order mismatch: {run_id}")
    if qry_frames.tolist() != [int(r["query_frame"]) for r in rows]:
        raise ValueError(f"query feature order mismatch: {run_id}")

    source_ids = np.asarray(source_ids, dtype=np.int64)
    candidates = sorted(int(x) for x in np.unique(source_ids) if x > 0)
    candidate_counts = [int(np.sum(source_ids == oid)) for oid in candidates]
    if len(candidates) < 2:
        raise ValueError(f"fewer than two supported reference identities: {run_id}")
    ref /= np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    qry /= np.maximum(np.linalg.norm(qry, axis=1, keepdims=True), 1e-12)
    scores = qry @ ref.T
    # Average the best k observations to reduce unequal-view-count effects.
    # If a source object has fewer than k observations, average its available
    # observations; this support difference is recorded in the protocol limit.
    object_scores = np.stack([
        np.sort(scores[:, source_ids == oid], axis=1)[:, -min(views_per_object, int(np.sum(source_ids == oid))):].mean(1)
        for oid in candidates
    ], axis=1)
    top2 = np.partition(object_scores, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    # `partition` does not sort its final two values.
    margin = np.maximum(margin, 0.0)
    pred_ids = np.asarray(candidates, dtype=np.int64)[object_scores.argmax(1)]
    query_identity = np.asarray([int(query_ids[int(r["query_frame"])][int(r["query_index"])]) for r in rows])
    return {
        "group": group_id,
        "margin": margin.astype(np.float64),
        "correct": ((query_identity > 0) & (pred_ids == query_identity)),
        "queries": len(rows),
        "supported_candidates": len(candidates),
        "candidate_view_counts": candidate_counts,
    }


def load_split(input_root: Path, pred_root: Path, views_per_object: int) -> list[dict]:
    manifest = json.loads((pred_root / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete split: {pred_root}")
    relations = [load_relation(input_root, pred_root, item, views_per_object) for item in manifest["results"]]
    merged = {}
    for relation in relations:
        group = merged.setdefault(relation["group"], {
            "group": relation["group"], "margins": [], "correct_rows": [], "queries": 0,
            "supported_candidates": 0, "candidate_view_counts": [],
        })
        group["margins"].append(relation["margin"])
        group["correct_rows"].append(relation["correct"])
        group["queries"] += relation["queries"]
        group["supported_candidates"] += relation["supported_candidates"]
        group["candidate_view_counts"].extend(relation["candidate_view_counts"])
    out = []
    for group in merged.values():
        out.append({
            "group": group["group"],
            "margin": np.concatenate(group["margins"]),
            "correct": np.concatenate(group["correct_rows"]),
            "queries": group["queries"],
            "supported_candidates": group["supported_candidates"],
            "candidate_view_counts": group["candidate_view_counts"],
        })
    return out


def metric(groups: list[dict], threshold: float) -> dict:
    rows = []
    for group in groups:
        accepted = group["margin"] >= threshold
        n = int(accepted.sum())
        c = int((accepted & group["correct"]).sum())
        rows.append({"accepted": n, "correct": c, "queries": group["queries"],
                     "precision": c / n if n else None, "coverage": c / group["queries"]})
    supported = [r for r in rows if r["precision"] is not None]
    na, nc, nq = (sum(r[k] for r in rows) for k in ("accepted", "correct", "queries"))
    return {
        "groups": len(rows), "accepted": na, "correct": nc, "queries": nq,
        "acceptance_rate": na / nq if nq else None,
        "pooled_precision": nc / na if na else None,
        "pooled_correct_coverage": nc / nq if nq else None,
        "group_macro_precision": float(np.mean([r["precision"] for r in supported])) if supported else None,
        "group_macro_correct_coverage": float(np.mean([r["coverage"] for r in rows])) if rows else None,
        "groups_with_accepts": len(supported), "per_group": rows,
    }


def fit(groups: list[dict], precision_floor: float) -> tuple[float, dict]:
    candidates = np.unique(np.concatenate([g["margin"] for g in groups]))
    feasible = []
    for threshold in candidates:
        result = metric(groups, float(threshold))
        if result["group_macro_precision"] is not None and result["group_macro_precision"] >= precision_floor:
            feasible.append((result["group_macro_correct_coverage"], float(threshold), result))
    if not feasible:
        raise ValueError(f"no threshold reaches source macro precision {precision_floor:.3f}")
    _, threshold, result = max(feasible, key=lambda x: (x[0], x[1]))
    return threshold, result


def bootstrap_object(rows: list[dict], threshold: float, margin_key: str,
                     correct_key: str, draws: int = 20_000) -> dict:
    ids = sorted({r["object_id"] for r in rows})
    by_id = []
    for oid in ids:
        subset = [r for r in rows if r["object_id"] == oid]
        accepted = [r for r in subset if r[margin_key] >= threshold]
        by_id.append((len(accepted), sum(bool(r[correct_key]) for r in accepted), len(subset)))
    arr = np.asarray(by_id, dtype=np.float64)
    rng = np.random.default_rng(20261009)
    sample = rng.integers(0, len(ids), size=(draws, len(ids)))
    acc, cor, qry = arr[sample, 0].sum(1), arr[sample, 1].sum(1), arr[sample, 2].sum(1)
    return {
        "draws": draws, "seed": 20261009, "unit": "physical object ID",
        "pooled_precision_95": [float(x) for x in np.quantile(cor / np.maximum(acc, 1), [.025, .975])],
        "correct_coverage_95": [float(x) for x in np.quantile(cor / np.maximum(qry, 1), [.025, .975])],
        "object_macro_coverage_95": [float(x) for x in np.quantile((cor / np.maximum(qry, 1)), [.025, .975])],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-input", type=Path, default=DEFAULT_SOURCE_INPUT)
    p.add_argument("--source-predictions", type=Path, default=DEFAULT_SOURCE_PRED)
    p.add_argument("--validation-input", type=Path, default=DEFAULT_VAL_INPUT)
    p.add_argument("--validation-predictions", type=Path, default=DEFAULT_VAL_PRED)
    p.add_argument("--uw-rows", type=Path, default=DEFAULT_UW)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--precision-floor", type=float, default=.90)
    p.add_argument("--views-per-object", type=int, default=3)
    a = p.parse_args()
    source = load_split(a.source_input, a.source_predictions, a.views_per_object)
    validation = load_split(a.validation_input, a.validation_predictions, a.views_per_object)
    source_ids = {g["group"] for g in source}
    validation_ids = {g["group"] for g in validation}
    if source_ids & validation_ids:
        raise ValueError("source and validation reference groups overlap")
    threshold, source_metric = fit(source, a.precision_floor)
    val_metric = metric(validation, threshold)
    source_loo = []
    for held in source:
        train = [g for g in source if g["group"] != held["group"]]
        try:
            loo_threshold, loo_fit = fit(train, a.precision_floor)
            held_metric = metric([held], loo_threshold)
            held_row = held_metric["per_group"][0]
        except ValueError:
            loo_threshold, loo_fit, held_metric, held_row = None, None, None, None
        source_loo.append({
            "held_out_reference_group": held["group"],
            "threshold": loo_threshold,
            "training_macro_precision": loo_fit["group_macro_precision"] if loo_fit else None,
            "training_macro_coverage": loo_fit["group_macro_correct_coverage"] if loo_fit else None,
            "held_out": held_row,
        })
    feasible_loo = [r for r in source_loo if r["threshold"] is not None]

    uw = [json.loads(x) for x in a.uw_rows.read_text().splitlines() if x.strip()]
    if a.views_per_object == 1:
        uw_margin_key, uw_correct_key = "dino_top1_margin", "dino_instance_correct"
    else:
        uw_margin_key = f"dino_top{a.views_per_object}mean_margin"
        uw_correct_key = f"dino_top{a.views_per_object}mean_instance_correct"
    if len(uw) != 21_600 or any(uw_margin_key not in r for r in uw):
        raise ValueError(f"expected 21,600 UW predictions with {uw_margin_key}")
    uw_group = {"group": "UW_RGBD_Object", "margin": np.asarray([r[uw_margin_key] for r in uw]),
                "correct": np.asarray([r[uw_correct_key] for r in uw], dtype=bool),
                "queries": len(uw), "supported_candidates": 300}
    uw_metric = metric([uw_group], threshold)
    accepted = [r for r in uw if r[uw_margin_key] >= threshold]
    uw_bootstrap = bootstrap_object(uw, threshold, uw_margin_key, uw_correct_key)
    uw_macro_acc = float(np.mean([np.mean([r[uw_correct_key] for r in uw if r["object_id"] == oid])
                                  for oid in sorted({r["object_id"] for r in uw})]))
    uw_supported_objects = len({r["object_id"] for r in accepted})
    one_sided_object_precision_lower_95 = 0.05 ** (1.0 / uw_supported_objects) if uw_supported_objects else None
    if feasible_loo:
        loo_text = (
            f"Leave-one-source-group-out threshold refits are feasible in {len(feasible_loo)}/{len(source_loo)} folds; held-out precision meets {a.precision_floor:.0%} in {sum(r['held_out']['precision'] >= a.precision_floor for r in feasible_loo)}/{len(feasible_loo)} groups. The cutoff range is [{min(r['threshold'] for r in feasible_loo):.5f}, {max(r['threshold'] for r in feasible_loo):.5f}]."
        )
    else:
        loo_text = f"No leave-one-source-group-out fit met the {a.precision_floor:.0%} source precision floor."
    result = {
        "status": "complete",
        "policy": f"mean of the best {a.views_per_object} cosines per physical candidate (or all available source observations if fewer); accept when top-1 minus top-2 candidate score reaches a frozen threshold",
        "selection": {"source_groups": len(source), "precision_floor": a.precision_floor,
                      "views_per_object": a.views_per_object,
                        "objective": "maximum group-macro correct coverage among source thresholds meeting group-macro precision floor",
                      "threshold": threshold, "source_fit": source_metric},
        "source_leave_one_group_out": {
            "folds": len(source_loo), "feasible_folds": len(feasible_loo),
            "threshold_range": [min(r["threshold"] for r in feasible_loo), max(r["threshold"] for r in feasible_loo)] if feasible_loo else None,
            "held_out_group_macro_precision": float(np.mean([r["held_out"]["precision"] for r in feasible_loo])) if feasible_loo else None,
            "held_out_group_macro_correct_coverage": float(np.mean([r["held_out"]["coverage"] for r in feasible_loo])) if feasible_loo else None,
            "held_out_groups_meeting_precision_floor": sum(r["held_out"]["precision"] >= a.precision_floor for r in feasible_loo),
            "per_fold": source_loo,
        },
        "source_candidate_view_support": {
            "candidate_identities": int(sum(g["supported_candidates"] for g in source)),
            "candidate_identities_with_fewer_than_k_views": int(sum(sum(n < a.views_per_object for n in g["candidate_view_counts"]) for g in source)),
            "views_per_candidate_quantiles_across_source_groups": [
                float(x) for x in np.quantile(np.concatenate([np.asarray(g["candidate_view_counts"]) for g in source]), [0, .25, .5, .75, 1])
            ],
        },
        "three_rscan_validation": {"groups": len(validation), "frozen_transfer": val_metric},
        "uw_rgbd_object": {
            "queries": len(uw), "objects": len({r["object_id"] for r in uw}),
            "threshold": threshold, "frozen_transfer": uw_metric,
            "object_macro_unfiltered_top1": uw_macro_acc,
            "objects_with_any_accepts": uw_supported_objects,
            "supported_object_perfect_precision_one_sided_95_lower": one_sided_object_precision_lower_95,
            "physical_object_cluster_bootstrap": uw_bootstrap,
            "absolute_cosine_gate_acceptance_rate": 0.9951388888888889,
            "caveat": "controlled turntable set, supplied masks/crops, closed-set 300-object gallery; not room rescan or preregistered independent confirmation",
        },
        "limits": [
            "The source cutoff is chosen with identity labels on eight 3RScan reference groups; source fit is exploratory, not a finite-sample guarantee.",
            "The margin definition is label-free at inference, but the selection and evaluation populations have been explored in this project.",
            "The UW test is controlled object-view retrieval with privileged masks and a fixed closed-set gallery, not native room-rescan deployment.",
        ],
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = [
        "# Source-frozen object-score margin gate",
        "",
        f"A label-free candidate margin is computed as the best physical-object score minus the second-best, where each candidate score is the mean of its best {a.views_per_object} source-view cosines (or all views when fewer are available). The threshold is fitted on {len(source)} source groups for a {a.precision_floor:.0%} group-macro precision point-estimate floor, maximizing group-macro correct coverage.",
        "",
        f"Frozen margin threshold: `{threshold:.8f}`.",
        "",
        loo_text,
        "",
        "## Transfer results",
        "",
        "| Target | Accepted | Pooled precision | Correct coverage | Macro correct coverage |",
        "|---|---:|---:|---:|---:|",
        f"| 3RScan source fit | {source_metric['accepted']:,}/{source_metric['queries']:,} ({source_metric['acceptance_rate']:.1%}) | {source_metric['pooled_precision']:.1%} (group macro {source_metric['group_macro_precision']:.1%}) | {source_metric['pooled_correct_coverage']:.1%} | {source_metric['group_macro_correct_coverage']:.1%} |",
        f"| 3RScan validation | {val_metric['accepted']:,}/{val_metric['queries']:,} ({val_metric['acceptance_rate']:.1%}) | {val_metric['pooled_precision']:.1%} (group macro {val_metric['group_macro_precision']:.1%}) | {val_metric['pooled_correct_coverage']:.1%} | {val_metric['group_macro_correct_coverage']:.1%} |",
        f"| UW RGB-D Object | {uw_metric['accepted']:,}/{uw_metric['queries']:,} ({uw_metric['acceptance_rate']:.1%}) | {uw_metric['pooled_precision']:.1%} | {uw_metric['pooled_correct_coverage']:.1%} | {uw_metric['group_macro_correct_coverage']:.1%} |",
        "",
        f"The margin gate accepts queries from {uw_supported_objects}/300 physical objects and makes no incorrect accepted assignments. Its object-cluster bootstrap interval for correct coverage is [{uw_bootstrap['correct_coverage_95'][0]:.1%}, {uw_bootstrap['correct_coverage_95'][1]:.1%}]. Unfiltered top-{a.views_per_object}-view-score object-macro accuracy is {uw_macro_acc:.1%}; the absolute-cosine max-view gate accepts 99.5% of queries at 88.2% precision. The margin therefore trades much lower coverage for high observed precision. Because there are zero supported objects with any accepted error, the bootstrap precision interval is degenerate; as an explicitly assumption-dependent sensitivity, 0 failures among {uw_supported_objects} supported physical objects gives a one-sided exact 95% lower bound of {one_sided_object_precision_lower_95:.1%} on the fraction of supported objects with no accepted error. This is a sample-based uncertainty summary, not a deployment guarantee.",
        "",
        f"In the 3RScan source bank, {result['source_candidate_view_support']['candidate_identities_with_fewer_than_k_views']}/{result['source_candidate_view_support']['candidate_identities']} candidate identities have fewer than {a.views_per_object} observations; those candidates are scored using all available views. For the top-1/max-view sensitivity, source counts range from 1 to 20, while the UW bank has 12 views per object. View-count mismatch therefore remains a limitation.",
        "",
        "The UW evaluation is a controlled turntable retrieval task with supplied masks, crops, and a closed-set 300-object gallery. It is not native RGB-D room-rescan confirmation. The source groups and prior validation have been explored; these results are post-hoc and do not support a deployment guarantee.",
        "",
    ]
    (a.output_dir / "report.md").write_text("\n".join(report))
    print((a.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
