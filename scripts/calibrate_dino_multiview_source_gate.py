#!/usr/bin/env python3
"""Fit a cosine acceptance gate on source relations, then freeze it on validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reference_scan_id(record: dict) -> str:
    path = Path(record["input"])
    uuid_pattern = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    match = re.match(rf"^({uuid_pattern})_frame_\d+_input\.npz$", path.name)
    scan_id = match.group(1) if match else path.parent.name
    if not re.fullmatch(uuid_pattern, scan_id):
        raise ValueError(f"cannot determine reference scan UUID from {path}")
    return scan_id.lower()


def rows_for(run_id: str, input_root: Path, prediction_root: Path) -> dict:
    frozen = json.loads((input_root / "uniform20_inputs" / run_id / "manifest.json").read_text())
    prediction = json.loads((prediction_root / f"{run_id}.json").read_text())
    if prediction.get("status") != "complete" or prediction.get("labels_read") is not False:
        raise ValueError(f"invalid inference artifact: {run_id}")
    expected_sha = {}
    for role in ("reference", "rescan"):
        digest = hashlib.sha256()
        for record in frozen["records"][role]:
            digest.update(bytes.fromhex(record["input_sha256"]))
        expected_sha[role] = digest.hexdigest()
    if prediction.get("input_sha256") != expected_sha:
        raise ValueError(f"input checksum mismatch: {run_id}")
    for role in ("reference", "query"):
        key = "reference" if role == "reference" else "rescan"
        if prediction[f"{role}_frames"] != [int(r["frame_id"]) for r in frozen["records"][key]]:
            raise ValueError(f"frame order mismatch: {run_id}/{role}")

    source_ids = []
    for record in frozen["records"]["reference"]:
        path = Path(record["identity_targets"])
        if not path.is_absolute():
            path = ROOT / path
        ids = np.load(path).astype(np.int64)
        if len(ids) != int(record["masks"]):
            raise ValueError(f"source labels/masks mismatch: {path}")
        source_ids.extend(ids.tolist())
    query_ids = {}
    for record in frozen["records"]["rescan"]:
        path = Path(record["identity_targets"])
        if not path.is_absolute():
            path = ROOT / path
        ids = np.load(path).astype(np.int64)
        if len(ids) != int(record["masks"]):
            raise ValueError(f"query labels/masks mismatch: {path}")
        query_ids[int(record["frame_id"])] = ids

    source_identity_set = {int(value) for value in source_ids if int(value) > 0}
    correct, scores, assignment_valid, identity_available = [], [], [], []
    seen: dict[int, set[int]] = {}
    for row in prediction["query_rows"]:
        frame, index = int(row["query_frame"]), int(row["query_index"])
        if frame not in query_ids or not 0 <= index < len(query_ids[frame]):
            raise ValueError(f"query label index invalid: {run_id}/{frame}/{index}")
        if index in seen.setdefault(frame, set()):
            raise ValueError(f"duplicate query row: {run_id}/{frame}/{index}")
        seen[frame].add(index)
        assigned = row["assigned_source_observation"]
        score = row["assigned_similarity"]
        valid = assigned is not None and score is not None
        if valid and not 0 <= int(assigned) < len(source_ids):
            raise ValueError(f"source assignment invalid: {run_id}")
        qid = int(query_ids[frame][index])
        identity_available.append(bool(qid > 0 and qid in source_identity_set))
        correct.append(bool(valid and qid > 0 and qid == source_ids[int(assigned)]))
        # Keep assignment validity separate from score values: valid cosine
        # similarities may be negative, so a numeric sentinel is ambiguous.
        scores.append(float(score) if valid else 0.0)
        assignment_valid.append(bool(valid))

    expected = {int(r["frame_id"]): int(r["masks"]) for r in frozen["records"]["rescan"]}
    nonempty = {frame for frame, count in expected.items() if count}
    if set(seen) != nonempty or any(len(seen.get(frame, ())) != count for frame, count in expected.items()):
        raise ValueError(f"incomplete query labels: {run_id}")
    env = reference_scan_id(frozen["records"]["reference"][0])
    return {"run_id": run_id, "environment_id": env, "scores": np.asarray(scores),
            "valid": np.asarray(assignment_valid, dtype=bool),
            "correct": np.asarray(correct, dtype=bool),
            "identity_available": np.asarray(identity_available, dtype=bool),
            "queries": len(scores)}


def load_split(input_root: Path, prediction_root: Path) -> list[dict]:
    manifest = json.loads((prediction_root / "run_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete manifest: {prediction_root}")
    return [rows_for(x["run_id"], input_root, prediction_root) for x in manifest["results"]]


def grouped_metrics(relations: list[dict], threshold: float) -> tuple[dict, list[dict]]:
    groups: dict[str, dict] = {}
    for relation in relations:
        g = groups.setdefault(relation["environment_id"], {"accepted": 0, "correct": 0, "queries": 0})
        selected = relation["valid"] & (relation["scores"] >= threshold)
        g["accepted"] += int(selected.sum())
        g["correct"] += int((selected & relation["correct"]).sum())
        g["queries"] += relation["queries"]
    detail = []
    for env, g in sorted(groups.items()):
        detail.append({"environment_id": env, **g,
                       "precision": g["correct"] / g["accepted"] if g["accepted"] else None,
                       "coverage": g["correct"] / g["queries"] if g["queries"] else None})
    valid = [r for r in detail if r["precision"] is not None]
    summary = {"groups": len(detail), "groups_with_accepts": len(valid),
               "accepted": sum(r["accepted"] for r in detail),
               "correct": sum(r["correct"] for r in detail),
               "queries": sum(r["queries"] for r in detail),
               "macro_precision": float(np.mean([r["precision"] for r in valid])) if valid else None,
               "macro_coverage": float(np.mean([r["coverage"] for r in detail])) if detail else None}
    summary["pooled_precision"] = summary["correct"] / summary["accepted"] if summary["accepted"] else None
    summary["pooled_coverage"] = summary["correct"] / summary["queries"] if summary["queries"] else None
    return summary, detail


def bootstrap(relations: list[dict], threshold: float, draws: int, seed: int) -> dict:
    summary, detail = grouped_metrics(relations, threshold)
    if len(detail) < 2:
        return {}
    rng = np.random.default_rng(seed)
    n = len(detail)
    ix = rng.integers(0, n, size=(draws, n))
    prec = np.asarray([r["precision"] if r["precision"] is not None else np.nan for r in detail])
    cov = np.asarray([r["coverage"] for r in detail])
    return {"draws": draws, "seed": seed, "unit": "reference scan ID",
            "macro_precision_percentile_95": [float(x) for x in np.nanquantile(np.nanmean(prec[ix], axis=1), [.025, .975])],
            "macro_coverage_percentile_95": [float(x) for x in np.quantile(cov[ix].mean(axis=1), [.025, .975])]}


def fit_threshold(relations: list[dict], target_macro_precision: float) -> tuple[float | None, dict | None]:
    scores = np.concatenate([r["scores"] for r in relations])
    valid = np.concatenate([r["valid"] for r in relations])
    candidates = np.unique(scores[valid])
    feasible = []
    for threshold in candidates:
        metrics, _ = grouped_metrics(relations, float(threshold))
        if metrics["macro_precision"] is not None and metrics["macro_precision"] >= target_macro_precision:
            feasible.append((metrics["macro_coverage"], float(threshold), metrics))
    if not feasible:
        return None, None
    _, threshold, metrics = max(feasible, key=lambda x: (x[0], x[1]))
    return threshold, metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-input-root", type=Path, default=ROOT / "outputs/reports/20261009_unseen_oasis_equal_budget")
    p.add_argument("--source-prediction-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank_train8")
    p.add_argument("--validation-input-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    p.add_argument("--validation-prediction-root", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank")
    p.add_argument("--remind-summary", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_bank/summary.json")
    p.add_argument("--target-macro-precision", type=float, default=.90)
    p.add_argument("--draws", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reports/20261009_dino_multiview_source_gate")
    a = p.parse_args()
    source = load_split(a.source_input_root, a.source_prediction_root)
    validation = load_split(a.validation_input_root, a.validation_prediction_root)
    source_env = {r["environment_id"] for r in source}
    validation_env = {r["environment_id"] for r in validation}
    if source_env & validation_env:
        raise ValueError(f"source/validation environments overlap: {sorted(source_env & validation_env)}")

    scores = np.concatenate([r["scores"] for r in source])
    valid = np.concatenate([r["valid"] for r in source])
    candidates = np.unique(scores[valid])
    threshold, fit_metrics = fit_threshold(source, a.target_macro_precision)
    if threshold is None:
        raise ValueError(f"no source threshold meets macro precision {a.target_macro_precision}")
    source_metrics, source_groups = grouped_metrics(source, threshold)
    val_metrics, val_groups = grouped_metrics(validation, threshold)
    remind = json.loads(a.remind_summary.read_text())
    gate_by_group = {r["environment_id"]: r for r in val_groups}
    remind_by_group = {r["environment_id"]: r["remind_one_to_one"] for r in remind["reference_groups"]}
    if set(gate_by_group) != set(remind_by_group):
        raise ValueError("frozen-gate and REMIND reference groups differ")
    paired_ids = sorted(gate_by_group)
    rng = np.random.default_rng(a.seed + 1)
    sample = rng.integers(0, len(paired_ids), size=(a.draws, len(paired_ids)))
    paired_effects = {}
    for metric in ("precision", "coverage"):
        diff = np.asarray([gate_by_group[env][metric] - remind_by_group[env][metric] for env in paired_ids])
        boot = diff[sample].mean(axis=1)
        paired_effects[metric] = {
            "estimate": float(diff.mean()),
            "ordinary_percentile_95": [float(x) for x in np.quantile(boot, [.025, .975])],
            "bonferroni_2_outcomes_percentile_95": [float(x) for x in np.quantile(boot, [.0125, .9875])],
            "groups_gate_higher": int(np.sum(diff > 0)), "groups_tied": int(np.sum(diff == 0)),
            "groups_remind_higher": int(np.sum(diff < 0)),
        }
    paired_remind = {"comparison": "source-frozen DINO cosine gate minus REMIND-style one-to-one source map",
                     "same_reference_groups": len(paired_ids), "bootstrap_draws": a.draws,
                     "seed": a.seed + 1, "unit": "paired released reference scan ID",
                     "dino_gate": val_metrics,
                     "remind_macro_precision": float(np.mean([remind_by_group[e]["precision"] for e in paired_ids])),
                     "remind_macro_coverage": float(np.mean([remind_by_group[e]["coverage"] for e in paired_ids])),
                     "remind_total_accepted": int(remind["pooled"]["remind_one_to_one"]["accepted"]),
                     "remind_total_correct": int(remind["pooled"]["remind_one_to_one"]["correct"]),
                     "effects": paired_effects}
    loo_rows = []
    for held in source:
        training = [r for r in source if r["environment_id"] != held["environment_id"]]
        loo_threshold, training_metrics = fit_threshold(training, a.target_macro_precision)
        if loo_threshold is None:
            held_metrics, held_group = None, None
        else:
            held_metrics, held_groups = grouped_metrics([held], loo_threshold)
            held_group = held_groups[0]
        loo_rows.append({"held_out_environment_id": held["environment_id"],
                         "threshold": loo_threshold,
                         "training_macro_precision": training_metrics["macro_precision"] if training_metrics else None,
                         "training_macro_coverage": training_metrics["macro_coverage"] if training_metrics else None,
                         "held_out": held_metrics, "held_out_group": held_group})
    loo_valid = [r for r in loo_rows if r["held_out"] is not None]
    loo_summary = {"fold_count": len(loo_rows), "feasible_folds": len(loo_valid),
                   "threshold_range": [min(r["threshold"] for r in loo_valid), max(r["threshold"] for r in loo_valid)] if loo_valid else None,
                   "held_out_macro_precision": float(np.mean([r["held_out"]["macro_precision"] for r in loo_valid])) if loo_valid else None,
                   "held_out_macro_coverage": float(np.mean([r["held_out"]["macro_coverage"] for r in loo_valid])) if loo_valid else None,
                   "held_out_groups_meeting_90pct_precision": sum(r["held_out"]["macro_precision"] >= .90 for r in loo_valid),
                   "folds": loo_rows}
    result = {
        "status": "complete",
        "scope": "exploratory source-frozen threshold transfer for full-reference-view DINO exemplar matching on the same explored 3RScan benchmark",
        "selection_rule": {"target_source_macro_precision": a.target_macro_precision,
                           "objective": "maximum source macro correct coverage among thresholds meeting the source macro precision floor",
                           "comparison": "assigned cosine similarity >= threshold",
                           "fit_relations": len(source), "candidate_thresholds": len(candidates)},
        "frozen_threshold": threshold,
        "source_fit": {"metrics": source_metrics, "reference_groups": source_groups},
        "leave_one_source_group_out": loo_summary,
        "validation_transfer": {"metrics": val_metrics, "reference_groups": val_groups,
                                "bootstrap": bootstrap(validation, threshold, a.draws, a.seed)},
        "paired_remind_comparison": paired_remind,
        "split_audit": {"source_reference_groups": len(source_env), "validation_reference_groups": len(validation_env),
                        "overlap": sorted(source_env & validation_env)},
        "limits": [
            f"The source threshold is selected using identity labels on only {len(source_env)} reference groups; source-fit precision is exploratory and is not a finite-sample guarantee.",
            "Validation groups were not used to select the threshold, but the same 3RScan benchmark and query masks have been explored in prior analyses; this is not independent-site confirmation.",
            "All reference and query masks are annotation-derived, and the matcher pools reference mask exemplars without track memory or cross-frame query updates.",
            "Validation bootstrap intervals condition on the frozen threshold and quantify reference-group resampling only; they do not account for threshold-fitting uncertainty.",
        ]}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "source_frozen_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    m, v = source_metrics, val_metrics
    report = f"""# Source-frozen DINO multi-view acceptance gate

## Protocol

We selected one absolute cosine-similarity threshold using only {len(source_env)} source reference groups. Among observed assignment scores, the rule chooses the threshold with maximum source macro correct coverage subject to source macro precision of at least {a.target_macro_precision:.0%}. We then froze that threshold and applied it unchanged to {len(validation)} target relations ({len(validation_env)} reference groups). Labels are read only for source threshold fitting and post-hoc scoring; inference artifacts record `labels_read: false`. Source and target reference-group IDs are disjoint.

## Results

| Split | Reference groups | Accepted | Correct | Macro precision | Macro correct coverage | Pooled precision | Pooled correct coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source fit | {m['groups']} | {m['accepted']} | {m['correct']} | {m['macro_precision']:.1%} | {m['macro_coverage']:.1%} | {m['pooled_precision']:.1%} | {m['pooled_coverage']:.1%} |
| Frozen target transfer | {v['groups']} | {v['accepted']} | {v['correct']} | {v['macro_precision']:.1%} | {v['macro_coverage']:.1%} | {v['pooled_precision']:.1%} | {v['pooled_coverage']:.1%} |

Frozen cutoff: **{threshold:.6f}**. Target reference-group bootstrap 95% percentile intervals, conditional on this fitted cutoff: macro precision {result['validation_transfer']['bootstrap']['macro_precision_percentile_95'][0]:.1%}–{result['validation_transfer']['bootstrap']['macro_precision_percentile_95'][1]:.1%}; macro correct coverage {result['validation_transfer']['bootstrap']['macro_coverage_percentile_95'][0]:.1%}–{result['validation_transfer']['bootstrap']['macro_coverage_percentile_95'][1]:.1%}.

Against the saved REMIND-style one-to-one map on the same {len(paired_ids)} reference groups, the frozen DINO gate has macro precision/coverage of {val_metrics['macro_precision']:.1%}/{val_metrics['macro_coverage']:.1%}, versus {paired_remind['remind_macro_precision']:.1%}/{paired_remind['remind_macro_coverage']:.1%}. The gate accepts {val_metrics['accepted']:,} masks ({val_metrics['correct']:,} correct), compared with {paired_remind['remind_total_accepted']:,} accepted ({paired_remind['remind_total_correct']:,} correct) for REMIND. Paired macro differences are {paired_effects['precision']['estimate']*100:+.1f} pp precision (two-outcome-adjusted interval [{paired_effects['precision']['bonferroni_2_outcomes_percentile_95'][0]*100:+.1f},{paired_effects['precision']['bonferroni_2_outcomes_percentile_95'][1]*100:+.1f}]) and {paired_effects['coverage']['estimate']*100:+.1f} pp correct coverage (adjusted interval [{paired_effects['coverage']['bonferroni_2_outcomes_percentile_95'][0]*100:+.1f},{paired_effects['coverage']['bonferroni_2_outcomes_percentile_95'][1]*100:+.1f}]). This is not an equal-acceptance comparison: the association procedures differ, and both receive privileged masks on a previously explored benchmark.

As a source-only stability check, we refit the same rule in {loo_summary['fold_count']} leave-one-reference-group-out folds. A fold is feasible only when its remaining training groups contain a cutoff meeting the same 90% macro precision floor. Across feasible folds, {loo_summary['feasible_folds']}/{loo_summary['fold_count']} had such a cutoff; their thresholds ranged from {min(r['threshold'] for r in loo_valid):.6f} to {max(r['threshold'] for r in loo_valid):.6f}. Held-out source-group precision averaged {loo_summary['held_out_macro_precision']:.1%} and coverage {loo_summary['held_out_macro_coverage']:.1%}; {loo_summary['held_out_groups_meeting_90pct_precision']}/{loo_summary['feasible_folds']} held-out groups reached 90% precision. These folds are a stability diagnostic on previously explored source data, not independent replications.

## Limits

This is exploratory source-fit calibration with only {len(source_env)} source groups, not a distribution-free precision guarantee. Although target labels did not choose the cutoff, the same 3RScan benchmark and masks were examined in earlier analyses, so the transfer is not independent-site confirmation. Both sides use privileged annotation masks. The matcher has no track memory or query updates. Bootstrap intervals condition on the selected cutoff and omit fitting uncertainty.
"""
    (a.output / "report.md").write_text(report)
    (a.output / "leave_one_source_group_out.json").write_text(json.dumps(loo_summary, indent=2) + "\n")
    (a.output / "paired_remind_comparison.json").write_text(json.dumps(paired_remind, indent=2) + "\n")
    print(json.dumps({"status": "complete", "threshold": threshold, "source": source_metrics,
                      "validation": val_metrics, "bootstrap": result["validation_transfer"]["bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
