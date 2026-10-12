#!/usr/bin/env python3
"""Fit a single HILO DINO margin cutoff on hash8 and evaluate unchanged on hash16."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/20261009_hilo_pose_projection"
SOURCE_FILE = REPORT / "hash8_identity_pilot.json"
TARGET_FILE = REPORT / "hash16_identity_pilot.json"
SCORED_FILE = REPORT / "hash16_identity_margin.json"


def load_scenes(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    return data["scenes"]


def metric(rows: list[dict], mask: np.ndarray) -> dict:
    n_total = len(rows)
    accepted = mask
    n_accept = int(accepted.sum())
    n_correct = int(sum(bool(row["dino_top1_correct"]) for row, keep in zip(rows, accepted) if keep))
    return {
        "queries": n_total,
        "accepted": n_accept,
        "correct_accepted": n_correct,
        "precision": n_correct / n_accept if n_accept else None,
        "correct_coverage": n_correct / n_total if n_total else None,
        "acceptance_coverage": n_accept / n_total if n_total else None,
    }


def rows_for_scenes(rows: list[dict], scenes: list[str]) -> list[dict]:
    wanted = set(scenes)
    return [row for row in rows if row["scene"] in wanted]


def fit_threshold(rows: list[dict], precision_floor: float) -> dict:
    margins = np.asarray([float(row["dino_top1_margin"]) for row in rows])
    correct = np.asarray([bool(row["dino_top1_correct"]) for row in rows])
    candidates = np.unique(margins)
    best = None
    for threshold in candidates:
        accepted = margins >= threshold
        n = int(accepted.sum())
        if n == 0:
            continue
        precision = float(correct[accepted].mean())
        coverage = float((correct & accepted).mean())
        if precision + 1e-12 < precision_floor:
            continue
        # Maximize correct coverage, then accepted coverage, then prefer higher cutoff.
        key = (coverage, float(accepted.mean()), float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), precision, coverage, n)
    if best is None:
        return {"feasible": False, "precision_floor": precision_floor}
    _, threshold, precision, coverage, n = best
    return {
        "feasible": True,
        "precision_floor": precision_floor,
        "threshold": threshold,
        "source_precision": precision,
        "source_correct_coverage": coverage,
        "source_acceptance_coverage": n / len(rows),
        "source_accepted": n,
    }


def cluster_bootstrap(rows: list[dict], scenes: list[str], threshold: float,
                      draws: int, seed: int) -> dict:
    by_scene = {scene: [row for row in rows if row["scene"] == scene] for scene in scenes}
    rng = np.random.default_rng(seed)
    vals = {key: [] for key in ("precision", "correct_coverage", "acceptance_coverage")}
    paired = {key: [] for key in ("precision", "correct_coverage", "acceptance_coverage")}
    valid_precision = 0
    for _ in range(draws):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        batch = [row for scene in sampled for row in by_scene[str(scene)]]
        margins = np.asarray([float(row["dino_top1_margin"]) for row in batch])
        correct = np.asarray([bool(row["dino_top1_correct"]) for row in batch])
        accepted = margins >= threshold
        mutual = np.asarray([bool(row["dino_mutual_accepted"]) for row in batch])
        n_accept = int(accepted.sum())
        if n_accept:
            vals["precision"].append(float(correct[accepted].mean()))
            valid_precision += 1
        vals["correct_coverage"].append(float((correct & accepted).mean()))
        vals["acceptance_coverage"].append(float(accepted.mean()))
        for key, arm in (("margin", accepted), ("mutual", mutual)):
            if key == "margin":
                margin_metrics = {
                    "precision": float(correct[arm].mean()) if arm.any() else np.nan,
                    "correct_coverage": float((correct & arm).mean()),
                    "acceptance_coverage": float(arm.mean()),
                }
            else:
                mutual_metrics = {
                    "precision": float(correct[arm].mean()) if arm.any() else np.nan,
                    "correct_coverage": float((correct & arm).mean()),
                    "acceptance_coverage": float(arm.mean()),
                }
        for key in paired:
            if np.isfinite(margin_metrics[key]) and np.isfinite(mutual_metrics[key]):
                paired[key].append(margin_metrics[key] - mutual_metrics[key])
    return {
        "draws": draws,
        "seed": seed,
        "clusters": len(scenes),
        "precision_valid_draws": valid_precision,
        "intervals_95": {
            key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
            for key, value in vals.items() if value
        },
        "paired_margin_minus_mutual_intervals_95": {
            key: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
            for key, value in paired.items() if value
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision-floor", type=float, default=0.90)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path, default=REPORT / "hash8_to_hash16_gate.json")
    args = parser.parse_args()

    source_scenes = load_scenes(SOURCE_FILE)
    target_scenes = load_scenes(TARGET_FILE)
    scored = json.loads(SCORED_FILE.read_text())
    all_rows = scored["query_outcomes"]
    required = {"scene", "dino_top1_correct", "dino_top1_margin", "dino_mutual_accepted"}
    if not all_rows or not required.issubset(all_rows[0]):
        raise ValueError(f"{SCORED_FILE} lacks per-query margin fields; rerun the frozen scorer")
    if set(source_scenes) & set(target_scenes):
        raise ValueError("source and target scenes overlap")
    if len(source_scenes) != 8 or len(target_scenes) != 8:
        raise ValueError("expected the frozen 8+8 configuration split")
    source = rows_for_scenes(all_rows, source_scenes)
    target = rows_for_scenes(all_rows, target_scenes)
    fit = fit_threshold(source, args.precision_floor)
    if not fit["feasible"]:
        raise RuntimeError("source precision floor was infeasible")
    cutoff = fit["threshold"]

    target_margins = np.asarray([float(row["dino_top1_margin"]) for row in target])
    target_correct = np.asarray([bool(row["dino_top1_correct"]) for row in target])
    target_mutual = np.asarray([bool(row["dino_mutual_accepted"]) for row in target])
    margin_result = metric(target, target_margins >= cutoff)
    mutual_result = metric(target, target_mutual)
    top1_result = metric(target, np.ones(len(target), dtype=bool))
    per_scene = {}
    for scene in target_scenes:
        subset = [row for row in target if row["scene"] == scene]
        m = np.asarray([float(row["dino_top1_margin"]) for row in subset]) >= cutoff
        per_scene[scene] = metric(subset, m)

    result = {
        "question": "Does the source-fitted DINO margin threshold transfer unchanged from hash8 to hash16?",
        "split": {"source_scenes": source_scenes, "target_scenes": target_scenes},
        "split_hashes": {
            "source_ordered_scene_list_sha256": hashlib.sha256("\n".join(source_scenes).encode()).hexdigest(),
            "target_ordered_scene_list_sha256": hashlib.sha256("\n".join(target_scenes).encode()).hexdigest(),
        },
        "protocol": {
            "confidence": "top-1 cosine minus runner-up cosine",
            "source_fit": f"maximize correct accepted-query coverage subject to pooled precision >= {args.precision_floor:.3f}; tie-break on acceptance coverage then larger cutoff",
            "labels_used_for_matching": False,
            "labels_used_for_source_threshold_fit": True,
            "candidate_pool": "known ten object identities per configuration",
            "selection_note": "target cohort was hash-selected before extension scoring, but target top-1 outcomes were already reported; this margin analysis is post-hoc rather than preregistered",
        },
        "source_fit": fit,
        "target": {
            "margin_gate": margin_result,
            "mutual_nearest_neighbor": mutual_result,
            "unfiltered_top1": top1_result,
            "margin_gate_by_scene": per_scene,
            "configuration_cluster_bootstrap": cluster_bootstrap(target, target_scenes, cutoff, args.draws, args.seed),
        },
        "scope": "exploratory controlled HILO object-view stress test; not room-rescan or site generalization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
