#!/usr/bin/env python3
"""Score anonymous all-reference-view DINO retrieval against the REMIND arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "outputs/reports/20261008_segmast3r_validation"
DEFAULT_PREDICTION_ROOT = ROOT / "outputs/reports/20261009_dino_multiview_source_bank"
DEFAULT_REMIND_ROOT = ROOT / "outputs/reports/20261109_remind_style_3rscan_validation"


def read_relation(run_id: str, *, input_root: Path, prediction_root: Path, remind_root: Path) -> dict:
    frozen_path = input_root / "uniform20_inputs" / run_id / "manifest.json"
    frozen = json.loads(frozen_path.read_text())
    prediction = json.loads((prediction_root / f"{run_id}.json").read_text())
    remind = json.loads((remind_root / f"{run_id}.json").read_text())
    if prediction.get("status") != "complete" or prediction.get("labels_read") is not False:
        raise ValueError(f"inference artifact invalid for {run_id}")
    if remind.get("status") != "complete":
        raise ValueError(f"REMIND artifact invalid for {run_id}")
    expected_sha = {}
    for role in ("reference", "rescan"):
        digest = hashlib.sha256()
        for record in frozen["records"][role]:
            digest.update(bytes.fromhex(record["input_sha256"]))
        expected_sha[role] = digest.hexdigest()
    if prediction.get("input_sha256") != expected_sha:
        raise ValueError(f"prediction input checksum chain differs for {run_id}")
    for role, records in (("reference", frozen["records"]["reference"]), ("query", frozen["records"]["rescan"])):
        expected_frames = [int(record["frame_id"]) for record in records]
        if prediction[f"{role}_frames"] != expected_frames:
            raise ValueError(f"prediction frame order differs for {run_id}/{role}")

    source_targets = []
    for record in frozen["records"]["reference"]:
        target_path = Path(record["identity_targets"])
        if not target_path.is_absolute():
            target_path = ROOT / target_path
        ids = np.load(target_path).astype(np.int64)
        if len(ids) != int(record["masks"]):
            raise ValueError(f"source target/mask length mismatch: {target_path}")
        source_targets.extend(ids.tolist())
    query_records = frozen["records"]["rescan"]
    query_targets = {}
    for record in query_records:
        target_path = Path(record["identity_targets"])
        if not target_path.is_absolute():
            target_path = ROOT / target_path
        ids = np.load(target_path).astype(np.int64)
        if len(ids) != int(record["masks"]):
            raise ValueError(f"query target/mask length mismatch: {target_path}")
        query_targets[int(record["frame_id"])] = ids
    if len(source_targets) != prediction["reference_observations"]:
        raise ValueError(f"source observation count mismatch for {run_id}")
    if len(prediction["query_rows"]) != prediction["query_observations"]:
        raise ValueError(f"query row count mismatch for {run_id}")

    rows = []
    query_seen: dict[int, set[int]] = {}
    for row in prediction["query_rows"]:
        frame_id = int(row["query_frame"])
        query_index = int(row["query_index"])
        if frame_id not in query_targets or query_index >= len(query_targets[frame_id]):
            raise ValueError(f"query index out of range for {run_id}/{frame_id}/{query_index}")
        if query_index in query_seen.setdefault(frame_id, set()):
            raise ValueError(f"duplicate query index for {run_id}/{frame_id}/{query_index}")
        query_seen[frame_id].add(query_index)
        query_id = int(query_targets[frame_id][query_index])
        nearest = int(row["nearest_source_observation"])
        if nearest < 0 or nearest >= len(source_targets):
            raise ValueError(f"nearest source index out of range for {run_id}")
        assigned = row["assigned_source_observation"]
        assigned_id = int(source_targets[int(assigned)]) if assigned is not None else 0
        nearest_id = int(source_targets[nearest])
        rows.append({
            "query_id": query_id,
            "nearest_id": nearest_id,
            "nearest_score": float(row["nearest_similarity"]),
            "assigned_id": assigned_id,
            "assigned_score": float(row["assigned_similarity"]) if row["assigned_similarity"] is not None else -1.0,
            "shared": int(query_id > 0 and query_id in source_targets),
        })
    expected_frames = {
        int(record["frame_id"]): int(record["masks"])
        for record in query_records
    }
    expected_nonempty = {frame_id for frame_id, count in expected_frames.items() if count > 0}
    if set(query_seen) != expected_nonempty or any(
        len(query_seen.get(frame_id, set())) != count for frame_id, count in expected_frames.items()
    ):
        raise ValueError(f"missing query frames for {run_id}")

    budget = int(remind["query_links_to_source_tracks"])
    ranked = sorted(range(len(rows)), key=lambda i: (-rows[i]["assigned_score"], i))
    if budget > len(ranked):
        raise ValueError(f"multi-view assignment pool too small for borrowed budget in {run_id}")
    selected = set(ranked[:budget])
    accepted_correct = sum(
        int(rows[i]["query_id"] > 0 and rows[i]["query_id"] == rows[i]["assigned_id"])
        for i in selected
    )
    nearest_correct = sum(
        int(row["query_id"] > 0 and row["query_id"] == row["nearest_id"])
        for row in rows
    )
    shared_count = sum(int(row["shared"]) for row in rows)
    nearest_shared_correct = sum(
        int(row["shared"] and row["query_id"] == row["nearest_id"])
        for row in rows
    )
    return {
        "run_id": run_id,
        "environment_id": Path(frozen["records"]["reference"][0]["input"]).parent.name,
        "query_masks": len(rows),
        "shared_query_masks": shared_count,
        "multiview_nearest": {"accepted": len(rows), "correct": nearest_correct},
        "multiview_nearest_shared": {"accepted": shared_count, "correct": nearest_shared_correct},
        "multiview_exact_budget": {"accepted": budget, "correct": accepted_correct},
        "remind_one_to_one": {
            "accepted": budget,
            "correct": int(remind["correct_cross_scan_links"]),
        },
    }


def rate(pair: dict, denominator: int) -> dict:
    return {
        "accepted": int(pair["accepted"]),
        "correct": int(pair["correct"]),
        "precision": float(pair["correct"] / pair["accepted"]) if pair["accepted"] else None,
        "coverage": float(pair["correct"] / denominator) if denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--remind-root", type=Path, default=DEFAULT_REMIND_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREDICTION_ROOT / "summary.json")
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path
    input_root, prediction_root, remind_root = map(resolve, (args.input_root, args.prediction_root, args.remind_root))
    output = resolve(args.output)
    inference_manifest = json.loads((prediction_root / "run_manifest.json").read_text())
    if inference_manifest.get("status") != "complete":
        raise ValueError("multi-view inference manifest is not complete")
    rows = [read_relation(
        item["run_id"], input_root=input_root, prediction_root=prediction_root, remind_root=remind_root
    ) for item in inference_manifest["results"]]

    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(row["environment_id"], {
            "query_masks": 0, "shared_query_masks": 0,
            "multiview_nearest": {"accepted": 0, "correct": 0},
            "multiview_nearest_shared": {"accepted": 0, "correct": 0},
            "multiview_exact_budget": {"accepted": 0, "correct": 0},
            "remind_one_to_one": {"accepted": 0, "correct": 0},
        })
        group["query_masks"] += row["query_masks"]
        group["shared_query_masks"] += row["shared_query_masks"]
        for arm in ("multiview_nearest", "multiview_nearest_shared", "multiview_exact_budget", "remind_one_to_one"):
            for key in ("accepted", "correct"):
                group[arm][key] += row[arm][key]

    group_rows = []
    for env, group in sorted(groups.items()):
        item = {"environment_id": env, "query_masks": group["query_masks"]}
        for arm in ("multiview_nearest", "multiview_exact_budget", "remind_one_to_one"):
            item[arm] = rate(group[arm], group["query_masks"])
        item["multiview_nearest_shared"] = rate(group["multiview_nearest_shared"], group["shared_query_masks"])
        group_rows.append(item)

    rng = np.random.default_rng(args.seed)
    n_groups = len(group_rows)
    sample = rng.integers(0, n_groups, size=(args.draws, n_groups))
    intervals = {}
    for metric in ("precision", "coverage"):
        for arm in ("multiview_exact_budget", "remind_one_to_one"):
            vals = np.asarray([r[arm][metric] for r in group_rows], dtype=float)
            boot = vals[sample].mean(axis=1)
            intervals[f"macro_{arm}_{metric}"] = {
                "estimate": float(vals.mean()),
                "percentile_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            }
        diff = np.asarray([
            r["multiview_exact_budget"][metric] - r["remind_one_to_one"][metric]
            for r in group_rows
        ])
        boot = diff[sample].mean(axis=1)
        intervals[f"macro_multiview_minus_remind_{metric}"] = {
            "estimate": float(diff.mean()),
            "percentile_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "bonferroni_2_outcomes_percentile": [float(x) for x in np.quantile(boot, [0.0125, 0.9875])],
        }

    pooled = {}
    for arm in ("multiview_nearest", "multiview_nearest_shared", "multiview_exact_budget", "remind_one_to_one"):
        accepted = sum(row[arm]["accepted"] for row in rows)
        correct = sum(row[arm]["correct"] for row in rows)
        denominator = sum(row["shared_query_masks"] for row in rows) if arm == "multiview_nearest_shared" else sum(row["query_masks"] for row in rows)
        pooled[arm] = rate({"accepted": accepted, "correct": correct}, denominator)

    result = {
        "status": "complete",
        "scope": "post-hoc full-reference-view DINO exemplar retrieval on the same explored 3RScan validation masks as REMIND; not independent-site evidence",
        "relations": rows,
        "reference_groups": group_rows,
        "pooled": pooled,
        "macro_reference_group_bootstrap": intervals,
        "bootstrap": {"draws": args.draws, "seed": args.seed, "unit": "released reference scan ID"},
        "limits": [
            "The multi-view arm pools every annotated reference mask as an independent exemplar; it has no object-track memory, contextual association, or cross-frame query updates.",
            "The exact accepted count is borrowed from REMIND per relation after inference and selected by target-free DINO cosine score; it is a diagnostic operating point, not a calibrated policy.",
            "All masks are annotation-derived and privileged. This remains same-dataset validation with previously explored relations.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "relations": len(rows), "groups": n_groups,
                      "intervals": intervals}, indent=2))


if __name__ == "__main__":
    main()
