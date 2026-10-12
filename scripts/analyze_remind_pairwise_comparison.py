#!/usr/bin/env python3
"""Compare pairwise DINO matching with REMIND tracking on identical masks.

The comparison is limited to the eight frozen 3RScan relations for which both
anonymous prediction sets exist. It scores source-linked query-mask coverage
and precision; tracker IDs use the frozen run's source-only Hungarian mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRWISE = ROOT / "outputs/reports/20261009_unseen_oasis_equal_budget"
DEFAULT_TRACKER = ROOT / "outputs/reports/20261109_remind_style_3rscan_mapping_final"
DEFAULT_OUTPUT = ROOT / "outputs/reports/20261109_remind_pairwise_comparison"


def read_relation(pairwise_root: Path, tracker_root: Path, run_id: str) -> dict:
    tracker_path = tracker_root / f"{run_id}.json"
    tracker = json.loads(tracker_path.read_text())
    population_path = pairwise_root / "uniform20_inputs" / run_id / "manifest.json"
    population = json.loads(population_path.read_text())
    inference_path = pairwise_root / "dino_segment_predictions" / run_id / "inference_summary.json"
    inference = json.loads(inference_path.read_text())
    if tracker.get("status") != "complete" or inference.get("status") != "complete":
        raise ValueError(f"incomplete predictions for {run_id}")
    if inference.get("identity_targets_read") or inference.get("semantic_labels_read"):
        raise ValueError(f"DINO inference summary indicates labels were read for {run_id}")

    reference_records = population["records"]["reference"]
    query_records = population["records"]["rescan"]
    environment_id = Path(reference_records[0]["input"]).parent.name
    reference_targets = {
        int(record["frame_id"]): np.load(ROOT / record["identity_targets"])
        for record in reference_records
    }
    query_targets = {
        int(record["frame_id"]): np.load(ROOT / record["identity_targets"])
        for record in query_records
    }
    query_rows = tracker["query_mask_rows"]
    row_cursor = 0
    query_count = dino_accepts = dino_correct = 0
    tracker_accepts = tracker_correct = 0
    tracker_majority_accepts = tracker_majority_correct = 0
    dino_mutual_candidates: list[tuple[float, int, bool]] = []
    dino_correct_flags: list[bool] = []
    tracker_correct_flags: list[bool] = []
    tracker_majority_correct_flags: list[bool] = []

    if len(inference["records"]) != len(query_records):
        raise ValueError(f"DINO query-frame count differs from frozen masks for {run_id}")
    for prediction_record, query_record in zip(inference["records"], query_records):
        query_frame = int(query_record["frame_id"])
        if int(prediction_record["query_frame"]) != query_frame:
            raise ValueError(f"DINO query-frame order mismatch for {run_id}")
        query_ids = query_targets[query_frame]
        reference_ids = reference_targets[int(prediction_record["reference_frame"])]
        prediction_path = ROOT / prediction_record["predictions"]
        with np.load(prediction_path) as prediction:
            assigned = prediction["mutual_assigned"].astype(bool)
            links = prediction["query_to_reference"].astype(np.int64)
            scores = prediction["scores"]
        if len(query_ids) != len(assigned) or len(query_ids) != len(links):
            raise ValueError(f"DINO query-mask count mismatch for {run_id}/{query_frame}")
        if row_cursor + len(query_ids) > len(query_rows):
            raise ValueError(f"tracker query-mask rows ended early for {run_id}")

        for index, query_id_value in enumerate(query_ids.tolist()):
            tracker_row = query_rows[row_cursor]
            row_cursor += 1
            query_id = int(query_id_value)
            if int(tracker_row["gt_id"]) != query_id:
                raise ValueError(f"tracker/query identity order mismatch for {run_id} at row {row_cursor - 1}")
            query_count += 1

            dino_correct_link = bool(
                assigned[index]
                and query_id > 0
                and int(reference_ids[int(links[index])]) == query_id
            )
            if assigned[index]:
                dino_mutual_candidates.append((
                    float(scores[index, int(links[index])]),
                    query_count - 1,
                    dino_correct_link,
                ))
            dino_accepts += int(assigned[index])
            dino_correct += int(dino_correct_link)
            dino_correct_flags.append(dino_correct_link)

            track_accept = bool(tracker_row["predicted_source_link"])
            track_correct = bool(tracker_row["correct_source_link"])
            majority_accept = bool(tracker_row["predicted_source_link_majority"])
            majority_correct = bool(tracker_row["correct_source_link_majority"])
            tracker_accepts += int(track_accept)
            tracker_correct += int(track_correct)
            tracker_majority_accepts += int(majority_accept)
            tracker_majority_correct += int(majority_correct)
            tracker_correct_flags.append(track_correct)
            tracker_majority_correct_flags.append(majority_correct)

    if row_cursor != len(query_rows):
        raise ValueError(f"unused tracker query-mask rows remain for {run_id}")
    if tracker_accepts > len(dino_mutual_candidates):
        raise ValueError(f"DINO mutual cannot meet tracker acceptance budget for {run_id}")
    equal_budget_candidates = sorted(dino_mutual_candidates, key=lambda row: (-row[0], row[1]))[:tracker_accepts]
    dino_equal_budget_correct = sum(int(candidate[2]) for candidate in equal_budget_candidates)

    return {
        "run_id": run_id,
        "environment_id": environment_id,
        "query_masks": query_count,
        "dino_mutual": {"accepted": dino_accepts, "correct": dino_correct},
        "dino_equal_budget": {
            "accepted": tracker_accepts,
            "correct": dino_equal_budget_correct,
            "selection": "highest assigned-pair cosine score among mutual-accepted queries; exact per-relation REMIND source-link budget",
        },
        "remind_hungarian": {"accepted": tracker_accepts, "correct": tracker_correct},
        "remind_majority_sensitivity": {
            "accepted": tracker_majority_accepts,
            "correct": tracker_majority_correct,
        },
        "correctness_overlap": {
            "both_correct": int(sum(a and b for a, b in zip(dino_correct_flags, tracker_correct_flags))),
            "dino_only_correct": int(sum(a and not b for a, b in zip(dino_correct_flags, tracker_correct_flags))),
            "remind_only_correct": int(sum(b and not a for a, b in zip(dino_correct_flags, tracker_correct_flags))),
            "neither_correct": int(sum(not a and not b for a, b in zip(dino_correct_flags, tracker_correct_flags))),
            "remind_majority_only_correct": int(sum(b and not a for a, b in zip(dino_correct_flags, tracker_majority_correct_flags))),
        },
    }


def rates(counts: dict, denominator: int) -> dict:
    accepted, correct = counts["accepted"], counts["correct"]
    return {
        "accepted": accepted,
        "correct": correct,
        "precision": correct / accepted if accepted else None,
        "coverage": correct / denominator if denominator else None,
    }


def summarize(relations: list[dict], draws: int, seed: int) -> dict:
    arms = ("dino_mutual", "dino_equal_budget", "remind_hungarian", "remind_majority_sensitivity")
    per_relation = []
    for row in relations:
        item = {
            "run_id": row["run_id"],
            "environment_id": row["environment_id"],
            "query_masks": row["query_masks"],
        }
        for arm in arms:
            item[arm] = rates(row[arm], row["query_masks"])
        item["correctness_overlap"] = row["correctness_overlap"]
        per_relation.append(item)

    environment_groups: dict[str, dict] = {}
    for row in relations:
        env = row["environment_id"]
        group = environment_groups.setdefault(env, {
            "query_masks": 0,
            **{arm: {"accepted": 0, "correct": 0} for arm in arms},
        })
        group["query_masks"] += row["query_masks"]
        for arm in arms:
            group[arm]["accepted"] += row[arm]["accepted"]
            group[arm]["correct"] += row[arm]["correct"]
    environment_rows = []
    for environment_id, row in sorted(environment_groups.items()):
        item = {"environment_id": environment_id, "query_masks": row["query_masks"]}
        for arm in arms:
            item[arm] = rates(row[arm], row["query_masks"])
        environment_rows.append(item)

    rng = np.random.default_rng(seed)
    n_environments = len(environment_rows)
    bootstrap = rng.integers(0, n_environments, size=(draws, n_environments))
    intervals = {}
    for metric in ("precision", "coverage"):
        for arm in arms:
            values = np.asarray([item[arm][metric] for item in environment_rows], dtype=float)
            samples = values[bootstrap].mean(axis=1)
            intervals[f"macro_{arm}_{metric}"] = {
                "estimate": float(values.mean()),
                "percentile_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            }
        difference = np.asarray([
            item["dino_mutual"][metric] - item["remind_hungarian"][metric]
            for item in environment_rows
        ], dtype=float)
        samples = difference[bootstrap].mean(axis=1)
        intervals[f"macro_dino_minus_remind_{metric}"] = {
            "estimate": float(difference.mean()),
            "percentile_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        }
        exact_budget_difference = np.asarray([
            item["dino_equal_budget"][metric] - item["remind_hungarian"][metric]
            for item in environment_rows
        ], dtype=float)
        exact_budget_samples = exact_budget_difference[bootstrap].mean(axis=1)
        intervals[f"macro_dino_equal_budget_minus_remind_{metric}"] = {
            "estimate": float(exact_budget_difference.mean()),
            "percentile_95": [float(x) for x in np.quantile(exact_budget_samples, [0.025, 0.975])],
        }

    pooled = {}
    total_queries = sum(row["query_masks"] for row in relations)
    for arm in arms:
        accepted = sum(row[arm]["accepted"] for row in relations)
        correct = sum(row[arm]["correct"] for row in relations)
        pooled[arm] = rates({"accepted": accepted, "correct": correct}, total_queries)
    overlap_totals = {
        key: sum(row["correctness_overlap"][key] for row in relations)
        for key in relations[0]["correctness_overlap"]
    }
    return {
        "status": "complete",
        "scope": f"post-hoc same-mask comparison on {len(relations)} frozen 3RScan relations aggregated over {n_environments} released reference groups; not independent-dataset validation",
        "query_masks": total_queries,
        "environment_groups": environment_rows,
        "relations": per_relation,
        "macro_bootstrap": intervals,
        "pooled": pooled,
        "correctness_overlap_totals": overlap_totals,
        "oracle_union_coverage": (overlap_totals["both_correct"] + overlap_totals["dino_only_correct"] + overlap_totals["remind_only_correct"]) / total_queries,
        "bootstrap": {"draws": draws, "seed": seed, "unit": "released reference group"},
        "interpretation_limits": [
            "DINO mutual makes one independent frame-pair assignment; REMIND aggregates observations with a temporal memory and contextual association.",
            "The equal-budget DINO arm ranks only mutual-accepted links by their assigned-pair cosine and keeps the exact REMIND source-link count separately within each relation; the budget is diagnostic and is not source-calibrated.",
            "Both use the same DINOv2 ViT-G backbone and frozen masks, but the association protocols and source identity scoring maps differ.",
            "The tracker Hungarian identity map is fitted from source observations for evaluation; the majority-map result is a scoring sensitivity.",
            "The selected relations come from an already explored 3RScan cohort; reference groups do not establish independent physical sites. Bootstrap intervals describe reference-group sensitivity only.",
            "The oracle union is a post-hoc upper bound and is not a deployable combined method.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-root", type=Path, default=DEFAULT_PAIRWISE)
    parser.add_argument("--tracker-root", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7182)
    args = parser.parse_args()

    tracker_manifest = json.loads((args.tracker_root / "run_manifest.json").read_text())
    if tracker_manifest.get("status") != "complete":
        raise ValueError("REMIND-style run manifest is incomplete")
    run_ids = [Path(path).stem for path in tracker_manifest["results"]]
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError(f"expected unique frozen relation outputs, found {len(run_ids)}")
    relations = [read_relation(args.pairwise_root, args.tracker_root, run_id) for run_id in run_ids]
    result = summarize(relations, args.bootstrap_draws, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "query_masks": result["query_masks"],
        "macro_bootstrap": result["macro_bootstrap"],
        "pooled": result["pooled"],
        "correctness_overlap_totals": result["correctness_overlap_totals"],
        "oracle_union_coverage": result["oracle_union_coverage"],
        "output": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
