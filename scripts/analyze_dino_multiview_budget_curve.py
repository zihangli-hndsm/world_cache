#!/usr/bin/env python3
"""Source-view count curve for all-view DINO exemplar retrieval.

For every query frame, the rank-matched reference frame is always included.
Additional reference frames are sampled uniformly without replacement. The
score-only analysis uses the exact per-relation REMIND accept count, borrowed
after inference, at every bank size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "outputs/reports/20261008_segmast3r_validation"
DEFAULT_FEATURE_ROOT = ROOT / "outputs/reports/20261009_dino_multiview_source_bank"
DEFAULT_REMIND_ROOT = ROOT / "outputs/reports/20261109_remind_style_3rscan_validation"
DEFAULT_OUTPUT = DEFAULT_FEATURE_ROOT / "view_budget_curve.json"
BANK_SIZES = (1, 2, 4, 8, 12, 20)


def load_ids(records: list[dict]) -> dict[int, np.ndarray]:
    out = {}
    for record in records:
        path = Path(record["identity_targets"])
        if not path.is_absolute():
            path = ROOT / path
        ids = np.load(path).astype(np.int64)
        if len(ids) != int(record["masks"]):
            raise ValueError(f"target/mask count differs for {path}")
        out[int(record["frame_id"])] = ids
    return out


def relation_repeats(run_id: str, *, input_root: Path, feature_root: Path,
                     remind_root: Path, repeats: int, seed: int) -> dict:
    frozen_path = input_root / "uniform20_inputs" / run_id / "manifest.json"
    frozen = json.loads(frozen_path.read_text())
    feature_path = feature_root / f"{run_id}_features.npz"
    remind = json.loads((remind_root / f"{run_id}.json").read_text())
    paired_path = input_root / "dino_segment_predictions" / run_id / "inference_summary.json"
    paired = json.loads(paired_path.read_text())
    if remind.get("status") != "complete":
        raise ValueError(f"REMIND output incomplete for {run_id}")
    with np.load(feature_path) as data:
        source_features = data["source_embeddings"].astype(np.float32)
        source_frame_ids = data["source_frame_ids"].astype(np.int32)
        query_features = data["query_embeddings"].astype(np.float32)
        query_frame_ids = data["query_frame_ids"].astype(np.int32)
    source_norms = np.linalg.norm(source_features, axis=1, keepdims=True).clip(1e-8)
    query_norms = np.linalg.norm(query_features, axis=1, keepdims=True).clip(1e-8)
    source_features /= source_norms
    query_features /= query_norms

    ref_records = frozen["records"]["reference"]
    query_records = frozen["records"]["rescan"]
    paired_reference_by_query = {
        int(record["query_frame"]): int(record["reference_frame"])
        for record in paired["records"]
    }
    if set(paired_reference_by_query) != {int(record["frame_id"]) for record in query_records}:
        raise ValueError(f"paired-frame map does not cover frozen query frames for {run_id}")
    reference_targets = load_ids(ref_records)
    query_targets = load_ids(query_records)
    if sum(len(x) for x in reference_targets.values()) != len(source_features):
        raise ValueError(f"source feature/target count mismatch for {run_id}")
    if sum(len(x) for x in query_targets.values()) != len(query_features):
        raise ValueError(f"query feature/target count mismatch for {run_id}")

    source_ids = np.concatenate([reference_targets[int(r["frame_id"])] for r in ref_records])
    query_id_by_key = {
        (int(frame_id), int(index)): int(target)
        for frame_id, ids in query_targets.items()
        for index, target in enumerate(ids.tolist())
    }
    if len(query_id_by_key) != len(query_features):
        raise ValueError(f"query ID index count mismatch for {run_id}")
    query_keys = []
    for record in query_records:
        frame_id = int(record["frame_id"])
        query_keys.extend((frame_id, index) for index in range(int(record["masks"])))
    if len(query_keys) != len(query_features):
        raise ValueError(f"query frame order/feature count mismatch for {run_id}")

    source_index_by_frame = {
        int(frame_id): np.flatnonzero(source_frame_ids == int(frame_id))
        for frame_id in [int(r["frame_id"]) for r in ref_records]
    }
    query_index_by_frame = {
        int(frame_id): np.flatnonzero(query_frame_ids == int(frame_id))
        for frame_id in [int(r["frame_id"]) for r in query_records]
    }
    if any(len(query_index_by_frame[int(r["frame_id"])]) != int(r["masks"]) for r in query_records):
        raise ValueError(f"query embeddings do not align to frozen frame masks for {run_id}")

    budget = int(remind["query_links_to_source_tracks"])
    query_count = len(query_features)
    run_rows = []
    for bank_size in BANK_SIZES:
        n_repeats = 1 if bank_size == len(ref_records) else repeats
        for repeat in range(n_repeats):
            rng = np.random.default_rng(seed + bank_size * 1_000_003 + repeat * 9_176 + sum(run_id.encode()))
            candidate_rows = []
            for query_record in query_records:
                query_frame = int(query_record["frame_id"])
                paired_reference = paired_reference_by_query[query_frame]
                other_frames = [
                    int(record["frame_id"]) for record in ref_records
                    if int(record["frame_id"]) != paired_reference
                ]
                if bank_size - 1 > len(other_frames):
                    raise ValueError(f"bank size exceeds reference frames for {run_id}")
                additional = rng.choice(other_frames, size=bank_size - 1, replace=False).tolist() if bank_size > 1 else []
                selected_frames = {paired_reference, *(int(x) for x in additional)}
                source_idx = np.concatenate([
                    source_index_by_frame[frame_id] for frame_id in selected_frames
                ]) if selected_frames else np.zeros((0,), dtype=np.int64)
                query_idx = query_index_by_frame[query_frame]
                if not len(source_idx) or not len(query_idx):
                    continue
                sims = query_features[query_idx] @ source_features[source_idx].T
                rr, cc = linear_sum_assignment(-sims)
                for q_local, s_local in zip(rr.tolist(), cc.tolist()):
                    q_global = int(query_idx[q_local])
                    s_global = int(source_idx[s_local])
                    candidate_rows.append({
                        "query_global": q_global,
                        "query_id": query_id_by_key[query_keys[q_global]],
                        "source_id": int(source_ids[s_global]),
                        "score": float(sims[q_local, s_local]),
                    })
            if len(candidate_rows) < budget:
                raise ValueError(f"bank size {bank_size} has only {len(candidate_rows)} assignments < budget {budget} in {run_id}")
            selected = sorted(candidate_rows, key=lambda x: (-x["score"], x["query_global"]))[:budget]
            correct = sum(int(x["query_id"] > 0 and x["query_id"] == x["source_id"]) for x in selected)
            run_rows.append({
                "run_id": run_id,
                "environment_id": Path(ref_records[0]["input"]).parent.name,
                "bank_size": bank_size,
                "repeat": repeat,
                "query_masks": query_count,
                "accepted": budget,
                "correct": int(correct),
            })
    return run_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--remind-root", type=Path, default=DEFAULT_REMIND_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else ROOT / path
    input_root, feature_root, remind_root, output = map(
        resolve, (args.input_root, args.feature_root, args.remind_root, args.output)
    )
    inference_manifest = json.loads((feature_root / "run_manifest.json").read_text())
    if inference_manifest.get("status") != "complete":
        raise ValueError("all-reference-view inference is not complete")
    raw_rows = []
    for index, item in enumerate(inference_manifest["results"], start=1):
        raw_rows.extend(relation_repeats(
            item["run_id"], input_root=input_root, feature_root=feature_root,
            remind_root=remind_root, repeats=args.repeats, seed=args.seed,
        ))
        if index % 10 == 0:
            print(f"scored {index}/{len(inference_manifest['results'])} relations", flush=True)

    group_rows: dict[tuple[int, int, str], dict] = {}
    for row in raw_rows:
        key = (row["bank_size"], row["repeat"], row["environment_id"])
        group = group_rows.setdefault(key, {"query_masks": 0, "accepted": 0, "correct": 0})
        group["query_masks"] += row["query_masks"]
        group["accepted"] += row["accepted"]
        group["correct"] += row["correct"]

    allview_summary = json.loads((feature_root / "summary.json").read_text())
    tracker_by_group = {
        row["environment_id"]: row["remind_one_to_one"]
        for row in allview_summary["reference_groups"]
    }

    summaries = []
    rng = np.random.default_rng(args.seed + 71)
    for bank_size in BANK_SIZES:
        n_repeats = 1 if bank_size == 20 else args.repeats
        by_repeat = []
        for repeat in range(n_repeats):
            envs = sorted({env for size, rep, env in group_rows if size == bank_size and rep == repeat})
            if len(envs) != len(tracker_by_group):
                raise ValueError(
                    f"expected {len(tracker_by_group)} groups for bank size={bank_size}, repeat={repeat}; got {len(envs)}"
                )
            precision = np.asarray([
                group_rows[(bank_size, repeat, env)]["correct"] / group_rows[(bank_size, repeat, env)]["accepted"]
                for env in envs
            ])
            coverage = np.asarray([
                group_rows[(bank_size, repeat, env)]["correct"] / group_rows[(bank_size, repeat, env)]["query_masks"]
                for env in envs
            ])
            by_repeat.append({"precision": float(precision.mean()), "coverage": float(coverage.mean())})
        # For K<20, the point estimate averages independent label-free view
        # subsets. Bootstrap reference groups over repeat-averaged group rates.
        envs = sorted({env for size, _, env in group_rows if size == bank_size})
        group_precision = np.asarray([
            np.mean([
                group_rows[(bank_size, repeat, env)]["correct"] /
                group_rows[(bank_size, repeat, env)]["accepted"]
                for repeat in range(n_repeats)
            ]) for env in envs
        ])
        group_coverage = np.asarray([
            np.mean([
                group_rows[(bank_size, repeat, env)]["correct"] /
                group_rows[(bank_size, repeat, env)]["query_masks"]
                for repeat in range(n_repeats)
            ]) for env in envs
        ])
        idx = rng.integers(0, len(envs), size=(args.draws, len(envs)))
        tracker_precision = np.asarray([
            tracker_by_group[env]["precision"] for env in envs
        ])
        tracker_coverage = np.asarray([
            tracker_by_group[env]["coverage"] for env in envs
        ])
        diff_precision = group_precision - tracker_precision
        diff_coverage = group_coverage - tracker_coverage
        diff_precision_boot = diff_precision[idx].mean(axis=1)
        diff_coverage_boot = diff_coverage[idx].mean(axis=1)
        family_tail = 0.05 / (2 * len(BANK_SIZES) * 2)
        summaries.append({
            "reference_views_per_query": bank_size,
            "repeats": n_repeats,
            "macro_precision": float(group_precision.mean()),
            "macro_precision_reference_group_bootstrap_95": [
                float(x) for x in np.quantile(group_precision[idx].mean(axis=1), [0.025, 0.975])
            ],
            "macro_coverage": float(group_coverage.mean()),
            "macro_coverage_reference_group_bootstrap_95": [
                float(x) for x in np.quantile(group_coverage[idx].mean(axis=1), [0.025, 0.975])
            ],
            "view_subset_repeat_range_precision": [
                min(r["precision"] for r in by_repeat), max(r["precision"] for r in by_repeat)
            ],
            "view_subset_repeat_range_coverage": [
                min(r["coverage"] for r in by_repeat), max(r["coverage"] for r in by_repeat)
            ],
            "paired_difference_vs_remind": {
                "precision": {
                    "estimate": float(diff_precision.mean()),
                    "reference_group_bootstrap_95": [
                        float(x) for x in np.quantile(diff_precision_boot, [0.025, 0.975])
                    ],
                    "bonferroni_12_effect_family_interval": [
                        float(x) for x in np.quantile(diff_precision_boot, [family_tail, 1 - family_tail])
                    ],
                    "groups_higher_lower_tied": [
                        int(np.sum(diff_precision > 1e-12)),
                        int(np.sum(diff_precision < -1e-12)),
                        int(np.sum(np.abs(diff_precision) <= 1e-12)),
                    ],
                },
                "coverage": {
                    "estimate": float(diff_coverage.mean()),
                    "reference_group_bootstrap_95": [
                        float(x) for x in np.quantile(diff_coverage_boot, [0.025, 0.975])
                    ],
                    "bonferroni_12_effect_family_interval": [
                        float(x) for x in np.quantile(diff_coverage_boot, [family_tail, 1 - family_tail])
                    ],
                    "groups_higher_lower_tied": [
                        int(np.sum(diff_coverage > 1e-12)),
                        int(np.sum(diff_coverage < -1e-12)),
                        int(np.sum(np.abs(diff_coverage) <= 1e-12)),
                    ],
                },
            },
        })

    result = {
        "status": "complete",
        "protocol": "rank-matched reference frame always included; remaining K-1 of 19 source views sampled without replacement independently per query frame; per-frame Hungarian; exact per-relation REMIND acceptance count selected by cosine",
        "cohort": {
            "relations": len(inference_manifest["results"]),
            "reference_groups": len(tracker_by_group),
        },
        "bootstrap": {"draws": args.draws, "seed": args.seed, "unit": "reference scan ID", "view_subset_repeats": args.repeats},
        "bank_size_results": summaries,
        "relation_repeat_rows": raw_rows,
        "limits": [
            "Post-hoc on the explored official 3RScan validation population; group IDs do not establish independent physical sites.",
            "Frozen annotation-derived masks are privileged inputs; acceptance counts are borrowed from the REMIND adapter after inference.",
            "Intervals resample reference groups; repeat ranges summarize sampled view-subset sensitivity and are not a preregistered uncertainty analysis.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "results": summaries}, indent=2))


if __name__ == "__main__":
    main()
