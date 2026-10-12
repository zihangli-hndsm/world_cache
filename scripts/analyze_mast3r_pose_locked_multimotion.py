#!/usr/bin/env python3
"""Test pose-locked secondary-motion transfer on saved MASt3R top-20 sets.

No MASt3R inference is rerun. Candidate sets are frozen and annotations are
used only to score mapped moved-rigid object queries after geometric fitting.
All cached relations are official-train sources and are evaluated by
leave-one-relation-out selection. The predesignated target is not cached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_3rscan_multimotion_rescue_mechanisms import (
    MARGINS_M,
    MIN_SUPPORTS,
    MODE_COUNTS,
    predict,
    relation_bootstrap,
)
from evaluate_3rscan_multimotion import PairFeatures, assign_modes, fit_fixed_consensus, fit_modes


THRESHOLD_M = 0.10


def read_cache(cache_path: Path, seed: int, metadata_by_ref: dict[str, dict[str, object]]) -> dict[str, object]:
    run = cache_path.parent.name
    extracted = cache_path.parent / "seed_17_extracted"
    with np.load(cache_path) as cache:
        candidate_indices20 = cache["candidate_indices"].copy()
        candidate_count = cache["candidate_count"].copy()
        reference_points = cache["reference_points"].copy()
        query_points_all = cache["query_points"].copy()
        query_frame_ids_all = cache["query_frame_ids"].copy()
        query_object_ids_all = cache["query_object_ids"].copy()
        query_evaluable_all = cache["query_evaluable"].astype(bool)
        reference_object_ids = cache["reference_object_ids"].astype(np.int64)
        reference_id = ""
        rescan_id = ""

    summary_path = extracted / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing MASt3R per-seed summary for {run}")
    pair_run = Path(json.loads(summary_path.read_text(encoding="utf-8"))["pair_run"])
    pair_config = json.loads((pair_run / "config.json").read_text(encoding="utf-8"))
    reference_id = str(pair_config["reference"])
    rescan_id = str(pair_config["rescan"])

    if reference_id not in metadata_by_ref:
        raise ValueError(f"missing metadata relation for MASt3R cache {run}")
    relation = metadata_by_ref[reference_id]
    if str(relation.get("type", "unknown")) != "train":
        raise ValueError(f"expected an official-train source relation for {run}")
    scan = next((item for item in relation["scans"] if item.get("reference") == rescan_id), None)
    if scan is None:
        raise ValueError(f"missing rescan metadata for {run}: {rescan_id}")
    rigid_map = {
        int(item["instance_rescan"]): int(item["instance_reference"])
        for item in scan.get("rigid", [])
        if "instance_rescan" in item and "instance_reference" in item
        and int(item["instance_rescan"]) > 0 and int(item["instance_reference"]) > 0
    }
    moved_all = query_evaluable_all & np.isin(query_object_ids_all, list(rigid_map))
    expected_all = np.asarray([rigid_map.get(int(object_id), -1) for object_id in query_object_ids_all], dtype=np.int64)
    candidate_valid = candidate_count > 0
    moved = moved_all[candidate_valid]
    expected = expected_all[candidate_valid]
    query_points = query_points_all[candidate_valid]
    frame_ids = query_frame_ids_all[candidate_valid]
    candidate_indices = candidate_indices20[candidate_valid]
    candidate_indices50 = np.tile(candidate_indices, (1, 3))[:, :50]
    limits = np.full(len(query_points), 20, dtype=np.int64)

    features = PairFeatures(
        pair=run,
        reference_id=reference_id,
        rescan_id=rescan_id,
        pair_run=Path("."),
        reference_points=reference_points,
        query_points=query_points,
        query_frame_ids=frame_ids,
        candidate_indices=candidate_indices50,
        candidate_scores=np.tile(np.arange(50, 0, -1, dtype=np.float64), (len(query_points), 1)),
        top20_reference_points=reference_points[candidate_indices],
        top50_reference_points=reference_points[candidate_indices50],
        gap20=np.zeros(len(query_points), dtype=np.float64),
    )
    baseline = fit_fixed_consensus(
        query_points, reference_points, candidate_indices,
        threshold=THRESHOLD_M, iterations=1500, seed=seed,
    )

    diagnostics_path = extracted / "query_diagnostics.csv"
    if seed == 17 and diagnostics_path.is_file():
        diagnostics = pd.read_csv(diagnostics_path).sort_values("query_index").reset_index(drop=True)
        expected_all = np.flatnonzero(candidate_valid)
        diagnostics = diagnostics.iloc[expected_all].reset_index(drop=True)
        if len(diagnostics) != len(query_points):
            raise ValueError(f"MASt3R diagnostics/cache query count mismatch for {run}")
        _, selected, inliers = __import__(
            "evaluate_3rscan_spatial_consensus", fromlist=["consensus"]
        ).consensus(
            query_points, reference_points, candidate_indices,
            threshold=THRESHOLD_M, iterations=1500, seed=seed,
        )
        observed_object = reference_object_ids[selected]
        expected_object = diagnostics.selected_reference_object_id.to_numpy(dtype=np.int64)
        if not np.array_equal(observed_object, expected_object):
            raise ValueError(f"reconstructed MASt3R global assignments differ for {run}")
        if not np.array_equal(inliers, diagnostics.consensus_inlier.astype(bool).to_numpy()):
            raise ValueError(f"reconstructed MASt3R global inliers differ for {run}")
        if not np.allclose(baseline.residuals, diagnostics.spatial_residual_m, rtol=0.0, atol=1e-6):
            raise ValueError(f"reconstructed MASt3R residuals differ for {run}")

    modes = fit_modes(
        features, limits, threshold=THRESHOLD_M,
        iterations=1500, secondary_iterations=1000, seed=seed,
        max_modes=4, min_support=8, fixed_baseline=baseline,
    )
    _, _, _, _, selected_matrix, residual_matrix = assign_modes(
        features, limits, modes, THRESHOLD_M
    )
    base_residual = residual_matrix[:, 0].astype(np.float64)
    base_object = reference_object_ids[selected_matrix[:, 0]].astype(np.int64)
    baseline_object = reference_object_ids[baseline.selected].astype(np.int64)
    baseline_correct = moved & (baseline.residuals <= THRESHOLD_M) & (baseline_object == expected)

    secondary_residuals: list[np.ndarray] = []
    secondary_objects: list[np.ndarray] = []
    secondary_support: list[int] = []
    previous_unexplained = base_residual > THRESHOLD_M
    for mode in range(1, 4):
        if mode >= residual_matrix.shape[1]:
            secondary_residuals.append(np.full(len(base_residual), np.inf))
            secondary_objects.append(np.full(len(base_residual), -1, dtype=np.int64))
            secondary_support.append(0)
            continue
        residual = residual_matrix[:, mode].astype(np.float64)
        objects = reference_object_ids[selected_matrix[:, mode]].astype(np.int64)
        inlier = np.isfinite(residual) & (residual <= THRESHOLD_M)
        secondary_support.append(int((inlier & previous_unexplained).sum()))
        secondary_residuals.append(residual)
        secondary_objects.append(objects)
        previous_unexplained &= ~inlier

    candidate_objects = reference_object_ids[candidate_indices]
    candidate_hit = np.any(candidate_objects == expected[:, None], axis=1)
    return {
        "pair": run,
        "features": features,
        "limits": limits,
        "moved_all": moved_all,
        "candidate_valid": candidate_valid,
        "moved": moved,
        "expected": expected,
        "candidate_hit": candidate_hit,
        "base_residual": base_residual,
        "base_object": base_object,
        "baseline_correct": baseline_correct,
        "baseline_prediction": np.where(baseline.residuals <= THRESHOLD_M, baseline_object, -1),
        "secondary_residuals": secondary_residuals,
        "secondary_objects": secondary_objects,
        "secondary_independent_support": secondary_support,
        "reference_points": reference_points,
        "reference_object_ids": reference_object_ids,
        "query_points": query_points,
        "query_frame_ids": frame_ids,
        "original_query_indices": np.flatnonzero(candidate_valid),
    }


def select_source_only(data_by_pair: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, dict[str, tuple[int, float, int]]]:
    pairs = sorted(data_by_pair)
    selections: list[dict[str, object]] = []
    settings: dict[str, tuple[int, float, int]] = {}
    configs = [(m, g, s) for m in MODE_COUNTS for g in MARGINS_M for s in MIN_SUPPORTS]
    for heldout in pairs:
        training = [pair for pair in pairs if pair != heldout]
        scores = {
            config: sum(int(predict(data_by_pair[pair], *config)[0].sum()) for pair in training)
            for config in configs
        }
        best = max(scores.values())
        chosen = min(
            (config for config, score in scores.items() if score == best),
            key=lambda config: (config[0], -config[2], -config[1]),
        )
        settings[heldout] = chosen
        correct, mode, _, _ = predict(data_by_pair[heldout], *chosen)
        data = data_by_pair[heldout]
        selections.append({
            "pair": heldout, "modes": chosen[0], "margin_m": chosen[1],
            "min_independent_support": chosen[2], "training_relations": len(training),
            "training_correct_transfers": best,
            "moved_queries": int(data["moved_all"].sum()),
            "candidate_top20_hits": int((data["candidate_hit"] & data["moved"]).sum()),
            "baseline_correct": int(data["baseline_correct"].sum()),
            "method_correct": int(correct.sum()),
            "rescued": int((correct & ~data["baseline_correct"]).sum()),
            "lost": int((data["baseline_correct"] & ~correct).sum()),
            "selected_secondary_queries": int((mode > 0).sum()),
        })
    return pd.DataFrame(selections), settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    by_ref = {str(row["reference"]): row for row in metadata}
    cache_paths = sorted(args.cache_root.glob("*/candidate_cache.npz"))
    if not cache_paths:
        raise ValueError(f"no MASt3R candidate caches under {args.cache_root}")
    all_data = {
        path.parent.name: read_cache(path, args.seed, by_ref)
        for path in cache_paths
    }
    source_data = all_data
    if len(source_data) < 3:
        raise ValueError("source-only LOPO needs at least three relations")
    selections, source_settings = select_source_only(source_data)
    source_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    for pair, chosen in source_settings.items():
        data = source_data[pair]
        correct, mode, _, prediction = predict(data, *chosen)
        baseline = np.asarray(data["baseline_correct"], dtype=bool)
        moved = np.asarray(data["moved"], dtype=bool)
        source_rows.append({
            "pair": pair,
            "modes": chosen[0], "margin_m": chosen[1], "min_independent_support": chosen[2],
            "moved_queries": int(data["moved_all"].sum()),
            "candidate_top20_hits": int((data["candidate_hit"] & moved).sum()),
            "baseline_correct": int(baseline.sum()), "method_correct": int(correct.sum()),
            "rescued": int((correct & ~baseline).sum()), "lost": int((baseline & ~correct).sum()),
            "baseline_coverage": float(baseline.sum() / data["moved_all"].sum()) if data["moved_all"].sum() else np.nan,
            "method_coverage": float(correct.sum() / data["moved_all"].sum()) if data["moved_all"].sum() else np.nan,
        })
        for local_index, original_index in enumerate(data["original_query_indices"]):
            if not moved[local_index]:
                continue
            query_rows.append({
                "pair": pair, "query_index": int(original_index),
                "frame_id": int(data["query_frame_ids"][local_index]),
                "candidate_top20_hit": bool(data["candidate_hit"][local_index]),
                "baseline_correct": bool(baseline[local_index]),
                "method_correct": bool(correct[local_index]),
                "selected_mode": int(mode[local_index]),
                "rescued": bool(correct[local_index] and not baseline[local_index]),
                "lost": bool(baseline[local_index] and not correct[local_index]),
            })
    source_outcomes = pd.DataFrame(source_rows)
    bootstrap = relation_bootstrap(source_outcomes.assign(
        pair=source_outcomes.pair, moved_queries=source_outcomes.moved_queries
    ), seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections.to_csv(args.output_dir / "source_lopo_selection.csv", index=False)
    source_outcomes.to_csv(args.output_dir / "source_lopo_outcomes.csv", index=False)
    pd.DataFrame(query_rows).to_csv(args.output_dir / "source_query_outcomes.csv", index=False)
    bootstrap.to_csv(args.output_dir / "source_relation_bootstrap_ci.csv", index=False)
    denominator = int(source_outcomes.moved_queries.sum())
    source_base = int(source_outcomes.baseline_correct.sum())
    source_method = int(source_outcomes.method_correct.sum())
    summary = {
        "seed": args.seed,
        "descriptor": "frozen MASt3R dense confidence-ranked top-20 candidates",
        "source_relations": len(source_outcomes),
        "source_moved_rigid_queries": denominator,
        "source_candidate_top20_recall": float(source_outcomes.candidate_top20_hits.sum() / denominator),
        "source_baseline_correct": source_base,
        "source_method_correct": source_method,
        "source_gain_pp": float(100 * (source_method - source_base) / denominator),
        "source_rescued": int(source_outcomes.rescued.sum()),
        "source_lost": int(source_outcomes.lost.sum()),
        "source_nonnegative_relations": int((source_outcomes.method_correct >= source_outcomes.baseline_correct).sum()),
        "source_weighted_relation_bootstrap_ci95_pp": bootstrap.iloc[0][["ci95_lower", "ci95_upper"]].astype(float).tolist(),
        "source_macro_relation_bootstrap_ci95_pp": bootstrap.iloc[1][["ci95_lower", "ci95_upper"]].astype(float).tolist(),
        "selector_grid": {"modes": list(MODE_COUNTS), "residual_margin_m": list(MARGINS_M), "independent_support": list(MIN_SUPPORTS)},
        "selection": "leave-one-source-relation-out; each held-out source is excluded from its own selector fit",
        "candidate_missing_queries_remain_in_denominator": True,
        "target_labels_used_for_inference_or_selection": False,
        "predesignated_target_available": False,
        "baseline_seed17_reconstruction_checked_against_saved_diagnostics": args.seed == 17,
        "limits": [
            "official-train relations only; no independent official validation evaluation",
            "MASt3R candidate caches contain top-20 only; this tests secondary motion conditional on those frozen candidates",
            "physical-site overlap across reference relations has not been independently verified",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lo, hi = summary["source_weighted_relation_bootstrap_ci95_pp"]
    lines = [
        "# Pose-locked multi-motion on frozen MASt3R candidates", "",
        f"On {denominator:,} moved-rigid queries from {len(source_outcomes)} official-train relations, MASt3R top-20 candidates yield {source_base:,} correct transfers with global RANSAC and {source_method:,} after leave-one-source-relation-out selected secondary-motion switching ({summary['source_gain_pp']:+.2f} pp; relation-bootstrap 95% CI [{lo:+.2f},{hi:+.2f}] pp). Candidate Recall@20 is {100*summary['source_candidate_top20_recall']:.1f}%.",
        "", f"Source-only rescues/losses: {summary['source_rescued']}/{summary['source_lost']}; nonnegative relations: {summary['source_nonnegative_relations']}/{len(source_outcomes)}.",
        "", "| Source relation | Queries | Baseline | Method | Gain (pp) | Modes | Margin | Support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_outcomes.itertuples(index=False):
        delta_pp = 100 * (row.method_correct - row.baseline_correct) / row.moved_queries if row.moved_queries else 0.0
        lines.append(f"| {row.pair} | {row.moved_queries} | {row.baseline_correct} | {row.method_correct} | {delta_pp:+.2f} | {row.modes} | {row.margin_m:.2f} | {row.min_independent_support} |")
    diagnostics_note = (
        "For seed 17, reconstructed global assignments, inliers, and residuals are checked against the saved MASt3R diagnostics before secondary-mode fitting."
        if args.seed == 17 else
        "Seeds 29 and 41 rerun RANSAC from the same frozen candidate sets; saved seed-17 diagnostics are not used as their baseline output."
    )
    lines += ["", "All 14 cached relations are official-train sources. The predesignated held-out target is not present in the MASt3R cache set, so this source-only LOPO audit provides no separate target result and is not external validation. " + diagnostics_note]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
