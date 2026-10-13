"""LOPO analysis of pose-locked multi-motion correspondence mechanisms.

All retrieval and rigid hypotheses are loaded from the fixed 2026-09-28
descriptor cache. Annotation IDs are used only to stratify/evaluate candidate
availability and transfer outcomes, never to choose a hypothesis at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_3rscan_moved_instance_candidates import validate_candidate_query_alignment
from evaluate_3rscan_instance_correspondence import read_annotated_mesh
from evaluate_3rscan_multimotion import PairFeatures, assign_modes, fit_fixed_consensus, fit_modes


THRESHOLD_M = 0.10
MODE_COUNTS = (1, 2, 3, 4)
MARGINS_M = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15)
MIN_SUPPORTS = (0, 8, 16, 32, 64, 128)


def read_pair(
    output_dir: Path,
    pair: str,
    annotation_root: Path,
    seed: int = 17,
    method_arm: str = "combined",
) -> dict[str, object]:
    if method_arm not in {"combined", "multi_motion"}:
        raise ValueError(f"unsupported fixed-candidate motion arm: {method_arm}")
    diagnostics = pd.read_csv(output_dir / "query_diagnostics.csv")
    base = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == "global_ransac")]
    combined = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == method_arm)]
    base = base.sort_values("query_index").reset_index(drop=True)
    combined = combined.sort_values("query_index").reset_index(drop=True)
    if base.empty or combined.empty:
        raise ValueError(f"missing fixed-query diagnostics for {pair}")
    if not np.array_equal(base.query_index.to_numpy(), combined.query_index.to_numpy()):
        raise ValueError(f"query index mismatch for {pair}")
    if not np.array_equal(base.frame_id.to_numpy(), combined.frame_id.to_numpy()):
        raise ValueError(f"query frame mismatch for {pair}")

    cache_path = output_dir / "descriptor_candidates" / f"{pair}.npz"
    with np.load(cache_path) as cache:
        if str(cache["pair"].item()) != pair:
            raise ValueError(f"candidate cache pair mismatch: {cache_path}")
        candidate_indices = cache["candidate_indices"].copy()
        candidate_points = cache["reference_points"].copy()
        query_points = cache["query_points"].copy()
        query_frame_ids = cache["query_frame_ids"].copy()
        candidate_scores = cache["candidate_scores"].copy()
        gap20 = cache["gap20"].copy()
        reference_id = str(cache["reference_id"].item())
        rescan_id = str(cache["rescan_id"].item())
    validate_candidate_query_alignment(
        base.query_index.to_numpy(), base.frame_id.to_numpy(), candidate_indices, query_frame_ids
    )

    tree, reference_object_ids, _, _, _ = read_annotated_mesh(annotation_root, reference_id)
    _, nearest = tree.query(candidate_points.astype(np.float32), k=1)
    patch_object_ids = reference_object_ids[nearest]
    expected = base.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
    candidate_rank = np.full(len(base), 51, dtype=np.int64)
    has_expected = expected > 0
    for q in np.flatnonzero(has_expected):
        match = np.flatnonzero(patch_object_ids[candidate_indices[q]] == expected[q])
        if len(match):
            candidate_rank[q] = int(match[0]) + 1

    baseline = fit_fixed_consensus(
        query_points,
        candidate_points,
        candidate_indices[:, :20],
        threshold=THRESHOLD_M,
        iterations=1500,
        seed=seed,
    )
    if seed == 17:
        saved_selected = base.selected_reference_index.to_numpy(dtype=np.int64)
        if not np.array_equal(baseline.selected, saved_selected):
            raise ValueError(f"reconstructed global RANSAC assignments differ from saved baseline for {pair}")
        saved_residual = base.spatial_residual_m.to_numpy(dtype=np.float64)
        if not np.allclose(baseline.residuals, saved_residual, rtol=0.0, atol=1e-6):
            raise ValueError(f"reconstructed global RANSAC residuals differ from saved baseline for {pair}")
        saved_inliers = base.consensus_inlier.astype(bool).to_numpy()
        if not np.array_equal(baseline.inliers, saved_inliers):
            raise ValueError(f"reconstructed global RANSAC inliers differ from saved baseline for {pair}")

    limits = combined.candidate_limit.to_numpy(dtype=np.int64)
    if len(limits) != len(query_points) or np.any((limits < 20) | (limits > 50)):
        raise ValueError(f"invalid saved adaptive candidate limits for {pair}")
    features = PairFeatures(
        pair=pair,
        reference_id=reference_id,
        rescan_id=rescan_id,
        pair_run=Path("."),
        reference_points=candidate_points,
        query_points=query_points,
        query_frame_ids=query_frame_ids,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        top20_reference_points=candidate_points[candidate_indices[:, :20]],
        top50_reference_points=candidate_points[candidate_indices],
        gap20=gap20,
    )
    modes = fit_modes(
        features,
        limits,
        threshold=THRESHOLD_M,
        iterations=1500,
        secondary_iterations=1000,
        seed=seed,
        max_modes=4,
        min_support=8,
        fixed_baseline=baseline,
    )
    _, _, _, _, selected_matrix, residual_matrix = assign_modes(features, limits, modes, THRESHOLD_M)
    # The method's scene transform is top-20, while its assignment step uses
    # the frozen adaptive K=20..50 candidate limit. Score the comparison arm
    # separately from the exact top-20 RANSAC assignments.
    base_residual = residual_matrix[:, 0].astype(np.float64)
    base_object = reference_object_ids[nearest[selected_matrix[:, 0]]].astype(np.int64)
    moved = base.query_evaluable.astype(bool).to_numpy() & has_expected
    baseline_residual = baseline.residuals.astype(np.float64)
    baseline_object = reference_object_ids[nearest[baseline.selected]].astype(np.int64)
    # Score the baseline using the same seed-specific global hypothesis that
    # anchors the secondary modes. The saved query identities and labels are
    # still valid, but its assignments were generated with seed 17.
    baseline_correct = moved & (baseline_residual <= THRESHOLD_M) & (baseline_object == expected)
    baseline_prediction = np.where(baseline_residual <= THRESHOLD_M, baseline_object, -1)

    secondary_residuals: list[np.ndarray] = []
    secondary_objects: list[np.ndarray] = []
    secondary_independent_support: list[int] = []
    previous_unexplained = base_residual > THRESHOLD_M
    # The scene transform is exactly the fixed top-20 global RANSAC baseline.
    # Secondary transforms are refit from queries unexplained by that baseline;
    # each support count uses disjoint geometric residual support from all
    # preceding modes.
    for mode in range(1, 4):
        if mode >= residual_matrix.shape[1]:
            secondary_residuals.append(np.full(len(base), np.inf, dtype=np.float64))
            secondary_objects.append(np.full(len(base), -1, dtype=np.int64))
            secondary_independent_support.append(0)
            continue
        residual = residual_matrix[:, mode].astype(np.float64)
        object_id = reference_object_ids[nearest[selected_matrix[:, mode]]].astype(np.int64)
        inlier = np.isfinite(residual) & (residual <= THRESHOLD_M)
        support = int((inlier & previous_unexplained).sum())
        secondary_residuals.append(residual)
        secondary_objects.append(object_id)
        secondary_independent_support.append(support)
        previous_unexplained &= ~inlier

    return {
        "base": base,
        "combined": combined,
        "moved": moved,
        "expected": expected,
        "reference_object_ids": np.unique(reference_object_ids[reference_object_ids > 0]).astype(np.int64),
        "candidate_rank": candidate_rank,
        "base_residual": base_residual,
        "base_object": base_object,
        "baseline_correct": baseline_correct,
        "baseline_prediction": baseline_prediction,
        "secondary_residuals": secondary_residuals,
        "secondary_objects": secondary_objects,
        "secondary_independent_support": secondary_independent_support,
    }


def predict(data: dict[str, object], modes: int, margin: float, min_support: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the geometric switch rule and return label-free object predictions."""
    base_residual = data["base_residual"]
    expected = data["expected"]
    base_object = data["base_object"]
    residuals = data["secondary_residuals"]
    objects = data["secondary_objects"]
    supports = data["secondary_independent_support"]
    n = len(base_residual)
    valid_secondary: list[int] = []
    for slot in range(min(modes - 1, len(residuals))):
        if supports[slot] >= min_support:
            valid_secondary.append(slot)

    if valid_secondary:
        matrix = np.stack([residuals[slot] for slot in valid_secondary], axis=1)
        best_local = np.argmin(matrix, axis=1)
        rows = np.arange(n)
        best_residual = matrix[rows, best_local]
        best_slot = np.asarray(valid_secondary, dtype=np.int64)[best_local]
        finite = np.isfinite(best_residual) & (best_residual <= THRESHOLD_M)
        gap = base_residual - best_residual
        switch = finite & ((base_residual > THRESHOLD_M) | (gap >= margin))
        chosen_object = base_object.copy()
        chosen_mode = np.zeros(n, dtype=np.int64)
        for slot in valid_secondary:
            use = switch & (best_slot == slot)
            chosen_object[use] = objects[slot][use]
            chosen_mode[use] = slot + 1
    else:
        best_residual = np.full(n, np.inf, dtype=np.float64)
        gap = np.full(n, -np.inf, dtype=np.float64)
        switch = np.zeros(n, dtype=bool)
        chosen_object = base_object.copy()
        chosen_mode = np.zeros(n, dtype=np.int64)

    assigned = (base_residual <= THRESHOLD_M) | switch
    correct = data["moved"] & assigned & (chosen_object == expected)
    prediction = np.where(assigned, chosen_object, -1).astype(np.int64)
    return correct, chosen_mode, gap, prediction


def choose_lopo(data_by_pair: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_pairs = sorted(pair for pair in data_by_pair if pair != "target")
    if len(source_pairs) != 16:
        raise ValueError(f"expected 16 source relations, found {len(source_pairs)}")
    selection_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []

    for heldout in source_pairs:
        training = [pair for pair in source_pairs if pair != heldout]
        scores: dict[tuple[int, float, int], int] = {}
        for config in ((m, g, s) for m in MODE_COUNTS for g in MARGINS_M for s in MIN_SUPPORTS):
            modes, margin, support = config
            scores[config] = sum(
                int(predict(data_by_pair[pair], modes, margin, support)[0].sum())
                for pair in training
            )
        best = max(scores.values())
        # Deterministic conservative tie-break: fewer modes, stronger support,
        # then a larger residual margin.
        chosen = min((c for c, score in scores.items() if score == best), key=lambda c: (c[0], -c[2], -c[1]))
        correct, selected_mode, gap, prediction = predict(data_by_pair[heldout], *chosen)
        data = data_by_pair[heldout]
        moved = data["moved"]
        baseline = data["baseline_correct"]
        selection_rows.append({
            "pair": heldout,
            "modes": chosen[0],
            "margin_m": chosen[1],
            "min_independent_support": chosen[2],
            "training_relations": len(training),
            "training_correct_transfers": best,
            "heldout_queries": int(moved.sum()),
            "baseline_correct": int(baseline.sum()),
            "method_correct": int(correct.sum()),
            "rescued": int((correct & ~baseline).sum()),
            "lost": int((baseline & ~correct).sum()),
            "selected_secondary_queries": int((selected_mode > 0).sum()),
        })
        outcome_rows.append({
            "pair": heldout,
            "modes": chosen[0],
            "margin_m": chosen[1],
            "min_independent_support": chosen[2],
            "moved_queries": int(moved.sum()),
            "baseline_correct": int(baseline.sum()),
            "method_correct": int(correct.sum()),
            "rescued": int((correct & ~baseline).sum()),
            "lost": int((baseline & ~correct).sum()),
            "baseline_coverage": float(baseline.sum() / moved.sum()) if moved.any() else np.nan,
            "method_coverage": float(correct.sum() / moved.sum()) if moved.any() else np.nan,
            "selected_secondary_queries": int((selected_mode > 0).sum()),
        })

    selections = pd.DataFrame(selection_rows)
    outcomes = pd.DataFrame(outcome_rows)

    scores: dict[tuple[int, float, int], int] = {}
    for config in ((m, g, s) for m in MODE_COUNTS for g in MARGINS_M for s in MIN_SUPPORTS):
        scores[config] = sum(int(predict(data_by_pair[pair], *config)[0].sum()) for pair in source_pairs)
    best = max(scores.values())
    chosen = min((c for c, score in scores.items() if score == best), key=lambda c: (c[0], -c[2], -c[1]))
    target_correct, target_mode, _, _ = predict(data_by_pair["target"], *chosen)
    target = data_by_pair["target"]
    target_moved = target["moved"]
    outcomes = pd.concat([outcomes, pd.DataFrame([{
        "pair": "target",
        "modes": chosen[0],
        "margin_m": chosen[1],
        "min_independent_support": chosen[2],
        "moved_queries": int(target_moved.sum()),
        "baseline_correct": int(target["baseline_correct"].sum()),
        "method_correct": int(target_correct.sum()),
        "rescued": int((target_correct & ~target["baseline_correct"]).sum()),
        "lost": int((target["baseline_correct"] & ~target_correct).sum()),
        "baseline_coverage": float(target["baseline_correct"].sum() / target_moved.sum()) if target_moved.any() else np.nan,
        "method_coverage": float(target_correct.sum() / target_moved.sum()) if target_moved.any() else np.nan,
        "selected_secondary_queries": int((target_mode > 0).sum()),
    }])], ignore_index=True)
    target_selection = pd.DataFrame([{
        "pair": "target_once",
        "modes": chosen[0],
        "margin_m": chosen[1],
        "min_independent_support": chosen[2],
        "training_relations": 16,
        "training_correct_transfers": best,
        "heldout_queries": int(target_moved.sum()),
        "baseline_correct": int(target["baseline_correct"].sum()),
        "method_correct": int(target_correct.sum()),
        "rescued": int((target_correct & ~target["baseline_correct"]).sum()),
        "lost": int((target["baseline_correct"] & ~target_correct).sum()),
        "selected_secondary_queries": int((target_mode > 0).sum()),
    }])
    selections = pd.concat([selections, target_selection], ignore_index=True)
    return selections, outcomes


def add_query_diagnostics(data_by_pair: dict[str, dict[str, object]], selections: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    settings = {row.pair: (int(row.modes), float(row.margin_m), int(row.min_independent_support))
                for row in selections.itertuples(index=False) if row.pair != "target_once"}
    target_row = selections[selections.pair == "target_once"].iloc[0]
    settings["target"] = (
        int(target_row.modes), float(target_row.margin_m), int(target_row.min_independent_support)
    )
    for pair, data in data_by_pair.items():
        modes, margin, support = settings[pair]
        correct, selected_mode, gap, prediction = predict(data, modes, margin, support)
        best_secondary = np.full(len(data["base_residual"]), -1, dtype=np.int64)
        best_resid = np.full(len(best_secondary), np.inf)
        for slot in range(min(modes - 1, 3)):
            r = data["secondary_residuals"][slot]
            eligible = data["secondary_independent_support"][slot] >= support
            take = eligible & np.isfinite(r) & (r < best_resid)
            best_secondary[take] = slot + 1
            best_resid[take] = r[take]
        independent_support = np.zeros(len(best_secondary), dtype=np.int64)
        for slot in range(3):
            independent_support[best_secondary == slot + 1] = data["secondary_independent_support"][slot]
        baseline = data["baseline_correct"]
        moved = data["moved"]
        d = pd.DataFrame({
            "pair": pair,
            "role": "target" if pair == "target" else "source",
            "query_index": data["base"].query_index.to_numpy(),
            "moved_rigid": moved,
            "candidate_rank": data["candidate_rank"],
            "baseline_correct": baseline,
            "method_correct": correct,
            "method_prediction_object_id": prediction,
            "baseline_prediction_object_id": data["baseline_prediction"],
            "rescued": correct & ~baseline,
            "lost": baseline & ~correct,
            "selected_mode": selected_mode,
            "best_secondary_mode": best_secondary,
            "best_secondary_independent_support": independent_support,
            "residual_gap_m": gap,
            "chosen_modes": modes,
            "chosen_margin_m": margin,
            "chosen_min_independent_support": support,
        })
        rows.append(d[moved].copy())
    return pd.concat(rows, ignore_index=True)


def environment_groups(input_dir: Path, metadata_path: Path) -> dict[str, dict[str, str]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    by_scan: dict[str, tuple[str, str]] = {}
    for scene in metadata:
        environment_id = str(scene["reference"])
        split = str(scene.get("type", "unknown"))
        by_scan[environment_id] = (environment_id, split)
        for scan in scene.get("scans", []):
            by_scan[str(scan["reference"])] = (environment_id, split)
    result: dict[str, dict[str, str]] = {}
    for cache_path in sorted((input_dir / "descriptor_candidates").glob("*.npz")):
        pair = cache_path.stem
        with np.load(cache_path) as cache:
            reference_id = str(cache["reference_id"].item())
            rescan_id = str(cache["rescan_id"].item())
        if reference_id not in by_scan or rescan_id not in by_scan:
            raise ValueError(f"official 3RScan metadata is missing scans for {pair}")
        ref_group, ref_split = by_scan[reference_id]
        rescan_group, rescan_split = by_scan[rescan_id]
        if ref_group != rescan_group or ref_split != rescan_split:
            raise ValueError(f"reference/rescan environment or split mismatch for {pair}")
        result[pair] = {"environment_id": ref_group, "official_split": ref_split}
    return result


def relation_bootstrap(outcomes: pd.DataFrame, samples: int = 20000, seed: int = 17) -> pd.DataFrame:
    """Paired bootstrap over eligible environment groups, not query rows."""
    frame = outcomes[(outcomes.pair != "target") & (outcomes.moved_queries > 0)].copy()
    if len(frame) < 2:
        raise ValueError("relation bootstrap requires at least two eligible source groups")
    n = frame.moved_queries.to_numpy(dtype=np.float64)
    diff = (frame.method_correct - frame.baseline_correct).to_numpy(dtype=np.float64)
    pair_delta = diff / n
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(frame), size=(samples, len(frame)))
    weighted = 100.0 * diff[index].sum(axis=1) / n[index].sum(axis=1)
    macro = 100.0 * pair_delta[index].mean(axis=1)
    output = []
    for metric, point, values in (
        ("weighted_coverage_delta_pp", 100.0 * diff.sum() / n.sum(), weighted),
        ("macro_relation_delta_pp", 100.0 * pair_delta.mean(), macro),
    ):
        output.append({
            "metric": metric,
            "estimate": float(point),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "eligible_environment_groups": int(len(frame)),
            "bootstrap_samples": int(samples),
            "seed": seed,
        })
    return pd.DataFrame(output)


def stratified_effect(frame: pd.DataFrame, column: str, bins: list[tuple[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, selector in bins:
        group = frame[selector(frame)] if callable(selector) else frame.loc[selector]
        rows.append({
            "stratifier": column,
            "stratum": name,
            "queries": int(len(group)),
            "baseline_correct": int(group.baseline_correct.sum()),
            "method_correct": int(group.method_correct.sum()),
            "rescued": int(group.rescued.sum()),
            "lost": int(group.lost.sum()),
            "net": int(group.rescued.sum() - group.lost.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17,
                        help="RANSAC seed for the seed-robustness audit (default: 17)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(path.stem for path in (args.input_dir / "descriptor_candidates").glob("*.npz"))
    data_by_pair = {
        pair: read_pair(args.input_dir, pair, args.annotation_root, seed=args.seed)
        for pair in pairs
    }
    selections, outcomes = choose_lopo(data_by_pair)
    queries = add_query_diagnostics(data_by_pair, selections)
    groups = environment_groups(args.input_dir, args.metadata)
    outcomes["environment_id"] = outcomes.pair.map(lambda pair: groups[pair]["environment_id"])
    outcomes["official_split"] = outcomes.pair.map(lambda pair: groups[pair]["official_split"])
    selections["environment_id"] = selections.pair.map(lambda pair: groups[pair]["environment_id"] if pair in groups else "")
    selections["official_split"] = selections.pair.map(lambda pair: groups[pair]["official_split"] if pair in groups else "")
    source_groups = [groups[pair]["environment_id"] for pair in data_by_pair if pair != "target"]
    if len(source_groups) != len(set(source_groups)):
        raise ValueError("source relations share environment groups; relation LOPO would leak groups")
    selections.to_csv(args.output_dir / "source_lopo_selection.csv", index=False)
    outcomes.to_csv(args.output_dir / "source_lopo_outcomes.csv", index=False)
    queries.to_csv(args.output_dir / "query_outcomes.csv", index=False)

    rank_bins = [
        ("rank_1", lambda x: x.candidate_rank == 1),
        ("rank_2_5", lambda x: x.candidate_rank.between(2, 5)),
        ("rank_6_20", lambda x: x.candidate_rank.between(6, 20)),
        ("rank_21_50", lambda x: x.candidate_rank.between(21, 50)),
        ("absent_top50", lambda x: x.candidate_rank > 50),
    ]
    support_bins = [
        ("no_valid_secondary", lambda x: x.best_secondary_independent_support == 0),
        ("support_1_15", lambda x: x.best_secondary_independent_support.between(1, 15)),
        ("support_16_31", lambda x: x.best_secondary_independent_support.between(16, 31)),
        ("support_32_63", lambda x: x.best_secondary_independent_support.between(32, 63)),
        ("support_64_plus", lambda x: x.best_secondary_independent_support >= 64),
    ]
    gap_bins = [
        ("no_valid_secondary", lambda x: x.best_secondary_mode < 0),
        ("gap_lt_0", lambda x: x.residual_gap_m < 0),
        ("gap_0_02", lambda x: x.residual_gap_m.between(0, 0.02, inclusive="left")),
        ("gap_02_05", lambda x: x.residual_gap_m.between(0.02, 0.05, inclusive="left")),
        ("gap_05_10", lambda x: x.residual_gap_m.between(0.05, 0.10, inclusive="left")),
        ("gap_ge_10", lambda x: x.residual_gap_m >= 0.10),
    ]
    stratified = pd.concat([
        stratified_effect(queries[queries.role == "source"], "candidate_rank", rank_bins),
        stratified_effect(queries[queries.role == "source"], "independent_support", support_bins),
        stratified_effect(queries[queries.role == "source"], "residual_gap_m", gap_bins),
    ], ignore_index=True)
    stratified.to_csv(args.output_dir / "source_rescue_loss_strata.csv", index=False)
    boot = relation_bootstrap(outcomes)
    boot.to_csv(args.output_dir / "relation_bootstrap_ci.csv", index=False)

    source = outcomes[outcomes.pair != "target"]
    denom = int(source.moved_queries.sum())
    base_n = int(source.baseline_correct.sum())
    method_n = int(source.method_correct.sum())
    eligible = int((source.moved_queries > 0).sum())
    nonnegative = int((source.loc[source.moved_queries > 0, "method_correct"]
                       >= source.loc[source.moved_queries > 0, "baseline_correct"]).sum())
    pose_table = pd.read_csv(args.input_dir / "summary.csv")
    base_pose = int(pose_table.loc[
        (pose_table.role == "source") & (pose_table.arm == "global_ransac"),
        "strict_scene_pose_success",
    ].sum()) if args.seed == 17 else None
    pose_loss = 0  # The method keeps each seed's baseline scene pose fixed.
    summary = {
        "source_relations": int(len(source)),
        "eligible_source_relations": eligible,
        "unique_source_environment_groups": int(len(set(source_groups))),
        "official_source_split": sorted({groups[pair]["official_split"] for pair in data_by_pair if pair != "target"}),
        "target_official_split": groups["target"]["official_split"],
        "source_moved_rigid_queries": denom,
        "baseline_correct": base_n,
        "method_correct": method_n,
        "baseline_coverage": base_n / denom if denom else None,
        "method_coverage": method_n / denom if denom else None,
        "coverage_delta_pp": 100 * (method_n - base_n) / denom if denom else None,
        "rescued": int(source.rescued.sum()),
        "lost": int(source.lost.sum()),
        "nonnegative_source_relations": nonnegative,
        "baseline_scene_pose_successes": base_pose,
        "method_scene_pose_successes": base_pose,
        "scene_pose_loss": pose_loss,
        "scene_pose_locked_to_seed_specific_baseline": True,
        "ransac_seed": args.seed,
        "stage1_gate_pass": bool(
            denom and (method_n - base_n) / denom >= 0.10 and nonnegative >= 10 and pose_loss <= 1
        ),
        "target_moved_rigid_queries": int(outcomes.loc[outcomes.pair == "target", "moved_queries"].iloc[0]),
        "target_baseline_correct": int(outcomes.loc[outcomes.pair == "target", "baseline_correct"].iloc[0]),
        "target_method_correct": int(outcomes.loc[outcomes.pair == "target", "method_correct"].iloc[0]),
        "target_used_for_selection": False,
        "weighted_delta_ci95_pp": [float(boot.iloc[0].ci95_lower), float(boot.iloc[0].ci95_upper)],
        "macro_delta_ci95_pp": [float(boot.iloc[1].ci95_lower), float(boot.iloc[1].ci95_upper)],
        "method": "global top-20 scene pose locked; source-LOPO selected secondary motion switch with independent residual support",
        "candidate_rank_bins": {"1": int((queries[queries.role == "source"].candidate_rank == 1).sum()),
                                "2-5": int(queries[queries.role == "source"].candidate_rank.between(2, 5).sum()),
                                "6-20": int(queries[queries.role == "source"].candidate_rank.between(6, 20).sum()),
                                "21-50": int(queries[queries.role == "source"].candidate_rank.between(21, 50).sum()),
                                "absent_top50": int((queries[queries.role == "source"].candidate_rank > 50).sum())},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    baseline_check = (
        "The reconstructed global baseline assignments and inliers match the saved pilot exactly; residuals match within 1e-6 m."
        if args.seed == 17
        else f"The global baseline and secondary modes were refit from the frozen candidate cache with RANSAC seed {args.seed}; baseline correctness is scored from that same seed-specific fit."
    )
    pose_text = (
        f"Strict scene-pose success: {base_pose}/16 to {base_pose}/16 by construction."
        if base_pose is not None
        else "The scene pose is locked to each seed-specific global baseline, so the motion branch incurs zero pose loss by construction; absolute pose success was not recomputed for this seed."
    )
    report_lines = [
        "# Pose-locked multi-motion rescue mechanism probe",
        "",
        f"RANSAC seed: {args.seed}. The scene pose is fixed to the seed-specific global top-20 RANSAC estimate. Secondary hypotheses are refit from the saved adaptive candidate sets after removing queries explained by that fixed scene mode and earlier secondary modes. {baseline_check} A secondary mode is eligible only when its independent geometric support reaches a source-selected threshold. Hypothesis choice and switching use residuals only; mapped object IDs are used for scoring and strata.",
        "",
        f"Source coverage: {base_n}/{denom} ({100*base_n/denom:.2f}%) baseline to {method_n}/{denom} ({100*method_n/denom:.2f}%), {summary['coverage_delta_pp']:+.2f} pp. Rescued/lost: {summary['rescued']}/{summary['lost']}. Nonnegative eligible source relations: {nonnegative}/{eligible}. {pose_text} Relation-bootstrap 95% CIs: weighted gain [{boot.iloc[0].ci95_lower:+.2f}, {boot.iloc[0].ci95_upper:+.2f}] pp; macro gain [{boot.iloc[1].ci95_lower:+.2f}, {boot.iloc[1].ci95_upper:+.2f}] pp. All {len(set(source_groups))} source relations have distinct 3RScan environment IDs and use official split {', '.join(summary['official_source_split'])}. Stage-one gate: {'PASS' if summary['stage1_gate_pass'] else 'FAIL'}.",
        "",
        "The source-only leave-one-relation-out selector chooses one mode-count, residual-margin, and minimum independent-support setting using the other 15 source relations. The target is evaluated once with the setting selected from all 16 sources.",
        "",
        "## LOPO choices and paired outcomes",
        "",
        "| Relation | Modes | Margin (m) | Min support | Queries | Baseline | Method | Rescued | Lost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in outcomes.itertuples(index=False):
        report_lines.append(f"| {row.pair} | {row.modes} | {row.margin_m:.2f} | {row.min_independent_support} | {row.moved_queries} | {row.baseline_correct} | {row.method_correct} | {row.rescued} | {row.lost} |")
    report_lines += ["", "## Source rescue/loss strata", "", "| Signal | Stratum | Queries | Baseline correct | Method correct | Rescued | Lost | Net |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in stratified.itertuples(index=False):
        report_lines.append(f"| {row.stratifier} | {row.stratum} | {row.queries} | {row.baseline_correct} | {row.method_correct} | {row.rescued} | {row.lost} | {row.net:+d} |")
    report_lines += ["", f"Target diagnostic only: {summary['target_baseline_correct']}/{summary['target_moved_rigid_queries']} to {summary['target_method_correct']}/{summary['target_moved_rigid_queries']}; target labels were not used for selection.", "", "Candidate-rank strata are evaluation-only. Inference does not inspect object IDs. See the CSV files for query-level outcomes and all relation selections.", ""]
    (args.output_dir / "mechanism_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
