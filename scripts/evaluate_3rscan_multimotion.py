"""Paired 2x2 multi-motion correspondence pilot on the fixed 3RScan set.

The four arms share the same depth-valid query and reference patch populations:

* global_ransac: original top-20 spatial-consensus baseline;
* multi_motion: the same global model plus residual rigid hypotheses;
* adaptive_k: per-query top-20/top-50 expansion from a source-calibrated score gap;
* combined: adaptive candidates with residual rigid hypotheses.

Inference functions receive descriptors and geometry only. Annotation loading,
official relation mappings, and the scene-pose ground truth enter in the
separate scoring phase after all hypotheses have been produced.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from audit_motion_stratified_failures import read_pair_table
from evaluate_3rscan_framelevel_layers import combine, normalized_tokens
from evaluate_3rscan_instance_correspondence import (
    load,
    patch_world_points,
    preprocess,
    read_annotated_mesh,
)
from evaluate_3rscan_pose_free_registration import rigid_fit, transform
from evaluate_3rscan_spatial_consensus import consensus
from worldcache.backbone.dinov2 import load_dinov2


@dataclass
class PairFeatures:
    pair: str
    reference_id: str
    rescan_id: str
    pair_run: Path
    reference_points: np.ndarray
    query_points: np.ndarray
    query_frame_ids: np.ndarray
    candidate_indices: np.ndarray
    candidate_scores: np.ndarray
    top20_reference_points: np.ndarray
    top50_reference_points: np.ndarray
    gap20: np.ndarray


@dataclass
class Hypothesis:
    matrix: np.ndarray
    selected: np.ndarray
    residuals: np.ndarray
    inliers: np.ndarray


def save_feature_cache(path: Path, features: PairFeatures, model_size: str, layers: tuple[int, ...], resolution: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pair=np.asarray(features.pair),
        reference_id=np.asarray(features.reference_id),
        rescan_id=np.asarray(features.rescan_id),
        model_size=np.asarray(model_size),
        layers=np.asarray(layers, dtype=np.int64),
        resolution=np.asarray(resolution, dtype=np.int64),
        reference_points=features.reference_points,
        query_points=features.query_points,
        query_frame_ids=features.query_frame_ids,
        candidate_indices=features.candidate_indices,
        candidate_scores=features.candidate_scores,
        gap20=features.gap20,
    )


def load_feature_cache(
    path: Path,
    pair: str,
    reference_id: str,
    rescan_id: str,
    pair_run: Path,
    model_size: str,
    layers: tuple[int, ...],
    resolution: int,
) -> PairFeatures | None:
    if not path.is_file():
        return None
    with np.load(path) as cached:
        metadata_matches = (
            str(cached["pair"].item()) == pair
            and str(cached["reference_id"].item()) == reference_id
            and str(cached["rescan_id"].item()) == rescan_id
            and str(cached["model_size"].item()) == model_size
            and tuple(cached["layers"].astype(int).tolist()) == layers
            and int(cached["resolution"].item()) == resolution
        )
        if not metadata_matches:
            raise ValueError(f"descriptor candidate cache metadata mismatch: {path}")
        reference_points = cached["reference_points"].copy()
        query_points = cached["query_points"].copy()
        query_frame_ids = cached["query_frame_ids"].copy()
        candidate_indices = cached["candidate_indices"].copy()
        candidate_scores = cached["candidate_scores"].copy()
        gap20 = cached["gap20"].copy()
    return PairFeatures(
        pair=pair,
        reference_id=reference_id,
        rescan_id=rescan_id,
        pair_run=pair_run,
        reference_points=reference_points,
        query_points=query_points,
        query_frame_ids=query_frame_ids,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        top20_reference_points=reference_points[candidate_indices[:, :20]],
        top50_reference_points=reference_points[candidate_indices],
        gap20=gap20,
    )


def find_pair_run(runs_root: Path, reference_id: str, rescan_id: str) -> Path:
    matches: list[Path] = []
    for config_path in runs_root.rglob("config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            config.get("dataset") == "3rscan"
            and config.get("reference") == reference_id
            and config.get("rescan") == rescan_id
            and config.get("coordinate_convention") == "native_scan_frames"
            and (config_path.parent / "manifest.json").is_file()
        ):
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"expected one native-frame cache for {reference_id}/{rescan_id}; found {matches}"
        )
    return matches[0]


def extract_pair_features(
    pair: str,
    reference_id: str,
    rescan_id: str,
    pair_run: Path,
    backbone: torch.nn.Module,
    layers: tuple[int, ...],
    device: torch.device,
    descriptor_chunk: int = 1024,
    preserve_top_k: int | None = None,
) -> PairFeatures:
    """Extract descriptor candidates without opening annotation files."""
    config = json.loads((pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError(f"{pair}: pair cache is not in native scan coordinates")
    manifest = json.loads((pair_run / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["reference"]) != 20 or not 1 <= len(manifest["rescan"]) <= 20:
        raise ValueError(f"{pair}: expected 20 reference frames and 1-20 fixed rescan frames")

    reference_features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    reference_points: list[np.ndarray] = []
    for entry in manifest["reference"]:
        observation = load(pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = normalized_tokens(backbone, observation, layers, device, valid)
        for layer in layers:
            reference_features[layer].append(features[layer])
        reference_points.append(points.numpy().astype(np.float64))

    query_features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    query_points: list[np.ndarray] = []
    query_frame_ids: list[np.ndarray] = []
    for entry in manifest["rescan"]:
        observation = load(pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = normalized_tokens(backbone, observation, layers, device, valid)
        for layer in layers:
            query_features[layer].append(features[layer])
        query_points.append(points.numpy().astype(np.float64))
        query_frame_ids.append(np.full(len(points), int(entry["frame_id"]), dtype=np.int64))

    reference_points_array = np.concatenate(reference_points, axis=0)
    query_points_array = np.concatenate(query_points, axis=0)
    query_frame_array = np.concatenate(query_frame_ids, axis=0)
    for layer in layers:
        reference_features[layer] = [torch.cat(reference_features[layer], dim=0)]
        query_features[layer] = [torch.cat(query_features[layer], dim=0)]
    ref_descriptor = combine({key: value[0] for key, value in reference_features.items()}, layers)
    query_descriptor = combine({key: value[0] for key, value in query_features.items()}, layers)

    # Chunked scoring bounds peak memory while preserving exact top-50 results.
    candidate_scores: list[torch.Tensor] = []
    candidate_indices: list[torch.Tensor] = []
    for start in range(0, len(query_descriptor), descriptor_chunk):
        scores = query_descriptor[start : start + descriptor_chunk] @ ref_descriptor.T
        values, indices = torch.topk(scores, k=min(50, scores.shape[1]), dim=1)
        if preserve_top_k is not None:
            if not 0 < preserve_top_k < min(50, scores.shape[1]):
                raise ValueError("preserve_top_k must be positive and smaller than the cached candidate count")
            values_k, indices_k = torch.topk(scores, k=preserve_top_k, dim=1)
            values_np = values.cpu().numpy()
            indices_np = indices.cpu().numpy()
            values_k_np = values_k.cpu().numpy()
            indices_k_np = indices_k.cpu().numpy()
            ordered_values = np.empty_like(values_np)
            ordered_indices = np.empty_like(indices_np)
            for row in range(len(indices_np)):
                tail = ~np.isin(indices_np[row], indices_k_np[row], assume_unique=True)
                remaining_values = values_np[row, tail]
                remaining_indices = indices_np[row, tail]
                ordered_values[row] = np.concatenate((values_k_np[row], remaining_values))[: values_np.shape[1]]
                ordered_indices[row] = np.concatenate((indices_k_np[row], remaining_indices))[: indices_np.shape[1]]
            values = torch.from_numpy(ordered_values)
            indices = torch.from_numpy(ordered_indices)
        candidate_scores.append(values.cpu())
        candidate_indices.append(indices.cpu())
    score_array = torch.cat(candidate_scores).numpy().astype(np.float64)
    index_array = torch.cat(candidate_indices).numpy().astype(np.int64)
    if index_array.shape[1] < 50:
        raise ValueError(f"{pair}: reference pool has fewer than 50 patches")
    return PairFeatures(
        pair=pair,
        reference_id=reference_id,
        rescan_id=rescan_id,
        pair_run=pair_run,
        reference_points=reference_points_array,
        query_points=query_points_array,
        query_frame_ids=query_frame_array,
        candidate_indices=index_array,
        candidate_scores=score_array,
        top20_reference_points=reference_points_array[index_array[:, :20]],
        top50_reference_points=reference_points_array[index_array],
        gap20=score_array[:, 0] - score_array[:, 19],
    )


def _candidate_match(
    source: np.ndarray,
    candidate_points: np.ndarray,
    candidate_indices: np.ndarray,
    limits: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = transform(source, matrix)
    selected_points = candidate_points[candidate_indices]
    squared = np.sum((predicted[:, None, :] - selected_points) ** 2, axis=2)
    squared[np.arange(squared.shape[1])[None, :] >= limits[:, None]] = np.inf
    slots = np.argmin(squared, axis=1)
    residuals = np.sqrt(squared[np.arange(len(source)), slots])
    return candidate_indices[np.arange(len(source)), slots], residuals, slots


def consensus_variable_k(
    source: np.ndarray,
    candidate_points: np.ndarray,
    candidate_indices: np.ndarray,
    limits: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
    batch_size: int = 64,
) -> Hypothesis:
    """Batched RANSAC with a per-query candidate limit (20 or 50)."""
    if len(source) < 3:
        raise ValueError("RANSAC requires at least three query points")
    if candidate_indices.shape != (len(source), 50) or limits.shape != (len(source),):
        raise ValueError("variable-K RANSAC expects N x 50 candidates and N limits")
    if np.any((limits < 1) | (limits > 50)):
        raise ValueError("candidate limits must lie within [1, 50]")

    rng = np.random.default_rng(seed)
    samples = np.stack([rng.choice(len(source), 3, replace=False) for _ in range(iterations)])
    slots = np.stack([rng.integers(0, limits[row], size=3) for row in samples])
    target_indices = candidate_indices[samples, slots]
    target = candidate_points[target_indices]
    sample_source = source[samples]
    source_center = sample_source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    covariance = np.einsum(
        "bni,bnj->bij", sample_source - source_center, target - target_center, optimize=True
    )
    u, _, vh = np.linalg.svd(covariance)
    rotation = np.matmul(vh.transpose(0, 2, 1), u.transpose(0, 2, 1))
    reflected = np.linalg.det(rotation) < 0
    if np.any(reflected):
        vh[reflected, -1, :] *= -1
        rotation[reflected] = np.matmul(vh[reflected].transpose(0, 2, 1), u[reflected].transpose(0, 2, 1))
    translation = target_center[:, 0, :] - np.einsum("bij,bj->bi", rotation, source_center[:, 0, :])

    candidate = candidate_points[candidate_indices]
    candidate_squared = np.sum(candidate * candidate, axis=2)
    candidate_valid = np.arange(50)[None, :] < limits[:, None]
    best_score = -1
    best_matrix = np.eye(4, dtype=np.float64)
    for start in range(0, iterations, batch_size):
        end = min(iterations, start + batch_size)
        batch_rotation = rotation[start:end]
        batch_translation = translation[start:end]
        predicted = np.einsum("bij,nj->bni", batch_rotation, source, optimize=True) + batch_translation[:, None, :]
        predicted_squared = np.sum(predicted * predicted, axis=2)
        dot = np.einsum("bni,nki->bnk", predicted, candidate, optimize=True)
        squared = np.maximum(predicted_squared[:, :, None] + candidate_squared[None, :, :] - 2.0 * dot, 0.0)
        # Per-row masking is needed when different queries use K=20 and K=50.
        if not np.all(limits == limits[0]):
            squared = np.where(candidate_valid[None, :, :], squared, np.inf)
        else:
            squared[:, :, limits[0] :] = np.inf
        inliers = squared.min(axis=2) <= threshold * threshold
        scores = inliers.sum(axis=1)
        winner = int(np.argmax(scores))
        if int(scores[winner]) > best_score:
            best_score = int(scores[winner])
            best_matrix[:3, :3] = batch_rotation[winner]
            best_matrix[:3, 3] = batch_translation[winner]

    selected, residuals, _ = _candidate_match(source, candidate_points, candidate_indices, limits, best_matrix)
    inliers = residuals <= threshold
    if inliers.sum() >= 3:
        best_matrix = rigid_fit(source[inliers], candidate_points[selected[inliers]])
        selected, residuals, _ = _candidate_match(source, candidate_points, candidate_indices, limits, best_matrix)
        inliers = residuals <= threshold
    return Hypothesis(best_matrix, selected, residuals, inliers)


def fit_fixed_consensus(
    source: np.ndarray,
    candidate_points: np.ndarray,
    candidate_indices: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> Hypothesis:
    matrix, selected, inliers = consensus(
        source,
        candidate_points,
        candidate_indices,
        threshold=threshold,
        iterations=iterations,
        seed=seed,
    )
    residuals = np.linalg.norm(transform(source, matrix) - candidate_points[selected], axis=1)
    return Hypothesis(matrix, selected, residuals, inliers)


def fit_modes(
    features: PairFeatures,
    limits: np.ndarray,
    threshold: float,
    iterations: int,
    secondary_iterations: int,
    seed: int,
    max_modes: int,
    min_support: int,
    fixed_baseline: Hypothesis | None = None,
) -> list[Hypothesis]:
    """Fit a scene mode, then explain residual queries with distinct motions."""
    if fixed_baseline is not None:
        modes = [fixed_baseline]
    elif np.all(limits == 20):
        modes = [fit_fixed_consensus(
            features.query_points,
            features.reference_points,
            features.candidate_indices[:, :20],
            threshold,
            iterations,
            seed,
        )]
    else:
        modes = [consensus_variable_k(
            features.query_points,
            features.reference_points,
            features.candidate_indices,
            limits,
            threshold,
            iterations,
            seed,
        )]

    for mode_index in range(1, max_modes):
        _, first_residuals, _ = _candidate_match(
            features.query_points,
            features.reference_points,
            features.candidate_indices,
            limits,
            modes[0].matrix,
        )
        # Remove query points explained by any already fitted motion.
        min_residual = first_residuals.copy()
        for previous in modes[1:]:
            _, residuals, _ = _candidate_match(
                features.query_points,
                features.reference_points,
                features.candidate_indices,
                limits,
                previous.matrix,
            )
            min_residual = np.minimum(min_residual, residuals)
        active = np.flatnonzero(min_residual > threshold)
        if len(active) < max(3, min_support):
            break
        local_indices = features.candidate_indices[active]
        local_limits = limits[active]
        mode = consensus_variable_k(
            features.query_points[active],
            features.reference_points,
            local_indices,
            local_limits,
            threshold,
            secondary_iterations,
            seed + 1009 * mode_index,
        )
        if int(mode.inliers.sum()) < min_support:
            break
        modes.append(mode)
    return modes


def assign_modes(
    features: PairFeatures,
    limits: np.ndarray,
    modes: list[Hypothesis],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual_columns: list[np.ndarray] = []
    selected_columns: list[np.ndarray] = []
    for mode in modes:
        selected, residuals, _ = _candidate_match(
            features.query_points,
            features.reference_points,
            features.candidate_indices,
            limits,
            mode.matrix,
        )
        selected_columns.append(selected)
        residual_columns.append(residuals)
    residual_matrix = np.stack(residual_columns, axis=1)
    selected_matrix = np.stack(selected_columns, axis=1)
    mode_assignment = np.argmin(residual_matrix, axis=1).astype(np.int64)
    query_rows = np.arange(len(mode_assignment))
    selected = selected_matrix[query_rows, mode_assignment]
    residuals = residual_matrix[query_rows, mode_assignment]
    assigned = residuals <= threshold
    mode_assignment[~assigned] = -1
    return mode_assignment, selected, residuals, assigned, selected_matrix, residual_matrix


def annotation_labels(annotation_root: Path, features: PairFeatures) -> dict[str, np.ndarray]:
    """Load labels only after the descriptor-only hypotheses have been made."""
    ref_tree, reference_object_ids, _, reference_semantics, _ = read_annotated_mesh(annotation_root, features.reference_id)
    query_tree, query_object_ids, _, query_semantics, _ = read_annotated_mesh(annotation_root, features.rescan_id)
    _, ref_nearest = ref_tree.query(features.reference_points.astype(np.float32), k=1)
    _, query_nearest = query_tree.query(features.query_points.astype(np.float32), k=1)
    return {
        "reference_object_id": reference_object_ids[ref_nearest],
        "query_object_id": query_object_ids[query_nearest],
        "reference_semantic_id": reference_semantics[ref_nearest],
        "query_semantic_id": query_semantics[query_nearest],
    }


def relation_record(metadata: list[dict[str, Any]], reference_id: str, rescan_id: str) -> dict[str, Any]:
    for relation in metadata:
        if relation.get("reference") != reference_id:
            continue
        for scan in relation.get("scans", []):
            if scan.get("reference") == rescan_id:
                return {"relation": relation, "scan": scan}
    raise ValueError(f"official relation missing for {reference_id}/{rescan_id}")


def rigid_object_mapping(scan_record: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in scan_record.get("rigid", []):
        ref_id = int(item.get("instance_reference", item.get("id", -1)))
        query_id = int(item.get("instance_rescan", item.get("id", -1)))
        if ref_id <= 0 or query_id <= 0:
            continue
        if query_id in mapping and mapping[query_id] != ref_id:
            raise ValueError(f"ambiguous official rigid mapping for rescan objectId={query_id}")
        mapping[query_id] = ref_id
    return mapping


def score_outputs(
    features: PairFeatures,
    outputs: dict[str, dict[str, Any]],
    annotation_root: Path,
    metadata: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    labels = annotation_labels(annotation_root, features)
    relation = relation_record(metadata, features.reference_id, features.rescan_id)
    changed_map = rigid_object_mapping(relation["scan"])
    reference_object = labels["reference_object_id"]
    query_object = labels["query_object_id"]
    reference_semantic = labels["reference_semantic_id"]
    query_semantic = labels["query_semantic_id"]
    evaluable = (query_object != 0) & (query_semantic != 0)
    moved = evaluable & np.isin(query_object, np.fromiter(changed_map.keys(), dtype=np.int64))
    expected_reference_object = np.full(len(query_object), -1, dtype=np.int64)
    for query_id, ref_id in changed_map.items():
        expected_reference_object[query_object == query_id] = ref_id

    ground_truth = np.asarray(config["rescan_to_reference"], dtype=np.float64)
    summary_rows: list[dict[str, Any]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for arm, item in outputs.items():
        selected = item["selected"]
        mode_assignment = item["mode_assignment"]
        residuals = item["residuals"]
        assigned = item["assigned"]
        selected_object = reference_object[selected]
        selected_semantic = reference_semantic[selected]
        semantic_correct = selected_semantic == query_semantic
        rigid_correct = selected_object == expected_reference_object
        correct_moved = moved & assigned & rigid_correct
        rotation_error = Rotation.from_matrix(item["scene_matrix"][:3, :3].T @ ground_truth[:3, :3]).magnitude()
        translation_error = np.linalg.norm(item["scene_matrix"][:3, 3] - ground_truth[:3, 3])
        moved_count = int(moved.sum())
        correct_moved_count = int(correct_moved.sum())
        inlier_evaluable = assigned & evaluable
        summary_rows.append({
            "pair": features.pair,
            "arm": arm,
            "role": "target" if features.pair == "target" else "source",
            "queries": int(evaluable.sum()),
            "matching_queries": int(len(evaluable)),
            "global_modes": len(item["modes"]),
            "evaluated_inliers": int(inlier_evaluable.sum()),
            "inlier_coverage": float(inlier_evaluable.sum() / max(1, evaluable.sum())),
            "semantic_accuracy_among_inliers": (
                float(semantic_correct[inlier_evaluable].mean()) if inlier_evaluable.any() else float("nan")
            ),
            "moved_rigid_queries": moved_count,
            "moved_rigid_correct_transfer_count": correct_moved_count,
            "moved_rigid_correct_transfer_coverage": (
                float(correct_moved_count / moved_count) if moved_count else float("nan")
            ),
            "moved_rigid_correct_transfer_precision": (
                float(correct_moved_count / int((moved & assigned).sum())) if (moved & assigned).any() else float("nan")
            ),
            "scene_rotation_error_deg": float(np.degrees(rotation_error)),
            "scene_translation_error_m": float(translation_error),
            "strict_scene_pose_success": bool(np.degrees(rotation_error) < 5.0 and translation_error < 0.25),
            "adaptive_threshold_gap20": float(item["adaptive_threshold"]),
            "adaptive_k50_fraction": float(item["k50_fraction"]),
        })
        diagnostic = {
            "pair": features.pair,
            "arm": arm,
            "query_index": np.arange(len(evaluable), dtype=np.int64),
            "frame_id": features.query_frame_ids,
            "query_evaluable": evaluable,
            "query_instance_id": query_object,
            "official_rigid_reference_instance_id": expected_reference_object,
            "query_semantic_id": query_semantic,
            "selected_reference_index": selected,
            "selected_reference_instance_id": selected_object,
            "selected_reference_semantic_id": selected_semantic,
            "assigned_mode": mode_assignment,
            "spatial_residual_m": residuals,
            "consensus_inlier": assigned,
            "semantic_correct": semantic_correct,
            "official_rigid_instance_correct": rigid_correct & moved,
            "descriptor_gap_top1_top20": features.gap20,
            "candidate_limit": item["limits"],
        }
        for mode_index in range(len(item["modes"])):
            mode_selected = item["mode_selected_matrix"][:, mode_index]
            diagnostic[f"hypothesis_{mode_index}_selected_reference_index"] = item["mode_selected_matrix"][:, mode_index]
            diagnostic[f"hypothesis_{mode_index}_residual_m"] = item["mode_residual_matrix"][:, mode_index]
            diagnostic[f"hypothesis_{mode_index}_selected_reference_instance_id"] = reference_object[mode_selected]
            diagnostic[f"hypothesis_{mode_index}_selected_reference_semantic_id"] = reference_semantic[mode_selected]
            diagnostic[f"hypothesis_{mode_index}_inlier"] = item["mode_residual_matrix"][:, mode_index] <= item["threshold"]
        diagnostic_frames.append(pd.DataFrame(diagnostic))
    return summary_rows, diagnostic_frames


def calibrate_gap_thresholds(features_by_pair: list[PairFeatures]) -> dict[str, float]:
    """Leave one source relation out; use a source-only confidence quantile."""
    sources = [features for features in features_by_pair if features.pair != "target"]
    if len(sources) != 16:
        raise ValueError(f"expected 16 source relations, found {len(sources)}")
    per_relation = {item.pair: float(np.median(item.gap20)) for item in sources}
    calibrated: dict[str, float] = {}
    for heldout in sources:
        training = [value for pair, value in per_relation.items() if pair != heldout.pair]
        calibrated[heldout.pair] = float(np.median(training))
    calibrated["target"] = float(np.median(list(per_relation.values())))
    return calibrated


def run_inference_arms(
    features: PairFeatures,
    adaptive_threshold: float,
    threshold: float,
    iterations: int,
    secondary_iterations: int,
    seed: int,
    max_modes: int,
    min_support: int,
) -> dict[str, dict[str, Any]]:
    """Produce all four arms before any annotations are opened."""
    limits_fixed = np.full(len(features.query_points), 20, dtype=np.int64)
    limits_adaptive = np.where(features.gap20 <= adaptive_threshold, 50, 20).astype(np.int64)
    base = fit_fixed_consensus(
        features.query_points,
        features.reference_points,
        features.candidate_indices[:, :20],
        threshold,
        iterations,
        seed,
    )
    adaptive_modes = fit_modes(
        features,
        limits_adaptive,
        threshold,
        iterations,
        secondary_iterations,
        seed,
        max_modes,
        min_support,
    )
    fixed_modes = fit_modes(
        features,
        limits_fixed,
        threshold,
        iterations,
        secondary_iterations,
        seed,
        max_modes,
        min_support,
        fixed_baseline=base,
    )
    baseline_assignment, baseline_selected, baseline_residual, baseline_inlier = 0, base.selected, base.residuals, base.inliers
    baseline_mode_assignment = np.where(baseline_inlier, baseline_assignment, -1).astype(np.int64)
    fixed_multi = assign_modes(features, limits_fixed, fixed_modes, threshold)
    adaptive_single = assign_modes(features, limits_adaptive, adaptive_modes[:1], threshold)
    adaptive_multi = assign_modes(features, limits_adaptive, adaptive_modes, threshold)

    def pack(
        modes: list[Hypothesis],
        assignment: np.ndarray,
        selected: np.ndarray,
        residuals: np.ndarray,
        inliers: np.ndarray,
        limits: np.ndarray,
        mode_selected_matrix: np.ndarray,
        mode_residual_matrix: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "modes": modes,
            "scene_matrix": modes[0].matrix,
            "mode_assignment": assignment,
            "selected": selected,
            "residuals": residuals,
            "assigned": inliers,
            "limits": limits,
            "mode_selected_matrix": mode_selected_matrix,
            "mode_residual_matrix": mode_residual_matrix,
            "adaptive_threshold": adaptive_threshold,
            "k50_fraction": float(np.mean(limits == 50)),
            "threshold": threshold,
        }

    baseline_selected_matrix = baseline_selected[:, None]
    baseline_residual_matrix = baseline_residual[:, None]
    return {
        "global_ransac": pack(
            [base], baseline_mode_assignment, baseline_selected, baseline_residual,
            baseline_inlier, limits_fixed, baseline_selected_matrix, baseline_residual_matrix,
        ),
        "multi_motion": pack(
            fixed_modes, *fixed_multi[:4], limits_fixed, *fixed_multi[4:]
        ),
        "adaptive_k": pack(
            adaptive_modes[:1], *adaptive_single[:4], limits_adaptive, *adaptive_single[4:]
        ),
        "combined": pack(
            adaptive_modes, *adaptive_multi[:4], limits_adaptive, *adaptive_multi[4:]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-table", type=Path, default=Path("paper/supplementary.tex"))
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--baseline-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--baseline-template",
        default="20260922_candidate_audit_{pair}_vitg_clean",
        help="saved 1,500-iteration ViT-G baseline directory template",
    )
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("small", "base", "large", "giant"), default="giant")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 20, 40])
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--secondary-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-modes", type=int, default=4)
    parser.add_argument("--min-mode-support", type=int, default=8)
    parser.add_argument("--pairs", nargs="*", help="optional pair subset for smoke runs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = read_pair_table(args.pair_table)
    if args.pairs:
        wanted = set(args.pairs)
        pair_rows = [row for row in pair_rows if row[0] in wanted]
        if {row[0] for row in pair_rows} != wanted:
            raise ValueError(f"pair subset missing from table: {sorted(wanted - {row[0] for row in pair_rows})}")
    if not args.pairs and (len(pair_rows) != 17 or pair_rows[-1][0] != "target"):
        raise ValueError("full run requires the exact 16 source + 1 held-out target table")
    layers = tuple(sorted(set(args.layers)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cache_dir = args.output_dir / "descriptor_candidates"
    pair_runs = [find_pair_run(args.runs_root, reference_id, rescan_id) for _, reference_id, rescan_id in pair_rows]
    pair_configs = [json.loads((pair_run / "config.json").read_text(encoding="utf-8")) for pair_run in pair_runs]
    resolutions = {int(config["resolution"]) for config in pair_configs}
    if len(resolutions) != 1:
        raise ValueError(f"pair caches use mixed image resolutions: {sorted(resolutions)}")
    resolution = resolutions.pop()
    cached_features: list[PairFeatures | None] = []
    for (pair, reference_id, rescan_id), pair_run in zip(pair_rows, pair_runs):
        cached_features.append(load_feature_cache(
            feature_cache_dir / f"{pair}.npz", pair, reference_id, rescan_id,
            pair_run, args.model_size, layers, resolution,
        ))
    if all(features is not None for features in cached_features):
        print(f"loading {len(cached_features)} cached descriptor candidate set(s)", flush=True)
        features_by_pair = [features for features in cached_features if features is not None]
    else:
        backbone = load_dinov2(args.model_size, resolution, device)
        features_by_pair: list[PairFeatures] = []
        for (pair, reference_id, rescan_id), pair_run, cached in zip(pair_rows, pair_runs, cached_features):
            if cached is not None:
                features_by_pair.append(cached)
                continue
            print(f"extracting {pair}: {reference_id}/{rescan_id}", flush=True)
            features = extract_pair_features(
                pair,
                reference_id,
                rescan_id,
                pair_run,
                backbone,
                layers,
                device,
                preserve_top_k=20,
            )
            save_feature_cache(feature_cache_dir / f"{pair}.npz", features, args.model_size, layers, resolution)
            features_by_pair.append(features)

    gap_thresholds = calibrate_gap_thresholds(features_by_pair) if not args.pairs else {
        item.pair: 0.0 for item in features_by_pair
    }
    summary_rows: list[dict[str, Any]] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for features in features_by_pair:
        print(f"fitting {features.pair}: gap20 threshold={gap_thresholds[features.pair]:.6f}", flush=True)
        # Inference ends here. Annotation files and official mappings are read
        # only inside score_outputs after all four arms exist.
        arms = run_inference_arms(
            features,
            gap_thresholds[features.pair],
            args.threshold,
            args.iterations,
            args.secondary_iterations,
            args.seed,
            args.max_modes,
            args.min_mode_support,
        )
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        config = json.loads((features.pair_run / "config.json").read_text(encoding="utf-8"))
        rows, diagnostics = score_outputs(features, arms, args.annotation_root, metadata, config)
        summary_rows.extend(rows)
        diagnostic_frames.extend(diagnostics)
        pd.DataFrame(rows).to_csv(args.output_dir / f"{features.pair}_summary.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    diagnostics.to_csv(args.output_dir / "query_diagnostics.csv", index=False)
    pd.DataFrame([
        {"pair": pair, "role": "target" if pair == "target" else "source", "gap20_threshold": value}
        for pair, value in gap_thresholds.items()
    ]).to_csv(args.output_dir / "source_lopo_thresholds.csv", index=False)

    source = summary[summary.role == "source"]
    target = summary[summary.role == "target"]
    source_pair_count = int(source.pair.nunique())
    full_protocol = len(pair_rows) == 17 and pair_rows[-1][0] == "target"
    aggregate_rows: list[dict[str, Any]] = []
    for arm, frame in summary.groupby("arm", sort=False):
        src = frame[frame.role == "source"]
        tgt = frame[frame.role == "target"]
        aggregate_rows.append({
            "arm": arm,
            "source_pairs": int(len(src)),
            "source_strict_scene_pose_successes": int(src.strict_scene_pose_success.sum()),
            "source_moved_rigid_queries": int(src.moved_rigid_queries.sum()),
            "source_moved_rigid_correct_transfers": int(src.moved_rigid_correct_transfer_count.sum()),
            "source_moved_rigid_correct_transfer_coverage": (
                float(src.moved_rigid_correct_transfer_count.sum() / src.moved_rigid_queries.sum())
                if src.moved_rigid_queries.sum() else float("nan")
            ),
            "source_mean_pair_moved_rigid_coverage": float(src.moved_rigid_correct_transfer_coverage.mean()),
            "target_moved_rigid_queries": int(tgt.moved_rigid_queries.sum()),
            "target_moved_rigid_correct_transfers": int(tgt.moved_rigid_correct_transfer_count.sum()),
            "target_moved_rigid_correct_transfer_coverage": (
                float(tgt.moved_rigid_correct_transfer_count.sum() / tgt.moved_rigid_queries.sum())
                if tgt.moved_rigid_queries.sum() else float("nan")
            ),
            "target_strict_scene_pose_successes": int(tgt.strict_scene_pose_success.sum()),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)
    baseline_rows: list[dict[str, Any]] = []
    for pair in summary.pair.unique():
        old_dir = args.baseline_root / args.baseline_template.format(pair=pair)
        old_path = old_dir / "summary.csv"
        if not old_path.is_file():
            continue
        old_frame = pd.read_csv(old_path)
        old_frame = old_frame[(old_frame.geometry_weight == 0.0) & (old_frame.top_k == 20)]
        if len(old_frame) != 1:
            raise ValueError(f"expected one saved top-20 baseline row in {old_path}")
        old = old_frame.iloc[0]
        current = summary[(summary.pair == pair) & (summary.arm == "global_ransac")].iloc[0]
        old_diag_path = old_dir / "query_diagnostics.csv"
        old_query_count = None
        if old_diag_path.is_file():
            old_diag = pd.read_csv(old_diag_path)
            old_diag = old_diag[(old_diag.geometry_weight == 0.0) & (old_diag.top_k == 20)]
            old_query_count = int(old_diag.query_index.nunique())
        query_count_match = int(current.queries) == int(old.queries) and (
            old_query_count is None or int(current.matching_queries) == old_query_count
        )
        inlier_count_match = int(current.evaluated_inliers) == int(old.inliers)
        rotation_delta = abs(float(current.scene_rotation_error_deg) - float(old.rotation_error_deg))
        translation_delta = abs(float(current.scene_translation_error_m) - float(old.translation_error_m))
        pose_match = rotation_delta <= 1e-6 and translation_delta <= 1e-6
        baseline_rows.append({
            "pair": pair,
            "queries_match": bool(query_count_match),
            "inliers_match": bool(inlier_count_match),
            "rotation_abs_delta_deg": rotation_delta,
            "translation_abs_delta_m": translation_delta,
            "pose_match": bool(pose_match),
            "baseline_regression_pass": bool(query_count_match and inlier_count_match and pose_match),
        })
    baseline_regression = pd.DataFrame(baseline_rows)
    baseline_regression.to_csv(args.output_dir / "baseline_regression.csv", index=False)
    baseline_regression_pass = bool(
        len(baseline_regression) == len(summary.pair.unique())
        and baseline_regression.baseline_regression_pass.all()
    )
    baseline = aggregate[aggregate.arm == "global_ransac"].iloc[0]
    go_nogo: dict[str, Any] = {}
    for _, row in aggregate.iterrows():
        arm = str(row.arm)
        if arm == "global_ransac":
            continue
        frame = summary[(summary.arm == arm) & (summary.role == "source")]
        base_frame = summary[(summary.arm == "global_ransac") & (summary.role == "source")]
        paired = frame.merge(base_frame, on="pair", suffixes=("", "_baseline"))
        nonnegative = int((paired.moved_rigid_correct_transfer_coverage >= paired.moved_rigid_correct_transfer_coverage_baseline).sum())
        pose_loss = int(base_frame.strict_scene_pose_success.sum() - frame.strict_scene_pose_success.sum())
        delta = float(row.source_moved_rigid_correct_transfer_coverage - baseline.source_moved_rigid_correct_transfer_coverage)
        go_nogo[arm] = {
            "source_moved_rigid_coverage_delta_pp": 100.0 * delta,
            "source_relations_nonnegative_vs_baseline": nonnegative,
            "source_scene_pose_success_loss": pose_loss,
            "passes_stage1_gate": bool(delta >= 0.10 and nonnegative >= 10 and pose_loss <= 1),
        }
    (args.output_dir / "go_nogo.json").write_text(json.dumps(go_nogo, indent=2) + "\n", encoding="utf-8")
    arm_lines = [
        "| Arm | Source strict scene pose | Moved rigid transfer coverage | Correct / eligible | Target coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        coverage = "--" if pd.isna(row.source_moved_rigid_correct_transfer_coverage) else f"{100 * row.source_moved_rigid_correct_transfer_coverage:.2f}%"
        target_coverage = "--" if pd.isna(row.target_moved_rigid_correct_transfer_coverage) else f"{100 * row.target_moved_rigid_correct_transfer_coverage:.2f}%"
        arm_lines.append(
            f"| {row.arm} | {row.source_strict_scene_pose_successes}/{source_pair_count} | {coverage} | "
            f"{row.source_moved_rigid_correct_transfers}/{row.source_moved_rigid_queries} | {target_coverage} |"
        )
    gate_lines = [
        "| Arm vs global RANSAC | Coverage change | Source pairs nonnegative | Scene pose successes lost | Pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for arm, values in go_nogo.items():
        gate_lines.append(
            f"| {arm} | {values['source_moved_rigid_coverage_delta_pp']:+.2f} pp | "
            f"{values['source_relations_nonnegative_vs_baseline']}/{source_pair_count} | "
            f"{values['source_scene_pose_success_loss']}/{source_pair_count} | "
            f"{'yes' if values['passes_stage1_gate'] else 'no'} |"
        )
    run_scope = (
        "This is the full 16-source-relation plus one target run."
        if full_protocol
        else f"This is a smoke subset covering {source_pair_count} source relation(s); it is not a full-protocol result."
    )
    adaptive_note = (
        "Adaptive K expands a query from 20 to 50 descriptor candidates when its top-1 minus top-20 score gap is at or below the median relation gap from the other 15 source relations. The target uses the median across all 16 source relations."
        if full_protocol
        else "For --pairs smoke runs, source gap calibration is bypassed and the adaptive-K gap threshold is fixed to 0.0; this arm is only an execution diagnostic."
    )
    baseline_note = (
        f"The original top-20 ViT-G evaluator was compared against its saved per-relation results at {args.iterations} iterations (the saved reference uses 1,500). Exact query and inlier counts and pose errors within 1e-6 are required. Status: **{'pass' if baseline_regression_pass else 'fail'}** ({int(baseline_regression.baseline_regression_pass.sum()) if len(baseline_regression) else 0}/{len(baseline_regression)} relations match)."
    )
    report = "\n".join([
        "# 3RScan multi-motion pilot",
        "",
        run_scope,
        "This fixed-set 2x2 experiment compares global RANSAC, residual multi-motion consensus, adaptive K, and their combination. Every arm uses the same depth-valid patch queries and the same reference pool. Semantic and instance annotations are read only after descriptor matching and hypotheses are complete.",
        "",
        adaptive_note + " Rigid-instance scoring uses local PLY objectId plus the official relation mapping; PLY globalId is not an instance identifier.",
        "",
        "## Baseline regression",
        "",
        baseline_note,
        "",
        "## Results",
        "",
        *arm_lines,
        "",
        "## Stage-one gate",
        "",
        "The frozen full-stage gate requires at least +10 percentage points of source weighted moved-rigid correct-transfer coverage, nonnegative coverage against baseline on at least 10 of 16 source relations, and no more than one lost source strict scene-pose success. Smoke subsets cannot pass this full-stage gate.",
        "",
        *gate_lines,
        "",
        "This run does not establish independent-environment validity: it uses only the existing 3RScan relation set, and no additional site-verified environment was evaluated. The manuscript was not changed.",
        "",
    ])
    (args.output_dir / "multimotion_report.md").write_text(report, encoding="utf-8")
    query_sets = {
        pair: [tuple(group.query_index.astype(int)) for _, group in pair_frame.groupby("arm", sort=False)]
        for pair, pair_frame in diagnostics.groupby("pair", sort=False)
    }
    same_query_set_all_arms = all(
        len(sets) == 4 and all(query_set == sets[0] for query_set in sets[1:])
        for sets in query_sets.values()
    )
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "full_protocol": full_protocol,
        "run_scope": "16 source relations plus target" if full_protocol else f"smoke subset: {source_pair_count} source relation(s)",
        "model_size": args.model_size,
        "layers": list(layers),
        "query_population": "all depth-valid patches; evaluation subset is nonzero local objectId and NYU40 label",
        "same_query_set_all_arms": bool(same_query_set_all_arms),
        "inference_uses_annotations": False,
        "physical_instance_identity": "PLY local objectId scored through official rigid relation mapping; globalId is not used",
        "adaptive_k_rule": (
            "per-query K=50 iff top1-minus-top20 descriptor score gap <= source-only median relation gap"
            if full_protocol else "smoke subset: fixed gap threshold 0.0; no source calibration"
        ),
        "lopo_thresholds": gap_thresholds,
        "target_parameter_selection": (
            "median threshold from 16 source relations; target labels not used for tuning"
            if full_protocol else "not applicable; smoke subset contains no target relation"
        ),
        "scene_pose_gate": "rotation < 5 degrees and translation < 0.25 metres",
        "correspondence_threshold_m": args.threshold,
        "iterations": args.iterations,
        "baseline_regression_rows": int(len(baseline_regression)),
        "baseline_regression_pass": baseline_regression_pass,
        "secondary_iterations": args.secondary_iterations,
        "seed": args.seed,
        "max_modes_including_scene": args.max_modes,
        "min_secondary_mode_support": args.min_mode_support,
        "paper_modified": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(aggregate.to_string(index=False))
    if go_nogo:
        print(json.dumps(go_nogo, indent=2))


if __name__ == "__main__":
    main()
