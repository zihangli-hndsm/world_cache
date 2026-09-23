"""Use top-K pose-free descriptor candidates with spatial-consensus RANSAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from evaluate_3rscan_framelevel_layers import combine, normalized_tokens
from evaluate_3rscan_instance_correspondence import (
    load,
    patch_world_points,
    preprocess,
    read_annotated_mesh,
    transform_points,
)
from evaluate_3rscan_pose_free_registration import rigid_fit, transform
from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2


def fpfh_descriptors(all_points: np.ndarray, selected_points: np.ndarray) -> torch.Tensor:
    """Map rigid-invariant local FPFH geometry descriptors to token points."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - optional research branch
        raise ImportError("--geometry-weights requires open3d==0.18.0") from exc
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    cloud = cloud.voxel_down_sample(0.08)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.16, max_nn=30))
    cloud.normalize_normals()
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.40, max_nn=100),
    )
    downsampled = np.asarray(cloud.points, dtype=np.float64)
    nearest = cKDTree(downsampled).query(selected_points, k=1)[1]
    values = np.asarray(feature.data, dtype=np.float32).T[nearest]
    return torch.nn.functional.normalize(torch.from_numpy(values), dim=1)


def consensus(
    source: np.ndarray,
    candidate_points: np.ndarray,
    candidate_indices: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
    assignment: str = "nearest",
    verification_weight: float = 0.0,
    verification_threshold: float | None = None,
    diversity_weight: float = 0.0,
    group_ids: np.ndarray | None = None,
    group_cap: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if assignment not in {"nearest", "unique"}:
        raise ValueError(f"unknown assignment mode: {assignment}")
    if group_cap < 0:
        raise ValueError("group_cap must be non-negative")
    if group_cap and group_ids is None:
        raise ValueError("group_ids are required when group_cap is non-zero")

    def select(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nearest = np.argmin(distances, axis=1)
        if assignment == "nearest":
            selected_distances = distances[np.arange(len(distances)), nearest]
            return nearest, selected_distances <= threshold
        # A query-to-reference patch correspondence should not explain many
        # source patches with the same reference patch.  Greedy assignment by
        # residual preserves the best-supported matches while making this
        # anti-collapse constraint explicit and deterministic.
        order = np.argsort(distances[np.arange(len(distances)), nearest])
        used: set[int] = set()
        assigned = np.full(len(distances), -1, dtype=np.int64)
        for query_index in order:
            for slot in np.argsort(distances[query_index]):
                slot = int(slot)
                candidate_index = int(candidate_indices[query_index, slot])
                if candidate_index not in used:
                    assigned[query_index] = slot
                    used.add(candidate_index)
                    break
        valid = assigned >= 0
        selected_distances = np.full(len(distances), np.inf, dtype=np.float64)
        selected_distances[valid] = distances[np.arange(len(distances))[valid], assigned[valid]]
        return assigned, selected_distances <= threshold

    generator = np.random.default_rng(seed)
    verification_threshold = threshold if verification_threshold is None else verification_threshold
    verification_tree = cKDTree(candidate_points) if verification_weight else None
    best_score = -np.inf

    def spread(points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
        # 3-D RGB-D surfaces are often close to planar; the product of the
        # two largest principal spreads measures occupied scene area without
        # requiring a volumetric cloud.
        return float(np.sqrt(max(0.0, eigenvalues[-1] * eigenvalues[-2])))

    full_spread = max(spread(source), np.finfo(np.float64).eps)
    query_count, top_k = candidate_indices.shape
    best_matrix = np.eye(4, dtype=np.float64)
    best_assignments = np.zeros(query_count, dtype=np.int64)
    best_inliers = np.zeros(query_count, dtype=bool)
    for _ in range(iterations):
        sample_queries = generator.choice(query_count, size=3, replace=False)
        sample_choices = generator.integers(0, top_k, size=3)
        sample_indices = candidate_indices[sample_queries, sample_choices]
        matrix = rigid_fit(source[sample_queries], candidate_points[sample_indices])
        transformed = transform(source, matrix)
        distances = np.linalg.norm(transformed[:, None, :] - candidate_points[candidate_indices], axis=2)
        assignments, inliers = select(distances)
        score = int(inliers.sum())
        if group_cap:
            group_counts = np.bincount(group_ids[inliers].astype(np.int64))
            score = float(np.minimum(group_counts, group_cap).sum())
        if diversity_weight:
            support_ratio = min(1.0, spread(source[inliers]) / full_spread)
            score *= (1.0 - diversity_weight) + diversity_weight * support_ratio * support_ratio
        if verification_tree is not None:
            geometric_distances = verification_tree.query(transform(source, matrix), k=1)[0]
            score += verification_weight * int((geometric_distances <= verification_threshold).sum())
        if score > best_score:
            best_matrix = matrix
            best_assignments = assignments
            best_inliers = inliers
            best_score = score
    if best_inliers.sum() >= 3:
        selected_indices = candidate_indices[np.arange(query_count), best_assignments]
        best_matrix = rigid_fit(source[best_inliers], candidate_points[selected_indices[best_inliers]])
        transformed = transform(source, best_matrix)
        distances = np.linalg.norm(transformed[:, None, :] - candidate_points[candidate_indices], axis=2)
        best_assignments, best_inliers = select(distances)
    selected_indices = candidate_indices[np.arange(query_count), best_assignments]
    return best_matrix, selected_indices, best_inliers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("small", "base", "large", "giant"), default="large")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 12, 24])
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument(
        "--descriptor-control",
        choices=("normal", "permuted_reference"),
        default="normal",
        help="optionally permute reference descriptor assignments as a negative control",
    )
    parser.add_argument("--assignment", choices=("nearest", "unique"), default="nearest")
    parser.add_argument("--verification-weight", type=float, default=0.0)
    parser.add_argument("--verification-threshold", type=float)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--geometry-weights", type=float, nargs="+", default=[0.0])
    parser.add_argument("--group-cap", type=int, default=0)
    parser.add_argument("--save-query-diagnostics", action="store_true")
    parser.add_argument(
        "--seed-sweep",
        type=int,
        nargs="+",
        help="run an optional post-hoc RANSAC seed sweep; default evaluation remains seed 17",
    )
    parser.add_argument(
        "--save-candidate-set-audit",
        action="store_true",
        help="save post-hoc semantic/instance hit indicators for each top-K candidate set",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(sorted(set(args.layers)))
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("pair manifest must store native scan coordinates; run migrate_3rscan_rescan_coordinates.py")
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2(args.model_size, config["resolution"], device)
    reference_tree, _, reference_global_ids, reference_semantic_ids, _ = read_annotated_mesh(args.annotation_root, config["reference"])
    rescan_tree, _, rescan_global_ids, rescan_semantic_ids, _ = read_annotated_mesh(args.annotation_root, config["rescan"])

    reference_features = {layer: [] for layer in layers}
    reference_points, reference_ids, reference_semantics = [], [], []
    reference_all_points = []
    for entry in manifest["reference"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        reference_all_points.append(points.numpy().astype(np.float64))
        features = normalized_tokens(backbone, observation, layers, device, valid)
        _, nearest = reference_tree.query(points.numpy(), k=1)
        ids, semantics = reference_global_ids[nearest], reference_semantic_ids[nearest]
        for layer in layers:
            # Annotation labels are never used to construct the matching
            # candidate set. They are carried only for post-hoc scoring.
            reference_features[layer].append(features[layer])
        reference_points.append(points.numpy().astype(np.float64))
        reference_ids.append(ids)
        reference_semantics.append(semantics)

    query_features = {layer: [] for layer in layers}
    query_points_native, query_ids, query_semantics, query_frame_ids = [], [], [], []
    query_all_points_native = []
    rescan_to_reference = np.asarray(config["rescan_to_reference"], dtype=np.float64)
    for entry in manifest["rescan"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = normalized_tokens(backbone, observation, layers, device, valid)
        points_native = points.numpy().astype(np.float64)
        query_all_points_native.append(points_native)
        _, nearest = rescan_tree.query(points_native, k=1)
        ids, semantics = rescan_global_ids[nearest], rescan_semantic_ids[nearest]
        for layer in layers:
            # Matching uses depth-valid points, not annotation availability.
            query_features[layer].append(features[layer])
        query_points_native.append(points_native)
        query_ids.append(ids)
        query_semantics.append(semantics)
        query_frame_ids.append(np.full(len(points_native), int(entry["frame_id"]), dtype=np.int64))

    reference_points = np.concatenate(reference_points, axis=0)
    reference_all_points = np.concatenate(reference_all_points, axis=0)
    reference_ids = np.concatenate(reference_ids, axis=0)
    reference_semantics = np.concatenate(reference_semantics, axis=0)
    query_points_native = np.concatenate(query_points_native, axis=0)
    query_all_points_native = np.concatenate(query_all_points_native, axis=0)
    query_ids = np.concatenate(query_ids, axis=0)
    query_semantics = np.concatenate(query_semantics, axis=0)
    query_frame_ids = np.concatenate(query_frame_ids, axis=0)
    query_evaluable = (query_ids != 0) & (query_semantics != 0)
    for layer in layers:
        reference_features[layer] = torch.cat(reference_features[layer], dim=0)
        query_features[layer] = torch.cat(query_features[layer], dim=0)
    reference_descriptor = combine(reference_features, layers)
    query_descriptor = combine(query_features, layers)
    descriptor_order = np.arange(len(reference_descriptor), dtype=np.int64)
    if args.descriptor_control == "permuted_reference":
        descriptor_order = np.random.default_rng(17).permutation(descriptor_order)
    descriptor_order_tensor = torch.from_numpy(descriptor_order).to(reference_descriptor.device)
    reference_matching_descriptor = reference_descriptor[descriptor_order_tensor]
    reference_geometry_tree = cKDTree(reference_points)
    query_geometry_tree = cKDTree(query_points_native)
    geometry_features = None
    if any(weight != 0.0 for weight in args.geometry_weights):
        geometry_features = (
            fpfh_descriptors(reference_all_points, reference_points),
            fpfh_descriptors(query_all_points_native, query_points_native),
        )

    rows = []
    query_diagnostics = []
    for geometry_weight in args.geometry_weights:
        if geometry_weight == 0.0:
            reference_matching = reference_matching_descriptor
            query_matching = query_descriptor
        else:
            reference_geometry, query_geometry = geometry_features
            reference_matching = torch.nn.functional.normalize(
                torch.cat([reference_matching_descriptor, geometry_weight * reference_geometry[descriptor_order]], dim=1), dim=1
            )
            query_matching = torch.nn.functional.normalize(
                torch.cat([query_descriptor, geometry_weight * query_geometry], dim=1), dim=1
            )
        scores = query_matching @ reference_matching.T
        for top_k in args.top_k:
            top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
            # The descriptor rows are deliberately decoupled from the point
            # rows for the negative control.  Candidate positions therefore
            # remain the original reference-point indices: the permutation is
            # not undone here.
            candidate_indices = top_indices.numpy()
            seeds = args.seed_sweep if args.seed_sweep is not None else [17]
            for ransac_seed in seeds:
                matrix, selected, inliers = consensus(
                    query_points_native,
                    reference_points,
                    candidate_indices,
                    threshold=args.threshold,
                    iterations=args.iterations,
                    seed=ransac_seed,
                    assignment=args.assignment,
                    verification_weight=args.verification_weight,
                    verification_threshold=args.verification_threshold,
                    diversity_weight=args.diversity_weight,
                    group_ids=query_frame_ids,
                    group_cap=args.group_cap,
                )
                residuals = np.linalg.norm(transform(query_points_native, matrix) - reference_points[selected], axis=1)
                evaluated_inliers = inliers & query_evaluable
                evaluated_query_count = int(query_evaluable.sum())
                transformed_query = transform(query_points_native, matrix)
                forward_geometry_distances = reference_geometry_tree.query(transformed_query, k=1)[0]
                reverse_geometry_distances = query_geometry_tree.query(
                    transform(reference_points, np.linalg.inv(matrix)), k=1
                )[0]
                rotation_error = Rotation.from_matrix(matrix[:3, :3].T @ rescan_to_reference[:3, :3]).magnitude()
                translation_error = np.linalg.norm(matrix[:3, 3] - rescan_to_reference[:3, 3])
                row = {
                    "geometry_weight": geometry_weight,
                    "top_k": top_k,
                    # The reported denominator is the annotated, depth-valid
                    # query subset. It is not used by matching; it defines only
                    # the post-hoc evaluation population.
                    "queries": evaluated_query_count,
                    "matching_query_points": len(selected),
                    "matching_reference_points": len(reference_points),
                    "matching_inliers": int(inliers.sum()),
                    "inliers": int(evaluated_inliers.sum()),
                    "inlier_coverage": float(evaluated_inliers.sum() / evaluated_query_count) if evaluated_query_count else 0.0,
                    "inlier_instance_accuracy": float(np.mean(reference_ids[selected[evaluated_inliers]] == query_ids[evaluated_inliers])) if evaluated_inliers.any() else float("nan"),
                    "inlier_semantic_accuracy": float(np.mean(reference_semantics[selected[evaluated_inliers]] == query_semantics[evaluated_inliers])) if evaluated_inliers.any() else float("nan"),
                    "median_inlier_residual_m": float(np.median(residuals[evaluated_inliers])) if evaluated_inliers.any() else float("nan"),
                    "forward_geometry_overlap": float(np.mean(forward_geometry_distances <= args.threshold)),
                    "reverse_geometry_overlap": float(np.mean(reverse_geometry_distances <= args.threshold)),
                    "bidirectional_geometry_overlap": float(
                        0.5 * (np.mean(forward_geometry_distances <= args.threshold) +
                                np.mean(reverse_geometry_distances <= args.threshold))
                    ),
                    "rotation_error_deg": float(np.degrees(rotation_error)),
                    "translation_error_m": float(translation_error),
                }
                if args.seed_sweep is not None:
                    row["ransac_seed"] = int(ransac_seed)
                rows.append(row)
                if args.save_query_diagnostics:
                    top1 = top_scores[:, 0].cpu().numpy()
                    margin = (top_scores[:, 0] - top_scores[:, min(1, top_scores.shape[1] - 1)]).cpu().numpy()
                    selected_scores = scores[
                        torch.arange(len(selected)), torch.from_numpy(selected).long()
                    ].detach().cpu().numpy()
                    diagnostic = {
                        "geometry_weight": geometry_weight,
                        "top_k": top_k,
                        "query_index": np.arange(len(selected), dtype=np.int64),
                        "frame_id": query_frame_ids,
                        "top1_descriptor_score": top1,
                        "top1_top2_margin": margin,
                        "selected_descriptor_score": selected_scores,
                        "spatial_residual_m": residuals,
                        "consensus_inlier": inliers,
                        "query_evaluable": query_evaluable,
                        "query_semantic_id": query_semantics,
                        "selected_reference_semantic_id": reference_semantics[selected],
                        "query_instance_id": query_ids,
                        "selected_reference_instance_id": reference_ids[selected],
                        "semantic_correct": reference_semantics[selected] == query_semantics,
                        "instance_correct": reference_ids[selected] == query_ids,
                    }
                    if args.seed_sweep is not None:
                        diagnostic["ransac_seed"] = int(ransac_seed)
                    if args.save_candidate_set_audit:
                        # These fields are strictly post-hoc: candidate_indices
                        # were constructed from descriptors before any labels
                        # entered the computation.  They quantify whether the
                        # correct semantic/instance identity was available to
                        # consensus at all.
                        candidate_semantics = reference_semantics[candidate_indices]
                        candidate_instances = reference_ids[candidate_indices]
                        semantic_matches = candidate_semantics == query_semantics[:, None]
                        instance_matches = candidate_instances == query_ids[:, None]
                        diagnostic.update({
                            "candidate_semantic_count": semantic_matches.sum(axis=1).astype(np.int64),
                            "candidate_instance_count": instance_matches.sum(axis=1).astype(np.int64),
                            "candidate_semantic_hit": semantic_matches.any(axis=1) & query_evaluable,
                            "candidate_instance_hit": instance_matches.any(axis=1) & query_evaluable,
                        })
                    query_diagnostics.append(pd.DataFrame(diagnostic))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    if query_diagnostics:
        pd.concat(query_diagnostics, ignore_index=True).to_csv(args.output_dir / "query_diagnostics.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "layers": list(layers),
        "known_alignment_used_for_selection": False,
        "threshold_m": args.threshold,
        "assignment": args.assignment,
        "verification_weight": args.verification_weight,
        "verification_threshold_m": args.verification_threshold,
        "diversity_weight": args.diversity_weight,
        "geometry_weights": args.geometry_weights,
        "group_cap": args.group_cap,
        "descriptor_control": args.descriptor_control,
        "candidate_set_audit_posthoc": bool(args.save_candidate_set_audit),
        "seed_sweep_posthoc": args.seed_sweep is not None,
        "ransac_seeds": args.seed_sweep if args.seed_sweep is not None else [17],
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
