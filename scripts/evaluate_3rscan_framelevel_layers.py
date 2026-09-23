"""Compare frame-level, pose-free DINO correspondence across ViT layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from evaluate_3rscan_instance_correspondence import (
    load,
    patch_world_points,
    preprocess,
    read_annotated_mesh,
    transform_points,
)
from evaluate_3rscan_pose_free_registration import ransac, transform
from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2


def normalized_tokens(backbone, observation, layers, device, valid):
    captured = extract_block_tokens(backbone, preprocess(observation["rgb"], device), layers)
    result = {}
    for layer in layers:
        tokens = captured[layer][0, backbone.num_prefix_tokens:].cpu()[valid]
        result[layer] = torch.nn.functional.normalize(tokens, dim=1)
    return result


def combine(features: dict[int, torch.Tensor], layers: tuple[int, ...]) -> torch.Tensor:
    if len(layers) == 1:
        return features[layers[0]]
    return torch.nn.functional.normalize(torch.cat([features[layer] for layer in layers], dim=1), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-size", choices=("small", "base", "large", "giant"), default="large")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 6, 12, 18, 24])
    parser.add_argument("--margin-thresholds", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument("--ransac-threshold", type=float, default=0.10)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(sorted(set(args.layers)))
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("pair manifest must store native scan coordinates; run migrate_3rscan_rescan_coordinates.py")
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2(args.model_size, config["resolution"], device)
    reference_tree, _, reference_global_ids, reference_semantic_ids, _ = read_annotated_mesh(
        args.annotation_root, config["reference"]
    )
    rescan_tree, _, rescan_global_ids, rescan_semantic_ids, _ = read_annotated_mesh(
        args.annotation_root, config["rescan"]
    )

    reference_features = {layer: [] for layer in layers}
    reference_points = []
    reference_ids = []
    reference_semantics = []
    for entry in manifest["reference"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = normalized_tokens(backbone, observation, layers, device, valid)
        _, nearest = reference_tree.query(points.numpy(), k=1)
        global_ids = reference_global_ids[nearest]
        semantics = reference_semantic_ids[nearest]
        for layer in layers:
            # Annotation labels are carried for scoring only. Every
            # depth-valid reference patch remains in the candidate pool.
            reference_features[layer].append(features[layer])
        reference_points.append(points.numpy().astype(np.float64))
        reference_ids.append(global_ids)
        reference_semantics.append(semantics)

    query_features = {layer: [] for layer in layers}
    query_points_ref = []
    query_points_native = []
    query_ids = []
    query_semantics = []
    rescan_to_reference = np.asarray(config["rescan_to_reference"], dtype=np.float64)
    for entry in manifest["rescan"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = normalized_tokens(backbone, observation, layers, device, valid)
        points_native = points.numpy().astype(np.float64)
        _, nearest = rescan_tree.query(points_native, k=1)
        global_ids = rescan_global_ids[nearest]
        semantics = rescan_semantic_ids[nearest]
        for layer in layers:
            # Matching is independent of annotation availability; labels are
            # used only after matching to define the evaluable query subset.
            query_features[layer].append(features[layer])
        query_points_ref.append(points_native)
        query_points_native.append(points_native)
        query_ids.append(global_ids)
        query_semantics.append(semantics)

    reference_points = np.concatenate(reference_points, axis=0)
    reference_ids = np.concatenate(reference_ids, axis=0)
    reference_semantics = np.concatenate(reference_semantics, axis=0)
    query_points_ref = np.concatenate(query_points_ref, axis=0)
    query_points_native = np.concatenate(query_points_native, axis=0)
    query_ids = np.concatenate(query_ids, axis=0)
    query_semantics = np.concatenate(query_semantics, axis=0)
    query_evaluable = (query_ids != 0) & (query_semantics != 0)
    evaluation_queries = int(query_evaluable.sum())
    for layer in layers:
        reference_features[layer] = torch.cat(reference_features[layer], dim=0)
        query_features[layer] = torch.cat(query_features[layer], dim=0)

    specifications = [(f"layer_{layer}", (layer,)) for layer in layers]
    if len(layers) >= 3:
        specifications.append(("layers_" + "_".join(map(str, (layers[0], layers[len(layers) // 2], layers[-1]))), (layers[0], layers[len(layers) // 2], layers[-1])))
    rows = []
    for name, selected_layers in specifications:
        reference_descriptor = combine(reference_features, selected_layers)
        query_descriptor = combine(query_features, selected_layers)
        scores = query_descriptor @ reference_descriptor.T
        top_scores, top_indices = torch.topk(scores, k=2, dim=1)
        selected = top_indices[:, 0].numpy()
        margin = (top_scores[:, 0] - top_scores[:, 1]).numpy()
        cache_to_query = torch.argmax(scores, dim=0).numpy()
        reciprocal = cache_to_query[selected] == np.arange(len(selected))
        masks = {
            "one_way": np.ones(len(selected), dtype=bool),
            "mutual": reciprocal,
        }
        for margin_threshold in args.margin_thresholds:
            masks[f"mutual_margin_{margin_threshold:g}"] = reciprocal & (margin >= margin_threshold)
        for method, mask in masks.items():
            count = int(mask.sum())
            evaluated = mask & query_evaluable
            evaluated_count = int(evaluated.sum())
            if count >= 3:
                source = query_points_native[mask]
                target = reference_points[selected[mask]]
                estimated, inliers = ransac(source, target, seed=17, threshold=args.ransac_threshold, iterations=args.ransac_iterations)
                residuals = np.linalg.norm(transform(source, estimated) - target, axis=1)
                rotation_error = Rotation.from_matrix(estimated[:3, :3].T @ rescan_to_reference[:3, :3]).magnitude()
                translation_error = np.linalg.norm(estimated[:3, 3] - rescan_to_reference[:3, 3])
                inlier_rate = float(inliers.mean())
                median_residual = float(np.median(residuals))
            else:
                inlier_rate = median_residual = rotation_error = translation_error = float("nan")
            rows.append({
                "feature": name,
                "method": method,
                "matches": count,
                "evaluated_matches": evaluated_count,
                "evaluation_queries": evaluation_queries,
                "coverage": float(evaluated_count / evaluation_queries) if evaluation_queries else 0.0,
                "instance_accuracy": float(np.mean(reference_ids[selected[evaluated]] == query_ids[evaluated])) if evaluated_count else float("nan"),
                "semantic_accuracy": float(np.mean(reference_semantics[selected[evaluated]] == query_semantics[evaluated])) if evaluated_count else float("nan"),
                "ransac_inlier_rate": inlier_rate,
                "median_registration_residual_m": median_residual,
                "rotation_error_deg": float(np.degrees(rotation_error)),
                "translation_error_m": float(translation_error),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "reference_patches": len(reference_points),
        "query_patches": len(query_points_ref),
        "layers": list(layers),
        "frame_level": True,
        "known_alignment_used_for_selection": False,
        "matching_uses_annotations": False,
        "evaluation_population": "nonzero_query_instance_and_semantic_labels",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
