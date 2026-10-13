#!/usr/bin/env python3
"""Exploratory HILO single-scene cross-view object-identity retrieval pilot.

The RGB-D frames, simplified meshes, poses, and intrinsics are expected to be
selectively extracted by the bounded audit script. Ground-truth mesh poses are
used only to make evaluation masks; the feature and matching stages do not
read object IDs. This is a feasibility/stress test, not a rescan benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2
from audit_hilo_pose_projection import CAPTURES


def dataset_path(root: Path, relative: str) -> Path:
    path = root / relative
    if path.exists():
        return path
    rel = Path(relative)
    if rel.parent.name in {"rgb_undistorted", "depth_undistorted"}:
        flat = root / f"{rel.parent.name}_{rel.name}"
    else:
        flat = root / rel.name
    if flat.exists():
        return flat
    raise FileNotFoundError(path)


def load_frame_features(model, rgb_path: Path, device: torch.device) -> np.ndarray:
    rgb = Image.open(rgb_path).convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    image = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float().div_(255).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    image = ((image - mean) / std).unsqueeze(0)
    captured = extract_block_tokens(model, image, (2, 20, 40))
    blocks = [F.normalize(captured[layer][0, model.num_prefix_tokens :], dim=1) for layer in (2, 20, 40)]
    tokens = F.normalize(torch.cat(blocks, dim=1), dim=1)
    return tokens.reshape(16, 16, -1).permute(2, 0, 1).contiguous().cpu().numpy()


def instance_masks(root: Path, ids: list[str], obj_poses: dict[str, np.ndarray],
                   camera_poses: dict, capture: str, K: np.ndarray,
                   scene_name: str) -> tuple[np.ndarray, dict]:
    depth_path = dataset_path(root, f"HILO_Scenes/{scene_name}/depth_undistorted/{capture}.png")
    depth = np.asarray(Image.open(depth_path), dtype=np.float32)
    height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    rays = np.stack(((xx - K[0, 2]) / K[0, 0],
                     (yy - K[1, 2]) / K[1, 1],
                     np.ones_like(xx)), axis=-1).astype(np.float32).reshape(-1, 3)
    ref_to_camera = np.linalg.inv(np.asarray(camera_poses[capture], dtype=np.float64))
    ray_scene = o3d.t.geometry.RaycastingScene()
    geometry_to_id = {}
    for oid in ids:
        mesh_path = dataset_path(root, f"HILO_Objects/simplified/{oid}_simplified.obj")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.transform(ref_to_camera @ obj_poses[oid])
        geometry_id = ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        geometry_to_id[geometry_id] = oid
    rays_o3d = np.concatenate((np.zeros_like(rays), rays), axis=1)
    cast = ray_scene.cast_rays(o3d.core.Tensor(rays_o3d, o3d.core.Dtype.Float32))
    z = cast["t_hit"].numpy().reshape(height, width)
    geometry = cast["geometry_ids"].numpy().reshape(height, width)
    valid = np.isfinite(z) & (z < 1e9) & (depth > 0) & (np.abs(z - depth) < 20)
    masks = np.zeros((len(ids), height, width), dtype=bool)
    area = {}
    for i, oid in enumerate(ids):
        for geometry_id, rendered_id in geometry_to_id.items():
            if rendered_id == oid:
                masks[i] = valid & (geometry == geometry_id)
                break
        area[oid] = int(masks[i].sum())
    return masks, area


def pool_features(tokens: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, list[int]]:
    # Match the main segment pipeline: bilinearly expand tokens, then average
    # over nearest-neighbor resized masks at 224x224.
    expanded = F.interpolate(
        torch.from_numpy(tokens).unsqueeze(0), size=(224, 224),
        mode="bilinear", align_corners=False,
    )[0].numpy()
    resized = []
    for mask in masks:
        img = Image.fromarray(mask.astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST)
        resized.append(np.asarray(img) > 0)
    pooled = []
    for mask in resized:
        if not mask.any():
            pooled.append(np.zeros(expanded.shape[0], dtype=np.float32))
        else:
            value = expanded[:, mask].mean(axis=1)
            pooled.append(value / max(np.linalg.norm(value), 1e-12))
    return np.stack(pooled), [int(mask.sum()) for mask in resized]


def color_histograms(rgb_path: Path, masks: np.ndarray) -> np.ndarray:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    descriptors = []
    for mask in masks:
        channels = [np.histogram(rgb[:, :, c][mask], bins=16, range=(0, 256))[0]
                    for c in range(3)]
        value = np.concatenate(channels).astype(np.float32)
        value /= max(value.sum(), 1.0)
        value /= max(np.linalg.norm(value), 1e-12)
        descriptors.append(value)
    return np.stack(descriptors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/tmp/hilo_pilot"))
    parser.add_argument("--scene", action="append", help="repeat for multi-scene runs")
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/20261009_hilo_pose_projection/identity_pilot.json"))
    args = parser.parse_args()
    root = args.data_root
    scene_names = args.scene or ["scene_08_s1_c1"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2("giant", 224, device)
    reference = CAPTURES[0]
    rows, areas, patch_counts, query_outcomes = [], {}, {}, []
    for scene_name in scene_names:
        scene_prefix = f"HILO_Scenes/{scene_name}/"
        ids = json.loads(dataset_path(root, scene_prefix + "object_id_list.json").read_text())
        poses = {item["Model"]: np.asarray(item["trans"], dtype=np.float64)
                 for item in json.loads(dataset_path(root, scene_prefix + "object_pose_wrt_arc0_image7.json").read_text())}
        camera_poses = json.loads(dataset_path(root, scene_prefix + "camera_poses.json").read_text())
        features, colors = {}, {}
        for capture in CAPTURES:
            arc = int(capture[3])
            param_name = "cam_params_arc0123.json" if arc < 4 else "cam_params_arc4567.json"
            param_path = dataset_path(root, "HILO_Scenes/" + param_name)
            K = np.asarray(json.loads(param_path.read_text())["depth_K"], dtype=np.float64)
            masks, areas[f"{scene_name}/{capture}"] = instance_masks(root, ids, poses, camera_poses, capture, K, scene_name)
            rgb_path = dataset_path(root, scene_prefix + f"rgb_undistorted/{capture}.png")
            tokens = load_frame_features(model, rgb_path, device)
            features[capture], patch_counts[f"{scene_name}/{capture}"] = pool_features(tokens, masks)
            colors[capture] = color_histograms(rgb_path, masks)

        for query in CAPTURES[1:]:
            scores = features[query] @ features[reference].T
            color_scores = colors[query] @ colors[reference].T
            pred_top1 = np.argmax(scores, axis=1)
            mutual_ref = np.argmax(scores, axis=0)
            mutual = mutual_ref[pred_top1] == np.arange(len(ids))
            color_pred = np.argmax(color_scores, axis=1)
            color_mutual = np.argmax(color_scores, axis=0)[color_pred] == np.arange(len(ids))
            row_ind, col_ind = linear_sum_assignment(-scores)
            hungarian_pred = col_ind[np.argsort(row_ind)]
            dino_correct = np.asarray([ids[pred_top1[i]] == ids[i] for i in range(len(ids))])
            color_correct = color_pred == np.arange(len(ids))
            for i, oid in enumerate(ids):
                query_outcomes.append({
                    "scene": scene_name,
                    "capture": query,
                    "arc": int(query[3]),
                    "object_id": oid,
                    "dino_top1_correct": bool(dino_correct[i]),
                    "dino_top1_cosine": float(scores[i, pred_top1[i]]),
                    "dino_top1_margin": float(scores[i, pred_top1[i]] - np.partition(scores[i], -2)[-2]),
                    "dino_mutual_accepted": bool(mutual[i]),
                    "dino_mutual_correct": bool(mutual[i] and dino_correct[i]),
                    "rgb_histogram_top1_correct": bool(color_correct[i]),
                    "rgb_histogram_mutual_accepted": bool(color_mutual[i]),
                    "rgb_histogram_mutual_correct": bool(color_mutual[i] and color_correct[i]),
                })
            rows.append({
                "scene": scene_name,
                "reference": reference,
                "query": query,
                "object_queries": len(ids),
                "top1_correct": int(dino_correct.sum()),
                "top1_accuracy": float(dino_correct.mean()),
                "mutual_accepts": int(mutual.sum()),
                "mutual_correct": int((mutual & dino_correct).sum()),
                "mutual_precision": float(dino_correct[mutual].mean()) if mutual.any() else None,
                "mean_correct_identity_cosine": float(np.mean([scores[i, i] for i in range(len(ids))])),
                "mean_best_wrong_cosine": float(np.mean([np.max(np.delete(scores[i], i)) for i in range(len(ids))])),
                "hungarian_correct": int(sum(ids[hungarian_pred[i]] == ids[i] for i in range(len(ids)))),
                "rgb_histogram_top1_correct": int(color_correct.sum()),
                "rgb_histogram_mutual_accepts": int(color_mutual.sum()),
                "rgb_histogram_mutual_correct": int((color_mutual & color_correct).sum()),
            })
    scene_rows = []
    for scene_name in scene_names:
        subset = [row for row in rows if row["scene"] == scene_name]
        q = sum(row["object_queries"] for row in subset)
        mutual_n = sum(row["mutual_accepts"] for row in subset)
        mutual_c = sum(row["mutual_correct"] for row in subset)
        color_n = sum(row["rgb_histogram_mutual_accepts"] for row in subset)
        color_c = sum(row["rgb_histogram_mutual_correct"] for row in subset)
        scene_rows.append({
            "scene": scene_name,
            "queries": q,
            "dino_top1_accuracy": sum(row["top1_correct"] for row in subset) / q,
            "dino_mutual_coverage": mutual_n / q,
            "dino_mutual_precision": mutual_c / mutual_n if mutual_n else None,
            "dino_hungarian_accuracy": sum(row["hungarian_correct"] for row in subset) / q,
            "rgb_histogram_top1_accuracy": sum(row["rgb_histogram_top1_correct"] for row in subset) / q,
            "rgb_histogram_mutual_coverage": color_n / q,
            "rgb_histogram_mutual_precision": color_c / color_n if color_n else None,
        })
    total_q = sum(row["object_queries"] for row in rows)
    total_mutual = sum(row["mutual_accepts"] for row in rows)
    total_mutual_correct = sum(row["mutual_correct"] for row in rows)
    total_color_mutual = sum(row["rgb_histogram_mutual_accepts"] for row in rows)
    total_color_correct = sum(row["rgb_histogram_mutual_correct"] for row in rows)
    summary = {
        "dataset": "HILO", "scenes": scene_names, "reference": reference,
        "backbone": "DINOv2 ViT-G, frozen, blocks 2/20/40, 224x224",
        "descriptor": "normalized block tokens concatenated; bilinear interpolation to 224x224; average pool over depth-validated mesh-rendered masks",
        "matching": "independent per-query cosine top-1; mutual nearest-neighbor reported as selective diagnostic",
        "identity_labels_used_during_features_or_matching": False,
        "ground_truth_mesh_poses_used_for_evaluation_masks": True,
        "mask_depth_tolerance_mm": 20,
        "visible_mask_pixels_by_frame": areas,
        "feature_patch_counts_by_frame": patch_counts,
        "scene_macro": scene_rows,
        "pooled": {
            "object_queries": total_q,
            "dino_top1_correct": sum(row["top1_correct"] for row in rows),
            "dino_top1_accuracy": sum(row["top1_correct"] for row in rows) / total_q,
            "dino_mutual_correct": total_mutual_correct,
            "dino_mutual_accepts": total_mutual,
            "dino_mutual_precision": total_mutual_correct / total_mutual if total_mutual else None,
            "rgb_histogram_top1_correct": sum(row["rgb_histogram_top1_correct"] for row in rows),
            "rgb_histogram_top1_accuracy": sum(row["rgb_histogram_top1_correct"] for row in rows) / total_q,
            "rgb_histogram_mutual_correct": total_color_correct,
            "rgb_histogram_mutual_accepts": total_color_mutual,
            "rgb_histogram_mutual_precision": total_color_correct / total_color_mutual if total_color_mutual else None,
        },
        "results": rows,
        "query_outcomes": query_outcomes,
        "scope": f"{len(scene_names)} hash-selected controlled tabletop scene configurations; exploratory object-centric view transfer, not room-rescan validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
