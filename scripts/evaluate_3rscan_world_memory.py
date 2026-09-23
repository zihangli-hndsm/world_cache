"""Evaluate world-aligned descriptor retrieval across a 3RScan revisit pair.

Reference-scan frames build the cache and rescan frames are queries.  The
rescan poses must already be transformed into the reference coordinate frame
by ``generate_3rscan_pairs.py``.  This is a descriptor/local-matching metric;
it does not execute a partial ViT or claim latency savings.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2_vitb14
from worldcache.geometry.projection import backproject_pixels


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((image - mean) / std).unsqueeze(0).to(device)


def patch_world_points(observation: dict[str, torch.Tensor], patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = observation["depth"].shape
    yy, xx = torch.meshgrid(
        torch.arange(patch_size / 2, height, patch_size),
        torch.arange(patch_size / 2, width, patch_size),
        indexing="ij",
    )
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
    depth = observation["depth"][pixels[:, 1].long(), pixels[:, 0].long()]
    valid = torch.isfinite(depth) & (depth > 0)
    return backproject_pixels(pixels[valid], depth[valid], observation["intrinsics"], observation["camera_to_world"]), valid


def voxel_key(point: torch.Tensor, voxel_size: float) -> tuple[int, int, int]:
    return tuple(torch.floor(point / voxel_size).to(torch.int64).tolist())


def neighbors(store: dict[tuple[int, int, int], list[torch.Tensor | int]], key: tuple[int, int, int], radius: int) -> list[list[torch.Tensor | int]]:
    values = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                value = store.get((key[0] + dx, key[1] + dy, key[2] + dz))
                if value is not None:
                    values.append(value)
    return values


def feature_frames(run_dir: Path, entries: list[dict[str, str]], backbone: torch.nn.Module, layers: list[int], device: torch.device):
    for entry in entries:
        observation = load(run_dir / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), layers)
        yield points, valid, {layer: features[layer][0, backbone.num_prefix_tokens :] for layer in layers}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4])
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=0.15)
    args = parser.parse_args()
    if args.voxel_size <= 0 or args.neighbor_radius < 0 or args.max_distance <= 0:
        raise ValueError("voxel size and distance must be positive; radius must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    backbone = load_dinov2_vitb14(config["resolution"], device)

    stores = {layer: defaultdict(lambda: [None, 0, None]) for layer in args.layers}
    reference_valid = 0
    for points, valid, features in feature_frames(args.pair_run, manifest["reference"], backbone, args.layers, device):
        reference_valid += int(valid.sum())
        for layer in args.layers:
            for point, feature in zip(points, features[layer][valid]):
                key = voxel_key(point, args.voxel_size)
                total, count, point_total = stores[layer][key]
                stores[layer][key] = [
                    feature.clone() if total is None else total + feature,
                    count + 1,
                    point.clone() if point_total is None else point_total + point,
                ]

    rows = []
    query_valid = 0
    for frame_index, (points, valid, features) in enumerate(feature_frames(args.pair_run, manifest["rescan"], backbone, args.layers, device)):
        query_valid += int(valid.sum())
        for layer in args.layers:
            for query_index, (point, query) in enumerate(zip(points, features[layer][valid])):
                candidates = neighbors(stores[layer], voxel_key(point, args.voxel_size), args.neighbor_radius)
                candidates = [
                    value for value in candidates
                    if torch.linalg.vector_norm(value[2] / value[1] - point).item() <= args.max_distance
                ]
                if not candidates:
                    continue
                centroids = torch.stack([value[2] / value[1] for value in candidates])
                descriptors = torch.stack([value[0] / value[1] for value in candidates])
                distances = torch.linalg.vector_norm(centroids - point, dim=1)
                target = int(torch.argmin(distances))
                scores = torch.nn.functional.cosine_similarity(descriptors, query[None], dim=1)
                order = torch.argsort(scores, descending=True)
                rank = int((order == target).nonzero(as_tuple=False)[0, 0]) + 1
                seed = 1_000_000 * frame_index + 10_000 * layer + query_index
                random_order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed))
                random_rank = int((random_order == target).nonzero(as_tuple=False)[0, 0]) + 1
                rows.append({
                    "frame_index": frame_index,
                    "query_index": query_index,
                    "layer": layer,
                    "candidate_count": len(candidates),
                    "target_rank": rank,
                    "random_target_rank": random_rank,
                    "target_distance_m": float(distances[target]),
                    "top1_distance_m": float(distances[order[0]]),
                    "target_cosine": float(scores[target]),
                    "top1_cosine": float(scores[order[0]]),
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no rescan queries passed the geometric cache gate")
    frame.to_parquet(args.output_dir / "local_matching_metrics.parquet", index=False)
    summary = frame.groupby("layer", as_index=False).agg(
        queries=("target_rank", "count"),
        candidate_count=("candidate_count", "median"),
        recall_at_1=("target_rank", lambda values: float((values == 1).mean())),
        recall_at_3=("target_rank", lambda values: float((values <= 3).mean())),
        random_recall_at_1=("random_target_rank", lambda values: float((values == 1).mean())),
        random_recall_at_3=("random_target_rank", lambda values: float((values <= 3).mean())),
        median_target_distance_m=("target_distance_m", "median"),
        median_top1_distance_m=("top1_distance_m", "median"),
        median_target_cosine=("target_cosine", "median"),
        median_top1_cosine=("top1_cosine", "median"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metrics = {
        "status": "complete",
        "reference_valid_patches": reference_valid,
        "rescan_valid_patches": query_valid,
        "geometrically_gated_queries": len(frame),
        "cache_hit_rate": len(frame) / query_valid,
        "config": vars(args),
    }
    (args.output_dir / "status.json").write_text(json.dumps(metrics, default=str, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metrics, default=str, indent=2))


if __name__ == "__main__":
    main()
