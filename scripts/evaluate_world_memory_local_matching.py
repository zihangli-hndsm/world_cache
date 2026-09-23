"""Evaluate whether a voxel feature memory resolves local world correspondences.

This diagnostic uses current frozen features only as retrieval queries.  It is
not a selective-forward implementation or a latency measurement.  The target
for each query is the geometrically nearest cached voxel centroid among a
small, pose/depth-gated neighborhood; feature retrieval is then scored by its
rank for that target.
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
    yy, xx = torch.meshgrid(torch.arange(patch_size / 2, height, patch_size), torch.arange(patch_size / 2, width, patch_size), indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
    depth = observation["depth"][pixels[:, 1].long(), pixels[:, 0].long()]
    valid = torch.isfinite(depth) & (depth > 0)
    return backproject_pixels(pixels[valid], depth[valid], observation["intrinsics"], observation["camera_to_world"]), valid


def keys(points: torch.Tensor, voxel_size: float) -> list[tuple[int, int, int]]:
    return [tuple(item) for item in torch.floor(points / voxel_size).to(torch.int64).tolist()]


def neighboring_values(store: dict[tuple[int, int, int], list[torch.Tensor | int]], key: tuple[int, int, int], radius: int) -> list[list[torch.Tensor | int]]:
    values = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                value = store.get((key[0] + dx, key[1] + dy, key[2] + dz))
                if value is not None:
                    values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=0.15)
    parser.add_argument("--train-pairs", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.pair_run / "config.json").read_text())
    records = [json.loads(line) for line in (args.pair_run / "pairs.jsonl").read_text().splitlines() if json.loads(line)["pair_type"] != "identity"]
    train, test = records[: args.train_pairs], records[args.train_pairs :]
    device = torch.device("cuda")
    backbone = load_dinov2_vitb14(config["resolution"], device)

    stores = {layer: defaultdict(lambda: [None, 0, None]) for layer in args.layers}
    for record in train:
        observation = load(args.pair_run / record["current"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), args.layers)
        for layer in args.layers:
            for key, point, feature in zip(keys(points, args.voxel_size), points, features[layer][0, backbone.num_prefix_tokens :][valid]):
                total, count, point_total = stores[layer][key]
                stores[layer][key] = [feature.clone() if total is None else total + feature, count + 1, point.clone() if point_total is None else point_total + point]

    rows: list[dict[str, float | int]] = []
    for record in test:
        observation = load(args.pair_run / record["current"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), args.layers)
        for layer in args.layers:
            for query_index, (key, point, query) in enumerate(zip(keys(points, args.voxel_size), points, features[layer][0, backbone.num_prefix_tokens :][valid])):
                candidates = neighboring_values(stores[layer], key, args.neighbor_radius)
                candidates = [value for value in candidates if torch.linalg.vector_norm(value[2] / value[1] - point).item() <= args.max_distance]
                if not candidates:
                    continue
                centroids = torch.stack([value[2] / value[1] for value in candidates])
                descriptors = torch.stack([value[0] / value[1] for value in candidates])
                spatial_distances = torch.linalg.vector_norm(centroids - point, dim=1)
                target_index = int(torch.argmin(spatial_distances))
                scores = torch.nn.functional.cosine_similarity(descriptors, query[None], dim=1)
                order = torch.argsort(scores, descending=True)
                rank = int((order == target_index).nonzero(as_tuple=False)[0, 0]) + 1
                generator = torch.Generator().manual_seed(1_000_000 * int(record["pair_id"]) + 10_000 * layer + query_index)
                random_order = torch.randperm(len(candidates), generator=generator)
                random_rank = int((random_order == target_index).nonzero(as_tuple=False)[0, 0]) + 1
                rows.append({
                    "pair_id": int(record["pair_id"]), "layer": layer, "candidate_count": len(candidates),
                    "target_rank": rank, "random_target_rank": random_rank, "target_distance_m": float(spatial_distances[target_index]),
                    "top1_distance_m": float(spatial_distances[order[0]]), "target_cosine": float(scores[target_index]),
                    "top1_cosine": float(scores[order[0]]),
                })

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output_dir / "local_matching_metrics.parquet", index=False)
    summary = frame.groupby("layer", as_index=False).agg(
        queries=("target_rank", "count"), candidate_count=("candidate_count", "median"), recall_at_1=("target_rank", lambda values: float((values == 1).mean())),
        recall_at_3=("target_rank", lambda values: float((values <= 3).mean())), median_rank=("target_rank", "median"),
        random_recall_at_1=("random_target_rank", lambda values: float((values == 1).mean())),
        random_recall_at_3=("random_target_rank", lambda values: float((values <= 3).mean())), random_median_rank=("random_target_rank", "median"),
        median_target_distance_m=("target_distance_m", "median"), median_top1_distance_m=("top1_distance_m", "median"),
        median_target_cosine=("target_cosine", "median"), median_top1_cosine=("top1_cosine", "median"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({"status": "complete", "rows": len(frame), "config": vars(args)}, default=str, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
