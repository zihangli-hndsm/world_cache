"""Explore 3D voxel-aggregated feature memory on held-out ReplicaCAD views.

This is a descriptor-memory diagnostic, not a selective-forward implementation.
It tests whether multiple world-aligned observations yield a more view-stable
token representation than one view-specific cached hidden state.
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
    world = backproject_pixels(pixels[valid], depth[valid], observation["intrinsics"], observation["camera_to_world"])
    return world, valid


def keys(points: torch.Tensor, voxel_size: float) -> list[tuple[int, int, int]]:
    return [tuple(item) for item in torch.floor(points / voxel_size).to(torch.int64).tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--voxel-sizes", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--neighbor-radii", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--max-distances",
        type=float,
        nargs="*",
        default=[],
        help="Optional maximum query-to-cache-centroid distances in metres. Empty includes every nearest candidate.",
    )
    parser.add_argument("--train-pairs", type=int, default=40)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text())
    records = [json.loads(line) for line in (args.pair_run / "pairs.jsonl").read_text().splitlines() if json.loads(line)["pair_type"] != "identity"]
    train, test = records[: args.train_pairs], records[args.train_pairs :]
    device = torch.device("cuda")
    backbone = load_dinov2_vitb14(config["resolution"], device)
    train_data, test_data = [], []
    for split, destination in ((train, train_data), (test, test_data)):
        for record in split:
            observation = load(args.pair_run / record["current"])
            points, valid = patch_world_points(observation, patch_size=14)
            features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), args.layers)
            destination.append((points, valid, {layer: features[layer][0, backbone.num_prefix_tokens :] for layer in args.layers}))
    rows = []
    for voxel_size in args.voxel_sizes:
        stores = {layer: defaultdict(lambda: [None, 0, None]) for layer in args.layers}
        for points, valid, features in train_data:
            for layer in args.layers:
                for key, point, feature in zip(keys(points, voxel_size), points, features[layer][valid]):
                    total, count, point_total = stores[layer][key]
                    stores[layer][key] = [
                        feature.clone() if total is None else total + feature,
                        count + 1,
                        point.clone() if point_total is None else point_total + point,
                    ]
        for layer in args.layers:
            for radius in args.neighbor_radii:
                queries = []
                for points, valid, features in test_data:
                    target_features = features[layer][valid]
                    for key, point, target in zip(keys(points, voxel_size), points, target_features):
                        candidates = []
                        for dx in range(-radius, radius + 1):
                            for dy in range(-radius, radius + 1):
                                for dz in range(-radius, radius + 1):
                                    value = stores[layer].get((key[0] + dx, key[1] + dy, key[2] + dz))
                                    if value is not None:
                                        candidates.append(value)
                        if candidates:
                            value = min(candidates, key=lambda item: torch.linalg.vector_norm(item[2] / item[1] - point).item())
                            aggregate = value[0] / value[1]
                            distance = torch.linalg.vector_norm(value[2] / value[1] - point).item()
                            similarity = torch.nn.functional.cosine_similarity(aggregate[None], target[None]).item()
                            queries.append((distance, similarity))
                test_tokens = sum(int(valid.sum()) for _, valid, _ in test_data)
                for max_distance in [None, *args.max_distances]:
                    accepted = [(distance, similarity) for distance, similarity in queries if max_distance is None or distance <= max_distance]
                    distances = [distance for distance, _ in accepted]
                    similarities = [similarity for _, similarity in accepted]
                    hits = len(accepted)
                    rows.append({"voxel_size_m": voxel_size, "neighbor_radius_voxels": radius, "max_distance_m": max_distance, "layer": layer, "world_tokens": len(stores[layer]), "test_tokens": test_tokens, "hits": hits, "hit_rate": hits / test_tokens, "median_query_distance_m": float(np.median(distances)) if distances else float("nan"), "p95_query_distance_m": float(np.percentile(distances, 95)) if distances else float("nan"), "median_cosine": float(np.median(similarities)) if similarities else float("nan"), "mean_cosine": float(np.mean(similarities)) if similarities else float("nan")})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "world_memory_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"train_pairs": len(train), "test_pairs": len(test), "results": rows}, indent=2) + "\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
