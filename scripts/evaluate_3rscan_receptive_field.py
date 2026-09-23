"""Evaluate 2-D receptive-field pooled descriptors for 3RScan local matching."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2_vitb14
from worldcache.geometry.projection import backproject_pixels


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image - mean) / std).unsqueeze(0)


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


def pooled_tokens(tokens: torch.Tensor, kernel: int) -> torch.Tensor:
    """Pool the token grid while preserving one descriptor per center token."""
    token_count = tokens.shape[0]
    side = int(token_count**0.5)
    if side * side != token_count:
        raise ValueError(f"expected a square patch grid, got {token_count} tokens")
    feature_map = tokens.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
    if kernel == 1:
        pooled = feature_map
    else:
        pooled = F.avg_pool2d(feature_map, kernel_size=kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    return pooled[0].permute(1, 2, 0).reshape(token_count, -1)


def voxel_key(point: torch.Tensor, size: float) -> tuple[int, int, int]:
    return tuple(torch.floor(point / size).to(torch.int64).tolist())


def neighbors(index: dict[tuple[int, int, int], list[int]], key: tuple[int, int, int], radius: int) -> list[int]:
    values: list[int] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                values.extend(index.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
    return values


def append_method(row: dict[str, object], prefix: str, distances: torch.Tensor, scores: torch.Tensor, selected: int, target: int) -> None:
    row[f"{prefix}_error_m"] = float(distances[selected])
    row[f"{prefix}_target_rank"] = int((torch.argsort(scores, descending=True) == target).nonzero(as_tuple=False)[0, 0]) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--kernels", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=0.15)
    args = parser.parse_args()
    if any(kernel < 1 or kernel % 2 == 0 for kernel in args.kernels):
        raise ValueError("kernels must be positive odd integers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2_vitb14(config["resolution"], device)

    stores = {
        kernel: defaultdict(lambda: [None, 0, None]) for kernel in args.kernels
    }
    reference_valid = 0
    for entry in manifest["reference"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        tokens = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()
        reference_valid += int(valid.sum())
        for kernel in args.kernels:
            features = pooled_tokens(tokens, kernel)[valid]
            for point, feature in zip(points, features):
                key = voxel_key(point, args.voxel_size)
                total, count, point_total = stores[kernel][key]
                stores[kernel][key] = [
                    feature.clone() if total is None else total + feature,
                    count + 1,
                    point.clone() if point_total is None else point_total + point,
                ]

    caches = {}
    for kernel, store in stores.items():
        points = torch.stack([value[2] / value[1] for value in store.values()])
        descriptors = torch.nn.functional.normalize(torch.stack([value[0] / value[1] for value in store.values()]), dim=1)
        index: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for item, key in enumerate(store):
            index[key].append(item)
        caches[kernel] = points, descriptors, index

    rows: list[dict[str, object]] = []
    query_valid = 0
    for frame_index, entry in enumerate(manifest["rescan"]):
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        tokens = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()
        query_valid += int(valid.sum())
        query_features = {kernel: pooled_tokens(tokens, kernel)[valid] for kernel in args.kernels}
        for query_index, point in enumerate(points):
            for kernel in args.kernels:
                cache_points, cache_descriptors, spatial_index = caches[kernel]
                local_indices = [
                    item for item in neighbors(spatial_index, voxel_key(point, args.voxel_size), args.neighbor_radius)
                    if float(torch.linalg.vector_norm(cache_points[item] - point)) <= args.max_distance
                ]
                if not local_indices:
                    continue
                local = torch.tensor(local_indices, dtype=torch.long)
                local_points = cache_points[local]
                distances = torch.linalg.vector_norm(local_points - point, dim=1)
                target = int(torch.argmin(distances))
                query = torch.nn.functional.normalize(query_features[kernel][query_index], dim=0)
                scores = cache_descriptors[local] @ query
                descriptor_selected = int(torch.argmax(scores))
                generator = torch.Generator().manual_seed(1_000_000 * frame_index + 10_000 * kernel + query_index)
                random_selected = int(torch.randperm(len(local), generator=generator)[0])
                row: dict[str, object] = {
                    "frame_index": frame_index,
                    "query_index": query_index,
                    "kernel": kernel,
                    "candidate_count": len(local_indices),
                }
                append_method(row, "geometry", distances, scores, target, target)
                append_method(row, "receptive_field", distances, scores, descriptor_selected, target)
                append_method(row, "random", distances, scores, random_selected, target)
                rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no rescan queries passed the geometric cache gate")
    frame.to_parquet(args.output_dir / "query_metrics.parquet", index=False)
    summary_rows = []
    for kernel, group in frame.groupby("kernel", sort=True):
        for label, prefix in (("Geometry-only", "geometry"), ("Receptive-field descriptor", "receptive_field"), ("Random", "random")):
            values = group[f"{prefix}_error_m"]
            ranks = group[f"{prefix}_target_rank"]
            summary_rows.append({
                "kernel": kernel,
                "method": label,
                "queries": len(group),
                "candidate_count_median": float(group["candidate_count"].median()),
                "error_median": float(values.median()),
                "error_mean": float(values.mean()),
                "error_cdf_05cm": float((values <= 0.05).mean()),
                "error_cdf_10cm": float((values <= 0.10).mean()),
                "target_recall_at_1": float((ranks == 1).mean()),
                "target_recall_at_3": float((ranks <= 3).mean()),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    metrics = {
        "status": "complete",
        "reference_valid_patches": reference_valid,
        "rescan_valid_patches": query_valid,
        "gated_rows": len(frame),
        "config": vars(args),
    }
    (args.output_dir / "status.json").write_text(json.dumps(metrics, default=str, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
