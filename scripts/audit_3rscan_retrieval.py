"""Audit fixed local and full-scene descriptor retrieval on a 3RScan pair."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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


def patch_world_points(observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = observation["depth"].shape
    yy, xx = torch.meshgrid(torch.arange(7, height, 14), torch.arange(7, width, 14), indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
    depth = observation["depth"][pixels[:, 1].long(), pixels[:, 0].long()]
    valid = torch.isfinite(depth) & (depth > 0)
    return backproject_pixels(pixels[valid], depth[valid], observation["intrinsics"], observation["camera_to_world"]), valid


def voxel_key(point: torch.Tensor, size: float) -> tuple[int, int, int]:
    return tuple(torch.floor(point / size).to(torch.int64).tolist())


def neighbors(index: dict[tuple[int, int, int], list[int]], key: tuple[int, int, int], radius: int) -> list[int]:
    result = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                result.extend(index.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
    return result


def append_method(row: dict, prefix: str, points: torch.Tensor, distances: torch.Tensor, scores: torch.Tensor, selected: int, target: int) -> None:
    row[f"{prefix}_error_m"] = float(distances[selected])
    row[f"{prefix}_cosine"] = float(scores[selected])
    row[f"{prefix}_selected_xyz_rank"] = int((torch.argsort(distances) == selected).nonzero(as_tuple=False)[0, 0]) + 1
    row[f"{prefix}_target_rank"] = int((torch.argsort(scores, descending=True) == target).nonzero(as_tuple=False)[0, 0]) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2_vitb14(config["resolution"], device)

    stores = defaultdict(lambda: [None, 0, None])
    for entry in manifest["reference"]:
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation)
        features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()[valid]
        for point, feature in zip(points, features):
            voxel = voxel_key(point, args.voxel_size)
            total, count, point_total = stores[voxel]
            stores[voxel] = [feature.clone() if total is None else total + feature, count + 1, point.clone() if point_total is None else point_total + point]
    cache_points = torch.stack([value[2] / value[1] for value in stores.values()])
    cache_descriptors = torch.nn.functional.normalize(torch.stack([value[0] / value[1] for value in stores.values()]), dim=1)
    spatial_index = defaultdict(list)
    for index, voxel in enumerate(stores):
        spatial_index[voxel].append(index)

    rows = []
    query_valid = 0
    for frame_index, entry in enumerate(manifest["rescan"]):
        observation = load(args.pair_run / entry["path"])
        points, valid = patch_world_points(observation)
        query_valid += int(valid.sum())
        features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()[valid]
        for query_index, (point, query) in enumerate(zip(points, features)):
            full_distances = torch.linalg.vector_norm(cache_points - point, dim=1)
            full_scores = cache_descriptors @ torch.nn.functional.normalize(query, dim=0)
            full_target = int(torch.argmin(full_distances))
            local_indices = [i for i in neighbors(spatial_index, voxel_key(point, args.voxel_size), args.neighbor_radius) if float(full_distances[i]) <= args.max_distance]
            row = {
                "frame_index": frame_index,
                "frame_id": int(entry["frame_id"]),
                "query_index": query_index,
                "full_candidate_count": len(cache_points),
                "local_candidate_count": len(local_indices),
                "local_has_candidate": int(bool(local_indices)),
            }
            for name, distances in (("full", full_distances), ("local", full_distances[torch.tensor(local_indices)] if local_indices else torch.empty(0))):
                row[f"{name}_nearest_xyz_error_m"] = float(distances.min()) if len(distances) else float("nan")
                for radius in (0.05, 0.10, 0.20, 0.50):
                    row[f"{name}_correct_fraction_{int(radius * 100):02d}cm"] = float((distances <= radius).float().mean()) if len(distances) else float("nan")
            generator = torch.Generator().manual_seed(args.seed + frame_index * 10000 + query_index)
            append_method(row, "full_geometry", cache_points, full_distances, full_scores, int(torch.argmin(full_distances)), full_target)
            append_method(row, "full_descriptor", cache_points, full_distances, full_scores, int(torch.argmax(full_scores)), full_target)
            full_random_order = torch.randperm(len(cache_points), generator=generator)
            append_method(row, "full_random", cache_points, full_distances, full_scores, int(full_random_order[0]), full_target)
            if local_indices:
                local = torch.tensor(local_indices, dtype=torch.long)
                local_points, local_distances, local_scores = cache_points[local], full_distances[local], full_scores[local]
                local_target = int(torch.argmin(local_distances))
                append_method(row, "local_geometry", local_points, local_distances, local_scores, int(torch.argmin(local_distances)), local_target)
                append_method(row, "local_descriptor", local_points, local_distances, local_scores, int(torch.argmax(local_scores)), local_target)
                local_random_order = torch.randperm(len(local), generator=generator)
                append_method(row, "local_random", local_points, local_distances, local_scores, int(local_random_order[0]), local_target)
            else:
                for prefix in ("local_geometry", "local_descriptor", "local_random"):
                    for suffix in ("error_m", "cosine", "selected_xyz_rank", "target_rank"):
                        row[f"{prefix}_{suffix}"] = float("nan")
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output_dir / "query_metrics.parquet", index=False)
    frame.to_json(args.output_dir / "query_metrics.jsonl", orient="records", lines=True)
    methods = {
        "Geometry-only (full scene)": "full_geometry",
        "Descriptor-only (full scene)": "full_descriptor",
        "Random (full scene)": "full_random",
        "Geometry-only (fixed local gate)": "local_geometry",
        "Geometry + Descriptor (fixed local gate)": "local_descriptor",
        "Random (fixed local gate)": "local_random",
    }
    summary_rows = []
    for label, prefix in methods.items():
        values = frame[f"{prefix}_error_m"].dropna()
        ranks = frame.loc[values.index, f"{prefix}_selected_xyz_rank"]
        descriptor_ranks = frame.loc[values.index, f"{prefix}_target_rank"]
        result = {
            "method": label,
            "queries": int(len(values)),
            "error_median": float(values.median()),
            "error_mean": float(values.mean()),
            "error_p90": float(values.quantile(0.90)),
            "recall_at_1": float((ranks <= 1).mean()),
            "recall_at_5": float((ranks <= 5).mean()),
            "descriptor_target_recall_at_1": float((descriptor_ranks <= 1).mean()),
            "descriptor_target_recall_at_5": float((descriptor_ranks <= 5).mean()),
        }
        for radius in (0.05, 0.10, 0.20, 0.50):
            result[f"error_cdf_{int(radius * 100):02d}cm"] = float((values <= radius).mean())
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    gated = frame[frame.local_has_candidate == 1]
    stats = {
        "dataset": "3RScan",
        "reference_scan": config["reference"],
        "query_scan": config["rescan"],
        "query_valid": query_valid,
        "gated_query_count": int(len(gated)),
        "geometry_gate_hit_rate": float(len(gated) / query_valid),
        "candidate_count_mean": float(gated.local_candidate_count.mean()) if len(gated) else float("nan"),
        "candidate_count_median": float(gated.local_candidate_count.median()) if len(gated) else float("nan"),
        "candidate_count_p10": float(gated.local_candidate_count.quantile(0.10)) if len(gated) else float("nan"),
        "candidate_count_p50": float(gated.local_candidate_count.quantile(0.50)) if len(gated) else float("nan"),
        "candidate_count_p90": float(gated.local_candidate_count.quantile(0.90)) if len(gated) else float("nan"),
        "correct_candidate_fraction_mean": {f"{int(r*100)}cm": float(gated[f"local_correct_fraction_{int(r*100):02d}cm"].mean()) for r in (0.05, 0.10, 0.20, 0.50)} if len(gated) else {},
        "correct_candidate_fraction_p10": {f"{int(r*100)}cm": float(gated[f"local_correct_fraction_{int(r*100):02d}cm"].quantile(0.10)) for r in (0.05, 0.10, 0.20, 0.50)} if len(gated) else {},
        "correct_candidate_fraction_p50": {f"{int(r*100)}cm": float(gated[f"local_correct_fraction_{int(r*100):02d}cm"].quantile(0.50)) for r in (0.05, 0.10, 0.20, 0.50)} if len(gated) else {},
        "correct_candidate_fraction_p90": {f"{int(r*100)}cm": float(gated[f"local_correct_fraction_{int(r*100):02d}cm"].quantile(0.90)) for r in (0.05, 0.10, 0.20, 0.50)} if len(gated) else {},
        "fixed_parameters": {"layer": args.layer, "voxel_size_m": args.voxel_size, "neighbor_radius": args.neighbor_radius, "distance_gate_m": args.max_distance},
        "full_candidate_pool_size": int(len(cache_points)),
    }
    (args.output_dir / "audit_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    grid = np.linspace(0, 0.50, 101)
    plt.figure(figsize=(7.5, 5))
    for label, prefix in methods.items():
        values = np.sort(frame[f"{prefix}_error_m"].dropna().to_numpy())
        if len(values):
            plt.plot(grid, np.searchsorted(values, grid, side="right") / len(values), label=label)
    plt.xlabel("3-D correspondence error (m)")
    plt.ylabel("P(error < r)")
    plt.xlim(0, 0.50)
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output_dir / "error_cdf.png", dpi=180)
    plt.close()
    print(summary.to_string(index=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
