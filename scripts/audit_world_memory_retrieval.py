"""Audit candidate-set bias in the ReplicaCAD world-memory retrieval metric.

The earlier local-matching diagnostic chooses the nearest cached centroid from a
small geometry-gated set as its target.  This script keeps that frozen protocol
but also evaluates the full scene cache.  For every valid query patch it logs
candidate counts, the fraction of candidates within several metric error
radii, selected 3-D locations/errors, and descriptor ranks.  The full-cache
protocol is intentionally harder: it does not use a tight GT-local candidate
filter before descriptor ranking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2_vitb14
from worldcache.geometry.projection import backproject_pixels


def load_observation(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image.to(device) - mean) / std).unsqueeze(0)


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


def candidate_indices(index: dict[tuple[int, int, int], list[int]], key: tuple[int, int, int], radius: int) -> list[int]:
    result: list[int] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                result.extend(index.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
    return result


def xyz_columns(prefix: str, point: torch.Tensor) -> dict[str, float]:
    return {f"{prefix}_x_m": float(point[0]), f"{prefix}_y_m": float(point[1]), f"{prefix}_z_m": float(point[2])}


def select_candidate(
    candidates: torch.Tensor,
    distances: torch.Tensor,
    scores: torch.Tensor,
    mode: str,
    generator: torch.Generator,
) -> int:
    if mode == "geometry":
        return int(torch.argmin(distances))
    if mode == "descriptor":
        return int(torch.argmax(scores))
    if mode == "random":
        return int(torch.randint(len(candidates), (1,), generator=generator))
    raise ValueError(mode)


def append_method(row: dict, prefix: str, candidate_points: torch.Tensor, distances: torch.Tensor, scores: torch.Tensor, selected: int, target: int) -> None:
    point = candidate_points[selected]
    row.update(xyz_columns(f"{prefix}_predicted", point))
    row[f"{prefix}_error_m"] = float(distances[selected])
    row[f"{prefix}_cosine"] = float(scores[selected])
    spatial_order = torch.argsort(distances)
    row[f"{prefix}_selected_xyz_rank"] = int((spatial_order == selected).nonzero(as_tuple=False)[0, 0]) + 1
    order = torch.argsort(scores, descending=True)
    row[f"{prefix}_target_rank"] = int((order == target).nonzero(as_tuple=False)[0, 0]) + 1
    row[f"{prefix}_target_error_m"] = float(distances[target])


def percentile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if len(values) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4])
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=0.15)
    parser.add_argument("--train-pairs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.voxel_size <= 0 or args.neighbor_radius < 0 or args.max_distance <= 0:
        raise ValueError("voxel size and max distance must be positive; radius must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.pair_run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["pair_type"] != "identity"
    ]
    train, test = records[: args.train_pairs], records[args.train_pairs :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_dinov2_vitb14(config["resolution"], device)

    # Each layer stores one pooled descriptor and one pooled 3-D centroid per voxel.
    stores = {layer: defaultdict(lambda: [None, 0, None]) for layer in args.layers}
    for record in train:
        observation = load_observation(args.pair_run / record["current"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), args.layers)
        for layer in args.layers:
            patch_features = features[layer][0, backbone.num_prefix_tokens :].cpu()[valid]
            for point, feature in zip(points, patch_features):
                key = voxel_key(point, args.voxel_size)
                total, count, point_total = stores[layer][key]
                stores[layer][key] = [
                    feature.clone() if total is None else total + feature,
                    count + 1,
                    point.clone() if point_total is None else point_total + point,
                ]

    rows: list[dict[str, float | int | str]] = []
    radii_m = (0.05, 0.10, 0.20, 0.50)
    for record in test:
        observation = load_observation(args.pair_run / record["current"])
        points, valid = patch_world_points(observation, patch_size=14)
        features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), args.layers)
        for layer in args.layers:
            cache_points = torch.stack([value[2] / value[1] for value in stores[layer].values()])
            cache_descriptors = torch.stack([value[0] / value[1] for value in stores[layer].values()])
            cache_descriptors = torch.nn.functional.normalize(cache_descriptors, dim=1)
            keys = list(stores[layer].keys())
            spatial_index: dict[tuple[int, int, int], list[int]] = {}
            for index, key in enumerate(keys):
                spatial_index.setdefault(key, []).append(index)
            query_features = features[layer][0, backbone.num_prefix_tokens :].cpu()[valid]
            for query_index, (point, query) in enumerate(zip(points, query_features)):
                full_distances = torch.linalg.vector_norm(cache_points - point, dim=1)
                query_normalized = torch.nn.functional.normalize(query[None], dim=1)[0]
                full_scores = cache_descriptors @ query_normalized
                full_target = int(torch.argmin(full_distances))
                local_indices = candidate_indices(spatial_index, voxel_key(point, args.voxel_size), args.neighbor_radius)
                local_indices = [index for index in local_indices if float(full_distances[index]) <= args.max_distance]
                local = torch.tensor(local_indices, dtype=torch.long)
                row: dict[str, float | int | str] = {
                    "pair_id": int(record["pair_id"]),
                    "layer": layer,
                    "query_index": query_index,
                    "candidate_protocol": "fixed_local_and_full_scene",
                    "full_candidate_count": len(cache_points),
                    "local_candidate_count": len(local_indices),
                    "local_has_candidate": int(bool(local_indices)),
                    "query_valid": 1,
                }
                row.update(xyz_columns("true_correspondence", point))
                for label, candidate_distances in (("full", full_distances), ("local", full_distances[local] if local_indices else torch.empty(0))):
                    row[f"{label}_nearest_xyz_error_m"] = float(candidate_distances.min()) if len(candidate_distances) else float("nan")
                    for radius in radii_m:
                        suffix = f"{int(radius * 100):02d}cm"
                        row[f"{label}_correct_fraction_{suffix}"] = float((candidate_distances <= radius).float().mean()) if len(candidate_distances) else float("nan")

                full_generator = torch.Generator().manual_seed(args.seed + 1_000_000 * int(record["pair_id"]) + 10_000 * layer + query_index)
                append_method(row, "full_geometry", cache_points, full_distances, full_scores, select_candidate(cache_points, full_distances, full_scores, "geometry", full_generator), full_target)
                append_method(row, "full_descriptor", cache_points, full_distances, full_scores, select_candidate(cache_points, full_distances, full_scores, "descriptor", full_generator), full_target)
                append_method(row, "full_random", cache_points, full_distances, full_scores, select_candidate(cache_points, full_distances, full_scores, "random", full_generator), full_target)
                if local_indices:
                    local_points = cache_points[local]
                    local_distances = full_distances[local]
                    local_scores = full_scores[local]
                    local_target = int(torch.argmin(local_distances))
                    local_generator = torch.Generator().manual_seed(args.seed + 2_000_000 * int(record["pair_id"]) + 10_000 * layer + query_index)
                    append_method(row, "local_geometry", local_points, local_distances, local_scores, select_candidate(local_points, local_distances, local_scores, "geometry", local_generator), local_target)
                    append_method(row, "local_descriptor", local_points, local_distances, local_scores, select_candidate(local_points, local_distances, local_scores, "descriptor", local_generator), local_target)
                    append_method(row, "local_random", local_points, local_distances, local_scores, select_candidate(local_points, local_distances, local_scores, "random", local_generator), local_target)
                else:
                    for prefix in ("local_geometry", "local_descriptor", "local_random"):
                        for suffix in ("predicted_x_m", "predicted_y_m", "predicted_z_m", "error_m", "cosine", "target_error_m"):
                            row[f"{prefix}_{suffix}"] = float("nan")
                        row[f"{prefix}_target_rank"] = float("nan")
                        row[f"{prefix}_selected_xyz_rank"] = float("nan")
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
        ranks = frame[f"{prefix}_target_rank"].dropna()
        result = {
            "layer": int(frame["layer"].iloc[0]),
            "method": label,
            "queries": int(len(values)),
            "error_median": float(values.median()),
            "error_mean": float(values.mean()),
            "error_p90": percentile(values, 0.90),
            # Recall@k here means whether the selected candidate is among the
            # k nearest candidates in 3-D.  This is distinct from the
            # descriptor rank of the nearest-XYZ target, which is logged per
            # query as ``*_target_rank``.
            "recall_at_1": float((frame.loc[values.index, f"{prefix}_selected_xyz_rank"] <= 1).mean()) if len(values) else float("nan"),
            "recall_at_5": float((frame.loc[values.index, f"{prefix}_selected_xyz_rank"] <= 5).mean()) if len(values) else float("nan"),
            "descriptor_target_recall_at_1": float((ranks <= 1).mean()) if len(ranks) else float("nan"),
            "descriptor_target_recall_at_5": float((ranks <= 5).mean()) if len(ranks) else float("nan"),
        }
        for radius in radii_m:
            result[f"error_cdf_{int(radius * 100):02d}cm"] = float((values <= radius).mean()) if len(values) else float("nan")
        summary_rows.append(result)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    gated = frame[frame["local_has_candidate"] == 1]
    audit_stats = {
        "candidate_count_mean": float(gated["local_candidate_count"].mean()),
        "candidate_count_median": float(gated["local_candidate_count"].median()),
        "candidate_count_p10": float(gated["local_candidate_count"].quantile(0.10)),
        "candidate_count_p50": float(gated["local_candidate_count"].quantile(0.50)),
        "candidate_count_p90": float(gated["local_candidate_count"].quantile(0.90)),
        "gated_query_count": int(len(gated)),
        "all_query_count": int(len(frame)),
        "geometry_gate_hit_rate": float(len(gated) / len(frame)),
        "correct_candidate_fraction_mean": {
            "5cm": float(gated["local_correct_fraction_05cm"].mean()),
            "10cm": float(gated["local_correct_fraction_10cm"].mean()),
            "20cm": float(gated["local_correct_fraction_20cm"].mean()),
            "50cm": float(gated["local_correct_fraction_50cm"].mean()),
        },
        "correct_candidate_fraction_p10": {
            "5cm": float(gated["local_correct_fraction_05cm"].quantile(0.10)),
            "10cm": float(gated["local_correct_fraction_10cm"].quantile(0.10)),
            "20cm": float(gated["local_correct_fraction_20cm"].quantile(0.10)),
            "50cm": float(gated["local_correct_fraction_50cm"].quantile(0.10)),
        },
        "correct_candidate_fraction_p50": {
            "5cm": float(gated["local_correct_fraction_05cm"].quantile(0.50)),
            "10cm": float(gated["local_correct_fraction_10cm"].quantile(0.50)),
            "20cm": float(gated["local_correct_fraction_20cm"].quantile(0.50)),
            "50cm": float(gated["local_correct_fraction_50cm"].quantile(0.50)),
        },
        "correct_candidate_fraction_p90": {
            "5cm": float(gated["local_correct_fraction_05cm"].quantile(0.90)),
            "10cm": float(gated["local_correct_fraction_10cm"].quantile(0.90)),
            "20cm": float(gated["local_correct_fraction_20cm"].quantile(0.90)),
            "50cm": float(gated["local_correct_fraction_50cm"].quantile(0.90)),
        },
        "fixed_parameters": {
            "layer": args.layers,
            "voxel_size_m": args.voxel_size,
            "neighbor_radius": args.neighbor_radius,
            "distance_gate_m": args.max_distance,
        },
        "full_candidate_pool_size": int(frame["full_candidate_count"].iloc[0]),
    }
    (args.output_dir / "audit_stats.json").write_text(json.dumps(audit_stats, indent=2) + "\n", encoding="utf-8")

    plt.figure(figsize=(7.5, 5.0))
    grid = np.linspace(0.0, 0.50, 101)
    for label, prefix in methods.items():
        values = np.sort(frame[f"{prefix}_error_m"].dropna().to_numpy())
        if not len(values):
            continue
        cdf = np.searchsorted(values, grid, side="right") / len(values)
        plt.plot(grid, cdf, label=label)
    plt.xlabel("3-D correspondence error (m)")
    plt.ylabel("P(error < r)")
    plt.xlim(0, 0.50)
    plt.ylim(0, 1.01)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.output_dir / "error_cdf.png", dpi=180)
    plt.close()

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    metadata = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dataset": "ReplicaCAD",
        "scene_id": config.get("scene_id", "apt_1"),
        "reference_scan": "",
        "query_scan": "",
        "backbone": "vit_base_patch14_dinov2.lvd142m",
        "backbone_layer": args.layers,
        "input_resolution": config["resolution"],
        "voxel_size": args.voxel_size,
        "neighbor_radius": args.neighbor_radius,
        "distance_gate": args.max_distance,
        "candidate_protocol": "fixed local geometry gate plus full scene cache",
        "descriptor_dim": int(cache_descriptors.shape[1]),
        "precision": "float32 features; CPU retrieval",
        "random_seed": args.seed,
        "num_queries": int(len(frame)),
        "num_candidates_full": int(frame["full_candidate_count"].iloc[0]),
        "num_candidates_local_median": float(frame["local_candidate_count"].median()),
        "metrics": str(args.output_dir / "summary.csv"),
        "wall_clock_ms": None,
        "gpu_peak_memory_mb": None,
        "train_pairs": len(train),
        "test_pairs": len(test),
        "args": vars(args),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "status.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({key: metadata[key] for key in ("num_queries", "num_candidates_full", "num_candidates_local_median")}, indent=2))


if __name__ == "__main__":
    main()
