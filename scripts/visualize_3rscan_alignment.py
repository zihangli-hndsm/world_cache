"""Generate qualitative 3RScan geometry/descriptor alignment diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2_vitb14
from worldcache.geometry.projection import backproject_pixels, project_world_points


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image - mean) / std).unsqueeze(0)


def patch_centers(height: int, width: int, patch_size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(patch_size / 2, height, patch_size), torch.arange(patch_size / 2, width, patch_size), indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()


def points_and_pixels(observation: dict[str, torch.Tensor], patch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pixels = patch_centers(*observation["depth"].shape, patch_size)
    depth = observation["depth"][pixels[:, 1].long(), pixels[:, 0].long()]
    valid = torch.isfinite(depth) & (depth > 0)
    points = backproject_pixels(pixels[valid], depth[valid], observation["intrinsics"], observation["camera_to_world"])
    return points, pixels[valid], valid


def neighbors(index: dict[tuple[int, int, int], list[int]], key: tuple[int, int, int], radius: int) -> list[int]:
    result = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                result.extend(index.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
    return result


def key(point: torch.Tensor, voxel_size: float) -> tuple[int, int, int]:
    return tuple(torch.floor(point / voxel_size).to(torch.int64).tolist())


def scatter(ax, image, uv, colors, title, s=8, cmap="turbo", vmin=None, vmax=None):
    ax.imshow(image)
    if len(uv):
        ax.scatter(uv[:, 0], uv[:, 1], c=colors, s=s, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.82, linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.20)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=0.15)
    parser.add_argument("--num-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))["resolution"]
    backbone = load_dinov2_vitb14(resolution, device)

    stores = defaultdict(lambda: [None, 0, None])
    reference_observations = []
    for entry in manifest["reference"]:
        observation = load(args.pair_run / entry["path"])
        reference_observations.append((entry, observation))
        points, _, valid = points_and_pixels(observation, 14)
        feature = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()[valid]
        for point, descriptor in zip(points, feature):
            voxel = key(point, args.voxel_size)
            total, count, point_total = stores[voxel]
            stores[voxel] = [descriptor.clone() if total is None else total + descriptor, count + 1, point.clone() if point_total is None else point_total + point]
    cache_points = torch.stack([v[2] / v[1] for v in stores.values()])
    cache_descriptors = torch.nn.functional.normalize(torch.stack([v[0] / v[1] for v in stores.values()]), dim=1)
    index = defaultdict(list)
    for i, voxel in enumerate(stores):
        index[voxel].append(i)

    entries = manifest["rescan"]
    selected_indices = np.linspace(0, len(entries) - 1, min(args.num_pairs, len(entries)), dtype=int).tolist()
    for sequence_index in selected_indices:
        entry = entries[sequence_index]
        observation = load(args.pair_run / entry["path"])
        points, query_pixels, valid = points_and_pixels(observation, 14)
        query_features = extract_block_tokens(backbone, preprocess(observation["rgb"], device), [args.layer])[args.layer][0, backbone.num_prefix_tokens :].cpu()[valid]
        gated_pixels, projected_query, selected_ref_points, errors, similarities = [], [], [], [], []
        for point, pixel, query in zip(points, query_pixels, query_features):
            candidates = [i for i in neighbors(index, key(point, args.voxel_size), args.neighbor_radius) if torch.linalg.vector_norm(cache_points[i] - point).item() <= args.max_distance]
            if not candidates:
                continue
            candidate_tensor = torch.tensor(candidates, dtype=torch.long)
            candidate_points = cache_points[candidate_tensor]
            distances = torch.linalg.vector_norm(candidate_points - point, dim=1)
            scores = cache_descriptors[candidate_tensor] @ torch.nn.functional.normalize(query, dim=0)
            selected = int(torch.argmax(scores))
            gated_pixels.append(pixel)
            projected_query.append(point)
            selected_ref_points.append(candidate_points[selected])
            errors.append(float(distances[selected]))
            similarities.append(float(scores[selected]))
        gated_pixels = torch.stack(gated_pixels) if gated_pixels else torch.empty((0, 2))
        projected_query = torch.stack(projected_query) if projected_query else torch.empty((0, 3))
        selected_ref_points = torch.stack(selected_ref_points) if selected_ref_points else torch.empty((0, 3))
        errors_array = np.asarray(errors, dtype=np.float32)
        similarities_array = np.asarray(similarities, dtype=np.float32)

        # Pick the reference frame in which the selected cache centroids are most visible.
        best_reference = reference_observations[0]
        best_visible = -1
        best_uv = torch.empty((0, 2))
        for candidate_reference in reference_observations:
            uv, depth = project_world_points(selected_ref_points, candidate_reference[1]["intrinsics"], candidate_reference[1]["camera_to_world"])
            visible = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < candidate_reference[1]["rgb"].shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < candidate_reference[1]["rgb"].shape[0])
            if int(visible.sum()) > best_visible:
                best_reference, best_visible, best_uv = candidate_reference, int(visible.sum()), uv
        reference_entry, reference_observation = best_reference
        projected_uv, projected_depth = project_world_points(projected_query, observation["intrinsics"], observation["camera_to_world"])
        rounded = projected_uv.round().long()
        in_bounds = (rounded[:, 0] >= 0) & (rounded[:, 0] < observation["depth"].shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < observation["depth"].shape[0])
        depth_residuals = torch.full((len(projected_depth),), float("nan"))
        if bool(in_bounds.any()):
            depth_residuals[in_bounds] = observation["depth"][rounded[in_bounds, 1], rounded[in_bounds, 0]] - projected_depth[in_bounds]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        scatter(axes[0, 0], reference_observation["rgb"].numpy(), best_uv, errors_array, f"Reference RGB + selected cache points\nframe {reference_entry['frame_id']}", vmin=0, vmax=args.max_distance)
        scatter(axes[0, 1], observation["rgb"].numpy(), gated_pixels, errors_array, f"Rescan RGB + gated query points\nframe {entry['frame_id']} ({len(gated_pixels)} gated)", vmin=0, vmax=args.max_distance)
        scatter(axes[0, 2], observation["rgb"].numpy(), projected_uv, projected_depth.numpy(), "Projected world points into rescan", cmap="viridis")
        scatter(axes[0, 3], observation["rgb"].numpy(), gated_pixels, similarities_array, "Descriptor correspondence confidence", cmap="viridis", vmin=-1, vmax=1)
        scatter(axes[1, 0], reference_observation["rgb"].numpy(), best_uv, similarities_array, "Descriptor-selected reference locations", cmap="viridis", vmin=-1, vmax=1)
        scatter(axes[1, 1], observation["rgb"].numpy(), gated_pixels, errors_array, "Descriptor correspondence 3-D error (m)", vmin=0, vmax=args.max_distance)
        finite_depth = depth_residuals[torch.isfinite(depth_residuals)].numpy()
        axes[1, 2].hist(finite_depth, bins=30, color="slateblue")
        axes[1, 2].axvline(0, color="black", linewidth=1)
        axes[1, 2].set_title("Depth consistency residual (m)", fontsize=8)
        axes[1, 2].set_xlabel("observed − projected")
        axes[1, 3].hist(errors_array, bins=20, color="darkorange", range=(0, args.max_distance))
        axes[1, 3].set_title("Correspondence error histogram", fontsize=8)
        axes[1, 3].set_xlabel("3-D error (m)")
        reference_id = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))["reference"]
        rescan_id = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))["rescan"]
        stem = f"reference_{reference_id}_rescan_{rescan_id}_reference_frame_{int(reference_entry['frame_id']):06d}_rescan_frame_{int(entry['frame_id']):06d}"
        fig.savefig(args.output_dir / f"{stem}.png", dpi=160)
        plt.close(fig)
        (args.output_dir / f"{stem}.json").write_text(json.dumps({
            "reference_frame": int(reference_entry["frame_id"]),
            "rescan_frame": int(entry["frame_id"]),
            "gated_queries": int(len(gated_pixels)),
            "valid_rescan_queries": int(len(query_pixels)),
            "gate_rate": float(len(gated_pixels) / len(query_pixels)) if len(query_pixels) else 0.0,
            "descriptor_error_median": float(np.median(errors_array)) if len(errors_array) else None,
            "descriptor_similarity_median": float(np.median(similarities_array)) if len(similarities_array) else None,
            "best_reference_visible_selected_points": best_visible,
            "depth_residual_median": float(np.nanmedian(depth_residuals.numpy())) if bool(torch.isfinite(depth_residuals).any()) else None,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
