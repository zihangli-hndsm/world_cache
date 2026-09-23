"""Measure and visualize geometric patch-center alignment for a saved pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from worldcache.geometry.projection import backproject_pixels, project_world_points
from worldcache.geometry.visibility import depth_visibility_mask


def load_observation(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def patch_centers(height: int, width: int, patch_size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(patch_size / 2, height, patch_size),
        torch.arange(patch_size / 2, width, patch_size),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=14)
    args = parser.parse_args()

    reference, current = load_observation(args.reference), load_observation(args.current)
    height, width = reference["depth"].shape
    pixels = patch_centers(height, width, args.patch_size)
    u, v = pixels[:, 0].long(), pixels[:, 1].long()
    reference_depth = reference["depth"][v, u]
    valid_reference = torch.isfinite(reference_depth) & (reference_depth > 0)
    world = backproject_pixels(
        pixels[valid_reference], reference_depth[valid_reference], reference["intrinsics"], reference["camera_to_world"]
    )
    projected, projected_depth = project_world_points(world, current["intrinsics"], current["camera_to_world"])
    visible = depth_visibility_mask(projected, projected_depth, current["depth"])
    residuals = torch.full_like(projected_depth, float("nan"))
    rounded = projected.round().long()
    in_bounds = (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    indices = torch.where(in_bounds)[0]
    residuals[indices] = current["depth"][rounded[indices, 1], rounded[indices, 0]] - projected_depth[indices]
    reference_reprojection, _ = project_world_points(world, reference["intrinsics"], reference["camera_to_world"])
    reprojection_error = torch.linalg.vector_norm(reference_reprojection - pixels[valid_reference], dim=1)
    metrics = {
        "patch_centers": int(len(pixels)),
        "valid_reference_centers": int(valid_reference.sum()),
        "visible_correspondences": int(visible.sum()),
        "visible_fraction_of_valid": float(visible.float().mean()),
        "median_abs_depth_residual_m": float(torch.nanmedian(residuals.abs())),
        "median_identity_reprojection_error_px": float(torch.median(reprojection_error)),
        "max_identity_reprojection_error_px": float(torch.max(reprojection_error)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(reference["rgb"].numpy())
    axes[0].scatter(pixels[valid_reference, 0], pixels[valid_reference, 1], s=5, c="cyan", alpha=0.7)
    axes[0].set_title("Reference patch centers")
    axes[1].imshow(current["rgb"].numpy())
    axes[1].scatter(projected[~visible, 0], projected[~visible, 1], s=8, c="tomato", alpha=0.75, label="invalid/occluded")
    axes[1].scatter(projected[visible, 0], projected[visible, 1], s=8, c="lime", alpha=0.85, label="depth-visible")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_title("Projected reference centers")
    finite_residuals = residuals[torch.isfinite(residuals)].numpy()
    axes[2].hist(finite_residuals, bins=40, color="slateblue")
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_title("Current depth − projected depth")
    axes[2].set_xlabel("meters")
    for axis in axes[:2]:
        axis.set_axis_off()
    fig.suptitle(
        f"visible {metrics['visible_correspondences']}/{metrics['valid_reference_centers']}; "
        f"median |depth residual|={metrics['median_abs_depth_residual_m']:.4f}m"
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
