"""Aggregate geometry validation metrics for an on-disk RGB-D pair run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from worldcache.geometry.projection import backproject_pixels, project_world_points
from worldcache.geometry.visibility import depth_visibility_mask


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def centers(height: int, width: int, patch_size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(patch_size / 2, height, patch_size),
        torch.arange(patch_size / 2, width, patch_size),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=14)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.run_dir / "pairs.jsonl").read_text().splitlines()]
    all_reprojection_errors: list[torch.Tensor] = []
    visible_fractions: list[float] = []
    depth_residuals: list[torch.Tensor] = []
    identity_errors: list[torch.Tensor] = []
    for record in records:
        reference = load(args.run_dir / record["reference"])
        current = load(args.run_dir / record["current"])
        height, width = reference["depth"].shape
        pixels = centers(height, width, args.patch_size)
        u, v = pixels[:, 0].long(), pixels[:, 1].long()
        depth = reference["depth"][v, u]
        valid = torch.isfinite(depth) & (depth > 0)
        world = backproject_pixels(pixels[valid], depth[valid], reference["intrinsics"], reference["camera_to_world"])
        reference_pixels, _ = project_world_points(world, reference["intrinsics"], reference["camera_to_world"])
        errors = torch.linalg.vector_norm(reference_pixels - pixels[valid], dim=1)
        all_reprojection_errors.append(errors)
        if record["pair_type"] == "identity":
            identity_errors.append(errors)
        projected, projected_depth = project_world_points(world, current["intrinsics"], current["camera_to_world"])
        visible = depth_visibility_mask(projected, projected_depth, current["depth"])
        visible_fractions.append(float(visible.float().mean()))
        rounded = projected.round().long()
        in_bounds = (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
        sample_indices = torch.where(in_bounds)[0]
        residual = current["depth"][rounded[sample_indices, 1], rounded[sample_indices, 0]] - projected_depth[sample_indices]
        depth_residuals.append(residual)

    def percentile(values: torch.Tensor, q: float) -> float:
        return float(torch.quantile(values, q))

    reprojection = torch.cat(all_reprojection_errors)
    depth = torch.cat(depth_residuals).abs()
    identity = torch.cat(identity_errors)
    summary = {
        "status": "complete",
        "pair_count": len(records),
        "identity_pair_count": sum(record["pair_type"] == "identity" for record in records),
        "median_reprojection_error_px": float(torch.median(reprojection)),
        "p99_reprojection_error_px": percentile(reprojection, 0.99),
        "max_reprojection_error_px": float(torch.max(reprojection)),
        "max_identity_reprojection_error_px": float(torch.max(identity)),
        "median_visible_fraction": float(np.median(visible_fractions)),
        "minimum_visible_fraction": float(np.min(visible_fractions)),
        "median_abs_depth_residual_m": float(torch.median(depth)),
        "p95_abs_depth_residual_m": percentile(depth, 0.95),
    }
    path = args.run_dir / "geometry_metrics.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
