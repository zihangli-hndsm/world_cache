"""Depth-based correspondence visibility tests."""

from __future__ import annotations

import torch


def depth_visibility_mask(
    projected_uv: torch.Tensor,
    projected_depth: torch.Tensor,
    current_depth: torch.Tensor,
    absolute_tolerance_m: float = 0.03,
    relative_tolerance: float = 0.01,
) -> torch.Tensor:
    """Return whether projected points are inside, in front of camera, and unoccluded.

    The depth image is sampled at nearest pixel for a conservative first-pass
    oracle. Invalid depths (non-finite or <=0) are always rejected.
    """
    if current_depth.ndim != 2:
        raise ValueError("current_depth must have shape [H, W]")
    if projected_uv.shape[-1] != 2:
        raise ValueError("projected_uv must have shape [N, 2]")
    if projected_depth.shape != projected_uv.shape[:1]:
        raise ValueError("projected_depth must have shape [N]")

    height, width = current_depth.shape
    u = projected_uv[:, 0].round().long()
    v = projected_uv[:, 1].round().long()
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    sampled = torch.zeros_like(projected_depth)
    valid_indices = torch.where(in_bounds)[0]
    sampled[valid_indices] = current_depth[v[valid_indices], u[valid_indices]]
    tolerance = absolute_tolerance_m + relative_tolerance * projected_depth.abs()
    return (
        in_bounds
        & torch.isfinite(projected_depth)
        & (projected_depth > 0)
        & torch.isfinite(sampled)
        & (sampled > 0)
        & ((sampled - projected_depth).abs() <= tolerance)
    )

