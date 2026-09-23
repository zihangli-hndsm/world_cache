"""Geometry-aware local receptive-field descriptor utilities."""

from __future__ import annotations

import torch


def encode_world_receptive_field(
    points: torch.Tensor,
    descriptors: torch.Tensor,
    radius: float,
    sigma: float | None = None,
) -> torch.Tensor:
    """Pool descriptors over a metric 3-D neighborhood around each point."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if descriptors.ndim != 2 or descriptors.shape[0] != points.shape[0]:
        raise ValueError("descriptors must have shape [N, C]")
    if radius <= 0:
        raise ValueError("radius must be positive")
    sigma = radius * 0.5 if sigma is None else sigma
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    distances = torch.cdist(points, points)
    weights = torch.exp(-0.5 * (distances / sigma).square()) * (distances <= radius)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = weights @ torch.nn.functional.normalize(descriptors, dim=1)
    return torch.nn.functional.normalize(pooled, dim=1)
