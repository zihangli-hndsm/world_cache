"""Low-capacity ranking rules for geometry-gated descriptor retrieval."""

from __future__ import annotations

import torch


def fuse_descriptor_geometry_scores(
    descriptor_scores: torch.Tensor,
    distances: torch.Tensor,
    alpha: float = 8.0,
) -> torch.Tensor:
    """Combine cosine similarity with metric proximity.

    ``alpha`` is measured in inverse metres because distances are metric.  The
    rule intentionally has no learned scene-specific parameters; alpha can be
    selected by scene-level cross-validation on training scenes.
    """
    if descriptor_scores.shape != distances.shape:
        raise ValueError("descriptor_scores and distances must have the same shape")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return descriptor_scores - float(alpha) * distances


def fuse_descriptor_geometry_ranks(
    descriptor_scores: torch.Tensor,
    distances: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Fuse within-candidate ranks, making the rule scale-invariant."""
    if descriptor_scores.shape != distances.shape:
        raise ValueError("descriptor_scores and distances must have the same shape")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    descriptor_order = torch.argsort(descriptor_scores, dim=-1, descending=True)
    distance_order = torch.argsort(distances, dim=-1, descending=False)
    descriptor_rank = torch.empty_like(descriptor_order)
    distance_rank = torch.empty_like(distance_order)
    positions = torch.arange(descriptor_scores.shape[-1], device=descriptor_scores.device)
    positions = positions.expand_as(descriptor_order)
    descriptor_rank.scatter_(-1, descriptor_order, positions)
    distance_rank.scatter_(-1, distance_order, positions)
    return -descriptor_rank.float() - float(beta) * distance_rank.float()
