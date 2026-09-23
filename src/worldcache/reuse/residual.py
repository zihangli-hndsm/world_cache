"""Lightweight token-wise residual predictor for cached ViT features."""

from __future__ import annotations

import torch


class TokenResidualMLP(torch.nn.Module):
    def __init__(self, dimension: int = 768, hidden_dimension: int = 1024) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * dimension),
            torch.nn.Linear(2 * dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, dimension),
        )

    def forward(self, cached: torch.Tensor, cheap_current: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((cached, cheap_current), dim=-1))
