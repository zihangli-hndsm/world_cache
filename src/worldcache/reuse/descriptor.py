"""Lightweight trainable projections for frozen visual descriptors."""

from __future__ import annotations

import torch
from torch import nn


class DescriptorProjection(nn.Module):
    """Project frozen backbone tokens into a correspondence metric space."""

    def __init__(self, input_width: int, hidden_width: int = 256, output_width: int = 128) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.output_width = int(output_width)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_width),
            nn.Linear(self.input_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.output_width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.network(tokens), dim=-1)


def save_descriptor_projection(model: DescriptorProjection, path: str) -> None:
    torch.save(
        {
            "input_width": model.input_width,
            "hidden_width": model.hidden_width,
            "output_width": model.output_width,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_descriptor_projection(path: str, device: torch.device) -> DescriptorProjection:
    payload = torch.load(path, map_location=device)
    model = DescriptorProjection(
        int(payload["input_width"]),
        int(payload["hidden_width"]),
        int(payload["output_width"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()
