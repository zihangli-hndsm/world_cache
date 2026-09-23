"""Patch-grid coordinate helpers for geometric token alignment."""

from __future__ import annotations

import torch


def pixel_to_patch_index(
    pixels_uv: torch.Tensor, image_height: int, image_width: int, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map pixels to row-major patch indices and return valid-pixel mask."""
    if patch_size <= 0 or image_height % patch_size or image_width % patch_size:
        raise ValueError("image dimensions must be positive multiples of patch_size")
    u, v = pixels_uv[:, 0], pixels_uv[:, 1]
    valid = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    cols = image_width // patch_size
    indices = (v.floor().long() // patch_size) * cols + (u.floor().long() // patch_size)
    indices = torch.where(valid, indices, torch.full_like(indices, -1))
    return indices, valid


def patch_sample_pixels(
    patch_row: int, patch_col: int, patch_size: int, samples_per_axis: int = 4
) -> torch.Tensor:
    """Return sub-patch sample positions as pixel centers in (u, v) order."""
    if samples_per_axis <= 0:
        raise ValueError("samples_per_axis must be positive")
    offsets = (torch.arange(samples_per_axis, dtype=torch.float32) + 0.5) * patch_size / samples_per_axis
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    return torch.stack((xx + patch_col * patch_size, yy + patch_row * patch_size), dim=-1).reshape(-1, 2)

