"""Frozen DINOv2 backbone access."""

from .dinov2 import load_dinov2_vitb14, extract_block_tokens

__all__ = ["extract_block_tokens", "load_dinov2_vitb14"]

