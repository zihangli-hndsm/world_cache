"""Inspectable frozen DINOv2 ViT extraction through timm."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import timm


MODEL_NAMES = {
    "small": "vit_small_patch14_dinov2.lvd142m",
    "base": "vit_base_patch14_dinov2.lvd142m",
    "large": "vit_large_patch14_dinov2.lvd142m",
    "giant": "vit_giant_patch14_dinov2.lvd142m",
}


def load_dinov2(model_size: str, resolution: int, device: torch.device) -> torch.nn.Module:
    """Load a frozen DINOv2 ViT-S/B/L/G with interpolated position embeddings."""
    try:
        model_name = MODEL_NAMES[model_size]
    except KeyError as exc:
        raise ValueError(f"unknown DINOv2 model size {model_size!r}; choose from {sorted(MODEL_NAMES)}") from exc
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=resolution)
    model.eval().to(device).requires_grad_(False)
    return model


def load_dinov2_vitb14(resolution: int, device: torch.device) -> torch.nn.Module:
    """Load frozen ViT-B/14 with interpolated positional embeddings as needed."""
    return load_dinov2("base", resolution, device)


@torch.inference_mode()
def extract_block_tokens(model: torch.nn.Module, image: torch.Tensor, layers: Iterable[int]) -> dict[int, torch.Tensor]:
    """Return requested 1-indexed transformer-block outputs, including prefix tokens."""
    requested = sorted(set(layers))
    if not requested or requested[0] < 1 or requested[-1] > len(model.blocks):
        raise ValueError(f"layers must be within [1, {len(model.blocks)}]")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook(layer: int):
        def capture(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captured[layer] = output.detach().float().cpu()

        return capture

    for layer in requested:
        handles.append(model.blocks[layer - 1].register_forward_hook(hook(layer)))
    try:
        model.forward_features(image)
    finally:
        for handle in handles:
            handle.remove()
    if sorted(captured) != requested:
        raise RuntimeError(f"missing requested features: expected {requested}, got {sorted(captured)}")
    return captured
