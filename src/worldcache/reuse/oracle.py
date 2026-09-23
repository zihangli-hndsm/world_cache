"""Oracle hybrid features and suffix execution for a frozen ViT."""

from __future__ import annotations

import torch


def build_oracle_hybrid(
    reference_tokens: torch.Tensor,
    current_tokens: torch.Tensor,
    source_patch_indices: torch.Tensor,
    target_patch_indices: torch.Tensor,
    reuse_count: int,
    prefix_tokens: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Copy a random subset of geometrically matched reference patches into current tokens.

    One source is retained per target patch to avoid overwrite-order dependence.
    Prefix tokens (including CLS) always remain current-frame tokens.
    """
    if reference_tokens.shape != current_tokens.shape or reference_tokens.ndim != 3:
        raise ValueError("reference_tokens and current_tokens must have identical [B, N, D] shape")
    if reference_tokens.shape[0] != 1:
        raise ValueError("oracle prototype currently accepts batch size 1")
    selected_source: list[int] = []
    selected_target: list[int] = []
    seen: set[int] = set()
    for source, target in zip(source_patch_indices.tolist(), target_patch_indices.tolist()):
        if target not in seen:
            selected_source.append(source)
            selected_target.append(target)
            seen.add(target)
    available = len(selected_target)
    reuse_count = min(max(reuse_count, 0), available)
    if reuse_count == 0:
        return current_tokens.clone(), 0
    selected = torch.randperm(available, generator=generator)[:reuse_count]
    source = torch.tensor(selected_source, dtype=torch.long)[selected].to(current_tokens.device)
    target = torch.tensor(selected_target, dtype=torch.long)[selected].to(current_tokens.device)
    hybrid = current_tokens.clone()
    hybrid[:, prefix_tokens + target] = reference_tokens[:, prefix_tokens + source]
    return hybrid, reuse_count


@torch.inference_mode()
def resume_transformer(model: torch.nn.Module, tokens_after_layer: torch.Tensor, insertion_layer: int) -> torch.Tensor:
    """Run remaining 1-indexed ViT blocks plus final norm after an insertion layer."""
    if insertion_layer < 1 or insertion_layer > len(model.blocks):
        raise ValueError(f"insertion_layer must be in [1, {len(model.blocks)}]")
    output = tokens_after_layer
    for block in model.blocks[insertion_layer:]:
        output = block(output)
    return model.norm(output)
