"""Evaluate an iterative sparse-update cache for a global ViT.

The cache stores each keyframe block input/output's K/V context.  For a new
frame, only selected patch tokens and CLS are updated at every block.  The
selected queries attend to cached K/V for unchanged tokens and current K/V for
selected tokens; unchanged token outputs are reused.  This is a fidelity
prototype for a cache-aware sparse kernel, not a production latency claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from worldcache.backbone.dinov2 import load_dinov2_vitb14


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image.to(device) - mean) / std).unsqueeze(0)


@torch.inference_mode()
def block_states(model: torch.nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    x = model.patch_embed(image)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    inputs = []
    outputs = []
    for block in model.blocks:
        inputs.append(x.detach().clone())
        x = block(x)
        outputs.append(x.detach().clone())
    return x, inputs, outputs


@torch.inference_mode()
def reference_kv(model: torch.nn.Module, block: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = block.norm1(x)
    qkv = block.attn.qkv(normalized).reshape(normalized.shape[0], normalized.shape[1], 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    return block.attn.q_norm(q), block.attn.k_norm(k), v


@torch.inference_mode()
def attention_residual(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    normalized = block.norm1(x)
    qkv = block.attn.qkv(normalized).reshape(1, normalized.shape[1], 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = block.attn.q_norm(q), block.attn.k_norm(k)
    attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    attended = attended.transpose(1, 2).reshape_as(x)
    attended = block.attn.norm(attended)
    if block.attn.gate is not None:
        attended = attended * block.attn.gate(normalized).sigmoid()
    return block.drop_path1(block.ls1(block.attn.proj_drop(block.attn.proj(attended))))


@torch.inference_mode()
def sparse_block(
    block: torch.nn.Module,
    current_input: torch.Tensor,
    reference_input: torch.Tensor,
    reference_output: torch.Tensor,
    selected: torch.Tensor,
    cached_kv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cached_mlp_delta: torch.Tensor,
    all_queries: bool,
) -> torch.Tensor:
    """Update selected token rows while reusing the reference block output elsewhere."""
    indices = selected.nonzero(as_tuple=False).flatten()
    current_selected = current_input[:, indices]
    norm_selected = block.norm1(current_selected)
    qkv = block.attn.qkv(norm_selected).reshape(1, len(indices), 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
    q_selected, k_selected, v_selected = qkv.unbind(0)
    q_selected, k_selected = block.attn.q_norm(q_selected), block.attn.k_norm(k_selected)
    cached_q, cached_k, cached_v = cached_kv
    k_all = cached_k.clone()
    v_all = cached_v.clone()
    k_all[:, :, indices] = k_selected
    v_all[:, :, indices] = v_selected
    if all_queries:
        normalized_all = block.norm1(current_input)
        q_all = block.attn.qkv(normalized_all).reshape(1, current_input.shape[1], 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)[0]
        q_all = block.attn.q_norm(q_all)
        attended = F.scaled_dot_product_attention(q_all, k_all, v_all, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(1, current_input.shape[1], block.attn.attn_dim)
    else:
        attended = F.scaled_dot_product_attention(q_selected, k_all, v_all, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(1, len(indices), block.attn.attn_dim)
    attended = block.attn.norm(attended)
    if block.attn.gate is not None:
        attended = attended * block.attn.gate(block.norm1(current_input) if all_queries else norm_selected).sigmoid()
    attended = block.attn.proj(attended)
    attended = block.attn.proj_drop(attended)
    if all_queries:
        after_attention = current_input + block.drop_path1(block.ls1(attended))
        output = after_attention + cached_mlp_delta
        selected_after_attention = after_attention[:, indices]
        selected_after_mlp = selected_after_attention + block.drop_path2(
            block.ls2(block.mlp(block.norm2(selected_after_attention)))
        )
        output[:, indices] = selected_after_mlp
    else:
        selected_after_attention = current_selected + block.drop_path1(block.ls1(attended))
        selected_after_mlp = selected_after_attention + block.drop_path2(
            block.ls2(block.mlp(block.norm2(selected_after_attention)))
        )
        output = reference_output.clone()
        output[:, indices] = selected_after_mlp
    return output


def patch_change(reference_rgb: torch.Tensor, current_rgb: torch.Tensor, patch_size: int) -> torch.Tensor:
    ref = reference_rgb.float().div(255.0)
    cur = current_rgb.float().div(255.0)
    change = (ref - cur).abs().mean(dim=2)
    height, width = change.shape
    values = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            values.append(change[top : top + patch_size, left : left + patch_size].mean())
    return torch.stack(values)


def select_tokens(change: torch.Tensor, patch_fraction: float, device: torch.device) -> torch.Tensor:
    count = max(1, min(len(change), int(np.ceil(len(change) * patch_fraction))))
    selected = torch.zeros(len(change) + 1, dtype=torch.bool)
    selected[0] = True  # CLS is globally coupled and always refreshed.
    selected[1:][torch.topk(change, count).indices] = True
    return selected.to(device)


def metrics(output: torch.Tensor, full: torch.Tensor, prefix_tokens: int) -> dict[str, float]:
    return {
        "output_cosine": float(F.cosine_similarity(output.flatten(1), full.flatten(1), dim=1).mean()),
        "cls_cosine": float(F.cosine_similarity(output[:, 0], full[:, 0], dim=1).mean()),
        "patch_cosine": float(F.cosine_similarity(output[:, prefix_tokens:], full[:, prefix_tokens:], dim=2).mean()),
        "output_l2": float(torch.sqrt(torch.mean((output - full).square()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.50, 1.0])
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--update-mode", choices=("selected_queries", "all_queries"), default="all_queries")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.pair_run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["pair_type"] != "identity"
    ][: args.max_pairs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_vitb14(int(config["resolution"]), device)
    rows = []
    for ordinal, record in enumerate(records, start=1):
        reference = load(args.pair_run / record["reference"])
        current = load(args.pair_run / record["current"])
        with torch.inference_mode():
            reference_final, reference_inputs, reference_outputs = block_states(model, preprocess(reference["rgb"], device))
            current_final, _, _ = block_states(model, preprocess(current["rgb"], device))
        reference_input = reference_inputs[0]
        current_input = block_states(model, preprocess(current["rgb"], device))[1][0]
        change = 1.0 - F.cosine_similarity(reference_input[:, 1:], current_input[:, 1:], dim=-1).flatten().cpu()
        full = model.norm(current_final)
        for fraction in args.fractions:
            selected = select_tokens(change, fraction, device)
            current_state = reference_inputs[0].clone()
            current_state[:, selected] = current_input[:, selected]
            for index, block in enumerate(model.blocks):
                cached_kv = reference_kv(model, block, reference_inputs[index])
                reference_after_attention = reference_inputs[index] + attention_residual(block, reference_inputs[index])
                cached_mlp_delta = reference_outputs[index] - reference_after_attention
                current_state = sparse_block(
                    block,
                    current_state,
                    reference_inputs[index],
                    reference_outputs[index],
                    selected,
                    cached_kv,
                    cached_mlp_delta,
                    args.update_mode == "all_queries",
                )
            output = model.norm(current_state)
            row = {
                "pair_id": int(record["pair_id"]),
                "patch_fraction": fraction,
                "selected_token_fraction": float(selected.float().mean()),
                "selected_patch_count": int(selected[1:].sum()),
            }
            row.update(metrics(output, full, model.num_prefix_tokens))
            rows.append(row)
        print(f"[{ordinal}/{len(records)}] pair={record['pair_id']}")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "iterative_cache_metrics.csv", index=False)
    summary = frame.groupby("patch_fraction", as_index=False).agg(
        queries=("pair_id", "count"),
        selected_token_fraction=("selected_token_fraction", "mean"),
        output_cosine=("output_cosine", "mean"),
        output_cosine_median=("output_cosine", "median"),
        output_cosine_p10=("output_cosine", lambda values: float(values.quantile(0.10))),
        cls_cosine=("cls_cosine", "mean"),
        patch_cosine=("patch_cosine", "mean"),
        patch_cosine_p10=("patch_cosine", lambda values: float(values.quantile(0.10))),
        output_l2=("output_l2", "mean"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "status.json").write_text(
        json.dumps({"status": "complete", "method": "iterative_sparse_token_cache", "update_mode": args.update_mode, "pairs": len(records), "summary": str(args.output_dir / "summary.csv")}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
