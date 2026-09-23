"""Train and evaluate a low-cost final-output residual cache.

Unlike the rejected layer-4 pivot, this adapter predicts the final output
directly.  The online path computes only an early prefix of the current image,
then applies a token-wise residual adapter to a cached keyframe output.  No
global-attention suffix is executed after cache insertion.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

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
def forward_to_layer(model: nn.Module, image: torch.Tensor, layer: int) -> torch.Tensor:
    """Run the real online prefix and return post-block tokens."""
    if layer < 1 or layer > len(model.blocks):
        raise ValueError(f"layer must be in [1, {len(model.blocks)}]")
    tokens = model.patch_embed(image)
    tokens = model._pos_embed(tokens)
    tokens = model.patch_drop(tokens)
    tokens = model.norm_pre(tokens)
    for block in model.blocks[:layer]:
        tokens = block(tokens)
    return tokens.detach().float()


class ResidualAdapter(nn.Module):
    def __init__(self, width: int = 768, hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Linear(width * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )

    def forward(self, early: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((early, cached), dim=-1))


def extract_training_tensors(
    model: nn.Module,
    pair_root: Path,
    records: list[dict[str, object]],
    early_layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    early_rows, cached_rows, target_rows = [], [], []
    for ordinal, record in enumerate(records, start=1):
        reference = load(pair_root / record["reference"])
        current = load(pair_root / record["current"])
        with torch.inference_mode():
            early = forward_to_layer(model, preprocess(current["rgb"], device), early_layer)[0].cpu()
            cached = model.forward_features(preprocess(reference["rgb"], device))[0].detach().float().cpu()
            target = model.forward_features(preprocess(current["rgb"], device))[0].detach().float().cpu()
        early_rows.append(early)
        cached_rows.append(cached)
        target_rows.append(target)
        print(f"features {ordinal}/{len(records)} pair={record['pair_id']}")
    return torch.stack(early_rows), torch.stack(cached_rows), torch.stack(target_rows)


def train_adapter(
    early: torch.Tensor,
    cached: torch.Tensor,
    target: torch.Tensor,
    prefix_tokens: int,
    device: torch.device,
    epochs: int,
    seed: int,
) -> ResidualAdapter:
    torch.manual_seed(seed)
    adapter = ResidualAdapter(width=int(target.shape[-1])).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-4, weight_decay=1e-4)
    # Train patches only. CLS is evaluated as a separate, conservative cache
    # baseline because it is globally coupled and should not be overclaimed.
    early = early[:, prefix_tokens:].reshape(-1, early.shape[-1])
    cached = cached[:, prefix_tokens:].reshape(-1, cached.shape[-1])
    target = target[:, prefix_tokens:].reshape(-1, target.shape[-1])
    generator = torch.Generator().manual_seed(seed)
    batch_size = 2048
    for epoch in range(epochs):
        order = torch.randperm(len(target), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch_early = early[index].to(device)
            batch_cached = cached[index].to(device)
            batch_target = target[index].to(device)
            prediction = batch_cached + adapter(batch_early, batch_cached)
            loss = torch.mean((prediction - batch_target).square())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        if epoch in (0, epochs // 4, epochs // 2, epochs - 1):
            print(f"epoch={epoch + 1}/{epochs} patch_mse={total / len(target):.6f}")
    return adapter.eval()


def evaluate(
    adapter: ResidualAdapter,
    early: torch.Tensor,
    cached: torch.Tensor,
    target: torch.Tensor,
    prefix_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    with torch.inference_mode():
        predicted = cached.clone()
        patch_delta = adapter(early[:, prefix_tokens:].to(device), cached[:, prefix_tokens:].to(device)).cpu()
        predicted[:, prefix_tokens:] = cached[:, prefix_tokens:] + patch_delta
    def metrics(left: torch.Tensor, right: torch.Tensor, label: str) -> dict[str, float]:
        result = {
            f"{label}_output_cosine": float(torch.nn.functional.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).median()),
            f"{label}_cls_cosine": float(torch.nn.functional.cosine_similarity(left[:, 0], right[:, 0], dim=1).median()),
            f"{label}_patch_cosine": float(torch.nn.functional.cosine_similarity(left[:, prefix_tokens:], right[:, prefix_tokens:], dim=2).mean(dim=1).median()),
            f"{label}_output_l2": float(torch.sqrt(torch.mean((left - right).square(), dim=(1, 2))).median()),
        }
        return result
    result = metrics(cached, target, "hard_cache")
    result.update(metrics(predicted, target, "residual_cache"))
    result["residual_patch_mse"] = float(torch.mean((predicted[:, prefix_tokens:] - target[:, prefix_tokens:]).square()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--early-layer", type=int, default=1)
    parser.add_argument("--train-pairs", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (args.pair_run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["pair_type"] != "identity"
    ]
    train_records = records[: args.train_pairs]
    test_records = records[args.train_pairs :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_vitb14(int(config["resolution"]), device)
    prefix_tokens = int(model.num_prefix_tokens)

    train_early, train_cached, train_target = extract_training_tensors(model, args.pair_run, train_records, args.early_layer, device)
    test_early, test_cached, test_target = extract_training_tensors(model, args.pair_run, test_records, args.early_layer, device)
    adapter = train_adapter(train_early, train_cached, train_target, prefix_tokens, device, args.epochs, args.seed)
    result = evaluate(adapter, test_early, test_cached, test_target, prefix_tokens, device)
    result.update(
        {
            "status": "complete",
            "method": "final_output_residual_adapter",
            "early_layer": args.early_layer,
            "train_pairs": len(train_records),
            "test_pairs": len(test_records),
            "epochs": args.epochs,
            "device": str(device),
            "backbone": "vit_base_patch14_dinov2.lvd142m",
            "patch_recompute_floor_approx": args.early_layer / len(model.blocks),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "complete", "summary": str(args.output_dir / "summary.json")}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
