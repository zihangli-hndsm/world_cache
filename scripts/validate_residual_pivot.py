"""One-hour feasibility check for cheap-current residual correction at layer 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2_vitb14
from worldcache.geometry.correspondence import pixel_to_patch_index
from worldcache.geometry.projection import backproject_pixels, project_world_points
from worldcache.geometry.visibility import depth_visibility_mask
from worldcache.reuse.oracle import resume_transformer
from worldcache.reuse.residual import TokenResidualMLP


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((image - mean) / std).unsqueeze(0).to(device)


def correspond(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor], patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = reference["depth"].shape
    yy, xx = torch.meshgrid(torch.arange(patch_size / 2, height, patch_size), torch.arange(patch_size / 2, width, patch_size), indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
    src = torch.arange(len(pixels)); depth = reference["depth"][pixels[:, 1].long(), pixels[:, 0].long()]
    valid = torch.isfinite(depth) & (depth > 0)
    world = backproject_pixels(pixels[valid], depth[valid], reference["intrinsics"], reference["camera_to_world"])
    projected, z = project_world_points(world, current["intrinsics"], current["camera_to_world"])
    visible = depth_visibility_mask(projected, z, current["depth"])
    dst, grid = pixel_to_patch_index(projected, height, width, patch_size)
    src, dst = src[valid][visible & grid], dst[visible & grid]
    # Deterministic one-source-per-target map.
    keep, seen = [], set()
    for i, target in enumerate(dst.tolist()):
        if target not in seen:
            seen.add(target); keep.append(i)
    keep_t = torch.tensor(keep, dtype=torch.long)
    return src[keep_t], dst[keep_t]


def metric(predicted: torch.Tensor, full: torch.Tensor, prefix: int) -> tuple[float, float]:
    cls = torch.nn.functional.cosine_similarity(predicted[:, 0], full[:, 0], dim=-1).mean()
    patch = torch.nn.functional.cosine_similarity(predicted[:, prefix:], full[:, prefix:], dim=-1).mean()
    return float(cls), float(patch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-pairs", type=int, default=45)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text())
    records = [json.loads(x) for x in (args.pair_run / "pairs.jsonl").read_text().splitlines() if json.loads(x)["pair_type"] != "identity"]
    device = torch.device("cuda")
    backbone = load_dinov2_vitb14(config["resolution"], device)
    examples = []
    for record in records:
        ref, cur = load(args.pair_run / record["reference"]), load(args.pair_run / record["current"])
        src, dst = correspond(ref, cur, 14)
        ref_layers = extract_block_tokens(backbone, preprocess(ref["rgb"], device), [1, 4])
        cur_layers = extract_block_tokens(backbone, preprocess(cur["rgb"], device), [1, 4])
        examples.append((record, src, dst, ref_layers, cur_layers, ref, cur))
    train, test = examples[: args.train_pairs], examples[args.train_pairs :]
    cached = torch.cat([x[3][4][0, 1 + x[1]] for x in train]).to(device)
    cheap = torch.cat([x[4][1][0, 1 + x[2]] for x in train]).to(device)
    target = torch.cat([x[4][4][0, 1 + x[2]] - x[3][4][0, 1 + x[1]] for x in train]).to(device)
    model = TokenResidualMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(args.epochs):
        order = torch.randperm(len(target), device=device)
        for indices in order.split(1024):
            loss = torch.nn.functional.mse_loss(model(cached[indices], cheap[indices]), target[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"epoch={epoch + 1} loss={float(loss):.6f}")
    model.eval()
    rows = []
    with torch.inference_mode():
        for record, src, dst, ref_layers, cur_layers, ref, cur in test:
            full = backbone.forward_features(preprocess(cur["rgb"], device))
            cur4 = cur_layers[4].to(device); cache4 = ref_layers[4].to(device); cheap1 = cur_layers[1].to(device)
            hard = cur4.clone(); learned = cur4.clone()
            hard[:, 1 + dst.to(device)] = cache4[:, 1 + src.to(device)]
            delta = model(cache4[0, 1 + src.to(device)], cheap1[0, 1 + dst.to(device)])
            learned[:, 1 + dst.to(device)] = cache4[:, 1 + src.to(device)] + delta
            hard_final = resume_transformer(backbone, hard, 4); learned_final = resume_transformer(backbone, learned, 4)
            hard_cls, hard_patch = metric(hard_final, full, backbone.num_prefix_tokens)
            learned_cls, learned_patch = metric(learned_final, full, backbone.num_prefix_tokens)
            rows.append({"pair_id": record["pair_id"], "reused_tokens": len(dst), "actual_recompute_ratio": 1 - len(dst) / 256, "hard_cls": hard_cls, "hard_patch": hard_patch, "learned_cls": learned_cls, "learned_patch": learned_patch})
    frame = pd.DataFrame(rows); frame.to_csv(args.output_dir / "test_metrics.csv", index=False)
    summary = frame.median(numeric_only=True).to_dict(); summary.update({"train_pairs": len(train), "test_pairs": len(test), "epochs": args.epochs})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "summary": summary}, args.output_dir / "residual_mlp.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
