"""Compute the oracle quality/recompute frontier from saved ReplicaCAD pairs.

This is deliberately a representation-fidelity upper bound: current features
are computed in full before hybrid insertion. It must not be interpreted as a
latency result; real selective execution begins only after this gate.
"""

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
from worldcache.reuse.oracle import build_oracle_hybrid, resume_transformer


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((image - mean) / std).unsqueeze(0).to(device)


def correspondence(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor], patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = reference["depth"].shape
    centers_1d = torch.arange(patch_size / 2, height, patch_size)
    yy, xx = torch.meshgrid(centers_1d, centers_1d, indexing="ij")
    pixels = torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()
    source = torch.arange(len(pixels))
    u, v = pixels[:, 0].long(), pixels[:, 1].long()
    depth = reference["depth"][v, u]
    valid_depth = torch.isfinite(depth) & (depth > 0)
    world = backproject_pixels(pixels[valid_depth], depth[valid_depth], reference["intrinsics"], reference["camera_to_world"])
    projected, projected_depth = project_world_points(world, current["intrinsics"], current["camera_to_world"])
    visible = depth_visibility_mask(projected, projected_depth, current["depth"])
    target, valid_grid = pixel_to_patch_index(projected, height, width, patch_size)
    keep = visible & valid_grid
    return source[valid_depth][keep], target[keep]


def token_metrics(hybrid: torch.Tensor, full: torch.Tensor, prefix_tokens: int) -> dict[str, float]:
    patch_hybrid, patch_full = hybrid[:, prefix_tokens:], full[:, prefix_tokens:]
    return {
        "final_cls_cosine": float(torch.nn.functional.cosine_similarity(hybrid[:, 0], full[:, 0], dim=-1).mean()),
        "final_patch_cosine": float(torch.nn.functional.cosine_similarity(patch_hybrid, patch_full, dim=-1).mean()),
        "final_l2": float(torch.sqrt(torch.mean((hybrid - full).square()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8, 10])
    parser.add_argument("--recompute-ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--selection-policy", choices=("random", "feature_oracle"), default="random")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text())
    records = [json.loads(line) for line in (args.pair_run / "pairs.jsonl").read_text().splitlines()]
    if args.limit is not None:
        records = records[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_vitb14(int(config["resolution"]), device)
    rows: list[dict[str, object]] = []
    for ordinal, record in enumerate(records, start=1):
        reference, current = load(args.pair_run / record["reference"]), load(args.pair_run / record["current"])
        source, target = correspondence(reference, current, args.patch_size)
        reference_layers = extract_block_tokens(model, preprocess(reference["rgb"], device), args.layers)
        current_layers = extract_block_tokens(model, preprocess(current["rgb"], device), args.layers)
        with torch.inference_mode():
            full_final = model.forward_features(preprocess(current["rgb"], device))
        patch_count = full_final.shape[1] - model.num_prefix_tokens
        for layer in args.layers:
            cached = reference_layers[layer].to(device)
            current_at_layer = current_layers[layer].to(device)
            selected_source, selected_target = source, target
            if args.selection_policy == "feature_oracle":
                prefix = model.num_prefix_tokens
                scores = torch.nn.functional.cosine_similarity(
                    cached[0, prefix + source.to(device)], current_at_layer[0, prefix + target.to(device)], dim=-1
                )
                order = torch.argsort(scores, descending=True).cpu()
                selected_source, selected_target = source[order], target[order]
            for ratio in args.recompute_ratios:
                desired_reuse = round(patch_count * (1.0 - ratio))
                generator = torch.Generator().manual_seed(10_000 * int(record["pair_id"]) + 100 * layer + round(ratio * 100))
                hybrid, reused = build_oracle_hybrid(
                    cached,
                    current_at_layer,
                    selected_source,
                    selected_target,
                    desired_reuse,
                    model.num_prefix_tokens,
                    generator,
                )
                final = resume_transformer(model, hybrid, layer)
                row: dict[str, object] = {
                    "pair_id": record["pair_id"],
                    "pair_type": record["pair_type"],
                    "selection_policy": args.selection_policy,
                    "layer": layer,
                    "requested_recompute_ratio": ratio,
                    "actual_recompute_ratio": 1.0 - reused / patch_count,
                    "reused_tokens": reused,
                    "available_oracle_tokens": len(target),
                }
                row.update(token_metrics(final, full_final, model.num_prefix_tokens))
                rows.append(row)
        print(f"[{ordinal}/{len(records)}] pair={record['pair_id']} valid_oracle_tokens={len(target)}")
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output_dir / "oracle_metrics.parquet", index=False)
    summary = frame.groupby(["layer", "requested_recompute_ratio"], as_index=False).agg(
        final_cls_cosine=("final_cls_cosine", "median"),
        final_patch_cosine=("final_patch_cosine", "median"),
        final_l2=("final_l2", "median"),
        actual_recompute_ratio=("actual_recompute_ratio", "median"),
        count=("pair_id", "count"),
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({"status": "complete", "rows": len(frame)}, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
