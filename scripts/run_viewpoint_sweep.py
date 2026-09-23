"""Extract frozen DINOv2 layer features and compare geometric controls."""

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


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((image - mean) / std).unsqueeze(0).to(device)


def patch_centers(height: int, width: int, patch_size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(patch_size / 2, height, patch_size),
        torch.arange(patch_size / 2, width, patch_size),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2).float()


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(left, right, dim=-1)


def correspondence(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor], patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = reference["depth"].shape
    source_pixels = patch_centers(height, width, patch_size)
    source_index = torch.arange(len(source_pixels))
    u, v = source_pixels[:, 0].long(), source_pixels[:, 1].long()
    source_depth = reference["depth"][v, u]
    depth_valid = torch.isfinite(source_depth) & (source_depth > 0)
    world = backproject_pixels(
        source_pixels[depth_valid], source_depth[depth_valid], reference["intrinsics"], reference["camera_to_world"]
    )
    projected, projected_depth = project_world_points(world, current["intrinsics"], current["camera_to_world"])
    visible = depth_visibility_mask(projected, projected_depth, current["depth"])
    target_index, in_grid = pixel_to_patch_index(projected, height, width, patch_size)
    keep = visible & in_grid
    return source_index[depth_valid][keep], target_index[keep]


def relative_motion(record: dict[str, object]) -> tuple[float, float]:
    transform = np.asarray(record["relative_pose"], dtype=np.float64)
    translation = float(np.linalg.norm(transform[:3, 3]))
    cos_angle = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(np.degrees(np.arccos(cos_angle)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12])
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (args.pair_run / "pairs.jsonl").read_text().splitlines()]
    if args.limit is not None:
        records = records[: args.limit]
    config = json.loads((args.pair_run / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_vitb14(int(config["resolution"]), device)
    rows: list[dict[str, object]] = []
    for ordinal, record in enumerate(records, start=1):
        reference, current = load(args.pair_run / record["reference"]), load(args.pair_run / record["current"])
        src, dst = correspondence(reference, current, args.patch_size)
        reference_layers = extract_block_tokens(model, preprocess(reference["rgb"], device), args.layers)
        current_layers = extract_block_tokens(model, preprocess(current["rgb"], device), args.layers)
        translation_m, rotation_degrees = relative_motion(record)
        generator = torch.Generator().manual_seed(int(record["pair_id"]))
        random_target = torch.randperm((config["resolution"] // args.patch_size) ** 2, generator=generator)[: len(src)]
        for layer in args.layers:
            prefix_tokens = model.num_prefix_tokens
            reference_tokens = reference_layers[layer][0, prefix_tokens:]
            current_tokens = current_layers[layer][0, prefix_tokens:]
            values = {
                "aligned": cosine(reference_tokens[src], current_tokens[dst]),
                "same_index": cosine(reference_tokens[src], current_tokens[src]),
                "random": cosine(reference_tokens[src], current_tokens[random_target]),
            }
            for control, scores in values.items():
                rows.extend(
                    {
                        "pair_id": record["pair_id"],
                        "pair_type": record["pair_type"],
                        "layer": layer,
                        "control": control,
                        "translation_m": translation_m,
                        "rotation_degrees": rotation_degrees,
                        "cosine": float(score),
                    }
                    for score in scores.tolist()
                )
        print(f"[{ordinal}/{len(records)}] pair={record['pair_id']} correspondences={len(src)}")
    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output_dir / "token_similarity.parquet", index=False)
    summary = frame.groupby(["layer", "control"], as_index=False)["cosine"].agg(["median", "mean", "count"]).reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "status.json").write_text(
        json.dumps({"status": "complete", "pairs": len(records), "rows": len(frame), "layers": args.layers}, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
