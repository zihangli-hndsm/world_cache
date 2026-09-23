"""Evaluate two-sided keyframe interpolation on a recorded RGB-D sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from worldcache.backbone.dinov2 import load_dinov2_vitb14


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image.to(device) - mean) / std).unsqueeze(0)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0))


def metric_row(estimate: torch.Tensor, output: torch.Tensor, prefix_tokens: int, recomputed: int) -> dict[str, float | int]:
    return {
        "full_cosine": cosine(estimate, output),
        "cls_cosine": float(torch.nn.functional.cosine_similarity(estimate[:, 0], output[:, 0], dim=1).item()),
        "patch_cosine": float(torch.nn.functional.cosine_similarity(estimate[:, prefix_tokens:], output[:, prefix_tokens:], dim=2).mean()),
        "recomputed": recomputed,
    }


def evaluate(outputs: list[torch.Tensor], stride: int) -> dict[str, object]:
    keyframes = list(range(0, len(outputs), stride))
    if keyframes[-1] != len(outputs) - 1:
        keyframes.append(len(outputs) - 1)
    values = []
    for index, output in enumerate(outputs):
        if index in keyframes:
            estimate = output
            recomputed = 1
        else:
            left = max(item for item in keyframes if item < index)
            right = min(item for item in keyframes if item > index)
            alpha = (index - left) / (right - left)
            estimate = (1.0 - alpha) * outputs[left] + alpha * outputs[right]
            recomputed = 0
        values.append(metric_row(estimate, output, 1, recomputed))
    frame = pd.DataFrame(values)
    result: dict[str, object] = {
        "stride": stride,
        "frames": len(frame),
        "keyframes": len(keyframes),
        "recompute_fraction": float(frame["recomputed"].mean()),
        "reuse_fraction": float(1.0 - frame["recomputed"].mean()),
    }
    for metric in ("full_cosine", "cls_cosine", "patch_cosine"):
        result[f"{metric}_mean"] = float(frame[metric].mean())
        result[f"{metric}_median"] = float(frame[metric].median())
        result[f"{metric}_p10"] = float(frame[metric].quantile(0.10))
        result[f"{metric}_min"] = float(frame[metric].min())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--sequence", choices=("reference", "rescan"), default="reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 3, 5, 10])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest[args.sequence]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_vitb14(int(config["resolution"]), device)
    outputs = []
    for ordinal, entry in enumerate(entries, start=1):
        observation = load(args.pair_run / entry["path"])
        with torch.inference_mode():
            outputs.append(model.forward_features(preprocess(observation["rgb"], device)).detach().float().cpu())
        print(f"features {ordinal}/{len(entries)} frame={entry['frame_id']}")
    summary = pd.DataFrame([evaluate(outputs, stride) for stride in args.strides])
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    torch.save(outputs, args.output_dir / "sequence_features.pt")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "complete", "sequence": args.sequence, "frames": len(outputs), "summary": str(args.output_dir / "summary.csv")}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
