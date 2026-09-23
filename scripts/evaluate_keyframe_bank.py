"""Evaluate keyframe-bank reuse and pose-aware interpolation on a trajectory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from worldcache.backbone.dinov2 import load_dinov2


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image.to(device) - mean) / std).unsqueeze(0)


def pose_from_record(record: dict[str, object]) -> np.ndarray:
    return np.linalg.inv(np.asarray(record["relative_pose"], dtype=np.float64))


def pose_delta(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    relative = np.linalg.inv(left) @ right
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(math.degrees(math.acos(cosine)))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0))


def metric_row(estimate: torch.Tensor, output: torch.Tensor, recomputed: int) -> dict[str, float | int]:
    return {
        "full_cosine": cosine(estimate, output),
        "cls_cosine": float(torch.nn.functional.cosine_similarity(estimate[:, 0], output[:, 0], dim=1).item()),
        "patch_cosine": float(torch.nn.functional.cosine_similarity(estimate[:, 1:], output[:, 1:], dim=2).mean()),
        "recomputed": recomputed,
    }


def evaluate_nearest(
    poses: list[np.ndarray],
    outputs: list[torch.Tensor],
    translation_threshold: float,
    rotation_threshold: float,
) -> dict[str, object]:
    keyframes = [0]
    predicted = [outputs[0]]
    rows = [{"frame": 0, "cache_index": 0, "translation_from_cache_m": 0.0, "rotation_from_cache_deg": 0.0, **metric_row(outputs[0], outputs[0], 1)}]
    for index in range(1, len(outputs)):
        translation, rotation = pose_delta(poses[keyframes[-1]], poses[index])
        refresh = translation > translation_threshold or rotation > rotation_threshold
        if refresh:
            keyframes.append(index)
            cached = outputs[index]
            recomputed = 1
            cache_index = index
        else:
            cached = outputs[keyframes[-1]]
            recomputed = 0
            cache_index = keyframes[-1]
        rows.append({"frame": index, "cache_index": cache_index, "translation_from_cache_m": translation, "rotation_from_cache_deg": rotation, **metric_row(cached, outputs[index], recomputed)})
    frame = pd.DataFrame(rows)
    result: dict[str, object] = {
        "policy": f"nearest_pose<={translation_threshold:g}m,{rotation_threshold:g}deg",
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


def evaluate_interpolation(poses: list[np.ndarray], outputs: list[torch.Tensor], stride: int) -> dict[str, object]:
    keyframes = list(range(0, len(outputs), stride))
    if keyframes[-1] != len(outputs) - 1:
        keyframes.append(len(outputs) - 1)
    rows = []
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
        rows.append({"frame": index, **metric_row(estimate, output, recomputed)})
    frame = pd.DataFrame(rows)
    result: dict[str, object] = {
        "policy": f"two_sided_linear_interpolation_stride_{stride}",
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


def evaluate_causal_extrapolation(outputs: list[torch.Tensor], stride: int) -> dict[str, object]:
    """Use only past keyframes; keep two initial keyframes for velocity."""
    keyframes = [0, 1]
    keyframes.extend(range(1 + stride, len(outputs), stride))
    keyframes = sorted(set(keyframes))
    rows = []
    for index, output in enumerate(outputs):
        if index in keyframes:
            estimate = output
            recomputed = 1
        else:
            left = max(item for item in keyframes if item < index)
            previous_candidates = [item for item in keyframes if item < left]
            previous = previous_candidates[-1] if previous_candidates else 0
            step = max(1, left - previous)
            alpha = (index - left) / step
            estimate = outputs[left] + alpha * (outputs[left] - outputs[previous])
            recomputed = 0
        rows.append({"frame": index, **metric_row(estimate, output, recomputed)})
    frame = pd.DataFrame(rows)
    result: dict[str, object] = {
        "policy": f"causal_extrapolation_stride_{stride}",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--translation-thresholds", type=float, nargs="+", default=[0.002, 0.004, 0.006, 0.010, 0.020])
    parser.add_argument("--rotation-thresholds", type=float, nargs="+", default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="base")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (args.pair_run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    # Pair 0 is the origin keyframe; identity controls are excluded from the
    # ordered trajectory after selecting their first identical frame.
    origin = records[0]
    trajectory = [record for record in records if record["pair_type"] != "identity"]
    records = [origin] + trajectory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(args.model_size, int(config["resolution"]), device)
    poses, outputs = [], []
    for ordinal, record in enumerate(records, start=1):
        observation = load(args.pair_run / record["current"])
        with torch.inference_mode():
            feature = model.forward_features(preprocess(observation["rgb"], device)).detach().float().cpu()
        poses.append(pose_from_record(record))
        outputs.append(feature)
        print(f"features {ordinal}/{len(records)} frame={record['pair_id']}")
    results = []
    for translation in args.translation_thresholds:
        for rotation in args.rotation_thresholds:
            results.append(evaluate_nearest(poses, outputs, translation, rotation))
    for stride in args.strides:
        results.append(evaluate_interpolation(poses, outputs, stride))
        results.append(evaluate_causal_extrapolation(outputs, stride))
    summary = pd.DataFrame(results)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    torch.save({"poses": poses, "outputs": outputs}, args.output_dir / "trajectory_features.pt")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "complete", "method": "pose_keyframe_bank", "model_size": args.model_size, "frames": len(outputs), "summary": str(args.output_dir / "summary.csv")}, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
