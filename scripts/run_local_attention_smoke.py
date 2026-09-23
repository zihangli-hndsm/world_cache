"""Oracle local-attention reuse smoke test with a pretrained Swin backbone."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch


MODEL_NAME = "swin_tiny_patch4_window7_224.ms_in1k"


def load(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image - mean) / std).unsqueeze(0)


def run_with_captures(model: torch.nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for index, stage in enumerate(model.layers):
        handles.append(stage.register_forward_hook(lambda _, __, output, index=index: captures.__setitem__(index, output.detach())))
    try:
        features = model.forward_features(image)
    finally:
        for handle in handles:
            handle.remove()
    output = model.forward_head(features, pre_logits=True)
    return output.detach(), captures


def suffix_output(model: torch.nn.Module, stage_output: torch.Tensor, stage_index: int) -> torch.Tensor:
    x = stage_output
    for layer in list(model.layers)[stage_index + 1 :]:
        x = layer(x)
    x = model.norm(x)
    return model.forward_head(x, pre_logits=True)


def full_output(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    return model.forward_head(model.forward_features(image), pre_logits=True)


def window_mask(current: torch.Tensor, reference: torch.Tensor, window_size: int, ratio: float) -> tuple[torch.Tensor, int, int]:
    height, width = current.shape[1:3]
    score_map = 1.0 - torch.nn.functional.cosine_similarity(current, reference, dim=-1)
    windows = []
    for top in range(0, height, window_size):
        for left in range(0, width, window_size):
            windows.append((float(score_map[:, top : top + window_size, left : left + window_size].mean()), top, left))
    windows.sort(reverse=True)
    count = max(1, int(np.ceil(len(windows) * ratio)))
    mask = torch.zeros((1, height, width), dtype=torch.bool, device=current.device)
    for _, top, left in windows[:count]:
        mask[:, top : top + window_size, left : left + window_size] = True
    return mask[..., None], count, len(windows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (args.pair_run / "pairs.jsonl").read_text().splitlines() if json.loads(line)["pair_type"] != "identity"][: args.max_pairs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).eval().to(device).requires_grad_(False)
    window_size_value = model.layers[0].blocks[0].window_size
    window_size = int(window_size_value[0] if isinstance(window_size_value, tuple) else window_size_value)
    rows = []
    for record in records:
        reference = preprocess(load(args.pair_run / record["reference"])["rgb"], device)
        current = preprocess(load(args.pair_run / record["current"])["rgb"], device)
        with torch.inference_mode():
            reference_output, reference_stages = run_with_captures(model, reference)
            current_output, current_stages = run_with_captures(model, current)
            full_cosine = float(torch.nn.functional.cosine_similarity(reference_output, current_output).item())
            stage_costs = {stage: int(value.shape[1] * value.shape[2] * len(model.layers[stage].blocks)) for stage, value in current_stages.items()}
            for stage_index, current_stage in current_stages.items():
                reference_stage = reference_stages[stage_index]
                stage_window_size = min(window_size, current_stage.shape[1], current_stage.shape[2])
                suffix_cost = sum(cost for stage, cost in stage_costs.items() if stage > stage_index)
                stage_cost = stage_costs[stage_index]
                for ratio in args.ratios:
                    mask, recomputed_windows, total_windows = window_mask(current_stage, reference_stage, stage_window_size, ratio)
                    hybrid_stage = torch.where(mask, current_stage, reference_stage)
                    hybrid_output = suffix_output(model, hybrid_stage, stage_index)
                    output_cosine = float(torch.nn.functional.cosine_similarity(hybrid_output, current_output).item())
                    stage_cosine = float(torch.nn.functional.cosine_similarity(hybrid_stage.flatten(1), current_stage.flatten(1)).item())
                    for _ in range(args.warmup):
                        suffix_output(model, hybrid_stage, stage_index)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        suffix_output(model, hybrid_stage, stage_index)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    suffix_ms = (time.perf_counter() - start) * 1000.0 / args.repeats
                    full_ms_values = []
                    for _ in range(args.warmup):
                        full_output(model, current)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.repeats):
                        full_output(model, current)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    full_ms_values.append((time.perf_counter() - start) * 1000.0 / args.repeats)
                    theoretical_fraction = (stage_cost * (recomputed_windows / total_windows) + suffix_cost) / max(1, stage_cost + suffix_cost)
                    rows.append({
                        "pair_id": int(record["pair_id"]),
                        "stage": stage_index,
                        "stage_height": current_stage.shape[1],
                        "stage_width": current_stage.shape[2],
                        "window_size": stage_window_size,
                        "ratio_requested": ratio,
                        "recomputed_windows": recomputed_windows,
                        "total_windows": total_windows,
                        "recomputed_window_fraction": recomputed_windows / total_windows,
                        "theoretical_stage_suffix_compute_fraction": theoretical_fraction,
                        "stage_feature_cosine": stage_cosine,
                        "output_feature_cosine": output_cosine,
                        "reference_current_output_cosine": full_cosine,
                        "oracle_suffix_latency_ms": suffix_ms,
                        "full_forward_latency_ms": full_ms_values[0],
                        "latency_ratio_suffix_over_full": suffix_ms / full_ms_values[0],
                        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "local_attention_metrics.csv", index=False)
    frame.to_json(args.output_dir / "local_attention_metrics.jsonl", orient="records", lines=True)
    frame.groupby(["stage", "recomputed_window_fraction"], as_index=False).agg(
        queries=("pair_id", "count"),
        output_feature_cosine=("output_feature_cosine", "mean"),
        output_feature_cosine_std=("output_feature_cosine", "std"),
        theoretical_stage_suffix_compute_fraction=("theoretical_stage_suffix_compute_fraction", "mean"),
        oracle_suffix_latency_ms=("oracle_suffix_latency_ms", "mean"),
        full_forward_latency_ms=("full_forward_latency_ms", "mean"),
        latency_ratio_suffix_over_full=("latency_ratio_suffix_over_full", "mean"),
    ).to_csv(args.output_dir / "summary.csv", index=False)
    metadata = {
        "status": "complete",
        "dataset": "ReplicaCAD",
        "backbone": MODEL_NAME,
        "input_resolution": config["resolution"],
        "oracle_policy": "select highest reference/current stage-output difference windows; reuse reference features in all other windows",
        "latency_note": "oracle_suffix_latency excludes the cost of selectively recomputing the chosen windows; sparse window kernel not implemented",
        "pairs": len(records),
        "ratios": args.ratios,
        "window_size": window_size,
        "device": str(device),
        "metrics": str(args.output_dir / "summary.csv"),
    }
    (args.output_dir / "status.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(frame.groupby(["stage", "recomputed_window_fraction"], as_index=False).agg(output_feature_cosine=("output_feature_cosine", "mean"), theoretical_stage_suffix_compute_fraction=("theoretical_stage_suffix_compute_fraction", "mean"), latency_ratio_suffix_over_full=("latency_ratio_suffix_over_full", "mean")).to_string(index=False))


if __name__ == "__main__":
    main()
