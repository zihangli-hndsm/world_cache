"""Measure the small-router overhead against one frozen DINOv2 forward.

This is a runtime proxy for the long-horizon cache prototype.  The trajectory
experiments precompute teacher outputs, so they cannot by themselves claim
wall-clock speedup.  This benchmark measures the two online costs separately:
the frozen backbone refresh and the learned refresh-router decision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_long_horizon_cache_router import RefreshRouter
from worldcache.backbone.dinov2 import load_dinov2


def preprocess(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = rgb.permute(2, 0, 1).float().div(255.0).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image - mean) / std).unsqueeze(0)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, repetitions: int, warmups: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmups):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions):
        fn()
    synchronize(device)
    elapsed = time.perf_counter() - start
    per_call_ms = elapsed * 1000.0 / repetitions
    return per_call_ms, 1000.0 / per_call_ms if per_call_ms else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="base")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (args.pair_run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    record = next(record for record in records if record["pair_type"] != "identity")
    with np.load(args.pair_run / record["current"]) as data:
        rgb = torch.from_numpy(data["rgb"].copy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = preprocess(rgb, device)

    model = load_dinov2(args.model_size, args.resolution, device)
    with torch.inference_mode():
        output = model.forward_features(image)
    full_ms, full_per_second = timed(
        lambda: model.forward_features(image), args.repetitions, args.warmups, device
    )

    payload = torch.load(args.router, map_location="cpu", weights_only=False)
    router = RefreshRouter(int(payload["input_width"]))
    router.load_state_dict(payload["model"])
    router.eval().to(device)
    router_input = torch.randn(1, int(payload["input_width"]), device=device)
    router_ms, router_per_second = timed(
        lambda: router(router_input), args.repetitions * 10, args.warmups, device
    )
    cache = output.detach()
    cache_read_ms, cache_read_per_second = timed(
        lambda: cache.clone(), args.repetitions * 10, args.warmups, device
    )

    result = {
        "status": "complete",
        "device": str(device),
        "pair_run": str(args.pair_run),
        "resolution": args.resolution,
        "backbone": f"vit_{args.model_size}_patch14_dinov2.lvd142m",
        "output_shape": list(output.shape),
        "output_bytes_fp32": int(output.numel() * 4),
        "router_parameters": int(sum(parameter.numel() for parameter in router.parameters())),
        "frozen_backbone_forward_ms": full_ms,
        "frozen_backbone_forward_per_second": full_per_second,
        "router_decision_ms": router_ms,
        "router_decision_per_second": router_per_second,
        "cached_output_clone_ms": cache_read_ms,
        "cached_output_clone_per_second": cache_read_per_second,
        "router_to_backbone_time_ratio": router_ms / full_ms,
        "note": "Backbone and cache output are measured separately; this is not an end-to-end sparse VLM benchmark.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
