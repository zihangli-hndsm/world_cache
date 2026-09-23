"""Evaluate an input-gated whole-frame keyframe cache.

The cache stores the complete frozen-backbone output for a keyframe.  A cheap
RGB/pose gate decides whether a new view can reuse that output; the backbone
is run only when the gate rejects reuse.  This is a temporal/keyframe cache,
not hidden-state token substitution.
"""

from __future__ import annotations

import argparse
import json
import math
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


def pose_delta(record: dict[str, object]) -> tuple[float, float]:
    transform = np.asarray(record["relative_pose"], dtype=np.float64)
    translation = float(np.linalg.norm(transform[:3, 3]))
    cosine = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation = float(math.degrees(math.acos(cosine)))
    return translation, rotation


def image_delta(reference: torch.Tensor, current: torch.Tensor) -> tuple[float, float]:
    left = reference.float().div(255.0)
    right = current.float().div(255.0)
    difference = (left - right).abs()
    return float(difference.mean()), float(difference.quantile(0.95))


def output_metrics(cached: torch.Tensor, full: torch.Tensor, prefix_tokens: int) -> dict[str, float]:
    flat_cached, flat_full = cached.flatten(1), full.flatten(1)
    patch_cached, patch_full = cached[:, prefix_tokens:], full[:, prefix_tokens:]
    return {
        "output_cosine": float(torch.nn.functional.cosine_similarity(flat_cached, flat_full, dim=1).mean()),
        "cls_cosine": float(torch.nn.functional.cosine_similarity(cached[:, 0], full[:, 0], dim=1).mean()),
        "patch_cosine": float(torch.nn.functional.cosine_similarity(patch_cached, patch_full, dim=2).mean()),
        "output_l2": float(torch.sqrt(torch.mean((cached - full).square()))),
    }


def summarize_policy(frame: pd.DataFrame, gate: pd.Series, label: str) -> dict[str, object]:
    reused = frame[gate]
    result: dict[str, object] = {
        "policy": label,
        "queries": int(len(frame)),
        "reused": int(len(reused)),
        "recompute": int(len(frame) - len(reused)),
        "reuse_fraction": float(len(reused) / len(frame)) if len(frame) else float("nan"),
    }
    for metric in ("output_cosine", "cls_cosine", "patch_cosine", "output_l2"):
        values = reused[metric]
        result[f"{metric}_median"] = float(values.median()) if len(values) else float("nan")
        result[f"{metric}_p10"] = float(values.quantile(0.10)) if len(values) else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rgb-thresholds", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.10])
    parser.add_argument("--translation-thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--rotation-thresholds", type=float, nargs="+", default=[1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (args.pair_run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        records = records[: args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_vitb14(int(config["resolution"]), device)

    rows: list[dict[str, object]] = []
    for ordinal, record in enumerate(records, start=1):
        reference = load(args.pair_run / record["reference"])
        current = load(args.pair_run / record["current"])
        with torch.inference_mode():
            cached = model.forward_features(preprocess(reference["rgb"], device)).detach().float()
            full = model.forward_features(preprocess(current["rgb"], device)).detach().float()
        translation, rotation = pose_delta(record)
        rgb_mean, rgb_p95 = image_delta(reference["rgb"], current["rgb"])
        row: dict[str, object] = {
            "pair_id": int(record["pair_id"]),
            "pair_type": record["pair_type"],
            "translation_m": translation,
            "rotation_degrees": rotation,
            "rgb_mean_abs": rgb_mean,
            "rgb_p95_abs": rgb_p95,
        }
        row.update(output_metrics(cached, full, model.num_prefix_tokens))
        rows.append(row)
        print(f"[{ordinal}/{len(records)}] pair={record['pair_id']} rgb={rgb_mean:.5f} output={row['output_cosine']:.5f}")

    frame = pd.DataFrame(rows)
    frame.to_parquet(args.output_dir / "keyframe_metrics.parquet", index=False)
    summaries: list[dict[str, object]] = []
    non_identity = frame[frame["pair_type"] != "identity"].reset_index(drop=True)
    for threshold in args.rgb_thresholds:
        summaries.append(summarize_policy(non_identity, non_identity["rgb_mean_abs"] <= threshold, f"rgb_mean<={threshold:g}"))
    for threshold in args.translation_thresholds:
        summaries.append(summarize_policy(non_identity, non_identity["translation_m"] <= threshold, f"translation<={threshold:g}m"))
    for threshold in args.rotation_thresholds:
        summaries.append(summarize_policy(non_identity, non_identity["rotation_degrees"] <= threshold, f"rotation<={threshold:g}deg"))
    for translation in args.translation_thresholds:
        for rotation in args.rotation_thresholds:
            gate = (non_identity["translation_m"] <= translation) & (non_identity["rotation_degrees"] <= rotation)
            summaries.append(summarize_policy(non_identity, gate, f"pose<={translation:g}m,{rotation:g}deg"))
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "keyframe_summary.csv", index=False)
    (args.output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "whole_frame_keyframe_reuse_with_cheap_gate",
                "backbone": "vit_base_patch14_dinov2.lvd142m",
                "resolution": int(config["resolution"]),
                "pairs": len(frame),
                "non_identity_pairs": len(non_identity),
                "metrics": str(args.output_dir / "keyframe_metrics.parquet"),
                "summary": str(args.output_dir / "keyframe_summary.csv"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
