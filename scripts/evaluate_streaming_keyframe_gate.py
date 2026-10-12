#!/usr/bin/env python3
"""Replay a causal RGB-change keyframe refresh policy on ReplicaCAD routes.

The policy compares each current RGB frame with the last frame whose full
DINOv2 output was refreshed. It refreshes on a source-selected mean-RGB-change
threshold and otherwise reuses the last output. Scene-level leave-one-out
selection keeps the held-out route out of threshold fitting.
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


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/runs"
DEFAULT_RUNS = [
    "20260920_scaleup_apt1_traj101_s1",
    "20260920_scaleup_v3sc0_traj101_s1",
    "20260920_scaleup_v3sc1_traj101_s1",
    "20260920_scaleup_v3sc2_traj101_s1",
    "20260920_scaleup_v3sc3_traj101_s1",
]


def preprocess(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    image = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    image = image.permute(2, 0, 1).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    return ((image - mean) / std).unsqueeze(0)


def load_rgb(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        return archive["rgb"].copy()


def load_sequences(run_names: list[str]) -> dict[str, list[tuple[np.ndarray, dict[str, object]]]]:
    sequences = {}
    for run_name in run_names:
        run = RUNS / run_name
        rows = [json.loads(line) for line in (run / "pairs.jsonl").read_text().splitlines()]
        identity = sorted((row for row in rows if row["pair_type"] == "identity"), key=lambda row: row["pair_id"])
        motion = sorted((row for row in rows if row["pair_type"] == "trajectory"), key=lambda row: row["pair_id"])
        if not identity or not motion:
            raise ValueError(f"{run_name} needs an identity anchor and trajectory rows")
        # The identity records are exact copies of the same anchor. Keep one.
        selected = [identity[0], *motion]
        records = [(load_rgb(run / row["current"]), row) for row in selected]
        if len({tuple(rgb.shape) for rgb, _ in records}) != 1:
            raise ValueError(f"inconsistent RGB shape in {run_name}")
        sequences[run_name] = records
    return sequences


@torch.inference_mode()
def extract_features(sequences, device: torch.device, resolution: int):
    model = load_dinov2_vitb14(resolution, device).eval()
    outputs = {}
    for run_name, records in sequences.items():
        seq_outputs = []
        for index, (rgb, _) in enumerate(records):
            output = model.forward_features(preprocess(rgb, device)).detach().float().cpu()
            seq_outputs.append(output)
            print(f"features {run_name} {index + 1}/{len(records)}", flush=True)
        outputs[run_name] = seq_outputs
    return model.num_prefix_tokens, outputs


def pairwise_scores(outputs: list[torch.Tensor], prefix_tokens: int, device: torch.device):
    stack = torch.cat(outputs, dim=0).to(device)
    flat = F.normalize(stack.flatten(1), dim=1)
    output = (flat @ flat.T).clamp(-1.0, 1.0).cpu().numpy()
    patch = F.normalize(stack[:, prefix_tokens:], dim=2)
    patch = torch.einsum("ntd,mtd->nm", patch, patch).div_(patch.shape[1]).clamp(-1.0, 1.0)
    return output, patch.cpu().numpy()


def pairwise_rgb_change(records):
    # Downsample by 2 in each spatial axis for a low-cost gate signal.
    images = np.stack([rgb[::2, ::2].astype(np.float32) / 255.0 for rgb, _ in records])
    matrix = np.empty((len(images), len(images)), dtype=np.float32)
    for left in range(len(images)):
        matrix[left] = np.abs(images - images[left]).mean(axis=(1, 2, 3))
    return matrix


def replay(run_name, records, rgb_changes, scores, threshold: float | None, stride: int | None):
    keyframe = 0
    rows = [{
        "run": run_name, "frame_index": 0, "change_from_cache": 0.0,
        "refreshed": 1, "output_cosine": 1.0, "patch_cosine": 1.0,
        "keyframe_index": 0,
    }]
    for index in range(1, len(records)):
        rgb_change = float(rgb_changes[keyframe, index])
        if threshold is not None:
            refresh = rgb_change > threshold
        else:
            refresh = index % int(stride) == 0
        if refresh:
            keyframe = index
        rows.append({
            "run": run_name, "frame_index": index, "change_from_cache": rgb_change,
            "refreshed": int(refresh),
            "output_cosine": float(scores[0][keyframe, index]),
            "patch_cosine": float(scores[1][keyframe, index]),
            "keyframe_index": keyframe,
        })
    return rows


def summarize(rows: list[dict[str, object]], policy: str, threshold: float | None, stride: int | None):
    frame = pd.DataFrame(rows)
    return {
        "policy": policy,
        "threshold": threshold,
        "stride": stride,
        "frames": int(len(frame)),
        "recompute_fraction": float(frame["refreshed"].mean()),
        "reuse_fraction": float(1.0 - frame["refreshed"].mean()),
        "output_cosine_mean": float(frame["output_cosine"].mean()),
        "output_cosine_p10": float(frame["output_cosine"].quantile(0.10)),
        "output_cosine_min": float(frame["output_cosine"].min()),
        "patch_cosine_p10": float(frame["patch_cosine"].quantile(0.10)),
        "patch_cosine_min": float(frame["patch_cosine"].min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[round(x * 0.0025, 4) for x in range(1, 81)])
    parser.add_argument("--quality-floor", type=float, default=0.95)
    parser.add_argument("--fixed-strides", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences(args.runs)
    resolution = int(json.loads((RUNS / args.runs[0] / "config.json").read_text())["resolution"])
    device = torch.device(args.device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    prefix_tokens, features = extract_features(sequences, device, resolution)
    score_matrices = {name: pairwise_scores(values, prefix_tokens, device) for name, values in features.items()}
    rgb_change_matrices = {name: pairwise_rgb_change(records) for name, records in sequences.items()}

    # Each held-out scene gets a threshold selected only from the other scenes.
    heldout_rows = []
    fold_thresholds = []
    for target in args.runs:
        source_runs = [name for name in args.runs if name != target]
        feasible = []
        for threshold in args.thresholds:
            source_summaries = [summarize(
                replay(name, sequences[name], rgb_change_matrices[name], score_matrices[name], threshold, None),
                "rgb_change", threshold, None,
            ) for name in source_runs]
            if all(item["output_cosine_p10"] >= args.quality_floor for item in source_summaries):
                feasible.append((float(np.mean([item["reuse_fraction"] for item in source_summaries])), threshold))
        selected = max(feasible)[1] if feasible else 0.0
        fold_thresholds.append({"target": target, "source_runs": source_runs,
                                "threshold": selected, "feasible_thresholds": len(feasible),
                                "source_quality_floor": args.quality_floor})
        target_rows = replay(target, sequences[target], rgb_change_matrices[target], score_matrices[target], selected, None)
        heldout_rows.extend(target_rows)
        print(f"LOSO target={target} selected_threshold={selected:.4f} "
              f"feasible={len(feasible)}", flush=True)

    prediction_frame = pd.DataFrame(heldout_rows)
    prediction_frame.to_csv(args.output_dir / "heldout_frame_rows.csv", index=False)
    fold_summaries = [summarize(
        prediction_frame[prediction_frame["run"] == item["target"]].to_dict("records"),
        "source_selected_rgb_change", item["threshold"], None,
    ) | {"run": item["target"]} for item in fold_thresholds]
    baseline_summaries = []
    for stride in args.fixed_strides:
        for run_name, records in sequences.items():
            rows = replay(run_name, records, rgb_change_matrices[run_name], score_matrices[run_name], None, stride)
            baseline_summaries.append(summarize(rows, "fixed_stride_last_output", None, stride) | {"run": run_name})
    all_summaries = fold_summaries + baseline_summaries
    pd.DataFrame(all_summaries).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "fold_thresholds.json").write_text(json.dumps(fold_thresholds, indent=2) + "\n")
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "status": "complete",
        "backbone": "DINOv2 ViT-B/14, frozen final-layer outputs",
        "dataset": "five ReplicaCAD routes; scene-group leave-one-out",
        "inference_policy": "mean absolute RGB change from last refreshed image; refresh iff score exceeds source-selected threshold",
        "quality_floor": args.quality_floor,
        "future_frames_used_by_policy": False,
        "target_features_used_only_for_scoring": True,
        "run_names": args.runs,
        "output_rows": str(args.output_dir / "heldout_frame_rows.csv"),
        "summary": str(args.output_dir / "summary.csv"),
    }, indent=2) + "\n")
    print(pd.DataFrame(all_summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
