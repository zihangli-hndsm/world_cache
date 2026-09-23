"""Evaluate compact cache states with a train-split PCA bottleneck.

The router prototype previously reported a CLS+mean-patch memory budget but did
not implement compression.  This script makes that budget concrete: fit PCA on
the train trajectories, store only a low-dimensional code, reconstruct the
CLS+mean-patch task state, and evaluate long-horizon reuse on held-out scenes.
The full token tensor remains available only as an evaluation teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from train_long_horizon_cache_router import (
    feature_for,
    load_sequence,
    pose_delta,
    stitch_sequences,
)


def summary_state(output: torch.Tensor) -> torch.Tensor:
    """Return the 1536-d CLS + mean-patch state used by the cache proxy."""
    return torch.cat([output[:, 0], output[:, 1:].mean(dim=1)], dim=1).squeeze(0)


def fit_pca(states: torch.Tensor, max_rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = states.mean(dim=0)
    centered = states - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(max_rank, vh.shape[0])
    return mean, vh[:rank].T.contiguous(), centered.square().mean().sqrt()


def encode_decode(states: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor, width: int) -> torch.Tensor:
    if width >= states.shape[1]:
        # The direct CLS+mean-patch cache is the 3 KB bf16 baseline.  It does
        # not need a learned decoder and should be reported separately from
        # lossy PCA latents.
        return states
    basis = basis[:, :width]
    code = (states - mean) @ basis
    return code @ basis.T + mean


def cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=1)


def sequence_metrics(
    sequence: dict[str, object],
    reconstructed: torch.Tensor,
    policy: str,
    state_threshold: float,
    router=None,
    router_mean=None,
    router_std=None,
    router_threshold: float = 0.30,
) -> dict[str, object]:
    originals = torch.stack([summary_state(output) for output in sequence["outputs"]])
    cache_index = 0
    rows = []
    for index in range(len(originals)):
        if index == 0 or policy == "full":
            refresh = True
        elif policy == "fixed5":
            refresh = index % 5 == 0
        elif policy == "oracle":
            refresh = bool(cosine(originals[cache_index : cache_index + 1], originals[index : index + 1])[0] < state_threshold)
        elif policy == "learned":
            features = feature_for(sequence, index, cache_index)
            features = (features - router_mean) / router_std
            probability = float(torch.sigmoid(router(features)).item())
            refresh = probability >= router_threshold
        else:
            raise ValueError(policy)
        cached = reconstructed[cache_index]
        current = originals[index]
        state_cosine = float(cosine(cached.unsqueeze(0), current.unsqueeze(0))[0])
        stale = state_cosine < state_threshold and not refresh
        rows.append({"refresh": int(refresh), "state_cosine": state_cosine, "stale": int(stale)})
        if refresh:
            cache_index = index
    frame = pd.DataFrame(rows)
    return {
        "scene": sequence["name"],
        "policy": policy,
        "frames": len(frame),
        "recompute_fraction": float(frame["refresh"].mean()),
        "reuse_fraction": float(1.0 - frame["refresh"].mean()),
        "state_cosine_mean": float(frame["state_cosine"].mean()),
        "state_cosine_p10": float(frame["state_cosine"].quantile(0.10)),
        "stale_violation_rate": float(frame["stale"].mean()),
        "max_cache_age": int(frame.groupby(frame["refresh"].cumsum()).size().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, nargs="+", required=True)
    parser.add_argument("--test-run", type=Path, nargs="+", required=True)
    parser.add_argument("--switch-run", type=Path, nargs=2, default=None)
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--feature-map", type=Path, default=None, help="JSON map from pair-run name to trajectory_features.pt")
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-threshold", type=float, default=0.97)
    parser.add_argument("--router-threshold", type=float, default=0.30)
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_map = json.loads(args.feature_map.read_text(encoding="utf-8")) if args.feature_map else {}

    def feature_path(run: Path) -> Path:
        if run.name in feature_map:
            return Path(feature_map[run.name])
        scene = run.name.replace("_traj101_s1", "_traj101_eval_s1")
        return args.feature_root / scene / "trajectory_features.pt"

    train_sequences = [load_sequence(run, feature_path(run)) for run in args.train_run]
    test_sequences = [load_sequence(run, feature_path(run)) for run in args.test_run]
    if args.switch_run is not None:
        first = load_sequence(args.switch_run[0], feature_path(args.switch_run[0]))
        second = load_sequence(args.switch_run[1], feature_path(args.switch_run[1]))
        test_sequences.append(stitch_sequences(first, second))
    train_states = torch.cat(
        [torch.stack([summary_state(output) for output in sequence["outputs"]]) for sequence in train_sequences]
    )
    mean, basis, train_rms = fit_pca(train_states, max(args.widths))

    payload = torch.load(args.router, map_location="cpu", weights_only=False)
    from train_long_horizon_cache_router import RefreshRouter

    router = RefreshRouter(int(payload["input_width"]))
    router.load_state_dict(payload["model"])
    router.eval()

    policies = ["fixed5", "oracle", "learned"]
    rows = []
    for width in args.widths:
        if width > basis.shape[1] and width < train_states.shape[1]:
            continue
        for sequence in test_sequences:
            originals = torch.stack([summary_state(output) for output in sequence["outputs"]])
            reconstructed = encode_decode(originals, mean, basis, width)
            direct = cosine(reconstructed, originals)
            for policy in policies:
                result = sequence_metrics(
                    sequence,
                    reconstructed,
                    policy,
                    args.state_threshold,
                    router,
                    payload["mean"],
                    payload["std"],
                    args.router_threshold,
                )
                result.update({
                    "split": "test",
                    "latent_width": width,
                    "state_bytes_fp16": width * 2,
                    "reconstruction_cosine_mean": float(direct.mean()),
                    "reconstruction_cosine_p10": float(direct.quantile(0.10)),
                    "train_projection_rms": float(train_rms),
                })
                rows.append(result)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "train_split_pca_on_cls_mean_patch_state",
                "train_scenes": [sequence["name"] for sequence in train_sequences],
                "test_scenes": [sequence["name"] for sequence in test_sequences],
                "full_teacher_output_bytes_fp32": int(train_sequences[0]["outputs"][0].numel() * 4),
                "uncompressed_task_state_dim": int(train_states.shape[1]),
                "pca_rank": int(basis.shape[1]),
                "state_threshold": args.state_threshold,
                "summary": str(args.output_dir / "summary.csv"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
