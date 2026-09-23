"""Train/evaluate a tiny causal refresh router on frozen trajectory features.

This is a policy prototype for a much larger frozen VLM. The router never
sees the teacher feature at decision time; frozen outputs are used only to
create oracle labels and evaluate stale-state error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def load_npz(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def pose_from_record(record: dict[str, object]) -> np.ndarray:
    return np.linalg.inv(np.asarray(record["relative_pose"], dtype=np.float64))


def pose_delta(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    relative = np.linalg.inv(left) @ right
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return translation, float(math.degrees(math.acos(cosine)))


def records_for_run(run_dir: Path) -> list[dict[str, object]]:
    records = [json.loads(line) for line in (run_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    return [records[0]] + [record for record in records[1:] if record["pair_type"] != "identity"]


def load_sequence(run_dir: Path, feature_path: Path) -> dict[str, object]:
    records = records_for_run(run_dir)
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    outputs = [item.float() for item in payload["outputs"]]
    if len(records) != len(outputs):
        raise ValueError(f"record/teacher length mismatch for {run_dir}: {len(records)} vs {len(outputs)}")
    images = []
    poses = []
    for record in records:
        observation = load_npz(run_dir / record["current"])
        rgb = observation["rgb"].float().div(255.0).permute(2, 0, 1).unsqueeze(0)
        images.append(F.interpolate(rgb, size=(8, 8), mode="area").flatten())
        poses.append(pose_from_record(record))
    return {"name": run_dir.name, "images": torch.stack(images), "poses": poses, "outputs": outputs}


def stitch_sequences(first: dict[str, object], second: dict[str, object], cut: int = 46) -> dict[str, object]:
    cut = min(cut, len(first["outputs"]), len(second["outputs"]))
    return {
        "name": f"switch_{first['name']}_to_{second['name']}",
        "images": torch.cat([first["images"][:cut], second["images"][:cut]]),
        "poses": first["poses"][:cut] + second["poses"][:cut],
        "outputs": first["outputs"][:cut] + second["outputs"][:cut],
    }


def state_metrics(cached: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    return {
        "full_cosine": float(F.cosine_similarity(cached.flatten(), full.flatten(), dim=0)),
        "cls_cosine": float(F.cosine_similarity(cached[:, 0], full[:, 0], dim=1).mean()),
        "patch_cosine": float(F.cosine_similarity(cached[:, 1:], full[:, 1:], dim=2).mean()),
    }


def feature_for(sequence: dict[str, object], index: int, cache_index: int) -> torch.Tensor:
    images = sequence["images"]
    current = images[index]
    cached = images[cache_index]
    translation, rotation = pose_delta(sequence["poses"][cache_index], sequence["poses"][index])
    age = float(index - cache_index)
    return torch.cat([
        current,
        cached,
        (current - cached).abs(),
        torch.tensor([translation, rotation / 10.0, age / 20.0], dtype=torch.float32),
    ])


def oracle_examples(sequence: dict[str, object], threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    cache_index = 0
    for index, output in enumerate(sequence["outputs"]):
        features.append(feature_for(sequence, index, cache_index))
        if index == 0:
            labels.append(1.0)
            continue
        similarity = state_metrics(sequence["outputs"][cache_index], output)["full_cosine"]
        refresh = similarity < threshold
        labels.append(float(refresh))
        if refresh:
            cache_index = index
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


class RefreshRouter(nn.Module):
    def __init__(self, input_width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def train_router(features: torch.Tensor, labels: torch.Tensor, epochs: int, seed: int) -> RefreshRouter:
    torch.manual_seed(seed)
    model = RefreshRouter(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    positive_weight = (labels == 0).sum().float() / (labels == 1).sum().clamp_min(1.0)
    for _ in range(epochs):
        logits = model(features)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def evaluate_policy(sequence: dict[str, object], policy: str, router: RefreshRouter | None, mean: torch.Tensor, std: torch.Tensor, threshold: float, state_threshold: float) -> dict[str, object]:
    cache_index = 0
    rows = []
    for index, output in enumerate(sequence["outputs"]):
        if index == 0:
            refresh = True
            probability = 1.0
        elif policy == "full":
            refresh = True
            probability = 1.0
        elif policy == "fixed5":
            refresh = index % 5 == 0
            probability = float(refresh)
        elif policy == "pose":
            translation, rotation = pose_delta(sequence["poses"][cache_index], sequence["poses"][index])
            refresh = translation > 0.01 or rotation > 1.0
            probability = float(refresh)
        elif policy == "rgb":
            delta = float((sequence["images"][index] - sequence["images"][cache_index]).abs().mean())
            refresh = delta > threshold
            probability = float(delta)
        elif policy == "oracle":
            refresh = state_metrics(sequence["outputs"][cache_index], output)["full_cosine"] < state_threshold
            probability = float(refresh)
        elif policy == "learned":
            features = (feature_for(sequence, index, cache_index) - mean) / std
            probability = float(torch.sigmoid(router(features)).item())
            refresh = probability >= threshold
        else:
            raise ValueError(policy)
        cached = output if refresh else sequence["outputs"][cache_index]
        metrics = state_metrics(cached, output)
        actual_violation = metrics["full_cosine"] < state_threshold and not refresh
        rows.append({"frame": index, "refresh": int(refresh), "probability": probability, "cache_index": cache_index, "stale_violation": int(actual_violation), **metrics})
        if refresh:
            cache_index = index
    frame = pd.DataFrame(rows)
    return {
        "scene": sequence["name"],
        "policy": policy,
        "frames": len(frame),
        "recompute_fraction": float(frame["refresh"].mean()),
        "reuse_fraction": float(1.0 - frame["refresh"].mean()),
        "full_cosine_mean": float(frame["full_cosine"].mean()),
        "full_cosine_p10": float(frame["full_cosine"].quantile(0.10)),
        "cls_cosine_mean": float(frame["cls_cosine"].mean()),
        "patch_cosine_mean": float(frame["patch_cosine"].mean()),
        "stale_violation_rate": float(frame["stale_violation"].mean()),
        "max_cache_age": int(frame.groupby((frame["refresh"].cumsum())).size().max()),
    }


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, nargs="+", required=True)
    parser.add_argument("--test-run", type=Path, nargs="+", required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--feature-map", type=Path, default=None, help="JSON map from pair-run name to trajectory_features.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-threshold", type=float, default=0.97)
    parser.add_argument("--router-thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--rgb-threshold", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--augment-switch-training", action="store_true")
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
    oracle_sequences = list(train_sequences)
    if args.augment_switch_training:
        oracle_sequences.extend(
            stitch_sequences(first, second)
            for first, second in zip(train_sequences[:-1], train_sequences[1:])
        )
    train_features, train_labels = zip(*(oracle_examples(sequence, args.state_threshold) for sequence in oracle_sequences))
    features = torch.cat(train_features)
    labels = torch.cat(train_labels)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-4)
    router = train_router((features - mean) / std, labels, args.epochs, args.seed)
    full_state_bytes = int(train_sequences[0]["outputs"][0].numel() * 4)
    task_state_dim = int(train_sequences[0]["outputs"][0][:, 0].numel() + train_sequences[0]["outputs"][0][:, 1:].mean(dim=1).numel())
    compact_state_bytes_bf16 = int(task_state_dim * 2)

    policies = [("full", None), ("fixed5", None), ("pose", None), ("rgb", None), ("oracle", None)]
    for threshold in args.router_thresholds:
        policies.append((f"learned_{threshold:g}", threshold))
    rows = []
    test_names = {sequence["name"] for sequence in test_sequences}
    for sequence in train_sequences + test_sequences:
        is_test = sequence["name"] in test_names
        for policy, threshold in policies:
            if policy.startswith("learned"):
                actual_policy, actual_threshold = "learned", float(threshold)
            else:
                actual_policy, actual_threshold = policy, args.rgb_threshold
            result = evaluate_policy(sequence, actual_policy, router, mean, std, actual_threshold, args.state_threshold)
            result["split"] = "test" if is_test else "train"
            result["reported_policy"] = policy
            rows.append(result)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    torch.save({"model": router.state_dict(), "mean": mean, "std": std, "input_width": features.shape[1]}, args.output_dir / "router.pt")
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "train_scenes": [sequence["name"] for sequence in train_sequences],
        "oracle_training_sequences": [sequence["name"] for sequence in oracle_sequences],
        "test_scenes": [sequence["name"] for sequence in test_sequences],
        "state_threshold": args.state_threshold,
        "router_parameters": sum(parameter.numel() for parameter in router.parameters()),
        "router_parameter_bytes_fp32": int(sum(parameter.numel() for parameter in router.parameters()) * 4),
        "full_teacher_output_bytes_fp32_per_slot": full_state_bytes,
        "compact_cls_mean_patch_state_dim": task_state_dim,
        "compact_state_bytes_bf16_per_slot_projection": compact_state_bytes_bf16,
        "compact_state_is_projected_budget_not_yet_reconstructed": True,
        "teacher_outputs_are_evaluation_only": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary[summary["split"] == "test"].sort_values(["reported_policy", "scene"]).to_string(index=False))


if __name__ == "__main__":
    main()
