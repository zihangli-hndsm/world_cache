"""Evaluate the trained refresh router across an abrupt scene switch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from train_long_horizon_cache_router import evaluate_policy, load_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--feature-map", type=Path, default=None, help="JSON map from pair-run name to trajectory_features.pt")
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--state-threshold", type=float, default=0.97)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_map = json.loads(args.feature_map.read_text(encoding="utf-8")) if args.feature_map else {}

    def feature_path(run: Path) -> Path:
        if run.name in feature_map:
            return Path(feature_map[run.name])
        scene = run.name.replace("_traj101_s1", "_traj101_eval_s1")
        return args.feature_root / scene / "trajectory_features.pt"

    first = load_sequence(args.first_run, feature_path(args.first_run))
    second = load_sequence(args.second_run, feature_path(args.second_run))
    cut = min(46, len(first["outputs"]), len(second["outputs"]))
    sequence = {
        "name": f"switch_{first['name']}_to_{second['name']}",
        "images": torch.cat([first["images"][:cut], second["images"][:cut]]),
        "poses": first["poses"][:cut] + second["poses"][:cut],
        "outputs": first["outputs"][:cut] + second["outputs"][:cut],
    }
    payload = torch.load(args.router, map_location="cpu", weights_only=False)
    from train_long_horizon_cache_router import RefreshRouter

    router = RefreshRouter(int(payload["input_width"]))
    router.load_state_dict(payload["model"])
    router.eval()
    rows = []
    for policy, threshold in [("full", args.threshold), ("fixed5", args.threshold), ("oracle", args.threshold), ("learned", args.threshold)]:
        result = evaluate_policy(sequence, policy, router, payload["mean"], payload["std"], threshold, args.state_threshold)
        rows.append(result)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
