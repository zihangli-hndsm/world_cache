#!/usr/bin/env python3
"""Aggregate full and paired object-pose FPFH RANSAC seed runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/reports"))
    ap.add_argument("--dino", type=Path, default=Path("outputs/reports/20261009_dino_object_pose_relocalization/per_object.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_fpfh_object_relocalization_deduplicated"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    frames = []
    for seed in args.seeds:
        folder = (args.root / "20261009_fpfh_object_relocalization_deduplicated" if seed == 17
                  else args.root / f"20261009_fpfh_object_relocalization_seed{seed}")
        frame = pd.read_csv(folder / "per_object.csv")
        frame = frame[(frame.status == "complete") & (frame.symmetry == 0)].copy()
        frame["ransac_seed"] = seed
        frames.append(frame)
    all_f = pd.concat(frames, ignore_index=True)
    keys = ["pair", "source_object_id", "target_object_id_gt", "symmetry"]
    if all_f.duplicated(keys + ["ransac_seed"]).any():
        raise ValueError("duplicate rigid-link predictions remain within a seed")
    dino = pd.read_csv(args.dino)
    dino = dino[(dino.arm == "combined") & (dino.symmetry == 0)].copy()
    dino["source_object_id"] = dino.reference_object_id_gt.astype(int)
    dino["target_object_id_gt"] = dino.query_object_id.astype(int)
    dino = dino.drop_duplicates(keys)
    paired = dino.merge(all_f, on=keys, suffixes=("_dino", "_fpfh"), validate="one_to_many")
    paired_base_path = args.output_dir / "paired_dino_fpfh_objects.csv"
    if not paired_base_path.is_file():
        raise FileNotFoundError(f"run compare_dino_fpfh_object_relocalization.py first: {paired_base_path}")
    paired_base = pd.read_csv(paired_base_path)
    dino_score_cols = ["dino_strict_10cm_10deg_ref_to_rescan", "dino_relaxed_20cm_20deg_ref_to_rescan"]
    paired = paired.merge(paired_base[keys + dino_score_cols], on=keys, how="left", validate="many_to_one")
    if paired[dino_score_cols].isna().any().any():
        raise ValueError("some paired DINO objects lack their forward-direction rescored pose")
    paired.to_csv(args.output_dir / "paired_dino_fpfh_objects_all_seeds.csv", index=False)

    metrics = ["strict_10cm_10deg", "relaxed_20cm_20deg"]
    rng = np.random.default_rng(args.seed)
    # One shared cluster-and-seed draw per bootstrap replicate.
    envs_full = sorted(all_f.environment_id.unique())
    envs_pair = sorted(paired.environment_id_dino.unique())
    seed_values = list(args.seeds)
    env_draw_full = rng.integers(0, len(envs_full), size=(args.draws, len(envs_full)))
    env_draw_pair = rng.integers(0, len(envs_pair), size=(args.draws, len(envs_pair)))
    seed_draw = rng.integers(0, len(seed_values), size=(args.draws, len(seed_values)))
    summary = {"status": "complete", "ransac_seeds": seed_values,
               "full_cohort_non_symmetric_objects_per_seed": int(all_f.groupby("ransac_seed").size().min()),
               "full_cohort_environments": len(envs_full), "paired_objects": int(paired.groupby("ransac_seed").size().min()),
               "paired_environments": len(envs_pair), "arms": {}, "paired_differences": {},
               "bootstrap_draws": args.draws, "bootstrap_seed": args.seed}
    for metric in metrics:
        seed_results = {}
        full_by_env_seed = []
        for method_seed, frame in all_f.groupby("ransac_seed"):
            by_env = frame.groupby("environment_id")[metric].mean().reindex(envs_full).fillna(0).to_numpy(float)
            full_by_env_seed.append(by_env)
            seed_results[str(method_seed)] = {"successes": int(frame[metric].sum()), "objects": len(frame),
                                               "pooled_rate": float(frame[metric].mean())}
        full_arr = np.stack(full_by_env_seed, axis=1)
        dino_col = f"dino_{metric}_ref_to_rescan"
        fpfh_col = f"{metric}_fpfh"
        # Paired object outcomes are aggregated by environment and seed.
        per_env_seed_f = np.stack([
            paired[paired.ransac_seed.eq(s)].groupby("environment_id_dino")[fpfh_col]
            .mean().reindex(envs_pair).fillna(0).to_numpy(float)
            for s in seed_values
        ], axis=1)
        per_env_d = paired.groupby("environment_id_dino")[dino_col].mean().reindex(envs_pair).fillna(0).to_numpy(float)
        dino_arr = np.repeat(per_env_d[:, None], len(seed_values), axis=1)
        # For each draw, resample environment clusters and RANSAC seeds.
        full_samples = np.empty(args.draws)
        fpfh_pair_samples = np.empty(args.draws)
        dino_pair_samples = np.empty(args.draws)
        for i in range(args.draws):
            selected_seeds = seed_draw[i]
            full_samples[i] = full_arr[env_draw_full[i]][:, selected_seeds].mean()
            fpfh_pair_samples[i] = per_env_seed_f[env_draw_pair[i]][:, selected_seeds].mean()
            dino_pair_samples[i] = dino_arr[env_draw_pair[i]][:, selected_seeds].mean()
        full_mean = float(full_arr.mean())
        fpfh_pair_mean = float(per_env_seed_f.mean())
        dino_pair_mean = float(dino_arr.mean())
        summary["arms"][metric] = {
            "fpfh_full_validation_by_seed": seed_results,
            "fpfh_full_environment_macro_mean_over_seeds": full_mean,
            "fpfh_full_ci95_environment_and_seed_bootstrap": np.quantile(full_samples, [.025, .975]).tolist(),
            "paired_dino_environment_macro": dino_pair_mean,
            "paired_fpfh_environment_macro_mean_over_seeds": fpfh_pair_mean,
        }
        diff_samples = dino_pair_samples - fpfh_pair_samples
        summary["paired_differences"][metric] = {
            "dino_minus_fpfh_macro": dino_pair_mean - fpfh_pair_mean,
            "ci95_environment_and_seed_bootstrap": np.quantile(diff_samples, [.025, .975]).tolist(),
        }
    summary["limitations"] = [
        "All results use previously explored 3RScan validation relations.",
        "The paired comparison joins identical official source/target object links; DINO predictions are inverted to reference-to-rescan coordinates.",
        "DINO uses frozen top-K point correspondences and grouped source-object query masks; FPFH uses a supplied reference-object mask and full target geometry.",
        "The comparison is task-level, not a descriptor-only ablation or official RIO reproduction.",
        "The bootstrap resamples environment clusters and three RANSAC runs; three seeds give limited algorithmic-variability characterization.",
        "Symmetric objects are excluded from pose success.",
    ]
    (args.output_dir / "multi_seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
