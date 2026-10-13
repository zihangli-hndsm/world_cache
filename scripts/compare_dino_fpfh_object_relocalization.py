#!/usr/bin/env python3
"""Paired object-level pose comparison on the shared DINO/FPFH cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def matrix(flat: list[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(4, 4).T


def inverse(t: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-root", type=Path, default=Path("data/3rscan"))
    ap.add_argument("--dino-csv", type=Path, default=Path("outputs/reports/20261009_dino_object_pose_relocalization/per_object.csv"))
    ap.add_argument("--fpfh-csv", type=Path, default=Path("outputs/reports/20261009_fpfh_object_relocalization_deduplicated/per_object.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_fpfh_object_relocalization_deduplicated"))
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261009)
    args = ap.parse_args()
    dino = pd.read_csv(args.dino_csv)
    dino = dino[(dino.arm == "combined") & (dino.symmetry == 0)].copy()
    fpfh = pd.read_csv(args.fpfh_csv)
    fpfh = fpfh[(fpfh.status == "complete") & (fpfh.symmetry == 0)].copy()
    # Both directions have an official per-object mapping. Join on the
    # reference and rescan object IDs, not merely a scan-level relation.
    dino["source_object_id"] = dino.reference_object_id_gt.astype(int)
    dino["target_object_id_gt"] = dino.query_object_id.astype(int)
    joined = dino.merge(
        fpfh,
        on=["pair", "source_object_id", "target_object_id_gt", "symmetry"],
        suffixes=("_dino", "_fpfh"),
        validate="one_to_one",
    )
    metadata = json.loads((args.annotation_root / "3RScan.json").read_text())
    by_reference = {x["reference"]: x for x in metadata}
    dino_strict, dino_relaxed, dino_rot, dino_trans = [], [], [], []
    for row in joined.itertuples(index=False):
        raw = getattr(row, "predicted_pose_rescan_to_reference")
        if not isinstance(raw, str) or not raw:
            pred = None
        else:
            pred = inverse(np.asarray(json.loads(raw), dtype=np.float64))
        if pred is None or not bool(getattr(row, "pose_available")):
            rot, trans = np.inf, np.inf
        else:
            ref_id = getattr(row, "reference_scan_id_dino")
            res_id = getattr(row, "rescan_scan_id_dino")
            qid = int(getattr(row, "query_object_id"))
            rid = int(getattr(row, "reference_object_id_gt"))
            scan = next(s for s in by_reference[ref_id]["scans"] if s["reference"] == res_id)
            link = next(x for x in scan["rigid"] if int(x["instance_rescan"]) == qid and int(x["instance_reference"]) == rid)
            truth = matrix(link["transform"])
            rot = float(np.degrees(Rotation.from_matrix(truth[:3, :3].T @ pred[:3, :3]).magnitude()))
            trans = float(np.linalg.norm(truth[:3, 3] - pred[:3, 3]))
        dino_rot.append(rot); dino_trans.append(trans)
        dino_strict.append(bool(rot <= 10 and trans <= .1))
        dino_relaxed.append(bool(rot <= 20 and trans <= .2))
    joined["dino_rotation_error_deg_ref_to_rescan"] = dino_rot
    joined["dino_translation_error_m_ref_to_rescan"] = dino_trans
    joined["dino_strict_10cm_10deg_ref_to_rescan"] = dino_strict
    joined["dino_relaxed_20cm_20deg_ref_to_rescan"] = dino_relaxed
    joined.to_csv(args.output_dir / "paired_dino_fpfh_objects.csv", index=False)
    envs = sorted(joined.environment_id_dino.unique())
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(envs), size=(args.draws, len(envs)))
    metrics = {
        "strict_10cm_10deg": ("dino_strict_10cm_10deg_ref_to_rescan", "strict_10cm_10deg_fpfh"),
        "relaxed_20cm_20deg": ("dino_relaxed_20cm_20deg_ref_to_rescan", "relaxed_20cm_20deg_fpfh"),
    }
    arms = {}
    paired_deltas = {}
    for metric, (dc, fc) in metrics.items():
        env_scores = {}
        for arm, column in (("dino_combined", dc), ("fpfh", fc)):
            values = joined.groupby("environment_id_dino")[column].mean().reindex(envs).to_numpy(float)
            samples = values[draws].mean(1)
            arms.setdefault(arm, {})[metric] = {
                "environment_macro": float(values.mean()),
                "pooled_successes": int(joined[column].astype(bool).sum()),
                "objects": int(len(joined)),
                "ci95_environment_bootstrap": np.quantile(samples, [.025, .975]).tolist(),
            }
            env_scores[arm] = values
        delta = env_scores["dino_combined"] - env_scores["fpfh"]
        samples = delta[draws].mean(1)
        paired_deltas[metric] = {"dino_minus_fpfh_environment_macro": float(delta.mean()),
                                 "ci95_environment_bootstrap": np.quantile(samples, [.025, .975]).tolist()}
    summary = {"status": "complete", "matched_objects": int(len(joined)), "environments": len(envs),
               "join": "same reference object, rescan object, and scan pair; DINO prediction inverted to official reference-to-rescan direction",
               "DINO_pose_availability_condition": "unavailable DINO poses count as failure",
               "arms": arms, "paired_differences": paired_deltas,
               "bootstrap_draws": args.draws, "bootstrap_seed": args.seed,
               "limitations": [
                   "Post-hoc same-dataset comparison, not independent validation.",
                   "DINO uses frozen rescan-to-reference top-K point correspondences and source masks to group query points; FPFH receives a supplied reference-object mask and matches against the full unsegmented rescan mesh.",
                   "Different inputs and candidate budgets prevent interpreting this as a controlled descriptor-only comparison.",
                   "The FPFH run uses one fixed seeded RANSAC run per object; no repeated-seed uncertainty is included.",
                   "Symmetric objects are excluded because symmetry-aware pose scoring is not implemented.",
               ]}
    (args.output_dir / "paired_dino_fpfh_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
