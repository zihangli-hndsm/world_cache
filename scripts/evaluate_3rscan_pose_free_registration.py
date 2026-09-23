"""Evaluate pose-free 3-D registration from global descriptor matches.

The correspondence evaluator stores rescan points in the reference frame so
that its geometry-gated controls are well-defined.  For the raw descriptor
field under a global candidate pool, the selected candidate depends only on
RGB descriptors, not on those coordinates.  This script changes the query
points back to their native rescan frame and estimates a rigid transform from
the descriptor matches with RANSAC.  The known 3RScan transform is used only
for that coordinate-frame conversion and for scoring the estimated transform;
it is not used to select descriptor matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = target_center - rotation @ source_center
    return matrix


def ransac(source: np.ndarray, target: np.ndarray, seed: int, threshold: float, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    if len(source) < 3:
        raise ValueError("at least three correspondences are required")
    generator = np.random.default_rng(seed)
    best_matrix = np.eye(4, dtype=np.float64)
    best_inliers = np.zeros(len(source), dtype=bool)
    for _ in range(iterations):
        sample = generator.choice(len(source), size=3, replace=False)
        candidate = rigid_fit(source[sample], target[sample])
        residuals = np.linalg.norm(transform(source, candidate) - target, axis=1)
        inliers = residuals <= threshold
        if inliers.sum() > best_inliers.sum():
            best_matrix = candidate
            best_inliers = inliers
    if best_inliers.sum() >= 3:
        best_matrix = rigid_fit(source[best_inliers], target[best_inliers])
        residuals = np.linalg.norm(transform(source, best_matrix) - target, axis=1)
        best_inliers = residuals <= threshold
    return best_matrix, best_inliers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    rescan_to_reference = np.asarray(config["rescan_to_reference"], dtype=np.float64)
    reference_to_rescan = np.linalg.inv(rescan_to_reference)
    metrics = pd.read_parquet(args.metrics)
    # Only this field/method is pose-free under the global candidate protocol.
    metrics = metrics[(metrics["field"] == "raw") & (metrics["method"] == "descriptor")].copy()
    coordinate_columns = [
        "query_point_ref_x", "query_point_ref_y", "query_point_ref_z",
        "selected_point_ref_x", "selected_point_ref_y", "selected_point_ref_z",
    ]
    missing = [column for column in coordinate_columns if column not in metrics]
    if missing:
        raise ValueError(f"metrics file lacks coordinate columns: {missing}; rerun correspondence evaluation")
    query_ref = metrics[["query_point_ref_x", "query_point_ref_y", "query_point_ref_z"]].to_numpy(float)
    target_ref = metrics[["selected_point_ref_x", "selected_point_ref_y", "selected_point_ref_z"]].to_numpy(float)
    query_native = transform(query_ref, reference_to_rescan)

    rows = []
    for threshold in args.thresholds:
        estimated, inliers = ransac(query_native, target_ref, seed=17, threshold=threshold, iterations=args.iterations)
        rotation_error = Rotation.from_matrix(estimated[:3, :3].T @ rescan_to_reference[:3, :3]).magnitude()
        translation_error = np.linalg.norm(estimated[:3, 3] - rescan_to_reference[:3, 3])
        residuals = np.linalg.norm(transform(query_native, estimated) - target_ref, axis=1)
        rows.append({
            "method": "raw_descriptor",
            "threshold_m": threshold,
            "correspondences": len(metrics),
            "inliers": int(inliers.sum()),
            "inlier_rate": float(inliers.mean()),
            "median_residual_m": float(np.median(residuals)),
            "median_inlier_residual_m": float(np.median(residuals[inliers])) if inliers.any() else float("nan"),
            "rotation_error_deg": float(np.degrees(rotation_error)),
            "translation_error_m": float(translation_error),
        })
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "registration.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "pair_run": str(args.pair_run),
        "metrics": str(args.metrics),
        "pose_free_method": "global raw descriptor-only matching",
        "coordinate_conversion_uses_ground_truth": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
