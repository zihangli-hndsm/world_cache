"""Evaluate a classical FPFH + RANSAC 3-D correspondence baseline.

The reference and rescan clouds are assembled from saved RGB-D patch points in
their native scan coordinate systems, and the matching stage uses only point
geometry. The known 3RScan transform is used only to score the estimated
transform; it is never used to prepare points or choose matches.

Open3D is an optional research dependency because the main WorldCache
experiments do not require it.  Install ``open3d==0.18.0`` in the
``worldcache`` environment before running this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise ImportError("FPFH baseline requires optional dependency open3d==0.18.0") from exc

from evaluate_3rscan_instance_correspondence import (
    load,
    patch_world_points,
    read_annotated_mesh,
    transform_points,
)
from evaluate_3rscan_pose_free_registration import ransac, transform


def aggregate_points(pair_run: Path, entries: list[dict], to_native: np.ndarray | None) -> np.ndarray:
    points = []
    for entry in entries:
        observation = load(pair_run / entry["path"])
        frame_points, _ = patch_world_points(observation, patch_size=14)
        values = frame_points.numpy().astype(np.float64)
        if to_native is not None:
            values = transform_points(values.astype(np.float32), to_native).astype(np.float64)
        points.append(values)
    if not points:
        raise ValueError("manifest contains no frames")
    return np.concatenate(points, axis=0)


def downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel_size)
    result = np.asarray(cloud.points, dtype=np.float64)
    if len(result) < 3:
        raise ValueError(f"voxel size {voxel_size} left fewer than three points")
    return result


def describe(points: np.ndarray, voxel_size: float, normal_radius: float, feature_radius: float):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    cloud.normalize_normals()
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    return cloud, np.asarray(feature.data, dtype=np.float64).T


def evaluate_matches(
    source_points: np.ndarray,
    source_features: np.ndarray,
    target_points: np.ndarray,
    target_features: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    source_semantics: np.ndarray,
    target_semantics: np.ndarray,
    source_evaluable: np.ndarray,
    transform_gt: np.ndarray,
    threshold: float,
    iterations: int,
    mutual: bool,
) -> dict[str, object]:
    target_tree = cKDTree(target_features)
    _, selected = target_tree.query(source_features, k=1)
    selected = selected.astype(np.int64)
    mask = np.ones(len(selected), dtype=bool)
    if mutual:
        source_tree = cKDTree(source_features)
        _, reverse = source_tree.query(target_features, k=1)
        mask = reverse[selected] == np.arange(len(selected))
    source = source_points[mask]
    target = target_points[selected[mask]]
    evaluated_matches = source_evaluable[mask]
    if len(source) < 3:
        raise ValueError(f"only {len(source)} FPFH matches remain")
    estimated, inliers = ransac(source, target, seed=17, threshold=threshold, iterations=iterations)
    residuals = np.linalg.norm(transform(source, estimated) - target, axis=1)
    evaluated_inliers = inliers & evaluated_matches
    evaluated_query_count = int(source_evaluable.sum())
    rotation_error = Rotation.from_matrix(estimated[:3, :3].T @ transform_gt[:3, :3]).magnitude()
    translation_error = np.linalg.norm(estimated[:3, 3] - transform_gt[:3, 3])
    result = {
        "method": "fpfh_mutual" if mutual else "fpfh_one_way",
        "cloud_source_points": int(len(source_points)),
        "cloud_target_points": int(len(target_points)),
        "evaluation_queries": evaluated_query_count,
        "matches": int(len(source)),
        "evaluated_matches": int(evaluated_matches.sum()),
        "coverage": float(evaluated_matches.sum() / evaluated_query_count) if evaluated_query_count else 0.0,
        "semantic_accuracy": float(np.mean(target_semantics[selected[mask]][evaluated_matches] == source_semantics[mask][evaluated_matches])) if evaluated_matches.any() else float("nan"),
        "instance_accuracy": float(np.mean(target_ids[selected[mask]][evaluated_matches] == source_ids[mask][evaluated_matches])) if evaluated_matches.any() else float("nan"),
        "ransac_inlier_rate": float(inliers.mean()),
        "matching_inliers": int(inliers.sum()),
        "inlier_coverage": float(evaluated_inliers.sum() / evaluated_query_count) if evaluated_query_count else 0.0,
        "inliers": int(evaluated_inliers.sum()),
        "inlier_semantic_accuracy": float(
            np.mean(target_semantics[selected[mask]][evaluated_inliers] == source_semantics[mask][evaluated_inliers])
        )
        if evaluated_inliers.any()
        else float("nan"),
        "inlier_instance_accuracy": float(
            np.mean(target_ids[selected[mask]][evaluated_inliers] == source_ids[mask][evaluated_inliers])
        )
        if evaluated_inliers.any()
        else float("nan"),
        "median_residual_m": float(np.median(residuals[evaluated_matches])) if evaluated_matches.any() else float("nan"),
        "median_inlier_residual_m": float(np.median(residuals[evaluated_inliers])) if evaluated_inliers.any() else float("nan"),
        "rotation_error_deg": float(np.degrees(rotation_error)),
        "translation_error_m": float(translation_error),
    }
    result["query_diagnostics"] = pd.DataFrame(
        {
            "query_semantic_id": source_semantics[mask],
            "selected_reference_semantic_id": target_semantics[selected[mask]],
            "query_instance_id": source_ids[mask],
            "selected_reference_instance_id": target_ids[selected[mask]],
            "query_evaluable": evaluated_matches,
            "consensus_inlier": inliers,
            "spatial_residual_m": residuals,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--normal-radius", type=float, default=0.10)
    parser.add_argument("--feature-radius", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--save-query-diagnostics", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("pair manifest must store native scan coordinates; run migrate_3rscan_rescan_coordinates.py")
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    rescan_to_reference = np.asarray(config["rescan_to_reference"], dtype=np.float64)
    reference_tree, reference_object_ids, reference_global_ids, reference_semantics, _ = read_annotated_mesh(
        args.annotation_root, config["reference"]
    )
    rescan_tree, rescan_object_ids, rescan_global_ids, rescan_semantics, _ = read_annotated_mesh(
        args.annotation_root, config["rescan"]
    )

    reference_raw = aggregate_points(args.pair_run, manifest["reference"], None)
    rescan_raw = aggregate_points(args.pair_run, manifest["rescan"], None)
    reference_points = downsample(reference_raw, args.voxel_size)
    rescan_points = downsample(rescan_raw, args.voxel_size)
    reference_ids = reference_global_ids[reference_tree.query(reference_points, k=1)[1]]
    rescan_ids = rescan_global_ids[rescan_tree.query(rescan_points, k=1)[1]]
    reference_semantic = reference_semantics[reference_tree.query(reference_points, k=1)[1]]
    rescan_semantic = rescan_semantics[rescan_tree.query(rescan_points, k=1)[1]]
    rescan_evaluable = (rescan_ids != 0) & (rescan_semantic != 0)

    _, reference_features = describe(reference_points, args.voxel_size, args.normal_radius, args.feature_radius)
    _, rescan_features = describe(rescan_points, args.voxel_size, args.normal_radius, args.feature_radius)
    rows = [
        evaluate_matches(
            rescan_points,
            rescan_features,
            reference_points,
            reference_features,
            rescan_ids,
            reference_ids,
            rescan_semantic,
            reference_semantic,
            rescan_evaluable,
            rescan_to_reference,
            args.threshold,
            args.iterations,
            mutual,
        )
        for mutual in (False, True)
    ]
    diagnostics = {row["method"]: row.pop("query_diagnostics") for row in rows}
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    if args.save_query_diagnostics:
        for method, frame in diagnostics.items():
            frame.to_csv(args.output_dir / f"query_diagnostics_{method}.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "pair_run": str(args.pair_run),
        "method": "Open3D FPFH descriptor nearest-neighbor matching with 3-point RANSAC",
        "known_alignment_used_for_selection": False,
        "voxel_size_m": args.voxel_size,
        "normal_radius_m": args.normal_radius,
        "feature_radius_m": args.feature_radius,
        "query_diagnostics_saved": args.save_query_diagnostics,
        "ransac_threshold_m": args.threshold,
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
