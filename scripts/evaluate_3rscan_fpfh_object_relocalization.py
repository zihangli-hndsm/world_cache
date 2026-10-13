#!/usr/bin/env python3
"""Object-level FPFH+RANSAC baseline for the 3RScan re-localization task.

The source object mask is given, as in RIO. The target scan is unsegmented:
target object annotations and transforms are used only after prediction to
score identity/pose. FPFH descriptors are computed on each complete scan so
source-object features retain surrounding geometric context.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def read_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = (path / "labels.instances.annotated.v2.ply").read_bytes()
    end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:end].decode("ascii")
    count = int(re.search(r"element vertex (\d+)", header).group(1))
    raw = np.fromstring(payload[end:].decode("ascii"), sep=" ", dtype=np.float32)
    vertices = raw[:count * 11].reshape(count, 11)
    return vertices[:, :3].astype(np.float64), vertices[:, 6].astype(np.int64)


def downsample(points: np.ndarray, ids: np.ndarray, voxel: float) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel)
    reduced = np.asarray(cloud.points, dtype=np.float64)
    nearest = cKDTree(points).query(reduced, k=1)[1]
    labels = ids[nearest]
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * voxel, max_nn=30))
    cloud.normalize_normals()
    return cloud, labels


def feature(cloud: o3d.geometry.PointCloud, radius: float) -> np.ndarray:
    f = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )
    return np.asarray(f.data, dtype=np.float64).T


def matrix(flat: list[float]) -> np.ndarray:
    return np.asarray(flat, dtype=np.float64).reshape(4, 4).T


def estimate(source: o3d.geometry.PointCloud, source_f: np.ndarray,
             target: o3d.geometry.PointCloud, target_f: np.ndarray,
             distance: float, iterations: int) -> tuple[np.ndarray, int, int]:
    # RIO's paper uses nearest-neighbour feature matches followed by RANSAC.
    _, nn = cKDTree(target_f).query(source_f, k=1)
    pairs = np.column_stack([np.arange(len(nn)), nn]).astype(np.int32)
    corr = o3d.utility.Vector2iVector(pairs)
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source, target, corr, distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(iterations, 0.999),
    )
    return np.asarray(result.transformation), int(len(pairs)), int(len(result.correspondence_set))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-root", type=Path, default=Path("data/3rscan"))
    ap.add_argument("--audit-root", type=Path, default=Path("outputs/reports/20261006_vitg_pose_locked_motion_validation"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_fpfh_object_relocalization"))
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--feature-radius", type=float, default=0.25)
    ap.add_argument("--ransac-distance", type=float, default=0.10)
    ap.add_argument("--ransac-iterations", type=int, default=5000)
    ap.add_argument("--max-relations", type=int, default=0, help="0 evaluates all cached validation relations")
    ap.add_argument("--ransac-seed", type=int, default=17)
    args = ap.parse_args()
    o3d.utility.random.seed(args.ransac_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.annotation_root / "3RScan.json").read_text())
    by_reference = {x["reference"]: x for x in metadata}
    cache_paths = sorted((args.audit_root / "descriptor_candidates").glob("*.npz"))
    if args.max_relations:
        cache_paths = cache_paths[:args.max_relations]
    rows = []
    for n, cache in enumerate(cache_paths, 1):
        with np.load(cache) as z:
            ref_id, res_id = str(z["reference_id"].item()), str(z["rescan_id"].item())
        rel = by_reference[ref_id]
        scan = next(s for s in rel["scans"] if s["reference"] == res_id)
        src_xyz, src_ids = read_scan(args.annotation_root / ref_id)
        tgt_xyz, tgt_ids = read_scan(args.annotation_root / res_id)
        src_cloud, src_down_ids = downsample(src_xyz, src_ids, args.voxel)
        tgt_cloud, _ = downsample(tgt_xyz, tgt_ids, args.voxel)
        src_f = feature(src_cloud, args.feature_radius)
        tgt_f = feature(tgt_cloud, args.feature_radius)
        ambiguity = set()
        for group in rel.get("ambiguity", []):
            for item in group:
                ambiguity.update(int(item[k]) for k in ("instance_source", "instance_target") if k in item)
        unique_links = {}
        for obj in scan.get("rigid", []):
            source_id = int(obj.get("instance_reference", -1))
            target_id = int(obj.get("instance_rescan", -1))
            symmetry = int(obj.get("symmetry", -1))
            transform_key = tuple(np.round(matrix(obj["transform"]), 8).ravel())
            unique_links[(source_id, target_id, symmetry, transform_key)] = obj
        for obj in unique_links.values():
            source_id = int(obj.get("instance_reference", -1))
            target_id = int(obj.get("instance_rescan", -1))
            symmetry = int(obj.get("symmetry", -1))
            if source_id <= 0 or target_id <= 0 or source_id in ambiguity or target_id in ambiguity:
                continue
            use = src_down_ids == source_id
            if use.sum() < 8:
                rows.append({"environment_id": ref_id, "pair": cache.stem, "source_object_id": source_id,
                             "target_object_id_gt": target_id, "symmetry": symmetry,
                             "source_points": int(use.sum()), "status": "insufficient_source_points"})
                continue
            sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(src_cloud.points)[use]))
            pred, matches, ransac_inliers = estimate(sub, src_f[use], tgt_cloud, tgt_f,
                                                      args.ransac_distance, args.ransac_iterations)
            truth = matrix(obj["transform"])
            rot = np.degrees(Rotation.from_matrix(truth[:3, :3].T @ pred[:3, :3]).magnitude())
            trans = np.linalg.norm(truth[:3, 3] - pred[:3, 3])
            rows.append({"environment_id": ref_id, "pair": cache.stem, "reference_scan_id": ref_id,
                         "rescan_scan_id": res_id, "source_object_id": source_id,
                         "target_object_id_gt": target_id, "symmetry": symmetry,
                         "source_points": int(use.sum()), "matches": matches, "ransac_inliers": ransac_inliers,
                         "rotation_error_deg": float(rot), "translation_error_m": float(trans),
                         "strict_10cm_10deg": bool(rot <= 10 and trans <= .1),
                         "relaxed_20cm_20deg": bool(rot <= 20 and trans <= .2), "status": "complete"})
        if n % 10 == 0:
            pd.DataFrame(rows).to_csv(args.output_dir / "per_object_partial.csv", index=False)
            print(f"relations={n}/{len(cache_paths)} object_rows={len(rows)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "per_object.csv", index=False)
    evaluable = frame[(frame.status == "complete") & (frame.symmetry == 0)].copy()
    summary = {"status": "complete", "relations_processed": len(cache_paths),
               "object_rows": len(frame), "non_symmetric_complete_objects": len(evaluable),
               "method": "source object mask + full-scan FPFH + one-way nearest feature matches + Open3D RANSAC",
               "source_object_mask_supplied": True, "target_segmentation_used_for_prediction": False,
               "parameters": {"voxel_m": args.voxel, "feature_radius_m": args.feature_radius,
                              "ransac_distance_m": args.ransac_distance,
                              "ransac_iterations": args.ransac_iterations,
                              "ransac_seed": args.ransac_seed}, "arms": {}}
    for metric in ("strict_10cm_10deg", "relaxed_20cm_20deg"):
        vals = evaluable[metric].astype(float).to_numpy()
        summary["arms"][metric] = {"successes": int(vals.sum()), "objects": len(vals),
                                     "pooled_rate": float(vals.mean()) if len(vals) else None}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete", "relations_processed": len(cache_paths),
        "source_object_mask_supplied": True, "target_segmentation_used_for_prediction": False,
        "method": summary["method"], "parameters": summary["parameters"],
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
