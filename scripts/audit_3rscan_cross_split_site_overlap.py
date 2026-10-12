#!/usr/bin/env python3
"""Screen available 3RScan train/validation reference scans for geometric overlap.

This is a screening audit, not proof of physical-site identity. It uses FPFH
feature matching, fast global registration, and symmetric geometric overlap.
Same-environment reference/rescan pairs are included as positive controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ANNOTATION_PLY = "labels.instances.annotated.v2.ply"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_features(path: Path, voxel: float) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        raise ValueError(f"empty point cloud: {path}")
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.5 * voxel, max_nn=30))
    features = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=4.0 * voxel, max_nn=100),
    )
    return cloud, features


def register(
    source_id: str,
    target_id: str,
    cache: dict[str, tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]],
    threshold: float,
) -> dict[str, object]:
    source, source_fpfh = cache[source_id]
    target, target_fpfh = cache[target_id]
    registration = o3d.pipelines.registration
    initial = registration.registration_fgr_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        registration.FastGlobalRegistrationOption(maximum_correspondence_distance=2.5 * threshold),
    )
    refined = registration.registration_icp(
        source,
        target,
        threshold,
        initial.transformation,
        registration.TransformationEstimationPointToPlane(),
    )
    refined_transform = refined.transformation
    forward = registration.evaluate_registration(source, target, threshold, refined_transform)
    backward = registration.evaluate_registration(target, source, threshold, np.linalg.inv(refined_transform))
    return {
        "source_id": source_id,
        "target_id": target_id,
        "source_points": len(source.points),
        "target_points": len(target.points),
        "fgr_fitness": float(initial.fitness),
        "fgr_rmse_m": float(initial.inlier_rmse),
        "icp_refined": True,
        "icp_forward_fitness": float(forward.fitness),
        "icp_backward_fitness": float(backward.fitness),
        "icp_symmetric_fitness": float(min(forward.fitness, backward.fitness)),
        "icp_forward_rmse_m": float(forward.inlier_rmse),
        "icp_backward_rmse_m": float(backward.inlier_rmse),
        "icp_symmetric_rmse_m": float(max(forward.inlier_rmse, backward.inlier_rmse)),
        "transform": refined_transform.tolist(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "source_id", "target_id", "source_points", "target_points", "fgr_fitness",
        "fgr_rmse_m", "icp_refined", "split", "icp_forward_fitness", "icp_backward_fitness",
        "icp_symmetric_fitness", "icp_forward_rmse_m", "icp_backward_rmse_m",
        "icp_symmetric_rmse_m", "transform",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "transform": json.dumps(row["transform"])})


def restore_rows(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    if not path.is_file():
        return {}
    restored: dict[tuple[str, str], dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("source_id") or not row.get("target_id"):
                continue
            row["transform"] = json.loads(row["transform"])
            for key in ("source_points", "target_points"):
                row[key] = int(row[key])
            for key in ("fgr_fitness", "fgr_rmse_m", "icp_forward_fitness", "icp_backward_fitness",
                        "icp_symmetric_fitness", "icp_forward_rmse_m", "icp_backward_rmse_m",
                        "icp_symmetric_rmse_m"):
                row[key] = float(row[key])
            # Historical cross-split checkpoint rows were produced before the
            # icp_refined column existed; the original run refined every pair.
            row["icp_refined"] = row.get("icp_refined", "True").lower() == "true"
            if "split" in row:
                row["split"] = row["split"] or None
            restored[(row["source_id"], row["target_id"])] = row
    return restored


def read_annotated_vertices(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        vertex_count = None
        for line in stream:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex declaration: {path}")
        return np.loadtxt(stream, max_rows=vertex_count)


def annotation_overlap(source_path: Path, target_path: Path, transform: list[list[float]]) -> dict[str, object]:
    """Compare stored color/instance/semantic fields for close aligned vertices."""
    source = read_annotated_vertices(source_path)
    target = read_annotated_vertices(target_path)
    matrix = np.asarray(transform)
    aligned = source[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    source_dist, source_idx = cKDTree(target[:, :3]).query(aligned, k=1)
    target_dist, target_idx = cKDTree(aligned).query(target[:, :3], k=1)
    threshold = 0.02
    keep = source_dist < threshold
    reverse_keep = target_dist < threshold
    if not np.any(keep):
        return {
            "distance_threshold_m": threshold,
            "source_fraction_with_match": 0.0,
            "target_fraction_with_match": 0.0,
            "matched_vertices": 0,
            "rgb_exact_agreement": None,
            "instance_and_semantic_fields_exact_agreement": None,
        }
    source_matches = source[keep]
    target_matches = target[source_idx[keep]]
    return {
        "distance_threshold_m": threshold,
        "source_fraction_with_match": float(np.mean(keep)),
        "target_fraction_with_match": float(np.mean(reverse_keep)),
        "matched_vertices": int(np.sum(keep)),
        "rgb_exact_agreement": float(np.mean(np.all(source_matches[:, 3:6] == target_matches[:, 3:6], axis=1))),
        "instance_and_semantic_fields_exact_agreement": float(np.mean(np.all(source_matches[:, 6:11] == target_matches[:, 6:11], axis=1))),
        "global_id_exact_agreement": float(np.mean(source_matches[:, 7] == target_matches[:, 7])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-m", type=float, default=0.08)
    parser.add_argument("--icp-threshold-m", type=float, default=0.10)
    parser.add_argument("--recompute-controls", action="store_true")
    parser.add_argument(
        "--only-validation-reference",
        action="append",
        default=None,
        help="screen one or more validation reference IDs against every available train reference",
    )
    args = parser.parse_args()
    if args.voxel_m <= 0 or args.icp_threshold_m <= 0:
        parser.error("voxel and ICP thresholds must be positive")

    dataset = json.loads(args.metadata.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.dataset_root
    existing: dict[str, Path] = {}
    for scene in dataset:
        for scan in [scene["reference"], *(row["reference"] for row in scene.get("scans", []))]:
            ply = root / scan / ANNOTATION_PLY
            if ply.is_file():
                existing[scan] = ply

    train_refs = sorted(
        scene["reference"] for scene in dataset
        if scene["type"] == "train" and scene["reference"] in existing
    )
    val_refs = sorted(
        scene["reference"] for scene in dataset
        if scene["type"] == "validation" and scene["reference"] in existing
    )
    # A registered scan feature cache is built once and reused for every pair.
    ids = sorted(set(train_refs + val_refs))
    cache: dict[str, tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]] = {}
    feature_times: dict[str, float] = {}
    for index, scan_id in enumerate(ids, start=1):
        started = time.time()
        cache[scan_id] = load_features(existing[scan_id], args.voxel_m)
        feature_times[scan_id] = time.time() - started
        if index % 10 == 0 or index == len(ids):
            print(f"prepared {index}/{len(ids)} reference clouds", flush=True)

    target_refs = val_refs
    if args.only_validation_reference:
        requested = set(args.only_validation_reference)
        unknown = requested - set(val_refs)
        if unknown:
            raise SystemExit(f"requested validation references are unavailable or not in validation split: {sorted(unknown)}")
        target_refs = sorted(requested)
    cross_pairs = [(train, val) for train in train_refs for val in target_refs]
    cross_path = args.output_dir / "cross_split_pairs.csv"
    cross_map = restore_rows(cross_path)
    for index, (train, val) in enumerate(cross_pairs, start=1):
        if (train, val) not in cross_map:
            cross_map[(train, val)] = register(train, val, cache, args.icp_threshold_m)
        if index % 100 == 0 or index == len(cross_pairs):
            write_csv(cross_path, list(cross_map.values()))
        if index % 100 == 0 or index == len(cross_pairs):
            print(f"registered {index}/{len(cross_pairs)} cross-split pairs ({len(cross_map)} restored/computed)", flush=True)
    cross_rows = list(cross_map.values())

    # Positive controls check sensitivity on available within-environment scans.
    control_pairs = []
    for scene in dataset:
        ref = scene["reference"]
        if ref not in cache:
            continue
        for row in scene.get("scans", []):
            query = row["reference"]
            if query in existing:
                control_pairs.append((scene["type"], ref, query))
    controls_path = args.output_dir / "same_environment_controls.csv"
    control_map = {} if args.recompute_controls else restore_rows(controls_path)
    for index, (split, ref, query) in enumerate(control_pairs, start=1):
        if query not in cache:
            cache[query] = load_features(existing[query], args.voxel_m)
        if (ref, query) not in control_map:
            result = register(ref, query, cache, args.icp_threshold_m)
            result["split"] = split
            control_map[(ref, query)] = result
        if index % 25 == 0 or index == len(control_pairs):
            write_csv(controls_path, list(control_map.values()))
            print(f"registered {index}/{len(control_pairs)} same-environment controls ({len(control_map)} restored/computed)", flush=True)
    control_rows = list(control_map.values())

    write_csv(cross_path, cross_rows)
    write_csv(controls_path, control_rows)
    # Explicit cutoffs are screening thresholds; preserve raw metrics for review.
    candidate_pairs = [
        row for row in cross_rows
        if row["icp_symmetric_fitness"] >= 0.30 and row["icp_symmetric_rmse_m"] <= 0.08
    ]
    control_scores = [row["icp_symmetric_fitness"] for row in control_rows]
    threshold_sensitivity = {}
    for cutoff in (0.30, 0.40, 0.50, 0.60):
        cross_count = sum(
            row["icp_symmetric_fitness"] >= cutoff and row["icp_symmetric_rmse_m"] <= 0.08
            for row in cross_rows
        )
        control_count = sum(
            row["icp_symmetric_fitness"] >= cutoff and row["icp_symmetric_rmse_m"] <= 0.08
            for row in control_rows
        )
        threshold_sensitivity[f"{cutoff:.2f}"] = {
            "cross_split_candidates": cross_count,
            "same_environment_controls_recalled": control_count,
            "same_environment_control_count": len(control_rows),
            "control_recall": control_count / len(control_rows) if control_rows else None,
        }
    high_confidence = [
        row for row in cross_rows
        if row["icp_symmetric_fitness"] >= 0.40 and row["icp_symmetric_rmse_m"] <= 0.08
    ]
    # Follow up all high-confidence geometric candidates with stored label data.
    high_confidence = sorted(
        high_confidence,
        key=lambda row: (-row["icp_symmetric_fitness"], row["icp_symmetric_rmse_m"]),
    )
    for row in high_confidence:
        row["annotation_overlap_2cm"] = annotation_overlap(
            existing[row["source_id"]], existing[row["target_id"]], row["transform"]
        )
    repeated_content = [
        row for row in high_confidence
        if row["annotation_overlap_2cm"]["instance_and_semantic_fields_exact_agreement"] is not None
        and row["annotation_overlap_2cm"]["instance_and_semantic_fields_exact_agreement"] >= 0.90
    ]
    official_train_count = sum(scene["type"] == "train" for scene in dataset)
    official_val_count = sum(scene["type"] == "validation" for scene in dataset)
    summary = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "metadata_file": args.metadata.name,
        "dataset_root": "<authorized-local-3RScan-root>",
        "method": "FPFH + fast global registration + point-to-plane ICP; symmetric overlap",
        "icp_refinement_policy": "all pair rows were refined by point-to-plane ICP",
        "voxel_m": args.voxel_m,
        "icp_threshold_m": args.icp_threshold_m,
        "screening_candidate_cutoff": {"symmetric_fitness_min": 0.30, "symmetric_rmse_m_max": 0.08},
        "official_environment_counts": {
            split: sum(scene["type"] == split for scene in dataset)
            for split in ["train", "validation"]
        },
        "available_reference_counts": {"train": len(train_refs), "validation": len(val_refs)},
        "validation_references_screened": target_refs,
        "cross_split_pairs_evaluated": len(cross_rows),
        "same_environment_controls_evaluated": len(control_rows),
        "threshold_sensitivity": threshold_sensitivity,
        "control_symmetric_fitness_summary": {
            "min": min(control_scores) if control_scores else None,
            "median": float(np.median(control_scores)) if control_scores else None,
            "p10": float(np.quantile(control_scores, 0.1)) if control_scores else None,
        },
        "screening_candidates": sorted(
            candidate_pairs,
            key=lambda row: (-row["icp_symmetric_fitness"], row["icp_symmetric_rmse_m"]),
        ),
        "annotation_overlap_followup_cutoff": {"symmetric_fitness_min": 0.40, "symmetric_rmse_m_max": 0.08},
        "high_confidence_candidate_annotation_checks": high_confidence,
        "repeated_content_candidates_90pct_annotation_agreement": repeated_content,
        "limitations": [
            f"This run screens {len(train_refs)} of {official_train_count} official train references and {len(target_refs)} of {official_val_count} validation references.",
            "Geometric similarity can suggest overlap but cannot establish physical-site identity.",
            "A missing match does not establish site independence; scenes may be spatially partial or modified.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# 3RScan cross-split geometric overlap screen",
        "",
        f"Screened {len(train_refs)} available train reference clouds against {len(target_refs)} selected validation reference clouds ({len(cross_rows)} pairs). The official split has {summary['official_environment_counts']['train']} train and {summary['official_environment_counts']['validation']} validation environments; local annotated PLY coverage is {len(train_refs)}/{summary['official_environment_counts']['train']} on train.",
        "",
        f"Same-environment reference/rescan controls: {len(control_rows)} pairs; median symmetric fitness {summary['control_symmetric_fitness_summary']['median']:.3f}, 10th percentile {summary['control_symmetric_fitness_summary']['p10']:.3f}. At the geometric candidate cutoff (fitness >= 0.30, RMSE <= 0.08 m), {len(candidate_pairs)} cross-split pairs remain.",
        "",
        "| Fitness cutoff | Cross-split candidates | Same-environment controls recalled | Control recall |",
        "|---:|---:|---:|---:|",
    ]
    for cutoff, counts in threshold_sensitivity.items():
        report_lines.append(
            f"| {float(cutoff):.2f} | {counts['cross_split_candidates']} | {counts['same_environment_controls_recalled']}/{counts['same_environment_control_count']} | {counts['control_recall']:.1%} |"
        )
    report_lines += [
        "",
        "## Strongest geometric candidates",
        "",
        "| Train reference | Validation reference | Symmetric fitness | Symmetric RMSE (m) | <2 cm source coverage | <2 cm target coverage | Exact RGB | Exact object/semantic fields |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in high_confidence:
        check = row["annotation_overlap_2cm"]
        report_lines.append(
            f"| `{row['source_id']}` | `{row['target_id']}` | {row['icp_symmetric_fitness']:.3f} | {row['icp_symmetric_rmse_m']:.3f} | {check['source_fraction_with_match']:.1%} | {check['target_fraction_with_match']:.1%} | {check['rgb_exact_agreement']:.1%} | {check['instance_and_semantic_fields_exact_agreement']:.1%} |"
        )
    report_lines += ["", "## Repeated-content candidates with exact annotation agreement", ""]
    if repeated_content:
        report_lines.extend([
            "The following cross-split pairs have at least 90% exact agreement on objectId, globalId, and semantic annotation fields among source vertices whose aligned nearest neighbor is within 2 cm:",
            "",
            "| Train reference | Validation reference | Source <2 cm coverage | Target <2 cm coverage | Exact RGB and annotation agreement |",
            "|---|---|---:|---:|---:|",
        ])
        for row in repeated_content:
            check = row["annotation_overlap_2cm"]
            report_lines.append(
                f"| `{row['source_id']}` | `{row['target_id']}` | {check['source_fraction_with_match']:.1%} | {check['target_fraction_with_match']:.1%} | {check['instance_and_semantic_fields_exact_agreement']:.1%} / {check['rgb_exact_agreement']:.1%} |"
            )
        report_lines += [
            "",
            "These are strong indicators of repeated or duplicated scan content across official environment IDs. They do not prove that the complete building or physical site is identical.",
        ]
    else:
        report_lines.append("No high-confidence pair crossed the 90% exact-annotation threshold in this run.")
    report_lines += [
        "",
        "## Scope and limitations",
        "",
        f"This is an automated geometric screen, not a site-identity benchmark. This run evaluates {len(train_refs)} of {official_train_count} train references against {len(target_refs)} of {official_val_count} validation references. Geometric similarity can produce false matches, and missing matches cannot establish independence. Same-environment controls show the screen has limited sensitivity for partial scans.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in [
        "available_reference_counts", "cross_split_pairs_evaluated",
        "same_environment_controls_evaluated", "control_symmetric_fitness_summary",
    ]}, indent=2))
    print(f"screening candidates: {len(candidate_pairs)}; report: {args.output_dir}")


if __name__ == "__main__":
    main()
