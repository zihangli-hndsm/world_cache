#!/usr/bin/env python3
"""Prepare LivingScenes-style labeled point clouds from 3RScan RGB-D scans.

This is an explicit adapter for feasibility work, not the authors' released
preprocessing pipeline. It samples a fixed number of frames uniformly, uses
an image-grid stride, and projects GT object IDs from the scan-local annotated
mesh for mask preparation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from worldcache.datasets import threerscan as scan_reader  # noqa: E402
from worldcache.datasets.threerscan import ThreeRScanSequence  # noqa: E402
from evaluate_3rscan_instance_correspondence import read_annotated_mesh  # noqa: E402


CATEGORIES = {
    "dining chair", "couch table", "dining table", "footstool", "cushion",
    "sofa", "bench", "table", "ottoman", "computer desk", "rocking chair",
    "armchair", "round table", "pillow", "side table", "stand", "sofa chair",
    "desk", "trash can", "couch", "coffee table", "chair", "bed",
}


def frame_member_map(names: list[str], prefix: str) -> dict[tuple[str, int], str]:
    members: dict[tuple[str, int], str] = {}
    for name in names:
        if not name.startswith(prefix):
            continue
        match = re.search(r"frame-(\d+)\.(color|depth|pose)\.", name, re.IGNORECASE)
        if match:
            members[(match.group(2).lower(), int(match.group(1)))] = name
            continue
        path = Path(name)
        parent = path.parent.name.lower()
        if parent in {"color", "depth", "pose"} and path.stem.isdigit():
            members[(parent, int(path.stem))] = name
    return members


def read_frame_depth(
    archive: zipfile.ZipFile,
    member_names: dict[tuple[str, int], str],
    frame_id: int,
    color_intrinsic: np.ndarray,
    depth_intrinsic: np.ndarray,
    depth_extrinsic: np.ndarray,
    depth_shift: float,
    has_depth_intrinsic: bool,
    color_shape: tuple[int, int],
) -> np.ndarray:
    depth_name = member_names[("depth", frame_id)]
    depth = np.asarray(Image.open(io.BytesIO(archive.read(depth_name))), dtype=np.float32)
    depth /= depth_shift
    if has_depth_intrinsic and depth.shape != color_shape:
        yy, xx = np.meshgrid(
            np.arange(depth.shape[0], dtype=np.float32),
            np.arange(depth.shape[1], dtype=np.float32),
            indexing="ij",
        )
        z = depth.reshape(-1)
        valid = np.isfinite(z) & (z > 0)
        pixels = np.column_stack((xx.reshape(-1)[valid], yy.reshape(-1)[valid], np.ones(valid.sum(), dtype=np.float32)))
        points_depth = (pixels @ np.linalg.inv(depth_intrinsic).T) * z[valid, None]
        points_depth_h = np.column_stack((points_depth, np.ones(len(points_depth), dtype=np.float32)))
        points_color = (points_depth_h @ depth_extrinsic.T)[:, :3]
        projected = points_color @ color_intrinsic.T
        uv = projected[:, :2] / projected[:, 2:3]
        u = np.rint(uv[:, 0]).astype(np.int64)
        v = np.rint(uv[:, 1]).astype(np.int64)
        visible = points_color[:, 2] > 0
        visible &= u >= 0
        visible &= v >= 0
        visible &= u < color_shape[1]
        visible &= v < color_shape[0]
        flat = v[visible] * color_shape[1] + u[visible]
        aligned = np.full(color_shape[0] * color_shape[1], np.inf, dtype=np.float32)
        np.minimum.at(aligned, flat, points_color[visible, 2])
        depth = aligned.reshape(color_shape)
        depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32, copy=False)


def choose_frames(frame_ids: list[int], max_frames: int) -> list[int]:
    if not frame_ids:
        return []
    count = min(max_frames, len(frame_ids))
    indices = np.linspace(0, len(frame_ids) - 1, num=count, dtype=np.int64)
    return [frame_ids[int(index)] for index in np.unique(indices)]


def scan_points(
    data_root: Path,
    output_root: Path,
    scan_id: str,
    max_frames: int,
    pixel_stride: int,
    max_mesh_distance_m: float,
    resume: bool,
) -> dict[str, int | float]:
    output_dir = output_root / scan_id
    ply_out = output_dir / "pointcloud.instances.align.ply"
    labels_out = output_dir / "pointcloud.labels.npz"
    stats_out = output_dir / "rgbd_support.json"
    if resume and ply_out.is_file() and labels_out.is_file() and stats_out.is_file():
        return json.loads(stats_out.read_text(encoding="utf-8"))

    tree, object_ids, _, _, object_to_label = read_annotated_mesh(data_root, scan_id)
    sequence = ThreeRScanSequence(scan_id, data_root)
    frame_ids = sequence.frame_ids()
    selected_ids = choose_frames(frame_ids, max_frames)
    if not selected_ids:
        raise ValueError(f"scan {scan_id} has no color frames")

    points_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    projected_count = 0
    accepted_count = 0
    with sequence._open() as archive:
        names = archive.namelist()
        prefix = sequence._prefix(archive)
        color_intrinsic, depth_intrinsic, depth_extrinsic, depth_shift, has_depth_intrinsic = sequence._info(archive, prefix)
        member_names = frame_member_map(names, prefix)
        first_color = Image.open(io.BytesIO(archive.read(member_names[("color", selected_ids[0])])))
        color_shape = (first_color.height, first_color.width)
        first_color.close()
        color_intrinsic_inverse = np.linalg.inv(color_intrinsic)
        for frame_id in selected_ids:
            depth = read_frame_depth(
                archive,
                member_names,
                frame_id,
                color_intrinsic,
                depth_intrinsic,
                depth_extrinsic,
                depth_shift,
                has_depth_intrinsic,
                color_shape,
            )
            pose_text = archive.read(member_names[("pose", frame_id)]).decode("utf-8")
            pose_values = scan_reader._FLOAT_RE.findall(pose_text)
            camera_to_world = scan_reader._matrix(pose_values, "camera pose")
            depth = depth[::pixel_stride, ::pixel_stride]
            height, width = depth.shape
            yy, xx = np.meshgrid(
                np.arange(0, height * pixel_stride, pixel_stride, dtype=np.float32),
                np.arange(0, width * pixel_stride, pixel_stride, dtype=np.float32),
                indexing="ij",
            )
            z = depth.reshape(-1)
            valid = np.isfinite(z) & (z > 0)
            if not np.any(valid):
                continue
            uv1 = np.column_stack((xx.reshape(-1)[valid], yy.reshape(-1)[valid], np.ones(valid.sum(), dtype=np.float32)))
            camera = (uv1 @ color_intrinsic_inverse.T) * z[valid, None]
            camera_h = np.column_stack((camera, np.ones(len(camera), dtype=np.float32)))
            world = (camera_h @ camera_to_world.T)[:, :3].astype(np.float32, copy=False)
            distances, nearest = tree.query(world, k=1, workers=1)
            keep = np.isfinite(distances) & (distances <= max_mesh_distance_m)
            projected_count += len(world)
            accepted_count += int(keep.sum())
            if np.any(keep):
                points_blocks.append(world[keep])
                label_blocks.append(object_ids[nearest[keep]].astype(np.int32, copy=False))

    if not points_blocks:
        raise ValueError(f"no RGB-D points projected onto the annotation mesh for {scan_id}")
    points = np.concatenate(points_blocks, axis=0)
    labels = np.concatenate(label_blocks, axis=0)
    object_counts = Counter(map(int, labels))
    eligible_ids = {
        object_id for object_id, count in object_counts.items()
        if object_id != 0 and count >= 1024 and object_to_label.get(object_id) in CATEGORIES
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_ply = ply_out.with_suffix(".ply.tmp")
    with temporary_ply.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(stream, points, fmt="%.6f %.6f %.6f")
    temporary_ply.replace(ply_out)
    temporary_npz = labels_out.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as stream:
        np.savez_compressed(stream, objectId=labels)
    temporary_npz.replace(labels_out)

    stats: dict[str, int | float] = {
        "scan_id": scan_id,
        "available_frames": len(frame_ids),
        "selected_frames": len(selected_ids),
        "pixel_stride": pixel_stride,
        "projected_depth_points": projected_count,
        "points_with_mesh_distance_le_threshold": accepted_count,
        "mesh_label_rate": accepted_count / projected_count if projected_count else 0.0,
        "mesh_distance_threshold_m": max_mesh_distance_m,
        "output_point_count": len(points),
        "eligible_instances_ge_1024_official_categories": len(eligible_ids),
        "eligible_instance_points": sum(object_counts[item] for item in eligible_ids),
        "eligible_object_ids": sorted(eligible_ids),
    }
    stats_out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--relation-limit", type=int, default=110)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--max-mesh-distance-m", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.relation_limit < 1 or args.max_frames < 1 or args.pixel_stride < 1:
        raise ValueError("relation limit, frame count, and pixel stride must be positive")
    if args.max_mesh_distance_m <= 0:
        raise ValueError("mesh distance threshold must be positive")

    with args.pair_manifest.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 110:
        raise ValueError(f"expected the official 110-relation validation manifest, got {len(pairs)}")
    pilot_pairs = pairs[: min(args.relation_limit, len(pairs))]
    scan_ids = sorted({row["reference_scan"] for row in pilot_pairs} | {row["rescan_scan"] for row in pilot_pairs})
    print(f"relations {len(pilot_pairs)}; unique scans {len(scan_ids)}", flush=True)

    scan_stats = []
    for index, scan_id in enumerate(scan_ids, start=1):
        scan_stats.append(
            scan_points(
                args.data_root,
                args.output_root,
                scan_id,
                args.max_frames,
                args.pixel_stride,
                args.max_mesh_distance_m,
                args.resume,
            )
        )
        if index % 5 == 0 or index == len(scan_ids):
            print(f"scans {index}/{len(scan_ids)}", flush=True)

    metadata = json.loads((args.data_root / "3RScan.json").read_text(encoding="utf-8"))
    metadata_by_pair = {
        (scene["reference"], scan["reference"]): scan
        for scene in metadata
        for scan in scene.get("scans", [])
    }
    scan_eligible = {item["scan_id"]: int(item["eligible_instances_ge_1024_official_categories"]) for item in scan_stats}
    eligible_object_ids = {
        scan_id: set(json.loads((args.output_root / scan_id / "rgbd_support.json").read_text(encoding="utf-8"))["eligible_object_ids"])
        for scan_id in scan_ids
    }
    relation_stats = []
    for row in pilot_pairs:
        ref, rescan = row["reference_scan"], row["rescan_scan"]
        record = metadata_by_pair.get((ref, rescan), {})
        mappings = record.get("rigid", [])
        eligible_links = 0
        for mapping in mappings:
            if not isinstance(mapping, dict):
                continue
            ref_id, rescan_id = int(mapping["instance_reference"]), int(mapping["instance_rescan"])
            eligible_links += int(
                ref_id in eligible_object_ids[ref]
                and rescan_id in eligible_object_ids[rescan]
            )
        relation_stats.append(
            {
                "reference_scan": ref,
                "rescan_scan": rescan,
                "official_rigid_mappings": len(mappings),
                "reference_eligible_instances": scan_eligible.get(ref, 0),
                "rescan_eligible_instances": scan_eligible.get(rescan, 0),
                "eligible_linked_instances": eligible_links,
            }
        )

    with (args.output_root / "pilot_scan_support.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scan_stats[0]))
        writer.writeheader()
        writer.writerows(scan_stats)
    with (args.output_root / "pilot_relation_support.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(relation_stats[0]))
        writer.writeheader()
        writer.writerows(relation_stats)
    relation_counts = [int(row["eligible_linked_instances"]) for row in relation_stats]
    summary = {
        "scans": len(scan_stats),
        "relations": len(relation_stats),
        "frames_per_scan_cap": args.max_frames,
        "pixel_stride": args.pixel_stride,
        "mesh_distance_threshold_m": args.max_mesh_distance_m,
        "projected_depth_points": sum(int(row["projected_depth_points"]) for row in scan_stats),
        "mesh_labeled_points": sum(int(row["points_with_mesh_distance_le_threshold"]) for row in scan_stats),
        "eligible_instances_total": sum(int(row["eligible_instances_ge_1024_official_categories"]) for row in scan_stats),
        "eligible_linked_instances_total": sum(relation_counts),
        "relations_with_ge1_eligible_linked_instance": sum(count >= 1 for count in relation_counts),
        "relations_with_ge3_eligible_linked_instances": sum(count >= 3 for count in relation_counts),
        "relations_with_ge5_eligible_linked_instances": sum(count >= 5 for count in relation_counts),
        "feasibility_gate_relations_ge1": 75,
        "feasibility_gate_linked_instances": 200,
        "feasibility_gate_passed": sum(count >= 1 for count in relation_counts) >= 75 and sum(relation_counts) >= 200,
        "interpretation": "input coverage only; no model inference or method accuracy was measured",
    }
    (args.output_root / "pointcloud_feasibility_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
