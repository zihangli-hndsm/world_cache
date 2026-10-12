#!/usr/bin/env python3
"""Check same-scan mesh-to-native-depth alignment on a bounded MultiScan pilot.

Reads the pinned scan ZIPs in place, decodes depth in memory, and raycasts the
annotated PLY mesh into uniformly sampled depth frames. No archive contents are
extracted to disk.
"""

from __future__ import annotations

import argparse
import json
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import open3d as o3d

from download_multiscan_pilot_pair import FILES, REVISION
from audit_multiscan_pilot_annotations import unique_member


def read_mesh(archive: zipfile.ZipFile, scan_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = unique_member(archive, scan_id + ".ply")
    with archive.open(name) as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"unsupported PLY encoding for {scan_id}")
        vertex_count = int(next(line.split()[2] for line in header if line.startswith("element vertex ")))
        face_count = int(next(line.split()[2] for line in header if line.startswith("element face ")))
        expected = [
            "property double x", "property double y", "property double z",
            "property double nx", "property double ny", "property double nz",
            "property uchar red", "property uchar green", "property uchar blue", "property uchar alpha",
        ]
        props = header[header.index("element vertex " + str(vertex_count)) + 1:]
        vertex_props = props[:10]
        if vertex_props != expected:
            raise ValueError(f"unexpected vertex PLY fields: {vertex_props}")
        vertex_bytes = stream.read(vertex_count * 52)
        if len(vertex_bytes) != vertex_count * 52:
            raise ValueError("truncated PLY vertices")
        vertex_dtype = np.dtype([("xyz", "<f8", (3,)), ("normal", "<f8", (3,)), ("rgba", "u1", (4,))])
        vertices = np.frombuffer(vertex_bytes, dtype=vertex_dtype)["xyz"].copy()
        faces = np.empty((face_count, 3), dtype=np.int32)
        object_ids = np.empty(face_count, dtype=np.uint16)
        face_fmt = struct.Struct("<BiiiHH")
        for index in range(face_count):
            record = stream.read(face_fmt.size)
            if len(record) != face_fmt.size:
                raise ValueError("truncated PLY face data")
            values = face_fmt.unpack(record)
            if values[0] != 3:
                raise ValueError("expected triangular faces")
            faces[index] = values[1:4]
            object_ids[index] = values[4]
    return vertices, faces, object_ids


def load_frames(archive: zipfile.ZipFile, scan_id: str) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    meta_name = unique_member(archive, scan_id + ".json")
    meta = json.loads(archive.read(meta_name))
    color = next(s for s in meta["streams"] if s["type"] == "color_camera")
    depth = next(s for s in meta["streams"] if s["type"] == "lidar_sensor")
    h, w = depth["resolution"]
    if depth["encoding"] != "float16_zlib" or meta["depth_unit"] != "m":
        raise ValueError(f"unsupported depth format in {scan_id}: {depth['encoding']} / {meta['depth_unit']}")
    depth_name = unique_member(archive, scan_id + ".depth.zlib")
    raw = zlib.decompress(archive.read(depth_name), -zlib.MAX_WBITS)
    frames = int(depth["number_of_frames"])
    if len(raw) != frames * h * w * 2:
        raise ValueError(f"depth byte count mismatch in {scan_id}: {len(raw)}")
    depth_array = np.frombuffer(raw, dtype="<f2").reshape(frames, h, w).astype(np.float32)
    confidence_name = unique_member(archive, scan_id + ".confidence.zlib")
    confidence_raw = zlib.decompress(archive.read(confidence_name), -zlib.MAX_WBITS)
    if len(confidence_raw) != frames * h * w:
        raise ValueError(f"confidence byte count mismatch in {scan_id}: {len(confidence_raw)}")
    confidence_array = np.frombuffer(confidence_raw, dtype=np.uint8).reshape(frames, h, w)
    pose_name = unique_member(archive, scan_id + ".jsonl")
    poses = [json.loads(line) for line in archive.read(pose_name).splitlines() if line]
    if len(poses) != int(color["number_of_frames"]):
        raise ValueError(f"camera pose count differs from color frame count in {scan_id}")
    count = min(frames, len(poses), int(color["number_of_frames"]))
    return {"meta": meta, "depth": depth, "color": color, "count": count}, poses[:count], depth_array[:count], confidence_array[:count]


def scaled_intrinsic_parameters(intrinsics: list[float], sx: float, sy: float) -> tuple[float, float, float, float]:
    """Read column-major RGB K and scale it to the depth image grid."""
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3, order="F")
    return K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy


def audit_scan(archive_path: Path, scan_id: str, sample_count: int) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        info, poses, depths, confidences = load_frames(archive, scan_id)
        vertices, triangles, object_ids = read_mesh(archive, scan_id)
        align = json.loads(archive.read(unique_member(archive, scan_id + ".align.json")))

    h, w = info["depth"]["resolution"]
    color_h, color_w = info["color"]["resolution"]
    sx, sy = w / color_w, h / color_h
    T_align = np.asarray(align["coordinate_transform"], dtype=np.float64).reshape(4, 4, order="F")
    vertices = (vertices @ T_align[:3, :3].T) + T_align[:3, 3]
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32),
    )
    scene = o3d.t.geometry.RaycastingScene()
    geom_id = scene.add_triangles(mesh)
    ys, xs = np.mgrid[0:h, 0:w]
    sample_ids = np.unique(np.linspace(0, info["count"] - 1, min(sample_count, info["count"]), dtype=int))
    results = []
    arkit_to_o3d = np.diag([1.0, -1.0, -1.0, 1.0])
    for frame_id in sample_ids:
        pose = poses[frame_id]
        T_arkit = np.asarray(pose["transform"], dtype=np.float64).reshape(4, 4, order="F")
        T_cw = T_arkit @ arkit_to_o3d
        fx, fy, cx, cy = scaled_intrinsic_parameters(pose["intrinsics"], sx, sy)
        dirs_cam = np.stack(((xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)), axis=-1).reshape(-1, 3)
        dirs_world = dirs_cam @ T_cw[:3, :3].T
        origins = np.broadcast_to(T_cw[:3, 3], dirs_world.shape)
        rays = np.concatenate((origins, dirs_world), axis=1).astype(np.float32)
        hits = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = hits["t_hit"].numpy().reshape(h, w)
        primitive = hits["primitive_ids"].numpy().reshape(h, w)
        obs = depths[frame_id]
        valid = np.isfinite(obs) & (obs > 0) & (obs < 8.0)
        projected = np.isfinite(t_hit) & (t_hit > 0)
        common = valid & projected
        high_conf_common = common & (confidences[frame_id] == 2)
        stable = high_conf_common.copy()
        temporal_neighbors = []
        if frame_id > 0:
            temporal_neighbors.append(depths[frame_id - 1])
        if frame_id + 1 < info["count"]:
            temporal_neighbors.append(depths[frame_id + 1])
        for neighbor in temporal_neighbors:
            neighbor_valid = np.isfinite(neighbor) & (neighbor > 0)
            stable &= neighbor_valid & (np.abs(neighbor - obs) <= 0.05)
        stable_residual = np.abs(t_hit[stable] - obs[stable])
        residual = np.abs(t_hit[common] - obs[common])
        high_conf_residual = np.abs(t_hit[high_conf_common] - obs[high_conf_common])
        signed_residual = t_hit[common] - obs[common]
        face_ids = np.zeros(int(common.sum()), dtype=np.uint16)
        if common.any():
            hit_faces = primitive[common]
            face_ids = object_ids[hit_faces]
        results.append({
            "frame_id": int(frame_id),
            "valid_depth_pixels": int(valid.sum()),
            "mesh_hit_pixels": int(projected.sum()),
            "depth_and_mesh_overlap_pixels": int(common.sum()),
            "mesh_hit_fraction_of_valid_depth": float((valid & projected).sum() / max(1, valid.sum())),
            "median_observed_depth_m": float(np.median(obs[common])) if common.any() else None,
            "median_mesh_hit_depth_m": float(np.median(t_hit[common])) if common.any() else None,
            "median_signed_depth_residual_m": float(np.median(signed_residual)) if common.any() else None,
            "median_abs_depth_residual_m": float(np.median(residual)) if residual.size else None,
            "p90_abs_depth_residual_m": float(np.quantile(residual, 0.90)) if residual.size else None,
            "p95_abs_depth_residual_m": float(np.quantile(residual, 0.95)) if residual.size else None,
            "within_2cm_fraction": float(np.mean(residual <= 0.02)) if residual.size else None,
            "within_5cm_fraction": float(np.mean(residual <= 0.05)) if residual.size else None,
            "within_10cm_fraction": float(np.mean(residual <= 0.10)) if residual.size else None,
            "within_20cm_fraction": float(np.mean(residual <= 0.20)) if residual.size else None,
            "high_confidence_overlap_pixels": int(high_conf_common.sum()),
            "high_confidence_median_abs_depth_residual_m": float(np.median(high_conf_residual)) if high_conf_residual.size else None,
            "high_confidence_within_5cm_fraction": float(np.mean(high_conf_residual <= 0.05)) if high_conf_residual.size else None,
            "high_conf_temporally_stable_pixels": int(stable.sum()),
            "high_conf_temporally_stable_median_abs_depth_residual_m": float(np.median(stable_residual)) if stable_residual.size else None,
            "high_conf_temporally_stable_within_5cm_fraction": float(np.mean(stable_residual <= 0.05)) if stable_residual.size else None,
            "face_object_id_counts_at_overlap": {str(int(k)): int(v) for k, v in zip(*np.unique(face_ids, return_counts=True))},
        })
        print(f"{scan_id} frame {frame_id}: overlap={common.sum()/max(1,valid.sum()):.3f}, median={np.median(residual) if residual.size else float('nan'):.3f}m", flush=True)

    overlap = np.asarray([r["mesh_hit_fraction_of_valid_depth"] for r in results])
    median_resid = [r["median_abs_depth_residual_m"] for r in results if r["median_abs_depth_residual_m"] is not None]
    return {
        "scan_id": scan_id,
        "archive": str(archive_path),
        "rgbd_frame_count": info["count"],
        "rgb_resolution": [color_w, color_h],
        "depth_resolution": [w, h],
        "mesh_vertices": int(len(vertices)),
        "mesh_triangles": int(len(triangles)),
        "sampled_frame_ids": [int(i) for i in sample_ids],
        "median_mesh_hit_fraction_of_valid_depth": float(np.median(overlap)),
        "median_abs_depth_residual_m_across_frames": float(np.median(median_resid)) if median_resid else None,
        "frames": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/multiscan_scene_00069"))
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scans = []
    for rel in sorted(FILES):
        scan_id = Path(rel).stem
        scans.append(audit_scan(args.data_dir / Path(rel).name, scan_id, args.sample_count))
    result = {
        "status": "complete",
        "dataset_revision": REVISION,
        "scene_id": "scene_00069",
        "method": "same-scan z-buffer mesh raycast against native float16 depth; frames sampled uniformly",
        "depth_tolerance_fractions_are_conditional_on_pixels_with_both_valid_depth_and_mesh_hit": True,
        "scans": scans,
        "interpretation": "A same-scan projection feasibility check only; one scene pair does not establish independent-scene generalization or baseline performance.",
    }
    output = args.output or args.data_dir / "rgbd_projection_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "scan_summary": [
        {"scan_id": s["scan_id"], "mesh_hit_fraction": s["median_mesh_hit_fraction_of_valid_depth"],
         "median_abs_depth_residual_m": s["median_abs_depth_residual_m_across_frames"]} for s in scans
    ]}, indent=2))


if __name__ == "__main__":
    main()
