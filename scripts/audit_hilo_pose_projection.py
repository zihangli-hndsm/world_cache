#!/usr/bin/env python3
"""Bounded HILO RGB-D / object-pose projection feasibility audit.

Reads only selected ZIP members with HTTP Range requests; it never downloads the
39.5 GB archive. Requires numpy, Pillow, Open3D, and network access.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import http.client
import json
import struct
import tempfile
import time
import urllib.request
import urllib.error
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from PIL import Image

RECORD = "https://zenodo.org/api/records/15263380"
PREFIX = "HILO_Dataset/"
CAPTURES = ("arc0_image7", "arc0_image12", "arc4_image7", "arc4_image12")


def ranged(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                payload = response.read()
                if response.status != 206 or len(payload) != end - start + 1:
                    raise RuntimeError(f"unexpected range response: {response.status}, {len(payload)} bytes")
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 7:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2 ** attempt, 30)
            time.sleep(delay)
        except (http.client.IncompleteRead, http.client.RemoteDisconnected,
                ConnectionResetError, TimeoutError, urllib.error.URLError):
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("unreachable range retry state")


def zip_entries(url: str, total: int, cache_path: Optional[Path] = None) -> dict[str, tuple[int, int, int, int, int]]:
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return {name: tuple(values) for name, values in cached.items()}
    tail = ranged(url, total - 2_000_000, total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    locator = struct.unpack_from("<4sIQ", tail, eocd - 20)
    if locator[0] != b"PK\x06\x07":
        raise RuntimeError("ZIP64 locator not found")
    zip64_offset = locator[2]
    z64 = ranged(url, zip64_offset, zip64_offset + 55)
    if z64[:4] != b"PK\x06\x06":
        raise RuntimeError("ZIP64 end record not found")
    fields = struct.unpack("<4sQ2H2L4Q", z64)
    count, cd_size, cd_offset = fields[-4], fields[-2], fields[-1]
    central = ranged(url, cd_offset, cd_offset + cd_size - 1)
    entries: dict[str, tuple[int, int, int, int, int]] = {}
    pos = 0
    while pos < len(central):
        row = struct.unpack_from("<4s6H3L5H2L", central, pos)
        if row[0] != b"PK\x01\x02":
            raise RuntimeError(f"bad central-directory entry at byte {pos}")
        _, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _, offset = row
        name = central[pos + 46 : pos + 46 + nlen].decode("utf-8")
        extra = central[pos + 46 + nlen : pos + 46 + nlen + xlen]
        if any(x == 0xFFFFFFFF for x in (usize, csize, offset)):
            ep = 0
            while ep < len(extra):
                xid, xsize = struct.unpack_from("<HH", extra, ep)
                val = extra[ep + 4 : ep + 4 + xsize]
                if xid == 1:
                    k = 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack_from("<Q", val, k)[0]
                        k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack_from("<Q", val, k)[0]
                        k += 8
                    if offset == 0xFFFFFFFF:
                        offset = struct.unpack_from("<Q", val, k)[0]
                    break
                ep += 4 + xsize
        entries[name] = (offset, csize, usize, method, crc)
        pos += 46 + nlen + xlen + clen
    if len(entries) != count:
        raise RuntimeError(f"central directory count mismatch: {len(entries)} != {count}")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(entries, separators=(",", ":")))
    return entries


def extract_member(url: str, entries: dict, name: str) -> bytes:
    offset, csize, usize, method, expected_crc = entries[name]
    header = ranged(url, offset, offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {name}")
    name_len, extra_len = struct.unpack_from("<HH", header, 26)
    compressed = ranged(url, offset + 30 + name_len + extra_len,
                        offset + 30 + name_len + extra_len + csize - 1)
    payload = zlib.decompress(compressed, -15) if method == 8 else compressed
    if len(payload) != usize or (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
        raise RuntimeError(f"size/CRC mismatch for {name}")
    return payload


def extract_or_reuse(url: str, entries: dict, name: str, destination: Path) -> bytes:
    expected_size, expected_crc = entries[name][2], entries[name][4]
    if destination.exists():
        payload = destination.read_bytes()
        if len(payload) == expected_size and (zlib.crc32(payload) & 0xFFFFFFFF) == expected_crc:
            return payload
    payload = extract_member(url, entries, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", help="scene folder; repeat to audit multiple scenes")
    parser.add_argument("--cache-dir", type=Path, help="persist the selected subset here for downstream analyses")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive-url", help="ZIP endpoint (defaults to the public record-file URL)")
    parser.add_argument("--index-cache", type=Path, help="cache the ZIP central directory as JSON")
    parser.add_argument("--workers", type=int, default=1,
                        help="maximum parallel ZIP-member downloads (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    scene_names = args.scene or ["scene_08_s1_c1"]
    record = json.load(urllib.request.urlopen(RECORD))
    license_id = record["metadata"]["license"]["id"]
    file_info = record["files"][0]
    # Use the public record-file URL directly. The API content endpoint can
    # trip the shared proxy's rate limiter during many small ZIP range reads.
    url = args.archive_url or "https://zenodo.org/records/15263380/files/HILO_Dataset.zip?download=1"
    total = file_info["size"]
    entries = zip_entries(url, total, args.index_cache)
    selected = [
        "HILO_Scenes/cam_params_arc0123.json",
        "HILO_Scenes/cam_params_arc4567.json",
    ]
    for scene_name in scene_names:
        scene_prefix = f"HILO_Scenes/{scene_name}/"
        selected.extend(scene_prefix + name for name in (
            "object_id_list.json", "object_pose_wrt_arc0_image7.json",
            "camera_poses.json",
        ))
        for capture in CAPTURES:
            selected.extend(scene_prefix + f"{field}/{capture}.png"
                            for field in ("rgb_undistorted", "depth_undistorted"))
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_context = nullcontext(args.cache_dir)
    else:
        cache_context = tempfile.TemporaryDirectory(prefix="hilo_pose_audit_")
    with cache_context as td:
        root = Path(td)
        local = {name: root / name for name in selected}
        bytes_read = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(extract_or_reuse, url, entries, PREFIX + name, path): name
                for name, path in local.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                payload = future.result()
                bytes_read += len(payload)
                print(f"scene-member {name} ({len(payload)} bytes)", flush=True)
        scene_data = {}
        all_ids = set()
        for scene_name in scene_names:
            scene_prefix = f"HILO_Scenes/{scene_name}/"
            ids = json.loads(local[scene_prefix + "object_id_list.json"].read_text())
            poses_list = json.loads(local[scene_prefix + "object_pose_wrt_arc0_image7.json"].read_text())
            obj_poses = {x["Model"]: np.asarray(x["trans"], dtype=np.float64) for x in poses_list}
            camera_poses = json.loads(local[scene_prefix + "camera_poses.json"].read_text())
            scene_data[scene_name] = (ids, obj_poses, camera_poses)
            all_ids.update(ids)
        object_members = [f"HILO_Objects/simplified/{oid}_simplified.obj"
                          for oid in sorted(all_ids)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(extract_or_reuse, url, entries, PREFIX + name, root / name): name
                for name in object_members
            }
            for future in as_completed(futures):
                name = futures[future]
                payload = future.result()
                bytes_read += len(payload)
                print(f"mesh-member {name} ({len(payload)} bytes)", flush=True)

        results = []
        for scene_name in scene_names:
          ids, obj_poses, camera_poses = scene_data[scene_name]
          for capture in CAPTURES:
              arc = int(capture[3])
              param_key = "HILO_Scenes/cam_params_arc0123.json" if arc < 4 else "HILO_Scenes/cam_params_arc4567.json"
              params = json.loads(local[param_key].read_text())
              K = np.asarray(params["depth_K"], dtype=np.float64)
              image_prefix = f"HILO_Scenes/{scene_name}/depth_undistorted/{capture}.png"
              height, width = np.asarray(Image.open(local[image_prefix])).shape
              yy, xx = np.mgrid[0:height, 0:width]
              rays = np.stack(((xx - K[0, 2]) / K[0, 0],
                               (yy - K[1, 2]) / K[1, 1],
                               np.ones_like(xx)), axis=-1).astype(np.float32).reshape(-1, 3)
              camera_to_ref = np.asarray(camera_poses[capture], dtype=np.float64)
              ref_to_camera = np.linalg.inv(camera_to_ref)
              ray_scene = o3d.t.geometry.RaycastingScene()
              for oid in ids:
                  mesh_path = root / f"HILO_Objects/simplified/{oid}_simplified.obj"
                  mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                  mesh.transform(ref_to_camera @ obj_poses[oid])
                  ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
              rays_o3d = np.concatenate((np.zeros_like(rays), rays), axis=1)
              result = ray_scene.cast_rays(o3d.core.Tensor(rays_o3d, o3d.core.Dtype.Float32))
              rendered_z = result["t_hit"].numpy().reshape(height, width)
              observed_z = np.asarray(Image.open(local[image_prefix]), dtype=np.float32)
              overlap = np.isfinite(rendered_z) & (rendered_z < 1e9) & (observed_z > 0)
              delta = rendered_z[overlap] - observed_z[overlap]
              results.append({
                  "scene": scene_name,
                  "capture": capture,
                  "rendered_pixels": int((np.isfinite(rendered_z) & (rendered_z < 1e9)).sum()),
                  "mesh_depth_overlap_pixels": int(overlap.sum()),
                  "fraction_within_20mm": float((np.abs(delta) < 20).mean()),
                  "median_rendered_minus_observed_mm": float(np.median(delta)),
                  "p90_absolute_depth_error_mm": float(np.percentile(np.abs(delta), 90)),
              })
    summary = {
        "dataset": "HILO", "license": license_id, "record": RECORD,
        "archive_size_bytes": total, "scenes": scene_names,
        "object_count": len(all_ids), "frame_count": len(results),
        "selected_uncompressed_bytes": bytes_read,
        "results": results,
        "scope": "bounded multi-scene pose/projection feasibility screen; not an identity benchmark",
    }
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
