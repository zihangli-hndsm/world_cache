#!/usr/bin/env python3
"""Generate frozen DINO/RGB top-20 and spatial-consensus predictions.

This stage reads only RGB video, native depth, stream metadata and per-frame
camera poses. It never opens annotation, mesh-label, or cross-scan alignment
members. Predictions are persisted before the separate scoring stage reads GT.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from audit_multiscan_rgbd_projection import REVISION, scaled_intrinsic_parameters, unique_member, load_frames  # noqa: E402
from evaluate_3rscan_spatial_consensus import consensus  # noqa: E402
from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2  # noqa: E402


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
LAYERS = (2, 20, 40)
TOP_K = 20
PATCH = 14
INPUT_SIZE = 224
FRAME_LIMIT = 20
THRESHOLD_M = 0.10
RANSAC_ITERATIONS = 1500
RANSAC_SEED = 17
PIL_RESAMPLING = getattr(Image, "Resampling", Image)
PREDS_DIR = REPORT_DIR / "predictions"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_indices(count: int) -> np.ndarray:
    return np.unique(np.linspace(0, count - 1, min(FRAME_LIMIT, count), dtype=int))


@torch.inference_mode()
def encode_scan(model, scan_data: tuple, device: torch.device) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    prepared = scan_data[4]
    images = prepared["images"].to(device)
    captured = extract_block_tokens(model, images, LAYERS)
    tokens = [F.normalize(captured[layer][:, model.num_prefix_tokens:], dim=2) for layer in LAYERS]
    full_desc = F.normalize(torch.cat(tokens, dim=2), dim=2).numpy().astype(np.float32)
    point_frames, patch_indices = scan_data[2], scan_data[3]
    selected_desc = []
    for frame_pos, frame_id in enumerate(prepared["stage"]["encoded_frame_ids"]):
        keep = point_frames == frame_id
        selected_desc.append(full_desc[frame_pos, patch_indices[keep]])
    return np.concatenate(selected_desc), time.perf_counter() - start


def topk(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    similarity = query @ reference.T
    candidates = np.argsort(-similarity, axis=1, kind="stable")[:, :TOP_K].astype(np.int32)
    return candidates, time.perf_counter() - start


def main() -> None:
    subset = json.loads((REPORT_DIR / "quota_subset_manifest.json").read_text())
    state = json.loads((REPORT_DIR / "download_state.json").read_text())
    gate_path = REPORT_DIR / "projection_preflight" / "per_environment_gate.csv"
    with gate_path.open(newline="") as stream:
        gate = {row["environment_id"]: row for row in csv.DictReader(stream)}
    identity_path = REPORT_DIR / "identity_preflight" / "per_environment.csv"
    with identity_path.open(newline="") as stream:
        identity = {row["environment_id"]: row for row in csv.DictReader(stream)}
    if set(gate) != set(subset["selected_scene_ids"]) or set(identity) != set(subset["selected_scene_ids"]):
        raise SystemExit("Projection or identity preflight does not cover the frozen subset exactly.")
    if any(row["relation_gate_pass"] != "True" for row in gate.values()):
        raise SystemExit("At least one selected relation failed the frozen 5 cm gate; refusing inference for this subset.")
    if any(row["identity_gate_pass"] != "True" for row in identity.values()):
        raise SystemExit("At least one selected relation lacks mapped persistent identities; refusing inference for this subset.")

    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("Frozen run requires the verified CUDA GPU; refusing silent CPU fallback.")
    weight_ref = Path.home() / ".cache/huggingface/hub/models--timm--vit_giant_patch14_dinov2.lvd142m/refs/main"
    expected_revision = "dd22bb6485eeb29d67273f76d983061298bdc7f0"
    if weight_ref.read_text().strip() != expected_revision:
        raise SystemExit("Cached ViT-G main ref is not the frozen model revision.")
    model_path = weight_ref.parent.parent / "snapshots" / expected_revision / "model.safetensors"
    expected_hash = "949077749c7a414f3443a007cf021d085bc11deb0b08959a6e32e170b1e8e359"
    if sha256(model_path) != expected_hash:
        raise SystemExit("Cached ViT-G weight hash differs from frozen manifest.")
    load_start = time.perf_counter()
    model = load_dinov2("giant", INPUT_SIZE, device)
    model_load_seconds = time.perf_counter() - load_start

    rows = []
    for relation in subset["relations"]:
        env = relation["environment_id"]
        output = PREDS_DIR / env / "predictions.npz"
        metadata_path = PREDS_DIR / env / "prediction_metadata.json"
        if output.exists() and metadata_path.exists():
            meta = json.loads(metadata_path.read_text())
            if meta.get("dataset_revision") != REVISION or meta.get("model_revision") != expected_revision:
                raise SystemExit(f"Existing predictions have mismatched frozen inputs: {env}")
            rows.append(meta["timings"] | {"environment_id": env, "status": "reused"})
            continue

        scan_paths = {}
        for scan_id in (relation["reference_scan_id"], relation["query_scan_id"]):
            filename = next(a["filename"] for a in subset["archives"] if Path(a["filename"]).stem == scan_id)
            scan_paths[scan_id] = ROOT / state["downloaded_archives"][filename]["cache_path"]

        decode_start = time.perf_counter()
        prepared = {}
        for role, scan_id in (("reference", relation["reference_scan_id"]), ("query", relation["query_scan_id"])):
            with zipfile.ZipFile(scan_paths[scan_id]) as archive:
                prepared[role] = prepare_scan_from_open_archive(archive, scan_paths[scan_id], scan_id)
        prepare_wall_seconds = time.perf_counter() - decode_start
        input_stages = [prepared[role][4]["stage"] for role in ("reference", "query")]
        data_decode_seconds = sum(s["depth_decode_seconds"] + s["video_decode_seconds"] for s in input_stages)
        geometry_projection_seconds = sum(s["geometry_projection_seconds"] for s in input_stages)

        dino_ref, time_ref = encode_scan(model, prepared["reference"], device)
        dino_query, time_query = encode_scan(model, prepared["query"], device)
        gpu_seconds = time_ref + time_query
        torch.cuda.synchronize()
        del prepared["reference"][4]["images"], prepared["query"][4]["images"]

        ref_points, ref_rgb, ref_frames, ref_patches = prepared["reference"][:4]
        query_points, query_rgb, query_frames, query_patches = prepared["query"][:4]
        match_start = time.perf_counter()
        dino_top20, dino_rank_seconds = topk(dino_query, dino_ref)
        rgb_ref = F.normalize(torch.from_numpy(ref_rgb), dim=1).numpy()
        rgb_query = F.normalize(torch.from_numpy(query_rgb), dim=1).numpy()
        rgb_top20, rgb_rank_seconds = topk(rgb_query, rgb_ref)
        cpu_match_seconds = time.perf_counter() - match_start

        consensus_start = time.perf_counter()
        dino_transform, dino_assign, dino_inliers = consensus(
            query_points.astype(np.float64), ref_points.astype(np.float64), dino_top20,
            THRESHOLD_M, RANSAC_ITERATIONS, RANSAC_SEED,
        )
        rgb_transform, rgb_assign, rgb_inliers = consensus(
            query_points.astype(np.float64), ref_points.astype(np.float64), rgb_top20,
            THRESHOLD_M, RANSAC_ITERATIONS, RANSAC_SEED,
        )
        consensus_seconds = time.perf_counter() - consensus_start
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            reference_points=ref_points, query_points=query_points,
            reference_frame_ids=ref_frames, query_frame_ids=query_frames,
            reference_patch_indices=ref_patches, query_patch_indices=query_patches,
            dino_top20=dino_top20, rgb_top20=rgb_top20,
            dino_assigned_reference=dino_assign, dino_inliers=dino_inliers,
            dino_transform_query_to_reference=dino_transform,
            rgb_assigned_reference=rgb_assign, rgb_inliers=rgb_inliers,
            rgb_transform_query_to_reference=rgb_transform,
        )
        torch.cuda.synchronize()
        timings = {
            "data_decode_seconds": data_decode_seconds,
            "geometry_projection_seconds": geometry_projection_seconds,
            "data_prepare_wall_seconds": prepare_wall_seconds,
            "gpu_feature_seconds": gpu_seconds,
            "cpu_rank_seconds": dino_rank_seconds + rgb_rank_seconds,
            "cpu_consensus_seconds": consensus_seconds,
            "cpu_total_match_seconds": cpu_match_seconds + consensus_seconds,
            "query_point_count": int(len(query_points)),
            "reference_point_count": int(len(ref_points)),
            "dino_rank_seconds": dino_rank_seconds,
            "rgb_rank_seconds": rgb_rank_seconds,
        }
        metadata = {
            "status": "predictions_saved_before_scoring",
            "environment_id": env,
            "reference_scan_id": relation["reference_scan_id"],
            "query_scan_id": relation["query_scan_id"],
            "dataset_revision": REVISION,
            "model": "DINOv2 ViT-G/14",
            "model_revision": expected_revision,
            "model_weights_sha256": expected_hash,
            "layers_1_indexed": list(LAYERS),
            "input_size": INPUT_SIZE,
            "top_k": TOP_K,
            "frame_sampling": "up to 20 uniformly spaced indices including endpoints",
            "consensus_threshold_m": THRESHOLD_M,
            "ransac_iterations": RANSAC_ITERATIONS,
            "ransac_seed": RANSAC_SEED,
            "input_reads": ["RGB mp4", "native depth zlib", "per-frame camera poses", "stream metadata"],
            "ground_truth_or_alignment_read_during_matching": False,
            "predictions_file": str(output.relative_to(ROOT)),
            "timings": timings,
            "model_load_seconds_shared_across_run": model_load_seconds,
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        rows.append({"environment_id": env, "status": "complete", **timings})
        print(f"saved predictions {env}: Q={len(query_points)} R={len(ref_points)}; GPU={gpu_seconds:.1f}s CPU={cpu_match_seconds+consensus_seconds:.1f}s", flush=True)

    write_csv(PREDS_DIR / "run_summary.csv", rows)
    run_info = {
        "dataset_revision": REVISION,
        "subset_manifest": "../quota_subset_manifest.json",
        "model_revision": expected_revision,
        "model_weights_sha256": expected_hash,
        "model_load_seconds": model_load_seconds,
        "environment_count": len(rows),
        "predictions_saved_before_label_scoring": True,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (PREDS_DIR / "run_manifest.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(json.dumps(run_info, indent=2))


def prepare_scan_from_open_archive(archive: zipfile.ZipFile, archive_path: Path, scan_id: str):
    """Prepare one scan, reading no annotation/mesh/alignment members."""
    depth_decode_started = time.perf_counter()
    info, poses, depth_all, _ = load_frames(archive, scan_id)
    depth_decode_seconds = time.perf_counter() - depth_decode_started
    color = info["color"]
    depth_meta = info["depth"]
    depth_w = int(depth_meta["resolution"][1])
    depth_h = int(depth_meta["resolution"][0])
    color_w = int(color["resolution"][1])
    color_h = int(color["resolution"][0])
    if len(poses) != info["count"] or depth_all.shape[0] != info["count"]:
        raise ValueError(f"RGB/depth/pose frame count mismatch for {scan_id}")
    indices = frame_indices(info["count"])
    video_name = unique_member(archive, scan_id + ".mp4")
    decode_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="multiscan_video_", dir=ROOT / "data") as temp:
        video_path = Path(temp) / "video.mp4"
        with archive.open(video_name) as source, video_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        expr = "+".join(f"eq(n\\,{int(i)})" for i in indices)
        process = subprocess.Popen([
            "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
            "-vf", f"select='{expr}'", "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None
        frame_bytes = color_w * color_h * 3
        rgb_images = []
        try:
            for frame_id in indices:
                raw = process.stdout.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise RuntimeError(f"short RGB decode for {scan_id} frame {frame_id}: {len(raw)} bytes")
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(color_h, color_w, 3)
                rgb_images.append(np.asarray(Image.fromarray(rgb).resize((INPUT_SIZE, INPUT_SIZE), PIL_RESAMPLING.BILINEAR)).copy())
            code = process.wait()
            error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            if code:
                raise RuntimeError(f"ffmpeg failed for {scan_id}: {error[-1000:]}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    video_decode_seconds = time.perf_counter() - decode_started

    geometry_started = time.perf_counter()
    depth_images = [np.asarray(Image.fromarray(depth_all[int(i)]).resize((INPUT_SIZE, INPUT_SIZE), PIL_RESAMPLING.NEAREST), dtype=np.float32) for i in indices]
    selected_poses = [poses[int(i)] for i in indices]
    points, rgb_desc, frame_ids, patch_ids, tensors, encoded_frame_ids = [], [], [], [], [], []
    patch_centers = np.arange(PATCH // 2, INPUT_SIZE, PATCH, dtype=np.float64)
    yy, xx = np.meshgrid(patch_centers, patch_centers, indexing="ij")
    patch_y, patch_x = yy.astype(np.int64), xx.astype(np.int64)
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    sx, sy = INPUT_SIZE / color_w, INPUT_SIZE / color_h
    arkit_to_o3d = np.diag([1.0, -1.0, -1.0, 1.0])
    for frame_id, rgb, depth, pose in zip(indices, rgb_images, depth_images, selected_poses):
        patch_depth = depth[patch_y, patch_x].reshape(-1)
        valid = np.isfinite(patch_depth) & (patch_depth > 0)
        if not valid.any():
            continue
        fx, fy, cx, cy = scaled_intrinsic_parameters(pose["intrinsics"], sx, sy)
        rays = np.stack(((xy[valid, 0]-cx)/fx, (xy[valid,1]-cy)/fy, np.ones(valid.sum())), axis=1)
        T_arkit = np.asarray(pose["transform"], dtype=np.float64).reshape(4,4,order="F")
        T_cw = T_arkit @ arkit_to_o3d
        pts = (rays * patch_depth[valid, None].astype(np.float64)) @ T_cw[:3,:3].T + T_cw[:3,3]
        points.append(pts.astype(np.float32))
        rgb_desc.append(rgb.reshape(16,PATCH,16,PATCH,3).mean(axis=(1,3)).reshape(-1,3)[valid].astype(np.float32))
        frame_ids.append(np.full(int(valid.sum()), int(frame_id), dtype=np.int32))
        patch_ids.append(np.flatnonzero(valid).astype(np.int32))
        encoded_frame_ids.append(int(frame_id))
        image = torch.from_numpy(rgb).permute(2,0,1).float().div_(255.0)
        tensors.append((image - torch.from_numpy(MEAN)[:,None,None]) / torch.from_numpy(STD)[:,None,None])
    if not points:
        raise ValueError(f"no finite positive depth patch points in {scan_id}")
    geometry_seconds = time.perf_counter() - geometry_started
    return (
        np.concatenate(points), np.concatenate(rgb_desc), np.concatenate(frame_ids), np.concatenate(patch_ids),
        {"images": torch.stack(tensors), "stage": {"selected_frame_ids": indices.tolist(),
          "rgb_resolution": [color_w,color_h], "depth_resolution": [depth_w,depth_h],
          "frame_count": info["count"], "depth_decode_seconds": depth_decode_seconds,
          "video_decode_seconds": video_decode_seconds, "geometry_projection_seconds": geometry_seconds,
          "encoded_frame_ids": encoded_frame_ids}},
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
