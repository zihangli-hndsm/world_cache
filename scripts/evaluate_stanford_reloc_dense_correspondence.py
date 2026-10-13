#!/usr/bin/env python3
"""Evaluate frozen RGB patch matches against global-pose RGB-D geometry.

This is a bounded external geometry/correspondence stress test on the public
Stanford Reloc Apartment 1 archive. It evaluates cross-sequence matches within
one room; there are no object identity annotations.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2
from stream_rescan_archive_subset import HTTPRangeReader
from probe_stanford_reloc_overlap import parse_info, parse_pose


ARCHIVE_URLS = {
    "apt1": "https://graphics.stanford.edu/projects/reloc/data/apt1.zip",
    "apt2": "https://graphics.stanford.edu/projects/reloc/data/apt2.zip",
    "office1": "https://graphics.stanford.edu/projects/reloc/data/office1.zip",
    "office2": "https://graphics.stanford.edu/projects/reloc/data/office2.zip",
}
MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def sequence_ranges(text: str) -> list[tuple[str, int, int]]:
    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Example: sequence0 [frames=357] [start=0 ; end=356]
        name, rest = line.split("[", 1)
        start = int(rest.split("start=", 1)[1].split(";", 1)[0].strip())
        end = int(rest.split("end=", 1)[1].split("]", 1)[0].strip())
        result.append((name.strip(), start, end))
    return result


def choose_frame_ids(ranges: list[tuple[str, int, int]], per_sequence: int) -> dict[str, list[int]]:
    chosen = {}
    for name, start, end in ranges:
        # Avoid sequence-boundary frames, which can contain transition artifacts.
        lo, hi = start + 20, end - 20
        if hi < lo:
            lo, hi = start, end
        if per_sequence == 1:
            ids = [int(round((lo + hi) / 2))]
        else:
            ids = np.linspace(lo, hi, per_sequence, dtype=int).tolist()
        chosen[name] = ids
    return chosen


def frame_geometry(depth_bytes: bytes, pose: np.ndarray, intr: tuple, color_size: tuple[int, int], patch_grid: int):
    w, h, fx, fy, cx, cy, _depth_shift = intr
    depth = np.asarray(Image.open(io.BytesIO(depth_bytes)), dtype=np.float32) * 0.001
    if depth.shape != (h, w):
        raise ValueError(f"depth shape {depth.shape}, expected {(h, w)}")
    color_w, color_h = color_size
    centers = (np.arange(patch_grid, dtype=np.float32) + 0.5) * (1.0 / patch_grid)
    xx, yy = np.meshgrid(centers * color_w * (w / color_w), centers * color_h * (h / color_h))
    # Image pixels map to the default depth grid by normalized image coordinates.
    du = np.clip(np.floor(xx).astype(int), 0, w - 1)
    dv = np.clip(np.floor(yy).astype(int), 0, h - 1)
    z = depth[dv, du]
    valid = np.isfinite(z) & (z >= 0.15) & (z <= 8.0)
    cam = np.stack(((du - cx) * z / fx, (dv - cy) * z / fy, z), axis=-1).reshape(-1, 3)
    world = cam @ pose[:3, :3].T + pose[:3, 3]
    return world.astype(np.float32), valid.reshape(-1)


@torch.inference_mode()
def frame_features(model, rgb_bytes: bytes, resolution: int, device: torch.device):
    rgb = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    original_size = rgb.size
    resized = rgb.resize((resolution, resolution), Image.Resampling.BICUBIC)
    rgb_array = np.asarray(resized).copy()
    image = torch.from_numpy(rgb_array).permute(2, 0, 1).float().div_(255.0).to(device)
    mean = torch.as_tensor(MEAN, device=device)[:, None, None]
    std = torch.as_tensor(STD, device=device)[:, None, None]
    captured = extract_block_tokens(model, ((image - mean) / std).unsqueeze(0), (2, 20, 40))
    block_tokens = []
    for layer in (2, 20, 40):
        tokens = captured[layer][0, model.num_prefix_tokens :]
        block_tokens.append(F.normalize(tokens, dim=1))
    desc = F.normalize(torch.cat(block_tokens, dim=1), dim=1).numpy()
    grid = resolution // 14
    if len(desc) != grid * grid:
        raise ValueError(f"feature count {len(desc)} does not match grid {grid}x{grid}")
    # A simple patch color-histogram baseline with the same spatial cells.
    cells = rgb_array.reshape(grid, 14, grid, 14, 3).transpose(0, 2, 1, 3, 4).reshape(grid * grid, 14 * 14, 3)
    histograms = []
    for patch in cells:
        hist = np.concatenate([
            np.histogram(patch[:, channel], bins=8, range=(0, 256))[0]
            for channel in range(3)
        ]).astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        hist /= max(np.linalg.norm(hist), 1e-12)
        histograms.append(hist)
    return desc.astype(np.float32), np.stack(histograms), original_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="apt1", help="top-level archive directory, e.g. apt1 or office1")
    parser.add_argument("--room", default="living", help="room directory listed in the archive split")
    parser.add_argument("--frames-per-sequence", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--model-size", choices=("base", "large", "giant"), default="giant")
    parser.add_argument("--max-sequences", type=int, default=3)
    parser.add_argument("--max-distance-cm", type=float, nargs="+", default=(2.0, 5.0, 10.0, 20.0))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-url", help="defaults to the official archive for --dataset-root")
    args = parser.parse_args()
    if args.frames_per_sequence < 1 or args.max_sequences < 2:
        raise ValueError("need at least one frame per sequence and two sequences")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive_url = args.archive_url or ARCHIVE_URLS.get(args.dataset_root)
    if archive_url is None:
        raise ValueError(f"pass --archive-url for dataset root {args.dataset_root!r}")
    remote = HTTPRangeReader(archive_url)
    room_root = f"{args.dataset_root}/{args.room}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ZipFile(remote) as archive:
        intr = parse_info(archive.read(f"{room_root}/info.txt").decode("utf-8"))
        ranges = sequence_ranges(archive.read(f"{room_root}/split.txt").decode("utf-8"))[: args.max_sequences]
        chosen = choose_frame_ids(ranges, args.frames_per_sequence)
        bounds = {name: (start, end) for name, start, end in ranges}
        model = load_dinov2(args.model_size, args.resolution, device)
        grid = args.resolution // 14
        frames = {}
        skipped_invalid_pose_frames = []
        for seq_name, ids in chosen.items():
            used_ids = set()
            actual_ids = []
            for frame_id in ids:
                start, end = bounds[seq_name]
                attempts = [frame_id]
                for offset in range(1, 31):
                    attempts.extend((frame_id + offset, frame_id - offset))
                valid_pose = None
                actual_id = None
                for candidate_id in attempts:
                    if candidate_id < start or candidate_id > end or candidate_id in used_ids:
                        continue
                    pose_member = f"{room_root}/data/frame-{candidate_id:06d}.pose.txt"
                    pose_bytes = archive.read(pose_member)
                    try:
                        valid_pose = parse_pose(pose_bytes.decode("utf-8"))
                        actual_id = candidate_id
                        break
                    except (ValueError, UnicodeDecodeError):
                        skipped_invalid_pose_frames.append(f"{seq_name}/{candidate_id}")
                if actual_id is None:
                    print(f"skip {seq_name}/{frame_id}: no valid pose within 30 frames", flush=True)
                    continue
                used_ids.add(actual_id)
                actual_ids.append(actual_id)
                stem = f"{room_root}/data/frame-{actual_id:06d}"
                rgb_bytes = archive.read(stem + ".color.jpg")
                depth_bytes = archive.read(stem + ".depth.png")
                descriptors, histograms, color_size = frame_features(model, rgb_bytes, args.resolution, device)
                points, valid = frame_geometry(depth_bytes, valid_pose, intr, color_size, grid)
                frames[(seq_name, frame_id)] = {
                    "desc": descriptors[valid],
                    "hist": histograms[valid],
                    "points": points[valid],
                    "n_patches": int(valid.sum()),
                    "valid": valid,
                }
                if actual_id != frame_id:
                    frames[(seq_name, actual_id)] = frames.pop((seq_name, frame_id))
                print(f"frame {args.room}/{seq_name}/{actual_id}: {int(valid.sum())}/{grid*grid} depth-supported patches", flush=True)
            chosen[seq_name] = actual_ids

    rows = []
    pair_rows = []
    sequence_names = list(chosen)
    thresholds = [float(x) / 100.0 for x in args.max_distance_cm]
    for source_seq in sequence_names:
        for target_seq in sequence_names:
            if source_seq == target_seq:
                continue
            for source_id in chosen[source_seq]:
                for target_id in chosen[target_seq]:
                    source = frames[(source_seq, source_id)]
                    target = frames[(target_seq, target_id)]
                    scores = source["desc"] @ target["desc"].T
                    prediction = scores.argmax(axis=1)
                    color_scores = source["hist"] @ target["hist"].T
                    color_prediction = color_scores.argmax(axis=1)
                    target_points = target["points"]
                    source_points = source["points"]
                    target_tree = cKDTree(target_points)
                    gt_dist, gt_idx = target_tree.query(source_points, k=1, workers=-1)
                    pred_dist = np.linalg.norm(source_points - target_points[prediction], axis=1)
                    color_pred_dist = np.linalg.norm(source_points - target_points[color_prediction], axis=1)
                    pair = {
                        "room": args.room,
                        "source_sequence": source_seq,
                        "target_sequence": target_seq,
                        "source_frame": source_id,
                        "target_frame": target_id,
                        "source_patches": int(len(source_points)),
                        "target_patches": int(len(target_points)),
                        "geometric_support_5cm": float(np.mean(gt_dist <= 0.05)),
                        "geometric_support_10cm": float(np.mean(gt_dist <= 0.10)),
                        "median_gt_nearest_cm": float(np.median(gt_dist) * 100),
                        "dino_top1_median_error_cm": float(np.median(pred_dist) * 100),
                        "rgbhist_top1_median_error_cm": float(np.median(color_pred_dist) * 100),
                    }
                    for threshold in thresholds:
                        label = f"{threshold*100:g}cm"
                        supported = gt_dist <= threshold
                        pair[f"support_{label}"] = float(np.mean(supported))
                        pair[f"hit_{label}_all_queries"] = float(np.mean(pred_dist <= threshold))
                        pair[f"rgbhist_hit_{label}_all_queries"] = float(np.mean(color_pred_dist <= threshold))
                        pair[f"hit_{label}_supported_queries"] = (
                            float(np.mean(pred_dist[supported] <= threshold)) if supported.any() else float("nan")
                        )
                        pair[f"rgbhist_hit_{label}_supported_queries"] = (
                            float(np.mean(color_pred_dist[supported] <= threshold)) if supported.any() else float("nan")
                        )
                    pair_rows.append(pair)
                    # Save compact query-level errors for later fixed-threshold analyses.
                    for qi in range(len(source_points)):
                        row = {
                            "room": args.room,
                            "source_sequence": source_seq,
                            "target_sequence": target_seq,
                            "source_frame": source_id,
                            "target_frame": target_id,
                            "query_patch": qi,
                            "gt_nearest_cm": float(gt_dist[qi] * 100),
                            "dino_top1_cm": float(pred_dist[qi] * 100),
                            "rgbhist_top1_cm": float(color_pred_dist[qi] * 100),
                            "gt_nearest_patch": int(gt_idx[qi]),
                            "dino_top1_patch": int(prediction[qi]),
                        }
                        rows.append(row)
                    print(f"pair {source_seq}:{source_id}->{target_seq}:{target_id}: support5={pair['geometric_support_5cm']:.3f}, dino<=5cm={pair['hit_5cm_all_queries']:.3f}", flush=True)

    with (args.output_dir / "pair_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        writer.writeheader(); writer.writerows(pair_rows)
    with (args.output_dir / "query_rows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "complete",
        "dataset": f"Stanford RGB Camera Relocalization Dataset, {args.dataset_root}",
        "source": "https://graphics.stanford.edu/projects/reloc/",
        "license": "CC BY-NC-SA 4.0",
        "room": args.room,
        "dataset_root": args.dataset_root,
        "archive_bytes": remote.size,
        "range_bytes_fetched": remote.bytes_downloaded,
        "sequences": chosen,
        "frames_per_sequence": args.frames_per_sequence,
        "skipped_invalid_pose_frames": skipped_invalid_pose_frames,
        "invalid_pose_replacement_policy": "for each planned frame, try it first, then search alternating +/- frame offsets up to 30",
        "model": f"DINOv2 ViT-{args.model_size}, concatenated normalized patch tokens from blocks 2/20/40",
        "resolution": args.resolution,
        "depth_intrinsics": {"width": intr[0], "height": intr[1], "fx": intr[2], "fy": intr[3], "cx": intr[4], "cy": intr[5]},
        "depth_scale_m_per_unit": 0.001,
        "queries": len(rows),
        "directed_frame_pairs": len(pair_rows),
        "pair_macro": {
            key: float(np.nanmean([row[key] for row in pair_rows]))
            for key in pair_rows[0] if key.startswith("hit_") or key.startswith("support_")
        },
        "pooled": {
            "support_5cm": float(np.mean([row["gt_nearest_cm"] <= 5 for row in rows])),
            "dino_hit_5cm_all_queries": float(np.mean([row["dino_top1_cm"] <= 5 for row in rows])),
            "rgbhist_hit_5cm_all_queries": float(np.mean([row["rgbhist_top1_cm"] <= 5 for row in rows])),
            "dino_hit_5cm_supported_queries": float(np.mean([row["dino_top1_cm"] <= 5 for row in rows if row["gt_nearest_cm"] <= 5])),
            "rgbhist_hit_5cm_supported_queries": float(np.mean([row["rgbhist_top1_cm"] <= 5 for row in rows if row["gt_nearest_cm"] <= 5])),
            "dino_median_error_cm": float(np.median([row["dino_top1_cm"] for row in rows])),
            "rgbhist_median_error_cm": float(np.median([row["rgbhist_top1_cm"] for row in rows])),
        },
        "limits": [
            "The official page says camera poses are globally bundle-adjusted across sequences; this audit does not independently verify them.",
            "The supplied intrinsics are defaults and uncalibrated; RGB-depth extrinsics are assumed identity as specified in info.txt.",
            "There are no object identity labels; this tests geometric surface correspondence, not physical-object recognition or room change.",
            "One room is evaluated; this is a domain stress test, not an independent multi-environment estimate.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
