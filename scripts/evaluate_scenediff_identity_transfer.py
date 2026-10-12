#!/usr/bin/env python3
"""Frozen object-descriptor transfer on the official SceneDiff benchmark.

This is an association-only diagnostic on changed-object annotations. It does
not reproduce the SceneDiff change-detection system. RGB videos and masks are
read from the pinned public ZIP via HTTP byte ranges; each selected pair's raw
videos are deleted after feature extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2
from stream_rescan_archive_subset import HTTPRangeReader


REVISION = "d9e2115fae13daf219ed547bfb072a875aae4d7c"
ARCHIVE_URL = (
    "https://huggingface.co/datasets/yuqun/SceneDiff/resolve/"
    f"{REVISION}/scenediff_benchmark.zip"
)
ARCHIVE_BYTES = 11_919_571_401
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
STRIDE = 15


def decode_compressed_counts(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    counts: list[int] = []
    position = 0
    while position < len(value):
        x, shift, more = 0, 0, True
        while more:
            c = ord(value[position]) - 48
            position += 1
            x |= (c & 0x1F) << (5 * shift)
            more = bool(c & 0x20)
            if not more and (c & 0x10):
                x |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_rle(rle: dict[str, object]) -> np.ndarray:
    height, width = (int(x) for x in rle["size"])
    counts = decode_compressed_counts(rle["counts"])
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    cursor, foreground = 0, False
    for run in counts:
        end = cursor + int(run)
        if end > total or run < 0:
            raise ValueError("invalid COCO RLE run length")
        if foreground:
            flat[cursor:end] = 1
        cursor = end
        foreground = not foreground
    if cursor != total:
        raise ValueError(f"RLE covers {cursor} of {total} pixels")
    return flat.reshape((height, width), order="F").astype(bool)


def canonical_label(value: str) -> str:
    return re.sub(r"[_\s]+\d+$", "", value.strip().lower())


def hash_order(items: list[str], salt: str) -> list[str]:
    return sorted(items, key=lambda x: hashlib.sha256(f"{salt}:{x}".encode()).hexdigest())


def extract_member(archive: ZipFile, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member, "r") as source, dest.open("wb") as target:
        shutil.copyfileobj(source, target, length=1 << 20)


def object_metadata(seg: dict[str, object]) -> dict[str, dict[str, object]]:
    result = {}
    for index, row in enumerate(seg["objects"]):
        key = str(row.get("original_obj_idx", index))
        result[key] = row
    return result


def sample_video(
    path: Path,
    objects: dict[str, object],
    metadata: dict[str, dict[str, object]],
    frame_stride: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, int]]:
    if not objects:
        return {}, {}, {}, {}
    first_id = next(iter(objects))
    first_frame = next(iter(objects[first_id].values()))
    height, width = (int(x) for x in first_frame["size"])
    frame_bytes = height * width * 3
    select = f"select='not(mod(n\\,{frame_stride}))'"
    command = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-vf", f"{select},scale={width}:{height}:flags=bilinear",
        "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    crops: dict[str, list[np.ndarray]] = defaultdict(list)
    masks: dict[str, list[np.ndarray]] = defaultdict(list)
    histograms: dict[str, list[np.ndarray]] = defaultdict(list)
    views: dict[str, int] = defaultdict(int)
    assert process.stdout is not None
    selected_index = 0
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"short decoded frame from {path}: {len(raw)} bytes")
            frame_id = selected_index * frame_stride
            selected_index += 1
            image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            for object_key, frame_map in objects.items():
                rle = frame_map.get(str(frame_id))
                if rle is None:
                    continue
                mask = decode_rle(rle)
                if mask.shape != image.shape[:2]:
                    raise ValueError(
                        f"mask/video shape mismatch for {path.name} frame {frame_id}: "
                        f"{mask.shape} vs {image.shape[:2]}"
                    )
                ys, xs = np.where(mask)
                if len(xs) < 100:
                    continue
                pad = 8
                y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad + 1)
                x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad + 1)
                crop = image[y0:y1, x0:x1]
                crop_mask = mask[y0:y1, x0:x1]
                resized = np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR))
                resized_mask = np.asarray(
                    Image.fromarray(crop_mask.astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST)
                ) > 0
                # Fill background with the model's RGB mean; patch pooling still
                # uses the resized annotation mask.
                mean_rgb = np.round(MEAN * 255).astype(np.uint8)
                resized = np.where(resized_mask[..., None], resized, mean_rgb[None, None, :])
                crops[str(object_key)].append(resized)
                masks[str(object_key)].append(resized_mask)
                pixels = image[mask]
                hist = []
                for channel in range(3):
                    bins = np.histogram(pixels[:, channel], bins=32, range=(0, 256))[0]
                    hist.append(bins.astype(np.float32) / max(len(pixels), 1))
                histograms[str(object_key)].append(
                    F.normalize(torch.from_numpy(np.concatenate(hist)), dim=0).numpy()
                )
                views[str(object_key)] += 1
        returncode = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {path}: {stderr[-1000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return dict(crops), dict(masks), dict(histograms), dict(views)


@torch.inference_mode()
def descriptors_for_objects(
    crops: dict[str, list[np.ndarray]], masks: dict[str, list[np.ndarray]],
    histograms: dict[str, list[np.ndarray]], model: torch.nn.Module,
    device: torch.device, batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    flat_keys, flat_crops, flat_masks = [], [], []
    for key in sorted(crops):
        for image, mask in zip(crops[key], masks[key]):
            flat_keys.append(key)
            flat_crops.append(image)
            flat_masks.append(mask)
    pooled_rows: list[np.ndarray] = []
    cls_rows: list[np.ndarray] = []
    for start in range(0, len(flat_crops), batch_size):
        images = np.stack(flat_crops[start:start + batch_size]).astype(np.float32) / 255.0
        images = (images - MEAN[None, None, None, :]) / STD[None, None, None, :]
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
        captured = extract_block_tokens(model, tensor, (2, 20, 40))
        tokens = [
            F.normalize(captured[layer][:, model.num_prefix_tokens:], dim=2)
            for layer in (2, 20, 40)
        ]
        patch_tokens = torch.cat(tokens, dim=2)
        mask_batch = np.stack(flat_masks[start:start + batch_size]).astype(np.float32)
        # extract_block_tokens returns hooked block activations on CPU.
        weights = torch.from_numpy(mask_batch)[:, None]
        weights = F.avg_pool2d(weights, kernel_size=14, stride=14).flatten(1)
        pooled = (patch_tokens * weights[:, :, None]).sum(1)
        pooled /= weights.sum(1).clamp_min(1e-6)[:, None]
        pooled_rows.append(F.normalize(pooled, dim=1).cpu().numpy())
        cls = F.normalize(captured[40][:, 0], dim=1).cpu().numpy()
        cls_rows.append(cls)
    pooled_all = np.concatenate(pooled_rows) if pooled_rows else np.empty((0, 0), np.float32)
    cls_all = np.concatenate(cls_rows) if cls_rows else np.empty((0, 0), np.float32)
    pooled_by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    cls_by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, pooled, cls in zip(flat_keys, pooled_all, cls_all):
        pooled_by_object[key].append(pooled)
        cls_by_object[key].append(cls)
    pool = {key: F.normalize(torch.tensor(np.mean(vals, axis=0)), dim=0).numpy()
            for key, vals in pooled_by_object.items()}
    cls = {key: F.normalize(torch.tensor(np.mean(vals, axis=0)), dim=0).numpy()
           for key, vals in cls_by_object.items()}
    color = {
        key: F.normalize(torch.tensor(np.mean(histograms[key], axis=0)), dim=0).numpy()
        for key in histograms if histograms[key]
    }
    return pool, cls, color


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--subsets", nargs="+", choices=("varied", "kitchen"), default=("varied", "kitchen"))
    parser.add_argument("--pair-count", type=int, default=40,
                        help="deterministic hash-selected pairs in each subset; 0 selects all")
    parser.add_argument("--frame-stride", type=int, default=STRIDE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/scenediff_pair_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frame_stride < 1 or args.pair_count < 0:
        raise ValueError("frame stride must be positive and pair count non-negative")

    remote = HTTPRangeReader(ARCHIVE_URL)
    if remote.size != ARCHIVE_BYTES:
        raise ValueError(f"pinned archive size changed: {remote.size} != {ARCHIVE_BYTES}")
    with ZipFile(remote) as archive:
        split_name = f"scenediff_benchmark/splits/{args.split}_split.json"
        splits = json.loads(archive.read(split_name))
        selected: list[tuple[str, str]] = []
        for subset in args.subsets:
            pairs = list(splits[subset])
            ordered = hash_order(pairs, f"worldcache-scenediff-{args.split}-v1")
            if args.pair_count:
                ordered = ordered[:args.pair_count]
            selected.extend((subset, pair) for pair in ordered)
        if not selected:
            raise ValueError("no sequence pairs selected")

        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_dinov2("giant", 224, device)
        all_rows: list[dict[str, object]] = []
        pair_summaries = []
        downloaded_members = []
        selected_ids = [pair for _, pair in selected]
        for index, (subset, pair) in enumerate(selected, start=1):
            prefix = f"scenediff_benchmark/data/{pair}/"
            segment_info = archive.getinfo(prefix + "segments.pkl")
            downloaded_members.append({
                "path": segment_info.filename,
                "compressed_bytes": segment_info.compress_size,
                "uncompressed_bytes": segment_info.file_size,
                "crc32": f"{segment_info.CRC:08x}",
            })
            seg = pickle.loads(archive.read(prefix + "segments.pkl"))
            info_by_name = {info.filename: info for info in archive.infolist()
                            if info.filename.startswith(prefix)}
            raw_videos = {}
            for video_index in (1, 2):
                candidates = [
                    name for name in info_by_name
                    if Path(name).name.lower().startswith(f"original_video{video_index}.")
                ]
                if len(candidates) != 1:
                    raise FileNotFoundError(f"expected one raw video{video_index} for {pair}; found {candidates}")
                member = candidates[0]
                dest = args.cache_dir / f"{pair}_video{video_index}{Path(member).suffix}"
                extract_member(archive, member, dest)
                raw_videos[video_index] = dest
                downloaded_members.append({
                    "path": member,
                    "compressed_bytes": info_by_name[member].compress_size,
                    "uncompressed_bytes": info_by_name[member].file_size,
                    "crc32": f"{info_by_name[member].CRC:08x}",
                })
            video_features = []
            try:
                for video_index in (1, 2):
                    object_key = f"video{video_index}_objects"
                    crops, masks, hists, views = sample_video(
                        raw_videos[video_index], seg[object_key], object_metadata(seg), args.frame_stride
                    )
                    desc = descriptors_for_objects(crops, masks, hists, model, device, args.batch_size)
                    video_features.append((desc, views))
                source_desc, source_views = video_features[0]
                target_desc, target_views = video_features[1]
                source_pool, source_cls, source_color = source_desc
                target_pool, target_cls, target_color = target_desc
                metadata = object_metadata(seg)
                candidates = sorted(set(source_pool) & set(source_cls) & set(source_color))
                candidate_set = set(candidates)
                query_keys = sorted(set(target_pool) & set(target_cls) & set(target_color))
                rows = []
                for query_key in query_keys:
                    if query_key not in candidate_set or len(candidates) < 2:
                        continue
                    row_meta = metadata.get(query_key, {})
                    label = str(row_meta.get("label", query_key))
                    query = {
                        "pair": pair, "subset": subset, "object_id": query_key,
                        "label": label, "canonical_label": canonical_label(label),
                        "candidate_count": len(candidates),
                        "same_label_candidates": sum(
                            canonical_label(str(metadata.get(candidate, {}).get("label", candidate))) == canonical_label(label)
                            for candidate in candidates
                        ),
                        "query_views": target_views.get(query_key, 0),
                        "gallery_views": source_views.get(query_key, 0),
                    }
                    for descriptor, source_map, target_map in (
                        ("maskpool", source_pool, target_pool),
                        ("cls", source_cls, target_cls),
                        ("rgbhist", source_color, target_color),
                    ):
                        scores = np.asarray([
                            float(np.dot(target_map[query_key], source_map[candidate]))
                            for candidate in candidates
                        ])
                        order = np.argsort(-scores)
                        ranked = [candidates[i] for i in order]
                        query[f"{descriptor}_correct"] = int(ranked[0] == query_key)
                        query[f"{descriptor}_top5"] = int(query_key in ranked[:5])
                        query[f"{descriptor}_rank"] = ranked.index(query_key) + 1
                        query[f"{descriptor}_margin"] = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
                        same = [c for c in candidates if canonical_label(str(metadata.get(c, {}).get("label", c))) == canonical_label(label)]
                        if len(same) >= 2:
                            same_rank = sorted(same, key=lambda c: float(np.dot(target_map[query_key], source_map[c])), reverse=True)
                            query[f"{descriptor}_same_label_correct"] = int(same_rank[0] == query_key)
                    rows.append(query)
                all_rows.extend(rows)
                pair_summaries.append({
                    "pair": pair, "subset": subset,
                    "annotated_objects": len(metadata),
                    "source_objects_with_features": len(source_pool),
                    "target_objects_with_features": len(target_pool),
                    "candidate_count": len(candidates),
                    "linked_queries_scored": len(rows),
                    "objects_present_both_sequences": sum(
                        bool(item.get("in_video1")) and bool(item.get("in_video2"))
                        for item in metadata.values()
                    ),
                    "scene_type": seg.get("scene_type", ""),
                })
            finally:
                for path in raw_videos.values():
                    path.unlink(missing_ok=True)
            print(f"[{index}/{len(selected)}] {subset}/{pair}: queries={len(rows)} candidates={len(candidates)}", flush=True)

    if not all_rows:
        raise ValueError("no linked object queries had features in both sequences")
    import pandas as pd

    frame = pd.DataFrame(all_rows)
    pair_names = sorted(frame.pair.unique())
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, len(pair_names), size=(args.bootstrap_samples, len(pair_names)))
    paired_diffs = {}
    for descriptor in ("maskpool", "cls"):
        pair_diffs = frame.groupby("pair").apply(
            lambda group: float(group[f"{descriptor}_correct"].mean() - group.rgbhist_correct.mean()),
            include_groups=False,
        ).reindex(pair_names).to_numpy()
        draws = pair_diffs[bootstrap_indices].mean(axis=1)
        paired_diffs[f"{descriptor}_minus_rgbhist"] = {
            "pair_macro_difference_pp": float(100 * pair_diffs.mean()),
            "pair_cluster_bootstrap_95_pp": [float(x) for x in 100 * np.quantile(draws, [0.025, 0.975])],
            "n_pairs": len(pair_names),
            "bootstrap_samples": args.bootstrap_samples,
        }
    metrics = {}
    for descriptor in ("maskpool", "cls", "rgbhist"):
        metrics[descriptor] = {
            "query_micro_top1": float(frame[f"{descriptor}_correct"].mean()),
            "query_pair_macro_top1": float(frame.groupby("pair")[f"{descriptor}_correct"].mean().mean()),
            "top5_micro": float(frame[f"{descriptor}_top5"].mean()),
            "mean_reciprocal_rank": float((1.0 / frame[f"{descriptor}_rank"]).mean()),
        }
    subset_metrics = {}
    for subset, subset_frame in frame.groupby("subset"):
        subset_metrics[str(subset)] = {
            "queries": int(len(subset_frame)),
            "pairs_scored": int(subset_frame.pair.nunique()),
            "metrics": {
                descriptor: {
                    "query_micro_top1": float(subset_frame[f"{descriptor}_correct"].mean()),
                    "query_pair_macro_top1": float(
                        subset_frame.groupby("pair")[f"{descriptor}_correct"].mean().mean()
                    ),
                    "top5_micro": float(subset_frame[f"{descriptor}_top5"].mean()),
                    "mean_reciprocal_rank": float(
                        (1.0 / subset_frame[f"{descriptor}_rank"]).mean()
                    ),
                }
                for descriptor in ("maskpool", "cls", "rgbhist")
            },
        }
    same_label = frame[frame.same_label_candidates >= 2]
    same_label_metrics = {
        "queries": int(len(same_label)),
        "fraction_with_same_label_distractor": float(len(same_label) / len(frame)),
        "top1": {
            d: float(same_label[f"{d}_same_label_correct"].mean()) if len(same_label) else None
            for d in ("maskpool", "cls", "rgbhist")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "query_rows.csv", index=False)
    pd.DataFrame(pair_summaries).to_csv(args.output_dir / "pair_rows.csv", index=False)
    summary = {
        "dataset": "SceneDiff",
        "dataset_revision": REVISION,
        "archive_bytes": ARCHIVE_BYTES,
        "dataset_license_metadata": "MIT",
        "official_split": args.split,
        "subsets": list(args.subsets),
        "selected_pairs": selected_ids,
        "pairs_processed": len(pair_summaries),
        "pairs_with_scored_queries": len(pair_names),
        "pairs_with_only_one_source_candidate": sum(
            int(row["candidate_count"] < 2) for row in pair_summaries
        ),
        "frame_stride": args.frame_stride,
        "model": "DINOv2 ViT-G/14; concatenated normalized layer-2/20/40 patch tokens",
        "task": "closed-set changed-object identity retrieval from video 2 to video 1; only objects annotated present in both videos are scored",
        "queries_scored": len(frame),
        "candidate_count_range": [int(frame.candidate_count.min()), int(frame.candidate_count.max())],
        "metrics": metrics,
        "subset_metrics": subset_metrics,
        "same_label_hard_negative_sensitivity": same_label_metrics,
        "paired_differences": paired_diffs,
        "range_reader_bytes_fetched": remote.bytes_downloaded,
        "downloaded_video_members": downloaded_members,
        "selected_members_compressed_bytes": int(sum(row["compressed_bytes"] for row in downloaded_members)),
        "limitations": [
            "This is an association-only diagnostic, not the official SceneDiff object-change AP evaluation or a reproduction of SceneDiff.",
            "Only annotated changed objects form the candidate pool; unchanged background objects are not modeled.",
            "Ground-truth masks provide privileged proposals; the RGB videos do not provide sensor depth or camera poses.",
            "The result tests frozen descriptor transfer, not a new method or real RGB-D rescan utility.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"queries": len(frame), "metrics": metrics, "paired_differences": paired_diffs,
                      "same_label": same_label_metrics, "range_bytes": remote.bytes_downloaded}, indent=2))


if __name__ == "__main__":
    main()
