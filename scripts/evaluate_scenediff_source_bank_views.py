#!/usr/bin/env python3
"""Test whether source-view count helps frozen DINO object association on SceneDiff.

This is an exploratory RGB-video transfer check. It uses the official test split,
SceneDiff's annotated changed-object masks, and the same DINOv2 ViT-G/14 patch
descriptor as the existing SceneDiff diagnostic. No depth, pose, or raw video is
retained. The primary contrast is four randomly sampled source views vs. one;
target descriptors and candidate sets stay fixed across the view-count curve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
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
from evaluate_scenediff_identity_transfer import (
    ARCHIVE_BYTES, ARCHIVE_URL, MEAN, STD, canonical_label, decode_rle,
    extract_member, object_metadata,
)
from stream_rescan_archive_subset import HTTPRangeReader


@torch.inference_mode()
def encode_views(crops: dict[str, list[np.ndarray]], masks: dict[str, list[np.ndarray]],
                 model: torch.nn.Module, device: torch.device, batch_size: int
                 ) -> dict[str, np.ndarray]:
    keys: list[str] = []
    images: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    for key in sorted(crops):
        for image, mask in zip(crops[key], masks[key]):
            keys.append(key)
            images.append(image)
            mask_rows.append(mask)
    result: dict[str, list[np.ndarray]] = defaultdict(list)
    for start in range(0, len(images), batch_size):
        batch = np.stack(images[start:start + batch_size]).astype(np.float32) / 255.0
        batch = (batch - MEAN[None, None, None, :]) / STD[None, None, None, :]
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        captured = extract_block_tokens(model, tensor, (2, 20, 40))
        tokens = [F.normalize(captured[layer][:, model.num_prefix_tokens:], dim=2)
                  for layer in (2, 20, 40)]
        patch_tokens = torch.cat(tokens, dim=2)
        mask_batch = np.stack(mask_rows[start:start + batch_size]).astype(np.float32)
        weights = F.avg_pool2d(torch.from_numpy(mask_batch)[:, None], kernel_size=14,
                               stride=14).flatten(1)
        pooled = (patch_tokens * weights[:, :, None]).sum(1)
        pooled /= weights.sum(1).clamp_min(1e-6)[:, None]
        pooled = F.normalize(pooled, dim=1).cpu().numpy()
        for key, vector in zip(keys[start:start + batch_size], pooled):
            result[key].append(vector)
    return {key: np.stack(values).astype(np.float32) for key, values in result.items()}


def sample_views(video: Path, objects: dict[str, object], stride: int
                 ) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, int]]:
    if not objects:
        return {}, {}, {}
    first_id = next(iter(objects))
    first_frame = next(iter(objects[first_id].values()))
    height, width = (int(x) for x in first_frame["size"])
    frame_bytes = height * width * 3
    command = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"select='not(mod(n\\,{stride}))',scale={width}:{height}:flags=bilinear",
        "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    crops: dict[str, list[np.ndarray]] = defaultdict(list)
    masks: dict[str, list[np.ndarray]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    assert process.stdout is not None
    selected_index = 0
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"short frame decode from {video}: {len(raw)} bytes")
            frame_id = selected_index * stride
            selected_index += 1
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            for key, frame_map in objects.items():
                rle = frame_map.get(str(frame_id))
                if rle is None:
                    continue
                mask = decode_rle(rle)
                if mask.shape != frame.shape[:2]:
                    raise ValueError(f"mask/video mismatch for {video.name}, frame {frame_id}")
                ys, xs = np.where(mask)
                if len(xs) < 100:
                    continue
                pad = 8
                y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad + 1)
                x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad + 1)
                crop = frame[y0:y1, x0:x1]
                crop_mask = mask[y0:y1, x0:x1]
                resized = np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR))
                resized_mask = np.asarray(Image.fromarray(crop_mask.astype(np.uint8)).resize(
                    (224, 224), Image.Resampling.NEAREST)) > 0
                mean_rgb = np.round(MEAN * 255).astype(np.uint8)
                resized = np.where(resized_mask[..., None], resized, mean_rgb[None, None, :])
                crops[str(key)].append(resized)
                masks[str(key)].append(resized_mask)
                counts[str(key)] += 1
        code = process.wait()
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if code:
            raise RuntimeError(f"ffmpeg failed for {video}: {error[-1000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return dict(crops), dict(masks), dict(counts)


def hash_order(items: list[str], salt: str) -> list[str]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{salt}:{item}".encode()).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-file", type=Path, default=Path(
        "outputs/reports/20261009_scenediff_identity_transfer/pair_rows.csv"))
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/scenediff_bank_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 2 or args.frame_stride < 1:
        raise ValueError("need at least two view-subset runs and positive frame stride")

    import pandas as pd
    pair_rows = pd.read_csv(args.pairs_file)
    pair_ids = sorted(str(x) for x in pair_rows.loc[pair_rows.linked_queries_scored > 0, "pair"].unique())
    pair_ids = hash_order(pair_ids, "worldcache-scenediff-source-bank-v1")
    remote = HTTPRangeReader(ARCHIVE_URL)
    if remote.size != ARCHIVE_BYTES:
        raise ValueError(f"pinned archive size changed: {remote.size} != {ARCHIVE_BYTES}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "pair_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2("giant", 224, device)
    bank_sizes = (1, 2, 4, 8, "all")
    rows: list[dict[str, object]] = []
    downloaded_members: list[dict[str, object]] = []
    resumed_range_bytes = 0
    with ZipFile(remote) as archive:
        all_names = [info.filename for info in archive.infolist()]
        for pair_idx, pair in enumerate(pair_ids, 1):
            checkpoint = checkpoint_dir / (hashlib.sha256(pair.encode()).hexdigest() + ".json")
            if checkpoint.is_file():
                record = json.loads(checkpoint.read_text())
                if record.get("pair") != pair or record.get("seed") != args.seed:
                    raise ValueError(f"checkpoint identity or seed mismatch: {checkpoint}")
                rows.extend(record["rows"])
                downloaded_members.extend(record["members"])
                resumed_range_bytes += int(record.get("range_bytes_fetched", 0))
                print(f"[{pair_idx}/{len(pair_ids)}] resume-skip {pair}: checkpoint present", flush=True)
                continue
            bytes_before_pair = remote.bytes_downloaded
            prefix = f"scenediff_benchmark/data/{pair}/"
            seg_member = prefix + "segments.pkl"
            seg_info = archive.getinfo(seg_member)
            seg = pickle.loads(archive.read(seg_member))
            pair_members = [{"path": seg_member, "compressed_bytes": seg_info.compress_size,
                             "uncompressed_bytes": seg_info.file_size, "crc32": f"{seg_info.CRC:08x}"}]
            pair_rows: list[dict[str, object]] = []
            videos: dict[int, Path] = {}
            try:
                descriptors: list[dict[str, np.ndarray]] = []
                view_counts: list[dict[str, int]] = []
                for video_index in (1, 2):
                    members = [name for name in all_names if name.startswith(prefix)
                               and Path(name).name.lower().startswith(f"original_video{video_index}.")]
                    if len(members) != 1:
                        raise FileNotFoundError(f"expected one video{video_index} in {pair}, found {members}")
                    member = members[0]
                    info = archive.getinfo(member)
                    pair_members.append({"path": member, "compressed_bytes": info.compress_size,
                                         "uncompressed_bytes": info.file_size, "crc32": f"{info.CRC:08x}"})
                    video = args.cache_dir / f"{pair}_v{video_index}{Path(member).suffix}"
                    extract_member(archive, member, video)
                    videos[video_index] = video
                    crops, masks, counts = sample_views(video, seg[f"video{video_index}_objects"], args.frame_stride)
                    descriptors.append(encode_views(crops, masks, model, device, args.batch_size))
                    view_counts.append(counts)
                source, target = descriptors
                source_counts, target_counts = view_counts
                # Match the existing SceneDiff diagnostic: every source-video
                # object with a DINO feature is in the gallery; only queries
                # must be observed in both videos. Requiring each gallery
                # object to also occur in the target video would leak target
                # availability into and shrink the candidate set.
                candidates = sorted(source)
                queries = sorted(set(source) & set(target))
                if len(candidates) < 2:
                    raise ValueError(f"source gallery has fewer than two feature-supported objects: {pair}")
                metadata = object_metadata(seg)
                labels = {key: canonical_label(str(metadata.get(key, {}).get("label", key))) for key in candidates}
                pair_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{pair}".encode()).digest()[:8], "big")
                rng = np.random.default_rng(pair_seed)
                for query in queries:
                    target_vec = F.normalize(torch.from_numpy(target[query].mean(axis=0)), dim=0).numpy()
                    same = [key for key in candidates if labels[key] == labels[query]]
                    for run in range(args.runs):
                        permutations = {key: rng.permutation(len(source[key])) for key in candidates}
                        for bank_size in bank_sizes:
                            scores = {}
                            for candidate in candidates:
                                order = permutations[candidate]
                                n = len(order) if bank_size == "all" else min(int(bank_size), len(order))
                                bank = source[candidate][order[:n]].mean(axis=0)
                                bank = F.normalize(torch.from_numpy(bank), dim=0).numpy()
                                scores[candidate] = float(np.dot(target_vec, bank))
                            ranked = sorted(candidates, key=lambda key: (-scores[key], key))
                            row = {
                                "pair": pair, "subset": str(seg.get("scene_type", "")),
                                "query_object": query, "run": run, "bank_views": bank_size,
                                "gallery_views": int(source_counts[query]),
                                "query_views": int(target_counts[query]),
                                "candidate_count": len(candidates),
                                "same_label_candidates": len(same),
                                "correct": int(ranked[0] == query),
                                "rank": ranked.index(query) + 1,
                                "same_label_correct": int(max(same, key=lambda k: scores[k]) == query)
                                    if len(same) >= 2 else "",
                            }
                            pair_rows.append(row)
            finally:
                for video in videos.values():
                    video.unlink(missing_ok=True)
            rows.extend(pair_rows)
            downloaded_members.extend(pair_members)
            record = {"pair": pair, "seed": args.seed, "rows": pair_rows,
                      "members": pair_members,
                      "range_bytes_fetched": remote.bytes_downloaded - bytes_before_pair}
            temp_checkpoint = checkpoint.with_suffix(".json.tmp")
            temp_checkpoint.write_text(json.dumps(record, separators=(",", ":")))
            temp_checkpoint.replace(checkpoint)
            print(f"[{pair_idx}/{len(pair_ids)}] {pair}: gallery={len(candidates)}, "
                  f"queries={len({row['query_object'] for row in pair_rows})}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "query_run_rows.csv", index=False)
    pair_run = frame.groupby(["bank_views", "run", "pair"], sort=False).correct.mean().reset_index()
    pair_run.to_csv(args.output_dir / "pair_run_scores.csv", index=False)
    # The macro estimand gives each video pair equal weight. Nested bootstrap
    # resamples pair clusters and observed source-view subset runs.
    pair_names = sorted(pair_run.pair.unique())
    if set(pair_names) != set(pair_ids):
        raise ValueError(f"scored pair set differs from frozen input: {len(pair_names)} vs {len(pair_ids)}")
    bank_order = ["1", "2", "4", "8", "all"]
    matrix = {bank: np.full((args.runs, len(pair_names)), np.nan) for bank in bank_order}
    for row in pair_run.itertuples(index=False):
        matrix[str(row.bank_views)][int(row.run), pair_names.index(row.pair)] = float(row.correct)
    if any(not np.isfinite(mat).all() for mat in matrix.values()):
        raise ValueError("missing pair-run score cells")
    summary: dict[str, object] = {
        "status": "complete", "dataset": "SceneDiff official test split",
        "revision": "d9e2115fae13daf219ed547bfb072a875aae4d7c",
        "primary_estimand": "pair-macro top-1 changed-object retrieval with fixed target multi-view mean and fixed candidate set",
        "primary_contrast": "four source views vs one source view; all query objects and pairs retained, available views capped by each object's support",
        "pairs_scored": len(pair_names),
        "queries_scored": int(frame[["pair", "query_object"]].drop_duplicates().shape[0]),
        "view_subset_runs": args.runs, "bank_sizes": list(bank_sizes), "seed": args.seed,
        "pair_level_estimates": {},
        "limitations": [
            "SceneDiff is RGB video without sensor depth or camera poses, and masks are privileged.",
            "The pair split has no released physical-scene cluster IDs; pair bootstrap does not model scene dependence.",
            "Queries are restricted to linked annotated objects, and the gallery contains source-video objects with available descriptors; unannotated scene objects are not distractors.",
            "This is a source-bank sensitivity on an independent RGB benchmark, not a new method or native RGB-D room-rescan validation.",
            "View subsets are Monte Carlo samples from frames already observed in each video, not independent captures.",
        ],
        "range_reader_bytes_fetched_this_invocation": remote.bytes_downloaded,
        "range_reader_bytes_from_completed_checkpoints": resumed_range_bytes,
        "range_reader_bytes_fetched_total": remote.bytes_downloaded + resumed_range_bytes,
        "downloaded_video_members": downloaded_members,
        "selected_members_compressed_bytes": int(sum(x["compressed_bytes"] for x in downloaded_members)),
    }
    # Resample pair clusters and, inside each selected pair, one of the observed
    # source-view subset runs. Batched arrays keep memory bounded.
    boot_rng = np.random.default_rng(args.seed + 1)
    draws = 50_000
    contrasts = {}
    batch = 1000
    boot = {bank: np.empty(draws, dtype=np.float32) for bank in bank_order}
    for start in range(0, draws, batch):
        count = min(batch, draws - start)
        groups = boot_rng.integers(0, len(pair_names), size=(count, len(pair_names)))
        runs = boot_rng.integers(0, args.runs, size=(count, len(pair_names)))
        for bank in bank_order:
            boot[bank][start:start + count] = matrix[bank][runs, groups].mean(axis=1)
    for bank in bank_order:
        point = float(matrix[bank].mean())
        ci = [float(x) for x in np.quantile(boot[bank], [.025, .975])]
        summary["pair_level_estimates"][bank] = {"estimate": point, "nested_pair_view_interval_95": ci}
    for bank in ("2", "4", "8", "all"):
        delta = boot[bank] - boot["1"]
        point_delta = float(matrix[bank].mean() - matrix["1"].mean())
        contrasts[bank] = {"difference_vs_one": point_delta,
                           "nested_pair_view_interval_95": [float(x) for x in np.quantile(delta, [.025, .975])],
                           "pair_count_positive": int(np.sum(matrix[bank].mean(axis=0) > matrix["1"].mean(axis=0)))}
    summary["contrasts_vs_one"] = contrasts
    summary["bootstrap"] = {"unit": "video pair, with within-pair resampling of 20 view-subset runs",
                            "draws": draws, "seed": args.seed + 1,
                            "primary_contrast": "four vs one source views; percentile 95% interval"}
    summary["same_label_sensitivity"] = {
        "queries": int(frame.loc[frame.same_label_candidates >= 2,
                                  ["pair", "query_object"]].drop_duplicates().shape[0]),
        "pairs": int(frame.loc[frame.same_label_candidates >= 2, "pair"].nunique()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "pairs": len(pair_names),
                      "queries": summary["queries_scored"], "contrasts_vs_one": contrasts,
                      "range_bytes": remote.bytes_downloaded}, indent=2))


if __name__ == "__main__":
    main()
