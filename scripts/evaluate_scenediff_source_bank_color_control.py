#!/usr/bin/env python3
"""Evaluate a paired RGB-histogram view-bank control on SceneDiff.

This control uses the same pair/query population, frames, source-view sampling
seed, bank sizes, and candidate-gallery rule as
evaluate_scenediff_source_bank_views.py. It compares the DINO view-count gain
with generic masked-color histogram averaging; it is not a new method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from evaluate_scenediff_identity_transfer import (
    ARCHIVE_BYTES, ARCHIVE_URL, decode_rle, extract_member,
)
from stream_rescan_archive_subset import HTTPRangeReader


def hash_order(items: list[str], salt: str) -> list[str]:
    return sorted(items, key=lambda item: hashlib.sha256(f"{salt}:{item}".encode()).hexdigest())


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def sample_histograms(video: Path, objects: dict[str, object], stride: int
                      ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Return per-view masked 32-bin/channel RGB histograms and support counts."""
    if not objects:
        return {}, {}
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
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    assert process.stdout is not None
    selected_index = 0
    try:
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(f"short decoded frame from {video}: {len(raw)} bytes")
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
                if int(mask.sum()) < 100:
                    continue
                pixels = frame[mask]
                hist = np.concatenate([
                    np.histogram(pixels[:, channel], bins=32, range=(0, 256))[0].astype(np.float32)
                    / max(len(pixels), 1)
                    for channel in range(3)
                ])
                rows[str(key)].append(normalize(hist).astype(np.float32))
                counts[str(key)] += 1
        code = process.wait()
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if code:
            raise RuntimeError(f"ffmpeg failed for {video}: {error[-1000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return {key: np.stack(values) for key, values in rows.items()}, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-file", type=Path, default=Path(
        "outputs/reports/20261009_scenediff_identity_transfer/pair_rows.csv"))
    parser.add_argument("--dino-rows", type=Path, default=Path(
        "outputs/reports/20261009_scenediff_source_bank_views/query_run_rows.csv"))
    parser.add_argument("--frame-stride", type=int, default=15)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/scenediff_color_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = pd.read_csv(args.pairs_file)
    pair_ids = sorted(str(x) for x in pair_rows.loc[pair_rows.linked_queries_scored > 0, "pair"].unique())
    pair_ids = hash_order(pair_ids, "worldcache-scenediff-source-bank-v1")
    dino_rows = pd.read_csv(args.dino_rows)
    if set(dino_rows.pair.unique()) != set(pair_ids):
        raise ValueError("DINO source-bank rows do not cover the frozen pair population")
    if set(dino_rows.bank_views.astype(str).unique()) != {"1", "2", "4", "8", "all"}:
        raise ValueError("unexpected bank sizes in the frozen DINO result")

    remote = HTTPRangeReader(ARCHIVE_URL)
    if remote.size != ARCHIVE_BYTES:
        raise ValueError(f"pinned archive size changed: {remote.size} != {ARCHIVE_BYTES}")
    checkpoints = args.output_dir / "pair_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    members: list[dict[str, object]] = []
    completed_range_bytes = 0
    bank_sizes: tuple[int | str, ...] = (1, 2, 4, 8, "all")
    with ZipFile(remote) as archive:
        names = [info.filename for info in archive.infolist()]
        for pair_idx, pair in enumerate(pair_ids, 1):
            checkpoint = checkpoints / (hashlib.sha256(pair.encode()).hexdigest() + ".json")
            if checkpoint.is_file():
                record = json.loads(checkpoint.read_text())
                if record.get("pair") != pair or record.get("seed") != args.seed:
                    raise ValueError(f"checkpoint identity or seed mismatch: {checkpoint}")
                rows.extend(record["rows"])
                members.extend(record["members"])
                completed_range_bytes += int(record.get("range_bytes_fetched", 0))
                print(f"[{pair_idx}/{len(pair_ids)}] resume-skip {pair}", flush=True)
                continue
            before = remote.bytes_downloaded
            prefix = f"scenediff_benchmark/data/{pair}/"
            seg_name = prefix + "segments.pkl"
            segment_info = archive.getinfo(seg_name)
            seg = pickle.loads(archive.read(seg_name))
            pair_members = [{
                "path": seg_name, "compressed_bytes": segment_info.compress_size,
                "uncompressed_bytes": segment_info.file_size,
                "crc32": f"{segment_info.CRC:08x}",
            }]
            video_paths: dict[int, Path] = {}
            try:
                descriptors: list[dict[str, np.ndarray]] = []
                counts: list[dict[str, int]] = []
                for video_index in (1, 2):
                    candidates = [name for name in names if name.startswith(prefix)
                                  and Path(name).name.lower().startswith(f"original_video{video_index}.")]
                    if len(candidates) != 1:
                        raise FileNotFoundError(f"expected one video{video_index} in {pair}, found {candidates}")
                    member = candidates[0]
                    info = archive.getinfo(member)
                    pair_members.append({
                        "path": member, "compressed_bytes": info.compress_size,
                        "uncompressed_bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                    })
                    video = args.cache_dir / f"{pair}_v{video_index}{Path(member).suffix}"
                    extract_member(archive, member, video)
                    video_paths[video_index] = video
                    desc, support = sample_histograms(video, seg[f"video{video_index}_objects"], args.frame_stride)
                    descriptors.append(desc)
                    counts.append(support)

                source, target = descriptors
                source_counts, target_counts = counts
                gallery = sorted(source)
                queries = sorted(set(source) & set(target))
                if len(gallery) < 2:
                    raise ValueError(f"source gallery has fewer than two candidates: {pair}")
                pair_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{pair}".encode()).digest()[:8], "big")
                rng = np.random.default_rng(pair_seed)
                pair_result = []
                for query in queries:
                    target_vector = normalize(target[query].mean(axis=0))
                    for run in range(args.runs):
                        permutations = {key: rng.permutation(len(source[key])) for key in gallery}
                        for bank_size in bank_sizes:
                            scores: dict[str, float] = {}
                            for candidate in gallery:
                                order = permutations[candidate]
                                n = len(order) if bank_size == "all" else min(int(bank_size), len(order))
                                bank = normalize(source[candidate][order[:n]].mean(axis=0))
                                scores[candidate] = float(np.dot(target_vector, bank))
                            ranked = sorted(gallery, key=lambda key: (-scores[key], key))
                            pair_result.append({
                                "pair": pair, "query_object": query, "run": run,
                                "bank_views": str(bank_size), "candidate_count": len(gallery),
                                "gallery_views": int(source_counts[query]),
                                "query_views": int(target_counts[query]),
                                "rgbhist_correct": int(ranked[0] == query),
                                "rgbhist_rank": ranked.index(query) + 1,
                            })
            finally:
                for video in video_paths.values():
                    video.unlink(missing_ok=True)
            rows.extend(pair_result)
            members.extend(pair_members)
            record = {"pair": pair, "seed": args.seed, "rows": pair_result,
                      "members": pair_members, "range_bytes_fetched": remote.bytes_downloaded - before}
            temporary = checkpoint.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, separators=(",", ":")))
            temporary.replace(checkpoint)
            print(f"[{pair_idx}/{len(pair_ids)}] {pair}: gallery={len(gallery)}, queries={len(queries)}", flush=True)

    color = pd.DataFrame(rows)
    color.to_csv(args.output_dir / "rgbhist_query_run_rows.csv", index=False)
    dino_keys = ["pair", "query_object", "run", "bank_views"]
    dino = dino_rows[dino_keys + ["correct"]].copy()
    dino["pair"] = dino.pair.astype(str)
    dino["query_object"] = dino.query_object.astype(str)
    color["pair"] = color.pair.astype(str)
    color["query_object"] = color.query_object.astype(str)
    dino["run"] = dino.run.astype(int)
    color["run"] = color.run.astype(int)
    dino["bank_views"] = dino.bank_views.astype(str)
    color["bank_views"] = color.bank_views.astype(str)
    paired = dino.merge(color, on=dino_keys, validate="one_to_one")
    if len(paired) != len(dino_rows) or paired.rgbhist_correct.isna().any():
        raise ValueError("DINO and RGB-histogram query/run rows failed one-to-one pairing")
    paired.to_csv(args.output_dir / "paired_query_run_rows.csv", index=False)
    pair_run = paired.groupby(["bank_views", "run", "pair"], sort=False)[
        ["correct", "rgbhist_correct"]].mean().reset_index()
    pair_run.to_csv(args.output_dir / "pair_run_scores.csv", index=False)
    pair_names = sorted(pair_run.pair.unique())
    if set(pair_names) != set(pair_ids):
        raise ValueError("paired scores differ from the frozen pair population")
    bank_order = ["1", "2", "4", "8", "all"]
    matrices = {metric: {bank: np.full((args.runs, len(pair_names)), np.nan)
                         for bank in bank_order}
                for metric in ("correct", "rgbhist_correct")}
    for row in pair_run.itertuples(index=False):
        j = pair_names.index(row.pair)
        matrices["correct"][str(row.bank_views)][int(row.run), j] = float(row.correct)
        matrices["rgbhist_correct"][str(row.bank_views)][int(row.run), j] = float(row.rgbhist_correct)
    if any(not np.isfinite(matrix).all() for values in matrices.values() for matrix in values.values()):
        raise ValueError("missing paired pair/run scores")

    rng = np.random.default_rng(args.seed + 1)
    draws = 50_000
    bootstrap = {"dino": np.empty(draws), "rgbhist": np.empty(draws), "interaction": np.empty(draws)}
    for start in range(0, draws, 1000):
        n = min(1000, draws - start)
        sampled_pairs = rng.integers(0, len(pair_names), size=(n, len(pair_names)))
        sampled_runs = rng.integers(0, args.runs, size=(n, len(pair_names)))
        d1 = matrices["correct"]["1"][sampled_runs, sampled_pairs]
        d4 = matrices["correct"]["4"][sampled_runs, sampled_pairs]
        h1 = matrices["rgbhist_correct"]["1"][sampled_runs, sampled_pairs]
        h4 = matrices["rgbhist_correct"]["4"][sampled_runs, sampled_pairs]
        bootstrap["dino"][start:start+n] = (d4 - d1).mean(axis=1)
        bootstrap["rgbhist"][start:start+n] = (h4 - h1).mean(axis=1)
        bootstrap["interaction"][start:start+n] = ((d4 - d1) - (h4 - h1)).mean(axis=1)

    def contrast(metric: str, bank: str) -> dict[str, object]:
        matrix = matrices[metric]
        difference = matrix[bank].mean() - matrix["1"].mean()
        sampled = bootstrap["dino"] if metric == "correct" and bank == "4" else None
        if metric == "rgbhist_correct" and bank == "4":
            sampled = bootstrap["rgbhist"]
        if bank == "4" and metric == "correct":
            positive = int(np.sum(matrix["4"].mean(axis=0) > matrix["1"].mean(axis=0)))
        elif bank == "4":
            positive = int(np.sum(matrix["4"].mean(axis=0) > matrix["1"].mean(axis=0)))
        else:
            positive = None
        return {"difference_vs_one": float(difference),
                "nested_pair_view_interval_95": [float(x) for x in np.quantile(sampled, [.025, .975])]
                if sampled is not None else None,
                "pair_count_positive": positive}

    dino4 = contrast("correct", "4")
    hist4 = contrast("rgbhist_correct", "4")
    interaction_point = float(dino4["difference_vs_one"] - hist4["difference_vs_one"])
    summary = {
        "status": "complete", "dataset": "SceneDiff official test split",
        "primary_estimand": "paired difference between DINO and RGB-histogram four-minus-one-source-view pair-macro top-1 gains",
        "pairs_scored": len(pair_names),
        "queries_scored": int(paired[["pair", "query_object"]].drop_duplicates().shape[0]),
        "runs": args.runs, "bank_sizes": bank_order, "seed": args.seed,
        "pair_macro_top1": {
            "dino": {bank: float(matrices["correct"][bank].mean()) for bank in bank_order},
            "rgbhist": {bank: float(matrices["rgbhist_correct"][bank].mean()) for bank in bank_order},
        },
        "four_vs_one": {"dino": dino4, "rgbhist": hist4,
                        "dino_minus_rgbhist_gain": interaction_point,
                        "nested_pair_view_interval_95": [float(x) for x in np.quantile(bootstrap["interaction"], [.025, .975])],
                        "dino_pairs_positive": dino4["pair_count_positive"],
                        "rgbhist_pairs_positive": hist4["pair_count_positive"]},
        "bootstrap": {"unit": "paired video pair and within-pair view-subset run",
                      "draws": draws, "seed": args.seed + 1},
        "range_reader_bytes_fetched_this_invocation": remote.bytes_downloaded,
        "range_reader_bytes_from_completed_checkpoints": completed_range_bytes,
        "range_reader_bytes_fetched_total": remote.bytes_downloaded + completed_range_bytes,
        "downloaded_video_members": members,
        "limitations": [
            "RGB histograms are a low-level appearance control, not a strong learned baseline.",
            "SceneDiff uses privileged masks and has no sensor depth or camera poses.",
            "The pair bootstrap cannot model physical-scene dependence because scene IDs are unavailable.",
            "Only 16 queries over nine pairs have same-label distractors; the closed set is mostly cross-category.",
            "This post-hoc control does not establish a new method or native RGB-D generalization.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "pairs": len(pair_names), "queries": summary["queries_scored"],
                      "four_vs_one": summary["four_vs_one"],
                      "range_bytes": remote.bytes_downloaded}, indent=2))


if __name__ == "__main__":
    main()
