#!/usr/bin/env python3
"""Benchmark DINO feature extraction for 12-view and 20-view source banks.

The benchmark reads only frozen RGB/mask inputs, never identity target files.
Disk reads and archive decoding happen before timing. Timed work includes GPU
DINOv2 ViT-G forward passes, mask-occupancy pooling, and feature normalization.
Query frames are held fixed while the source bank uses either 12 temporal-
uniform reference frames or all 20 reference frames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "outputs/reports/20261008_segmast3r_validation/uniform20_inputs"
SELECTIONS = ROOT / "outputs/reports/20261009_dino_source_bank_no_pair_condition/per_relation.csv"
OUT = ROOT / "outputs/reports/20261009_dino_source_bank_encoder_runtime"
MATCHING_SUMMARY = ROOT / "outputs/reports/20261009_dino_source_bank_runtime/summary.json"
SEED = 20261009


def records_for_run(run_id: str) -> dict:
    path = INPUT_ROOT / run_id / "manifest.json"
    data = json.loads(path.read_text())
    return data["records"]


def load_frozen_input(record: dict) -> tuple[np.ndarray, np.ndarray]:
    path = Path(record["input"])
    if not path.is_absolute():
        path = ROOT / path
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["input_sha256"]:
        raise ValueError(f"frozen input checksum mismatch: {path}")
    with np.load(path) as archive:
        rgb = archive["rgb"].copy()
        masks = archive["masks"].astype(bool, copy=True)
    if rgb.shape != (224, 224, 3) or masks.ndim != 3 or masks.shape[1:] != (224, 224):
        raise ValueError(f"unexpected RGB/mask shapes in {path}: {rgb.shape}, {masks.shape}")
    return rgb, masks


def select_runs(max_relations: int) -> tuple[list[dict], dict]:
    selected = []
    with SELECTIONS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["strategy"] == "temporal_uniform" and row["repeat"] == "0"
                    and row["bank_size"] in {"12", "20"}):
                selected.append(row)
    grouped: dict[str, dict[int, dict]] = {}
    for row in selected:
        grouped.setdefault(row["run_id"], {})[int(row["bank_size"])] = row
    if len(grouped) != 110 or any(set(rows) != {12, 20} for rows in grouped.values()):
        raise RuntimeError(f"expected 110 complete 12/20-view selections, got {len(grouped)}")
    run_ids = sorted(grouped)
    if max_relations:
        run_ids = run_ids[:max_relations]

    query_frames: list[tuple[np.ndarray, np.ndarray]] = []
    source_frames: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {12: [], 20: []}
    selected_frame_digest = hashlib.sha256()
    source_mask_count = {12: 0, 20: 0}
    query_mask_count = 0
    for run_id in run_ids:
        records = records_for_run(run_id)
        reference = {int(x["frame_id"]): x for x in records["reference"]}
        for bank_size in (12, 20):
            ids = [int(x) for x in grouped[run_id][bank_size]["selected_reference_frames"].split(",") if x]
            if len(ids) != bank_size or len(set(ids)) != bank_size:
                raise RuntimeError(f"bad {bank_size}-frame selection for {run_id}: {ids}")
            selected_frame_digest.update(f"{run_id}:{bank_size}:{','.join(map(str, ids))}\n".encode())
            for frame_id in ids:
                rgb, masks = load_frozen_input(reference[frame_id])
                source_frames[bank_size].append((rgb, masks))
                source_mask_count[bank_size] += len(masks)
        # Query encoding is identical for both bank sizes and performed once
        # per mode to preserve the actual query-side cost in the comparison.
        for record in records["rescan"]:
            rgb, masks = load_frozen_input(record)
            query_frames.append((rgb, masks))
            query_mask_count += len(masks)
    return (query_frames, source_frames), {
        "relation_count": len(run_ids),
        "query_frame_count": len(query_frames),
        "source_frame_count": {str(k): len(source_frames[k]) for k in (12, 20)},
        "query_mask_count": query_mask_count,
        "source_mask_count": {str(k): source_mask_count[k] for k in (12, 20)},
        "selection_sha256": selected_frame_digest.hexdigest(),
    }


@torch.inference_mode()
def encode(frames: list[tuple[np.ndarray, np.ndarray]], *, model, device, batch_size: int) -> None:
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]
        batch_np = np.stack([x[0] for x in batch_frames])
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
        batch = batch.permute(0, 3, 1, 2).div_(255.0).sub_(mean).div_(std)
        tokens = model.forward_features(batch)
        if isinstance(tokens, dict):
            tokens = tokens.get("x_norm_patchtokens", tokens.get("last_hidden_state"))
            if tokens is None:
                raise RuntimeError("unrecognized timm DINOv2 feature dictionary")
        patch_tokens = tokens[:, -256:, :].reshape(len(batch_frames), 16, 16, -1)
        for i, (_, masks) in enumerate(batch_frames):
            if not len(masks):
                continue
            mask_tensor = torch.from_numpy(masks.astype(np.float32, copy=False)).to(device=device).unsqueeze(1)
            weights = F.adaptive_avg_pool2d(mask_tensor, (16, 16)).squeeze(1).flatten(1)
            pooled = weights @ patch_tokens[i].reshape(256, -1)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            F.normalize(pooled, p=2, dim=1)


def timed_mode(query_frames, source_frames, bank_size, *, model, device, batch_size):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    encode(query_frames, model=model, device=device, batch_size=batch_size)
    encode(source_frames[bank_size], model=model, device=device, batch_size=batch_size)
    torch.cuda.synchronize(device)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-relations", type=int, default=0, help="pilot only; zero uses all 110 relations")
    args = parser.parse_args()
    if args.repeats < 1 or args.batch_size < 1:
        raise ValueError("repeats and batch size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("this benchmark is intended to measure GPU feature extraction")
    frames, inputs = select_runs(args.max_relations)
    query_frames, source_frames = frames
    model = timm.create_model(
        "vit_giant_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224
    ).to(device).eval()

    # One full warm-up of both paths, followed by paired alternating epochs.
    timed_mode(query_frames, source_frames, 12, model=model, device=device, batch_size=args.batch_size)
    timed_mode(query_frames, source_frames, 20, model=model, device=device, batch_size=args.batch_size)
    times = {12: [], 20: []}
    order_log = []
    for epoch in range(args.repeats):
        order = [12, 20]
        if epoch % 2:
            order.reverse()
        for bank_size in order:
            times[bank_size].append(timed_mode(
                query_frames, source_frames, bank_size,
                model=model, device=device, batch_size=args.batch_size,
            ))
        order_log.append(order)

    metrics = {}
    for bank_size in (12, 20):
        metrics[str(bank_size)] = {
            "epochs": len(times[bank_size]),
            "median_seconds": statistics.median(times[bank_size]),
            "mean_seconds": statistics.mean(times[bank_size]),
            "p10_seconds": float(np.quantile(times[bank_size], .10)),
            "p90_seconds": float(np.quantile(times[bank_size], .90)),
            "epoch_seconds": times[bank_size],
        }
    ratio = metrics["12"]["median_seconds"] / metrics["20"]["median_seconds"]
    matching = json.loads(MATCHING_SUMMARY.read_text())
    if matching.get("status") != "complete" or matching.get("relation_count") != inputs["relation_count"]:
        raise RuntimeError("cached-feature matching microbenchmark is missing or mismatched")
    matching_seconds = {
        str(x["bank_size"]): float(x["median_seconds"])
        for x in matching["results"]
    }
    summary = {
        "status": "complete",
        "protocol": {
            "input": "frozen 224x224 RGB frames and annotation-derived binary masks; identity target files are never opened",
            "timed_operation": "GPU DINOv2 ViT-G/14 forward pass, mask-occupancy pooling, and L2 normalization; disk I/O and NPZ decoding excluded",
            "comparison": "same query frames plus 12 temporal-uniform fixed source frames vs same query frames plus all 20 source frames per relation",
            "warmup": "one full pass through each variant",
            "paired_order": order_log,
            "batch_size": args.batch_size,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "software": {
                "torch": torch.__version__,
                "timm": timm.__version__,
                "cuda": torch.version.cuda,
            },
            "seed": SEED,
            "limits": [
                "Feature-extraction benchmark only; excludes disk I/O, cosine matching, robotics, and downstream task latency.",
                "Uses privileged annotation-derived masks and the already explored 3RScan validation cohort.",
                "Repeated relation inputs are counted per relation, matching the current relation-wise feature pipeline; a shared cross-relation cache could reduce both variants' costs.",
            ],
        },
        "inputs": inputs,
        "results": metrics,
        "ratios": {
            "median_feature_time_12_over_20": ratio,
            "median_feature_time_reduction": 1.0 - ratio,
            "source_frame_count_12_over_20": len(source_frames[12]) / len(source_frames[20]),
            "source_mask_count_12_over_20": inputs["source_mask_count"]["12"] / inputs["source_mask_count"]["20"],
        },
        "median_encoder_plus_cached_matching_seconds": {
            "12": metrics["12"]["median_seconds"] + matching_seconds["12"],
            "20": metrics["20"]["median_seconds"] + matching_seconds["20"],
        },
        "matching_microbenchmark_summary": str(MATCHING_SUMMARY.relative_to(ROOT)),
    }
    summary["ratios"]["median_encoder_plus_cached_matching_12_over_20"] = (
        summary["median_encoder_plus_cached_matching_seconds"]["12"]
        / summary["median_encoder_plus_cached_matching_seconds"]["20"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = [
        "# DINO source-bank feature extraction benchmark", "",
        "This benchmark runs on frozen 224x224 RGB/mask inputs for the 110 explored 3RScan validation relations. Each timed mode encodes the same rescan/query frames and either the 12-view temporal-uniform reference bank or all 20 reference views. Timed work is GPU DINOv2 ViT-G/14 inference, mask-occupancy pooling, and descriptor normalization. Disk reads and NPZ decoding happen before timing; identity-target files are never opened.", "",
        "| Bank | Median feature time / relations (s) | p10–p90 (s) | Source frames | Source masks |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in (12, 20):
        r = metrics[str(k)]
        md.append(f"| {k} | {r['median_seconds']:.3f} | {r['p10_seconds']:.3f}–{r['p90_seconds']:.3f} | {inputs['source_frame_count'][str(k)]} | {inputs['source_mask_count'][str(k)]} |")
    md += [
        "",
        f"Median feature-extraction time is {ratio:.1%} of the 20-view time. Adding the separate cached-feature matching microbenchmark yields {summary['median_encoder_plus_cached_matching_seconds']['12']:.3f}s vs {summary['median_encoder_plus_cached_matching_seconds']['20']:.3f}s ({1-summary['ratios']['median_encoder_plus_cached_matching_12_over_20']:.1%} lower) per 110 relations. This combined value joins separately measured feature-extraction and CPU matching stages; it excludes disk I/O and downstream robotics.",
        "",
        "The input contains annotation-derived masks and comes from an already explored validation cohort. Per-relation duplicate frames are counted as the relation-wise pipeline processes them; sharing a cache across relations could reduce both totals. Raw timings, selection digest, and device metadata are in `summary.json`. Reproduce with `HF_HUB_OFFLINE=1 conda run -n worldcache python scripts/benchmark_dino_source_bank_encoder.py --device cuda --batch-size 8 --repeats 3`.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md))
    print(json.dumps({"status": "complete", "results": {k: metrics[k] for k in ("12", "20")}, "ratios": summary["ratios"]}, indent=2))


if __name__ == "__main__":
    main()
