#!/usr/bin/env python3
"""Cross-view instance retrieval on the UW RGB-D Object Dataset eval set.

The official evaluation archive supplies object crops, binary masks, and three
view sequences for each of 300 physical instances. We use one sequence as a
label-blind reference bank and query the other two, rotating the reference
sequence over all three source choices. This is an object-view stress test,
not room-rescan or cross-site identity validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2


NAME_RE = re.compile(r"^(?P<oid>.+)_(?P<video>\d+)_(?P<frame>\d+)_(?P<kind>crop|maskcrop)\.png$")
VIDEO_IDS = ("1", "2", "4")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_archive(zf: zipfile.ZipFile) -> dict[tuple[str, str], dict[int, dict[str, str]]]:
    files: dict[tuple[str, str], dict[int, dict[str, str]]] = defaultdict(dict)
    for name in zf.namelist():
        m = NAME_RE.match(Path(name).name)
        if m is None:
            continue
        key = (m.group("oid"), m.group("video"))
        frame = int(m.group("frame"))
        files[key].setdefault(frame, {})[m.group("kind")] = name
    return files


def choose_frames(files: dict, max_views: int) -> list[tuple[str, str, int, str, str]]:
    selected = []
    for (oid, video), frames in sorted(files.items()):
        eligible = [(fid, row) for fid, row in sorted(frames.items())
                    if "crop" in row and "maskcrop" in row]
        if len(eligible) < max_views:
            raise ValueError(f"{oid}/{video} has only {len(eligible)} eligible views")
        picks = np.linspace(0, len(eligible) - 1, max_views, dtype=int)
        if len(set(picks.tolist())) != max_views:
            raise ValueError(f"frame sampling duplicated positions for {oid}/{video}")
        for i in picks:
            fid, row = eligible[int(i)]
            selected.append((oid, video, fid, row["crop"], row["maskcrop"]))
    return selected


def load_masked_crop(zf: zipfile.ZipFile, rgb_name: str, mask_name: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(BytesIO(zf.read(rgb_name))).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(BytesIO(zf.read(mask_name)))) > 0
    if rgb.shape[:2] != mask.shape or not mask.any():
        raise ValueError(f"invalid RGB/mask pair: {rgb_name}")
    # Remove background cues before the backbone. Preserve aspect ratio and
    # center the crop so the ViT sees a consistent 224 x 224 canvas.
    h, w = mask.shape
    scale = 224.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    rgb_im = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BICUBIC)
    mask_im = Image.fromarray(mask.astype(np.uint8) * 255).resize((nw, nh), Image.Resampling.NEAREST)
    canvas = np.round(MEAN * 255).astype(np.uint8)
    square = np.broadcast_to(canvas, (224, 224, 3)).copy()
    square_mask = np.zeros((224, 224), dtype=np.uint8)
    x0, y0 = (224 - nw) // 2, (224 - nh) // 2
    resized_rgb = np.asarray(rgb_im)
    resized_mask = np.asarray(mask_im) > 0
    # Pixel-level masking prevents turntable/background color from carrying
    # object identity; the explicit mask remains available for token pooling.
    square[y0:y0 + nh, x0:x0 + nw] = np.where(
        resized_mask[..., None], resized_rgb, square[y0:y0 + nh, x0:x0 + nw]
    )
    square_mask[y0:y0 + nh, x0:x0 + nw] = resized_mask.astype(np.uint8)
    tensor = torch.from_numpy(square).permute(2, 0, 1).float().div_(255)
    tensor = (tensor - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]
    return tensor, square_mask


def color_hist(rgb_bytes: bytes, mask_bytes: bytes) -> np.ndarray:
    rgb = np.asarray(Image.open(BytesIO(rgb_bytes)).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(BytesIO(mask_bytes))) > 0
    values = np.concatenate([np.histogram(rgb[:, :, c][mask], bins=16, range=(0, 256))[0]
                             for c in range(3)]).astype(np.float32)
    values /= max(values.sum(), 1.0)
    values /= max(np.linalg.norm(values), 1e-12)
    return values


@torch.inference_mode()
def extract_descriptors(zf: zipfile.ZipFile, rows: list[tuple], batch_size: int,
                        device: torch.device) -> tuple[list[dict], np.ndarray, np.ndarray]:
    model = load_dinov2("giant", 224, device)
    descriptors, colors = [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        images, masks = [], []
        for oid, video, frame, rgb_name, mask_name in batch_rows:
            image, mask = load_masked_crop(zf, rgb_name, mask_name)
            images.append(image)
            masks.append(mask)
            descriptors.append({"object_id": oid, "video": video, "frame": frame})
            colors.append(color_hist(zf.read(rgb_name), zf.read(mask_name)))
        image_batch = torch.stack(images).to(device)
        captured = extract_block_tokens(model, image_batch, (2, 20, 40))
        layer_tokens = [
            F.normalize(captured[layer][:, model.num_prefix_tokens:], dim=2)
            for layer in (2, 20, 40)
        ]
        tokens = torch.cat(layer_tokens, dim=2)
        mask_batch = torch.from_numpy(np.stack(masks).astype(np.float32))[:, None]
        patch_weights = F.avg_pool2d(mask_batch, kernel_size=14, stride=14).flatten(1)
        pooled = (tokens * patch_weights.to(tokens.device)[:, :, None]).sum(1)
        pooled /= patch_weights.sum(1).to(pooled.device).clamp_min(1e-6)[:, None]
        pooled = F.normalize(pooled, dim=1).cpu().numpy().astype(np.float32)
        if start == 0:
            dino = np.empty((len(rows), pooled.shape[1]), dtype=np.float32)
        dino[start:start + len(batch_rows)] = pooled
        if (start + len(batch_rows)) % 1200 == 0 or start + len(batch_rows) == len(rows):
            print(f"embedded {start + len(batch_rows)}/{len(rows)} crops", flush=True)
    del model
    if len(colors) != len(rows):
        raise RuntimeError("descriptor count mismatch")
    return descriptors, dino, np.stack(colors)


def score_fold(descriptors: list[dict], features: np.ndarray, colors: np.ndarray,
               source_video: str, device: torch.device) -> list[dict]:
    by_oid_video: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(descriptors):
        by_oid_video[(row["object_id"], row["video"])].append(i)
    objects = sorted({row["object_id"] for row in descriptors})
    ref_idx = [i for oid in objects for i in by_oid_video[(oid, source_video)]]
    query_idx = [i for i, row in enumerate(descriptors) if row["video"] != source_video]
    nref = len(by_oid_video[(objects[0], source_video)])
    if any(len(by_oid_video[(oid, source_video)]) != nref for oid in objects):
        raise ValueError("source reference-bank sizes are not balanced")
    object_to_idx = {oid: i for i, oid in enumerate(objects)}
    ref_dino = torch.as_tensor(features[ref_idx], device=device)
    ref_color = torch.as_tensor(colors[ref_idx], device=device)
    out = []
    for start in range(0, len(query_idx), 512):
        qidx = query_idx[start:start + 512]
        qd = torch.as_tensor(features[qidx], device=device)
        qc = torch.as_tensor(colors[qidx], device=device)
        dino_view_scores = (qd @ ref_dino.T).reshape(len(qidx), len(objects), nref)
        sd = dino_view_scores.max(2).values
        sd_top3 = dino_view_scores.topk(3, dim=2).values.mean(2)
        sc = (qc @ ref_color.T).reshape(len(qidx), len(objects), nref).max(2).values
        dino_top2, pred_d_tensor = sd.topk(2, dim=1)
        pred_d = pred_d_tensor[:, 0].cpu().numpy()
        dino_top3_top2, pred_d_top3_tensor = sd_top3.topk(2, dim=1)
        pred_d_top3 = pred_d_top3_tensor[:, 0].cpu().numpy()
        pred_c = sc.argmax(1).cpu().numpy()
        dino_top1_similarity = sd.max(1).values.cpu().numpy()
        dino_top2_similarity = dino_top2[:, 1].cpu().numpy()
        dino_top3mean_similarity = dino_top3_top2[:, 0].cpu().numpy()
        dino_top3mean_margin = (dino_top3_top2[:, 0] - dino_top3_top2[:, 1]).cpu().numpy()
        for j, i in enumerate(qidx):
            oid = descriptors[i]["object_id"]
            category = oid.rsplit("_", 1)[0]
            pred_d_oid, pred_c_oid = objects[int(pred_d[j])], objects[int(pred_c[j])]
            pred_d_top3_oid = objects[int(pred_d_top3[j])]
            out.append({
                "source_video": source_video,
                "query_video": descriptors[i]["video"],
                "frame": descriptors[i]["frame"],
                "object_id": oid,
                "category": category,
                "dino_predicted_object": pred_d_oid,
                "dino_top1_similarity": float(dino_top1_similarity[j]),
                "dino_top2_similarity": float(dino_top2_similarity[j]),
                "dino_top1_margin": float(dino_top1_similarity[j] - dino_top2_similarity[j]),
                "dino_top3mean_predicted_object": pred_d_top3_oid,
                "dino_top3mean_similarity": float(dino_top3mean_similarity[j]),
                "dino_top3mean_margin": float(dino_top3mean_margin[j]),
                "dino_top3mean_instance_correct": pred_d_top3_oid == oid,
                "color_predicted_object": pred_c_oid,
                "dino_instance_correct": pred_d_oid == oid,
                "color_instance_correct": pred_c_oid == oid,
                "dino_category_correct": pred_d_oid.rsplit("_", 1)[0] == category,
                "color_category_correct": pred_c_oid.rsplit("_", 1)[0] == category,
            })
    return out


def summarize(rows: list[dict], draws: int, seed: int) -> dict:
    object_ids = sorted({r["object_id"] for r in rows})
    metrics = {}
    per_object = {}
    for oid in object_ids:
        subset = [r for r in rows if r["object_id"] == oid]
        per_object[oid] = {
            "dino_instance": float(np.mean([r["dino_instance_correct"] for r in subset])),
            "color_instance": float(np.mean([r["color_instance_correct"] for r in subset])),
            "dino_category": float(np.mean([r["dino_category_correct"] for r in subset])),
            "color_category": float(np.mean([r["color_category_correct"] for r in subset])),
        }
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(object_ids), size=(draws, len(object_ids)))
    keys = list(per_object)
    for name, dkey, ckey in (
        ("instance", "dino_instance", "color_instance"),
        ("category", "dino_category", "color_category"),
    ):
        d = np.asarray([per_object[k][dkey] for k in keys])
        c = np.asarray([per_object[k][ckey] for k in keys])
        dd, cc = d[ix].mean(1), c[ix].mean(1)
        diff = dd - cc
        metrics[name] = {
            "dino_macro": float(d.mean()), "color_histogram_macro": float(c.mean()),
            "dino_minus_color_pp": float((d.mean() - c.mean()) * 100),
            "paired_object_cluster_bootstrap_95_pp": [float(x * 100) for x in np.quantile(diff, [.025, .975])],
        }
    fold_metrics = {}
    for source in VIDEO_IDS:
        subset = [r for r in rows if r["source_video"] == source]
        fold_metrics[source] = {
            "queries": len(subset),
            "dino_instance_accuracy": float(np.mean([r["dino_instance_correct"] for r in subset])),
            "color_instance_accuracy": float(np.mean([r["color_instance_correct"] for r in subset])),
            "dino_category_accuracy": float(np.mean([r["dino_category_correct"] for r in subset])),
            "color_category_accuracy": float(np.mean([r["color_category_correct"] for r in subset])),
        }
    return {
        "objects": len(object_ids), "categories": len({oid.rsplit("_", 1)[0] for oid in object_ids}),
        "query_rows": len(rows), "macro_metrics": metrics, "by_source_video": fold_metrics,
        "per_object": per_object,
        "bootstrap": {"draws": draws, "seed": seed, "unit": "physical object ID"},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_rgbd_object_eval_multiview"))
    p.add_argument("--views-per-sequence", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--bootstrap-draws", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261009)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as zf:
        files = parse_archive(zf)
        ids = sorted({oid for oid, _ in files})
        videos = {v for _, v in files}
        if len(ids) != 300 or videos != set(VIDEO_IDS):
            raise ValueError(f"expected 300 objects and videos {VIDEO_IDS}; got {len(ids)} and {sorted(videos)}")
        for oid in ids:
            if any((oid, v) not in files for v in VIDEO_IDS):
                raise ValueError(f"missing one of three videos for {oid}")
        selected = choose_frames(files, args.views_per_sequence)
        descriptors, dino, colors = extract_descriptors(zf, selected, args.batch_size, device)
        outcomes = []
        for source in VIDEO_IDS:
            outcomes.extend(score_fold(descriptors, dino, colors, source, device))
    summary = summarize(outcomes, args.bootstrap_draws, args.seed)
    metadata = {
        "status": "complete", "dataset": "UW RGB-D Object Dataset evaluation set",
        "dataset_use": "non-commercial research/educational use only; no data redistributed",
        "archive_sha256": sha256(args.archive), "archive_bytes": args.archive.stat().st_size,
        "model": "timm vit_giant_patch14_dinov2.lvd142m; normalized layers 2,20,40; mask-weighted mean",
        "descriptor_protocol": "12 evenly spaced masked crops per physical object per source/query video; each source sequence is used once; cosine max over equal-size per-object source views",
        "video_ids": list(VIDEO_IDS), "views_per_sequence": args.views_per_sequence,
        "device": str(device), "summary": summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output_dir / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in outcomes))
    metrics = summary["macro_metrics"]
    lines = [
        "# UW RGB-D Object Dataset: cross-view instance retrieval",
        "",
        "This is a frozen-feature, label-blind nearest-neighbor stress test on the official evaluation crops. It tests object identity across three turntable videos and is not a room-rescan or natural-site experiment.",
        "",
        f"- Archive SHA-256: `{metadata['archive_sha256']}` ({metadata['archive_bytes']:,} bytes).",
        f"- Cohort: {summary['objects']} physical objects in {summary['categories']} categories; {args.views_per_sequence} evenly spaced views per video; {summary['query_rows']:,} cross-view query predictions over three source-video folds.",
        "- Source and query videos are disjoint. Each source bank has the same number of views per object. Evaluation identity labels are used only after predictions are frozen.",
        "- DINOv2 ViT-G/14 uses the normalized layer-2/20/40 patch tokens pooled by the published binary masks. Baseline: masked 16-bin-per-channel RGB histograms. Background pixels are replaced with ImageNet mean before DINO inference.",
        "",
        "## Object-macro results",
        "",
        "| Outcome | DINOv2 ViT-G | RGB histogram | Difference | Object-cluster 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for outcome, label in (("instance", "Physical-instance top-1"), ("category", "Category top-1")):
        m = metrics[outcome]
        lo, hi = m["paired_object_cluster_bootstrap_95_pp"]
        lines.append(f"| {label} | {m['dino_macro']:.1%} | {m['color_histogram_macro']:.1%} | {m['dino_minus_color_pp']:+.1f} pp | [{lo:+.1f}, {hi:+.1f}] pp |")
    lines.extend(["", "## Source-video folds", "", "| Source video | Queries | DINO instance top-1 | RGB histogram top-1 | DINO category top-1 |", "|---|---:|---:|---:|---:|"])
    for source, m in summary["by_source_video"].items():
        lines.append(f"| {source} | {m['queries']:,} | {m['dino_instance_accuracy']:.1%} | {m['color_instance_accuracy']:.1%} | {m['dino_category_accuracy']:.1%} |")
    lines.extend([
        "", "## Scope and caveats", "",
        "The benchmark is small, controlled, and turntable-based. It contains tightly cropped objects rather than room scans, and the pretrained DINO model may have encountered these public images during pretraining. The object-cluster interval measures heterogeneity across these 300 objects, not uncertainty across capture sites. This result can support an independent object-view robustness claim only; it cannot validate RGB-D room-rescan mapping, physical identity under scene change, or a novel method contribution.",
        "", "Reproduce with `python scripts/evaluate_rgbd_object_eval_multiview.py --archive <path-to-rgbd-dataset_eval.zip>`. The dataset is used under the official non-commercial research/educational restriction; the raw archive is not included.", "",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(json.dumps({"objects": summary["objects"], "categories": summary["categories"],
                      "query_rows": summary["query_rows"], "macro_metrics": metrics,
                      "by_source_video": summary["by_source_video"]}, indent=2))


if __name__ == "__main__":
    main()
