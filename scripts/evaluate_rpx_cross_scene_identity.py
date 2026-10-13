#!/usr/bin/env python3
"""Frozen DINOv2 object re-identification on the public RPX phase data.

Downloads only RGB and primary instance-mask tar files for a deterministic
scene subset. Cross-scene mode excludes every query scene from its gallery.
Official masks/identity maps are used only to form object crops and score the
closed-set retrieval; no labels enter DINO or the descriptor ranking.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from worldcache.backbone.dinov2 import load_dinov2_vitb14


REVISION = "51432e5d48895bbafb490ae7c5e34e0f6b7da7d6"
REPO = "IRVLUTD/RPX"
MAP_PATH = "manifest/mos_mask_object_map_v1.csv"
MODEL = "vit_base_patch14_dinov2.lvd142m"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DOWNLOAD_RECORDS: dict[str, dict[str, object]] = {}
IDENTITY_MAP_BYTES = 0


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def fetch_bytes(path: str) -> bytes:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def load_identity_map() -> tuple[list[dict[str, str]], str]:
    global IDENTITY_MAP_BYTES
    raw = fetch_bytes(MAP_PATH)
    IDENTITY_MAP_BYTES = len(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    return rows, hashlib.sha256(raw).hexdigest()


def hash_order(scene_ids: list[str]) -> list[str]:
    return sorted(
        scene_ids,
        key=lambda scene: hashlib.sha256(
            f"worldcache-rpx-pilot-v1:{scene}".encode()
        ).hexdigest(),
    )


def download_tar(scene: str, phase: int, kind: str, cache: Path) -> Path:
    if kind == "rgb":
        rel = f"scenes/{scene}/{phase}/rgb.tar"
    elif kind == "masks":
        rel = f"scenes/{scene}/{phase}/labels/masks/v1.tar"
    else:
        raise ValueError(kind)
    dest = cache / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".partial")
        digest = hashlib.sha256()
        byte_count = 0
        with urllib.request.urlopen(
            f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{rel}",
            timeout=120,
        ) as response, tmp.open("wb") as out:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        tmp.replace(dest)
        DOWNLOAD_RECORDS[rel] = {"bytes": byte_count, "sha256": digest.hexdigest()}
    elif rel not in DOWNLOAD_RECORDS:
        digest = hashlib.sha256()
        byte_count = 0
        with dest.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                byte_count += len(chunk)
        DOWNLOAD_RECORDS[rel] = {"bytes": byte_count, "sha256": digest.hexdigest()}
    return dest


def tar_members(path: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(path, "r:") as archive:
        return {member.name: member for member in archive.getmembers() if member.isfile()}


def read_tar_member(path: Path, member_name: str, index: dict[str, tarfile.TarInfo]) -> bytes:
    with tarfile.open(path, "r:") as archive:
        stream = archive.extractfile(index[member_name])
        if stream is None:
            raise FileNotFoundError(member_name)
        return stream.read()


def make_hist(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = rgb[mask]
    if not len(pixels):
        return np.zeros(96, dtype=np.float32)
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=32, range=(0, 256))
        parts.append(hist.astype(np.float32) / len(pixels))
    return unit(np.concatenate(parts))


def collect_phase(
    scene: str,
    phase: int,
    mapping: dict[int, dict[str, str]],
    cache: Path,
    frame_stride: int,
) -> tuple[list[dict[str, object]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    rgb_path = download_tar(scene, phase, "rgb", cache)
    mask_path = download_tar(scene, phase, "masks", cache)
    rgb_index, mask_index = tar_members(rgb_path), tar_members(mask_path)
    frames = sorted(
        int(name.rsplit("/", 1)[-1].split(".", 1)[0])
        for name in rgb_index
        if name.startswith("rgb/") and name.endswith(".webp")
    )
    selected = frames[::frame_stride]
    records: list[dict[str, object]] = []
    crops: list[np.ndarray] = []
    crop_masks: list[np.ndarray] = []
    hists: list[np.ndarray] = []
    try:
        for frame_id in selected:
            rgb_name = f"rgb/{frame_id:05d}.webp"
            mask_name = f"sam2/masks/{frame_id:05d}.png"
            rgb = np.asarray(Image.open(io.BytesIO(read_tar_member(rgb_path, rgb_name, rgb_index))).convert("RGB"))
            labels = np.asarray(Image.open(io.BytesIO(read_tar_member(mask_path, mask_name, mask_index))))
            if rgb.shape[:2] != labels.shape:
                raise ValueError(f"{scene}/{phase}/{frame_id}: RGB/mask shape mismatch")
            for local_id in np.unique(labels):
                local_id = int(local_id)
                if local_id == 0 or local_id not in mapping:
                    continue
                mask = labels == local_id
                ys, xs = np.where(mask)
                if len(xs) < 100:
                    continue
                pad = 8
                y0, y1 = max(0, int(ys.min()) - pad), min(rgb.shape[0], int(ys.max()) + pad + 1)
                x0, x1 = max(0, int(xs.min()) - pad), min(rgb.shape[1], int(xs.max()) + pad + 1)
                crop = rgb[y0:y1, x0:x1]
                crop_mask = mask[y0:y1, x0:x1]
                resized = np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR))
                resized_mask = np.asarray(Image.fromarray(crop_mask.astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST)) > 0
                row = mapping[local_id]
                records.append({
                    "scene": scene,
                    "phase": phase,
                    "frame_id": frame_id,
                    "local_mask_id": local_id,
                    "global_object_id": int(row["global_object_id"]),
                    "object_id": row["object_id"],
                    "class_name": row["class_name"],
                    "mask_area_pixels": int(mask.sum()),
                })
                crops.append(resized)
                crop_masks.append(resized_mask)
                hists.append(make_hist(rgb, mask))
    finally:
        # Free per-scene tarballs before advancing to the next scene.
        for p in (rgb_path, mask_path):
            p.unlink(missing_ok=True)
    return records, crops, crop_masks, hists


@torch.inference_mode()
def extract_descriptors(
    crops: list[np.ndarray], masks: list[np.ndarray], model: torch.nn.Module,
    device: torch.device, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pooled, cls = [], []
    for start in range(0, len(crops), batch_size):
        batch_rgb = np.stack(crops[start : start + batch_size]).astype(np.float32) / 255.0
        batch = (batch_rgb - MEAN[None, None, None, :]) / STD[None, None, None, :]
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        outputs = model.forward_features(tensor)
        if isinstance(outputs, dict):
            patches = outputs.get("x_norm_patchtokens")
            cls_tokens = outputs.get("x_norm_clstoken")
            if patches is None:
                patches = outputs["last_hidden_state"][:, 1:]
            if cls_tokens is None:
                cls_tokens = outputs["last_hidden_state"][:, 0]
        else:
            patches, cls_tokens = outputs[:, 1:], outputs[:, 0]
        patches = patches.float().cpu().numpy().reshape(len(batch), 16, 16, -1)
        mask_batch = np.stack(masks[start : start + batch_size]).astype(np.float32)
        mask_patch = mask_batch.reshape(len(batch), 16, 14, 16, 14).mean(axis=(2, 4))
        pooled.append((patches * mask_patch[..., None]).sum((1, 2)) / (mask_patch.sum((1, 2))[:, None] + 1e-12))
        cls.append(cls_tokens.float().cpu().numpy())
    return unit(np.concatenate(pooled)), unit(np.concatenate(cls))


def aggregate_scene_objects(
    records: list[dict[str, object]], pooled: np.ndarray, cls: np.ndarray,
    hists: list[np.ndarray],
) -> dict[int, dict[str, object]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for i, row in enumerate(records):
        grouped[int(row["global_object_id"])].append(i)
    result = {}
    for oid, indices in grouped.items():
        result[oid] = {
            "object_id": records[indices[0]]["object_id"],
            "class_name": records[indices[0]]["class_name"],
            "views": len(indices),
            "maskpool": unit(pooled[indices].mean(axis=0)),
            "cls": unit(cls[indices].mean(axis=0)),
            "rgbhist": unit(np.stack([hists[i] for i in indices]).mean(axis=0)),
        }
    return result


def cluster_bootstrap(
    rows: list[dict[str, object]], metric_a: str, metric_b: str,
    samples: int, seed: int,
) -> dict[str, object]:
    scenes = sorted({str(r["scene"]) for r in rows})
    objects = sorted({int(r["global_object_id"]) for r in rows})
    by = {(str(r["scene"]), int(r["global_object_id"])): r for r in rows}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for b in range(samples):
        sc = rng.choice(scenes, size=len(scenes), replace=True)
        ob = rng.choice(objects, size=len(objects), replace=True)
        scw, obw = defaultdict(int), defaultdict(int)
        for x in sc: scw[x] += 1
        for x in ob: obw[int(x)] += 1
        vals, weights = [], []
        for (s, o), row in by.items():
            w = scw[s] * obw[o]
            if w:
                vals.append(float(row[metric_a]) - float(row[metric_b]))
                weights.append(w)
        draws[b] = np.average(vals, weights=weights) if weights else np.nan
    draws = draws[np.isfinite(draws)]
    return {
        "difference_pp": float(100 * np.mean(draws)),
        "crossed_scene_object_bootstrap_95_pp": [float(x) for x in 100 * np.quantile(draws, [0.025, 0.975])],
        "bootstrap_samples": int(len(draws)),
        "n_scenes": len(scenes),
        "n_objects": len(objects),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/rpx_identity_cache"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--scene-count", type=int, default=20)
    ap.add_argument("--scenes", nargs="*", help="explicit scene list; omit for deterministic hash order")
    ap.add_argument("--frame-stride", type=int, default=25,
                    help="sample every Nth synchronized frame in each phase")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bootstrap-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--within-scene-pilot", action="store_true",
                    help="allow phase-0 gallery from the query scene (pilot only)")
    args = ap.parse_args()
    if args.frame_stride < 1:
        raise ValueError("frame-stride must be positive")
    map_rows, map_sha = load_identity_map()
    by_scene_phase: dict[tuple[str, int], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in map_rows:
        phase = int(row["phase"].replace("phase", ""))
        by_scene_phase[(row["scene_id"], phase)][int(row["local_mask_id"])] = row
    all_scenes = sorted({row["scene_id"] for row in map_rows})
    class_to_ids: dict[str, set[int]] = defaultdict(set)
    for row in map_rows:
        class_to_ids[row["class_name"]].add(int(row["global_object_id"]))
    global_ids = {int(row["global_object_id"]) for row in map_rows}
    class_identity_confounded = (
        len(class_to_ids) == len(global_ids)
        and all(len(ids) == 1 for ids in class_to_ids.values())
    )
    scenes = args.scenes if args.scenes else hash_order(all_scenes)[: args.scene_count]
    if not scenes or any(scene not in all_scenes for scene in scenes):
        raise ValueError("invalid or empty scene list")

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2_vitb14(224, device)
    object_views: dict[tuple[str, int, int], dict[str, object]] = {}
    extraction_rows = []
    for scene in scenes:
        for phase in (0, 2):
            records, crops, crop_masks, hists = collect_phase(
                scene, phase, by_scene_phase[(scene, phase)], args.cache_dir,
                args.frame_stride,
            )
            if not records:
                raise ValueError(f"no mask crops recovered for {scene} phase {phase}")
            pooled, cls = extract_descriptors(crops, crop_masks, model, device, args.batch_size)
            aggregated = aggregate_scene_objects(records, pooled, cls, hists)
            for oid, row in aggregated.items():
                object_views[(scene, phase, oid)] = row
            extraction_rows.append({
                "scene": scene,
                "phase": phase,
                "sampled_frames": len(set(int(r["frame_id"]) for r in records)),
                "object_ids": len(aggregated),
                "object_views": len(records),
            })
            print(f"[{scene} phase{phase}] objects={len(aggregated)} views={len(records)}", flush=True)

    results = []
    for query_scene in scenes:
        query_objects = {
            oid: row for (scene, phase, oid), row in object_views.items()
            if scene == query_scene and phase == 2
        }
        gallery_by_id: dict[int, list[dict[str, object]]] = defaultdict(list)
        for (scene, phase, oid), row in object_views.items():
            if phase == 0 and (args.within_scene_pilot or scene != query_scene):
                gallery_by_id[oid].append(row)
        supported_ids = set(gallery_by_id)
        candidates = sorted(supported_ids)
        candidate_classes = {
            cid: str(gallery_by_id[cid][0]["class_name"]) for cid in candidates
        }
        for oid, query in query_objects.items():
            if oid not in supported_ids:
                continue
            gallery = {}
            for cid in candidates:
                sources = gallery_by_id[cid]
                for desc in ("maskpool", "cls", "rgbhist"):
                    gallery.setdefault(desc, {})[cid] = unit(
                        np.mean([np.asarray(r[desc]) for r in sources], axis=0)
                    )
            row = {
                "scene": query_scene,
                "global_object_id": int(oid),
                "object_id": query["object_id"],
                "class_name": query["class_name"],
                "gallery_scene_support": len(gallery_by_id[oid]),
                "candidate_count": len(candidates),
                "query_views": query["views"],
            }
            same_class_candidates = [
                cid for cid in candidates
                if candidate_classes[cid] == str(query["class_name"])
            ]
            row["same_class_candidate_count"] = len(same_class_candidates)
            for desc in ("maskpool", "cls", "rgbhist"):
                scores = np.array([
                    float(np.dot(np.asarray(query[desc]), np.asarray(gallery[desc][cid])))
                    for cid in candidates
                ])
                order = np.argsort(-scores)
                ranked = [candidates[i] for i in order]
                row[f"{desc}_correct"] = int(ranked[0] == oid)
                row[f"{desc}_top5"] = int(oid in ranked[:5])
                row[f"{desc}_rank"] = ranked.index(oid) + 1
                row[f"{desc}_margin"] = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
                if len(same_class_candidates) >= 2:
                    same_order = sorted(
                        same_class_candidates,
                        key=lambda cid: float(np.dot(
                            np.asarray(query[desc]), np.asarray(gallery[desc][cid])
                        )),
                        reverse=True,
                    )
                    row[f"{desc}_same_class_correct"] = int(same_order[0] == oid)
            results.append(row)

    if not results:
        raise ValueError("no query object had a cross-scene gallery identity")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    frame.to_csv(args.output_dir / "query_rows.csv", index=False)
    pd.DataFrame(extraction_rows).to_csv(args.output_dir / "extraction_rows.csv", index=False)
    summaries = {}
    for desc in ("maskpool", "cls", "rgbhist"):
        scene_acc = frame.groupby("scene")[f"{desc}_correct"].mean()
        summaries[desc] = {
            "query_count": len(frame),
            "top1_micro": float(frame[f"{desc}_correct"].mean()),
            "top1_scene_macro": float(scene_acc.mean()),
            "top5_micro": float(frame[f"{desc}_top5"].mean()),
            "mean_reciprocal_rank": float((1.0 / frame[f"{desc}_rank"]).mean()),
            "scene_accuracy": {str(k): float(v) for k, v in scene_acc.items()},
        }
    pairwise = {}
    for desc in ("maskpool", "cls"):
        pairwise[f"{desc}_minus_rgbhist"] = cluster_bootstrap(
            results, f"{desc}_correct", "rgbhist_correct",
            args.bootstrap_samples, args.seed + len(pairwise),
        )
    same_class = frame[frame.same_class_candidate_count >= 2]
    same_class_summary = {
        "query_count": int(len(same_class)),
        "queries_with_same_class_hard_negative_fraction": float(len(same_class) / len(frame)),
        "top1": {
            desc: float(same_class[f"{desc}_same_class_correct"].mean())
            if len(same_class) else None
            for desc in ("maskpool", "cls", "rgbhist")
        },
    }
    summary = {
        "dataset": "RPX Multi-Object Scenes",
        "dataset_revision": REVISION,
        "dataset_license": "CC BY 4.0",
        "identity_map_sha256": map_sha,
        "model": MODEL,
        "mode": "within-scene phase retrieval pilot" if args.within_scene_pilot else "cross-scene held-out object retrieval",
        "scenes": scenes,
        "frame_stride": args.frame_stride,
        "candidate_population": "phase-0 global_object_id identities observed in gallery scenes; query scenes are excluded unless --within-scene-pilot is used",
        "identity_map_entries": len(map_rows),
        "identity_class_audit": {
            "global_object_ids": len(global_ids),
            "class_names": len(class_to_ids),
            "classes_with_multiple_global_ids": sum(len(ids) > 1 for ids in class_to_ids.values()),
            "global_identity_and_class_are_one_to_one": class_identity_confounded,
            "class_name_oracle_top1": 1.0 if class_identity_confounded else None,
            "interpretation": "physical instance identity cannot be separated from category in this release subset" if class_identity_confounded else "some same-class identity negatives are available",
        },
        "queries_scored": len(results),
        "query_support_fraction": len(results) / sum(len([1 for s, p, o in object_views if s == scene and p == 2]) for scene in scenes),
        "cross_scene_bootstrap": "resamples scene and physical global_object_id clusters independently (two-way/pigeonhole bootstrap)",
        "metrics": summaries,
        "same_class_hard_negative_sensitivity": same_class_summary,
        "paired_differences": pairwise,
        "limitations": [
            "closed-set retrieval among the supported catalog objects, not open-set recognition",
            "each of the 70 global object IDs maps to a unique class in the identity map, so there are no same-class instance negatives; a class-name oracle achieves 100% and physical instance recognition is not identified",
            "official instance masks are privileged crop proposals and used only for crop construction and scoring",
            "only RGB descriptors are evaluated; this is an identity stress test, not RGB-D pose or cache-runtime validation",
            "RPX is a newly released benchmark whose dataset card says the paper citation will be added after release",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "download_manifest.json").write_text(json.dumps({
        "repository": REPO,
        "revision": REVISION,
        "license": "CC BY 4.0",
        "identity_map": {"path": MAP_PATH, "sha256": map_sha, "bytes": IDENTITY_MAP_BYTES},
        "tar_files": DOWNLOAD_RECORDS,
        "modalities_downloaded": ["RGB", "primary SAM2 instance masks"],
    }, indent=2) + "\n")
    print(json.dumps({"queries": len(results), "summaries": summaries, "differences": pairwise}, indent=2))


if __name__ == "__main__":
    main()
