#!/usr/bin/env python3
"""Temporal object-identity retrieval on the UW RGB-D Scenes Dataset.

Uses only official per-frame 2D boxes. Within each scene, the first temporal
third is the reference bank and the last third is queried; the middle third is
discarded as a temporal gap. This is a box-crop retrieval probe, not a 3D or
room-rescan evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from PIL import Image

from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def unwrap(value):
    while isinstance(value, np.ndarray):
        value = value.item()
    return value


def scalar(value):
    return int(unwrap(value))


def text(value):
    return str(unwrap(value))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_scene(path: Path):
    rows = []
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
        mat_member = next(m for m in members if m.name.endswith(".mat"))
        boxes = scipy.io.loadmat(io.BytesIO(archive.extractfile(mat_member).read()))["bboxes"].ravel()
        rgb_members = {}
        for member in members:
            name = Path(member.name).name
            if name.endswith(".png") and not name.endswith("_depth.png"):
                stem = name[:-4]
                try:
                    frame = int(stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                rgb_members[frame] = member
        ordered_frames = sorted(rgb_members)
        # The official MAT vector is in frame order, beginning at frame 1.
        if len(ordered_frames) != len(boxes) or ordered_frames != list(range(1, len(boxes) + 1)):
            raise ValueError(f"frame/annotation mismatch in {path.name}: {len(ordered_frames)} RGB, {len(boxes)} labels")
        n = len(boxes)
        for pos, cell in enumerate(boxes):
            if cell.size == 0:
                continue
            frame = pos + 1
            rgb_bytes = archive.extractfile(rgb_members[frame]).read()
            image = np.asarray(Image.open(io.BytesIO(rgb_bytes)).convert("RGB"))
            for ann in cell.ravel():
                category = text(ann["category"])
                instance = scalar(ann["instance"])
                # MATLAB boxes are 1-indexed inclusive [top,bottom,left,right].
                top, bottom = scalar(ann["top"]), scalar(ann["bottom"])
                left, right = scalar(ann["left"]), scalar(ann["right"])
                y0, y1 = max(0, top - 1), min(image.shape[0], bottom)
                x0, x1 = max(0, left - 1), min(image.shape[1], right)
                if y1 <= y0 or x1 <= x0:
                    continue
                rows.append({
                    "scene": path.stem, "frame": frame, "frame_position": pos,
                    "frame_count": n, "object_id": f"{category}_{instance}",
                    "category": category, "image": image[y0:y1, x0:x1].copy(),
                })
    return rows


def preprocess(image: np.ndarray) -> torch.Tensor:
    h, w = image.shape[:2]
    scale = 224.0 / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = np.asarray(Image.fromarray(image).resize((nw, nh), Image.Resampling.BICUBIC))
    canvas = np.round(MEAN * 255).astype(np.uint8)
    square = np.broadcast_to(canvas, (224, 224, 3)).copy()
    x0, y0 = (224 - nw) // 2, (224 - nh) // 2
    square[y0:y0 + nh, x0:x0 + nw] = resized
    tensor = torch.from_numpy(square).permute(2, 0, 1).float().div_(255)
    return (tensor - torch.from_numpy(MEAN)[:, None, None]) / torch.from_numpy(STD)[:, None, None]


def color_hist(image: np.ndarray) -> np.ndarray:
    values = np.concatenate([
        np.histogram(image[:, :, c], bins=16, range=(0, 256))[0]
        for c in range(3)
    ]).astype(np.float32)
    values /= max(values.sum(), 1.0)
    values /= max(np.linalg.norm(values), 1e-12)
    return values


@torch.inference_mode()
def embed(rows, batch_size: int, device: torch.device):
    model = load_dinov2("giant", 224, device)
    descriptors = []
    feats, colors = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        images = torch.stack([preprocess(row["image"]) for row in batch]).to(device)
        captured = extract_block_tokens(model, images, (2, 20, 40))
        layer_tokens = [
            F.normalize(captured[layer][:, model.num_prefix_tokens:], dim=2)
            for layer in (2, 20, 40)
        ]
        pooled = F.normalize(torch.cat(layer_tokens, dim=2).mean(1), dim=1)
        feats.append(pooled.cpu().numpy().astype(np.float32))
        colors.extend(color_hist(row["image"]) for row in batch)
        descriptors.extend({k: row[k] for k in ("scene", "frame", "frame_position", "frame_count", "object_id", "category")} for row in batch)
        if start + len(batch) == len(rows) or (start + len(batch)) % 256 < batch_size:
            print(f"embedded {start + len(batch)}/{len(rows)} annotations", flush=True)
    del model
    return descriptors, np.concatenate(feats), np.stack(colors)


def retrieve(meta, features, colors):
    groups = defaultdict(list)
    for i, row in enumerate(meta):
        groups[row["scene"]].append(i)
    out = []
    for scene, indices in sorted(groups.items()):
        n = meta[indices[0]]["frame_count"]
        ref, query = [], []
        for i in indices:
            row = meta[i]
            if row["frame_position"] < n // 3:
                ref.append(i)
            elif row["frame_position"] >= (2 * n) // 3:
                query.append(i)
        ref_ids = sorted({meta[i]["object_id"] for i in ref})
        eligible = sorted({meta[i]["object_id"] for i in ref} & {meta[i]["object_id"] for i in query})
        ref = [i for i in ref if meta[i]["object_id"] in eligible]
        query = [i for i in query if meta[i]["object_id"] in eligible]
        if len(eligible) < 2:
            continue
        ref_by_id = {oid: [i for i in ref if meta[i]["object_id"] == oid] for oid in eligible}
        ref_flat = [i for oid in eligible for i in ref_by_id[oid]]
        for qi in query:
            per_id_scores = [features[qi] @ features[ref_by_id[oid]].T for oid in eligible]
            dino = np.asarray([np.max(scores) for scores in per_id_scores])
            dino_top3mean = np.asarray([
                np.sort(scores)[-min(3, len(scores)):].mean() for scores in per_id_scores
            ])
            color = np.asarray([np.max(colors[qi] @ colors[ref_by_id[oid]].T) for oid in eligible])
            d_pred, c_pred = eligible[int(np.argmax(dino))], eligible[int(np.argmax(color))]
            d3_pred = eligible[int(np.argmax(dino_top3mean))]
            max_view_top2 = np.partition(dino, -2)[-2:]
            top3_mean_top2 = np.partition(dino_top3mean, -2)[-2:]
            target = meta[qi]["object_id"]
            out.append({**{k: meta[qi][k] for k in ("scene", "frame", "object_id", "category")},
                        "candidate_count": len(eligible), "dino_prediction": d_pred,
                        "dino_max_view_margin": float(max_view_top2[1] - max_view_top2[0]),
                        "dino_top3mean_prediction": d3_pred,
                        "dino_top3mean_margin": float(top3_mean_top2[1] - top3_mean_top2[0]),
                        "dino_top3mean_correct": d3_pred == target,
                        "color_prediction": c_pred, "dino_correct": d_pred == target,
                        "color_correct": c_pred == target,
                        "dino_category_correct": d_pred.rsplit("_", 1)[0] == meta[qi]["category"],
                        "color_category_correct": c_pred.rsplit("_", 1)[0] == meta[qi]["category"]})
        print(f"{scene}: candidates={len(eligible)} queries={len(query)}", flush=True)
    return out


def retrieve_leave_one_scene_out(meta, features, colors, max_refs_per_id: int | None = 12):
    """Query scenes against other-scene galleries, optionally balancing references."""
    groups = defaultdict(list)
    for i, row in enumerate(meta):
        groups[row["scene"]].append(i)
    all_indices = list(range(len(meta)))
    out = []
    for held_out in sorted(groups):
        query = groups[held_out]
        gallery = [i for i in all_indices if meta[i]["scene"] != held_out]
        gallery_ids = sorted({meta[i]["object_id"] for i in gallery})
        candidates_by_id = {oid: [i for i in gallery if meta[i]["object_id"] == oid] for oid in gallery_ids}
        if max_refs_per_id is None:
            by_id = candidates_by_id
        else:
            by_id = {}
            for oid, candidates in candidates_by_id.items():
                if len(candidates) < max_refs_per_id:
                    raise ValueError(f"{held_out}/{oid} has only {len(candidates)} refs; need {max_refs_per_id}")
                positions = np.linspace(0, len(candidates) - 1, max_refs_per_id, dtype=int)
                by_id[oid] = [candidates[int(pos)] for pos in positions]
        query = [i for i in query if meta[i]["object_id"] in by_id]
        if not query:
            continue
        for qi in query:
            per_id_scores = [features[qi] @ features[by_id[oid]].T for oid in gallery_ids]
            dino = np.asarray([np.max(scores) for scores in per_id_scores])
            dino_top3mean = np.asarray([
                np.sort(scores)[-min(3, len(scores)):].mean() for scores in per_id_scores
            ])
            color = np.asarray([np.max(colors[qi] @ colors[by_id[oid]].T) for oid in gallery_ids])
            d_pred, c_pred = gallery_ids[int(np.argmax(dino))], gallery_ids[int(np.argmax(color))]
            d3_pred = gallery_ids[int(np.argmax(dino_top3mean))]
            max_view_top2 = np.partition(dino, -2)[-2:]
            top3_mean_top2 = np.partition(dino_top3mean, -2)[-2:]
            target = meta[qi]["object_id"]
            out.append({**{k: meta[qi][k] for k in ("scene", "frame", "object_id", "category")},
                        "candidate_count": len(gallery_ids), "dino_prediction": d_pred,
                        "dino_max_view_margin": float(max_view_top2[1] - max_view_top2[0]),
                        "dino_top3mean_prediction": d3_pred,
                        "dino_top3mean_margin": float(top3_mean_top2[1] - top3_mean_top2[0]),
                        "dino_top3mean_correct": d3_pred == target,
                        "color_prediction": c_pred, "dino_correct": d_pred == target,
                        "color_correct": c_pred == target,
                        "dino_category_correct": d_pred.rsplit("_", 1)[0] == meta[qi]["category"],
                        "color_category_correct": c_pred.rsplit("_", 1)[0] == meta[qi]["category"]})
        ref_note = "all refs" if max_refs_per_id is None else f"refs/ID={max_refs_per_id}"
        print(f"leave-one-scene-out {held_out}: candidates={len(gallery_ids)} {ref_note} queries={len(query)}", flush=True)
    return out


def summarize(rows, draws: int, seed: int):
    def cluster_metrics(cluster_key):
        groups = defaultdict(list)
        for row in rows:
            groups[cluster_key(row)].append(row)
        units = sorted(groups)
        means = {key: np.asarray([
            np.mean([row[key] for row in groups[unit]]) for unit in units
        ]) for key in ("dino_correct", "color_correct", "dino_category_correct", "color_category_correct")}
        rng = np.random.default_rng(seed)
        ix = rng.integers(0, len(units), size=(draws, len(units)))
        return units, means, ix

    # Instance numbers are global to the source object set, so the primary
    # uncertainty unit is the physical category/instance ID across scenes.
    units, means, ix = cluster_metrics(lambda row: row["object_id"])
    metrics = {}
    for metric, dkey, ckey in (("instance", "dino_correct", "color_correct"),
                               ("category", "dino_category_correct", "color_category_correct")):
        d, c = means[dkey], means[ckey]
        delta = d - c
        metrics[metric] = {"dino_macro": float(d.mean()), "color_histogram_macro": float(c.mean()),
                           "difference_pp": float((d.mean() - c.mean()) * 100),
                           "paired_object_cluster_bootstrap_95_pp": [float(x * 100) for x in np.quantile(delta[ix].mean(1), [.025, .975])]}
    sensitivities = {}
    for name, key_fn in (("scene_x_object_id", lambda row: (row["scene"], row["object_id"])),
                         ("scene", lambda row: row["scene"])):
        clusters, cluster_means, cluster_ix = cluster_metrics(key_fn)
        d = cluster_means["dino_correct"]
        c = cluster_means["color_correct"]
        sensitivities[name] = {
            "clusters": len(clusters), "dino_macro": float(d.mean()),
            "color_histogram_macro": float(c.mean()), "difference_pp": float((d.mean() - c.mean()) * 100),
            "paired_cluster_bootstrap_95_pp": [float(x * 100) for x in np.quantile((d-c)[cluster_ix].mean(1), [.025, .975])]
        }
    by_scene = {}
    for scene in sorted({r["scene"] for r in rows}):
        subset = [r for r in rows if r["scene"] == scene]
        by_scene[scene] = {"queries": len(subset), "candidate_count": int(np.median([r["candidate_count"] for r in subset])),
                           "dino_instance": float(np.mean([r["dino_correct"] for r in subset])),
                           "color_instance": float(np.mean([r["color_correct"] for r in subset]))}
    return {"scenes": len(by_scene), "eligible_global_physical_object_ids": len(units), "query_annotations": len(rows),
            "metrics": metrics, "cluster_sensitivity": sensitivities, "by_scene": by_scene,
            "bootstrap": {"primary_unit": "global category_instance physical object ID",
                          "draws": draws, "seed": seed}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap-draws", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261009)
    args = p.parse_args()
    paths = sorted(args.archives.glob("*.tar"))
    if len(paths) != 8:
        raise ValueError(f"expected the eight official per-scene archives, got {len(paths)}")
    scenes = [row for path in paths for row in read_scene(path)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"parsed {len(scenes)} bbox annotations across {len(paths)} scenes; device={device}", flush=True)
    meta, features, colors = embed(scenes, args.batch_size, device)
    rows = retrieve(meta, features, colors)
    summary = summarize(rows, args.bootstrap_draws, args.seed)
    cross_scene_rows = retrieve_leave_one_scene_out(meta, features, colors)
    cross_scene_all_refs_rows = retrieve_leave_one_scene_out(meta, features, colors, max_refs_per_id=None)
    cross_scene_summary = summarize(cross_scene_rows, args.bootstrap_draws, args.seed + 1)
    cross_scene_summary.update({"evaluation": "leave-one-scene-out global identity retrieval",
                                "all_box_annotations": len(meta),
                                "eligible_query_annotations": len(cross_scene_rows),
                                "excluded_unseen_query_annotations": len(meta) - len(cross_scene_rows),
                                "protocol": {"gallery": "12 temporally spaced annotated box crops per global ID from the other seven scenes",
                                             "query": "all annotated box crops in the held-out scene whose global ID appears in the gallery",
                                             "candidate_set": "all global category_instance IDs observed in the other seven scenes",
                                             "descriptor": "frozen DINOv2 ViT-G/14; normalized patch tokens from blocks 2/20/40, mean pooled",
                                             "baseline": "16-bin per-channel RGB histograms, L2 normalized; same tight 2D boxes and retrieval rule",
                                             "caveat": "closed-set retrieval; query IDs absent from the gallery are excluded; same UW dataset family",
                                             "references_per_candidate": 12},
                                "input_archives": {path.name: sha256(path) for path in paths}})
    cross_scene_all_refs_summary = summarize(cross_scene_all_refs_rows, args.bootstrap_draws, args.seed + 1)
    cross_scene_all_refs_summary.update({"evaluation": "leave-one-scene-out global identity retrieval, all available references",
                                         "all_box_annotations": len(meta),
                                         "eligible_query_annotations": len(cross_scene_all_refs_rows),
                                         "excluded_unseen_query_annotations": len(meta) - len(cross_scene_all_refs_rows),
                                         "protocol": {"gallery": "all annotated box crops from the other seven scenes; unequal reference counts",
                                                      "candidate_set": "all global category_instance IDs observed in the other seven scenes",
                                                      "caveat": "closed-set retrieval; query IDs absent from the gallery are excluded; same UW dataset family"},
                                         "input_archives": {path.name: sha256(path) for path in paths}})
    summary.update({"dataset": "UW RGB-D Scenes Dataset", "input_archives": {p.name: sha256(p) for p in paths},
                    "protocol": {"reference": "first temporal third", "query": "last temporal third",
                                 "temporal_gap": "middle temporal third", "crop": "official 2D bbox, tight crop, 224 px",
                                 "descriptor": "frozen DINOv2 ViT-G/14; normalized patch tokens from blocks 2/20/40, mean pooled",
                                 "retrieval": "max cosine over reference annotations for each candidate object ID",
                                 "baseline": "16-bin per-channel RGB histograms, L2 normalized; same crop and retrieval rule",
                                 "caveat": "original benchmark may overlap DINOv2 pretraining; bbox crops are not 3D object masks"}})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "query_rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "cross_scene_summary.json").write_text(json.dumps(cross_scene_summary, indent=2) + "\n")
    with (args.output / "cross_scene_query_rows.jsonl").open("w") as f:
        for row in cross_scene_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output / "cross_scene_all_refs_summary.json").write_text(json.dumps(cross_scene_all_refs_summary, indent=2) + "\n")
    with (args.output / "cross_scene_all_refs_query_rows.jsonl").open("w") as f:
        for row in cross_scene_all_refs_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
