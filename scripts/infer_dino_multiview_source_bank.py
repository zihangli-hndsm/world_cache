#!/usr/bin/env python3
"""Anonymous all-reference-view DINO retrieval on the frozen 3RScan masks.

Each rescan mask is matched against every reference-mask observation in its
relation. A rectangular cosine Hungarian assignment is solved independently
for each query frame; query labels are never loaded. Scoring is a separate
post-inference step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import timm
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "outputs/reports/20261008_segmast3r_validation"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/reports/20261009_dino_multiview_source_bank"


def load_model(device: torch.device):
    model = timm.create_model(
        "vit_giant_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        img_size=224,
    ).to(device).eval()
    return model


@torch.inference_mode()
def extract_mask_features(records: list[dict], *, root: Path, model, device, batch_size: int):
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    frame_ids: list[int] = []
    frame_counts: list[int] = []
    manifest_hash = hashlib.sha256()

    for record in records:
        input_path = Path(record["input"])
        if not input_path.is_absolute():
            input_path = root / input_path
        input_bytes = input_path.read_bytes()
        got_sha = hashlib.sha256(input_bytes).hexdigest()
        if got_sha != record["input_sha256"]:
            raise ValueError(f"frozen input checksum mismatch: {input_path}")
        manifest_hash.update(bytes.fromhex(got_sha))
        with np.load(input_path) as data:
            rgb = data["rgb"].copy()
            frame_masks = data["masks"].astype(bool, copy=True)
        if rgb.shape != (224, 224, 3) or frame_masks.ndim != 3 or frame_masks.shape[1:] != (224, 224):
            raise ValueError(f"unexpected frozen input shapes in {input_path}: {rgb.shape}, {frame_masks.shape}")
        if int(record["frame_id"]) < 0:
            raise ValueError(f"negative frame id in {input_path}")
        images.append(rgb)
        masks.extend(frame_masks)
        frame_ids.append(int(record["frame_id"]))
        frame_counts.append(len(frame_masks))

    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    frame_features: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch_np = np.stack(images[start : start + batch_size])
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        batch = (batch - mean) / std
        tokens = model.forward_features(batch)
        if isinstance(tokens, dict):
            tokens = tokens.get("x_norm_patchtokens", tokens.get("last_hidden_state"))
            if tokens is None:
                raise RuntimeError("unrecognized timm DINOv2 feature dictionary")
        n_patches = 16 * 16
        n_prefix = int(tokens.shape[1] - n_patches)
        patch_tokens = tokens[:, n_prefix:, :].reshape(len(batch_np), 16, 16, -1)
        frame_features.extend(patch_tokens.unbind(0))

    pooled: list[torch.Tensor] = []
    cursor = 0
    for feature_map, count in zip(frame_features, frame_counts):
        frame_masks = np.stack(masks[cursor : cursor + count]) if count else np.zeros((0, 224, 224), bool)
        cursor += count
        if count == 0:
            pooled.append(torch.empty((0, feature_map.shape[-1]), device=device))
            continue
        mask_tensor = torch.from_numpy(frame_masks.astype(np.float32)).to(device=device).unsqueeze(1)
        weights = F.adaptive_avg_pool2d(mask_tensor, (16, 16)).squeeze(1).flatten(1)
        flat_features = feature_map.reshape(256, -1)
        embeddings = weights @ flat_features / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        pooled.append(embeddings)
    return frame_ids, pooled, manifest_hash.hexdigest()


def run_relation(run_id: str, manifest_path: Path, *, root: Path, model, device, batch_size: int,
                 feature_path: Path) -> dict:
    frozen = json.loads(manifest_path.read_text())
    reference_ids, reference_features, reference_sha = extract_mask_features(
        frozen["records"]["reference"], root=root, model=model, device=device, batch_size=batch_size
    )
    query_ids, query_features, query_sha = extract_mask_features(
        frozen["records"]["rescan"], root=root, model=model, device=device, batch_size=batch_size
    )
    source_embeddings = torch.cat(reference_features, dim=0)
    query_embeddings_all = torch.cat(query_features, dim=0)
    source_frame_for_index = np.concatenate([
        np.full(len(feats), frame_id, dtype=np.int32)
        for frame_id, feats in zip(reference_ids, reference_features)
    ]) if len(source_embeddings) else np.zeros((0,), dtype=np.int32)
    rows: list[dict] = []
    for query_frame_id, query_embeddings in zip(query_ids, query_features):
        if not len(query_embeddings) or not len(source_embeddings):
            continue
        similarities = query_embeddings @ source_embeddings.T
        row_idx, col_idx = linear_sum_assignment(-similarities.detach().cpu().numpy())
        assigned_by_query = {int(q): (int(s), float(similarities[q, s].item())) for q, s in zip(row_idx, col_idx)}
        best_scores, best_source = similarities.max(dim=1)
        for query_index in range(len(query_embeddings)):
            assignment = assigned_by_query.get(query_index)
            assigned_source, assigned_score = assignment if assignment is not None else (None, None)
            rows.append({
                "query_frame": int(query_frame_id),
                "query_index": int(query_index),
                "nearest_source_observation": int(best_source[query_index].item()),
                "nearest_source_frame": int(source_frame_for_index[int(best_source[query_index].item())]),
                "nearest_similarity": float(best_scores[query_index].item()),
                "assigned_source_observation": assigned_source,
                "assigned_source_frame": (
                    int(source_frame_for_index[assigned_source]) if assigned_source is not None else None
                ),
                "assigned_similarity": assigned_score,
            })
    np.savez_compressed(
        feature_path,
        source_embeddings=source_embeddings.detach().cpu().numpy().astype(np.float16),
        source_frame_ids=source_frame_for_index.astype(np.int32),
        query_embeddings=query_embeddings_all.detach().cpu().numpy().astype(np.float16),
        query_frame_ids=np.concatenate([
            np.full(len(feats), frame_id, dtype=np.int32)
            for frame_id, feats in zip(query_ids, query_features)
        ]),
    )
    return {
        "run_id": run_id,
        "status": "complete",
        "input_sha256": {"reference": reference_sha, "rescan": query_sha},
        "reference_frames": reference_ids,
        "query_frames": query_ids,
        "reference_observations": int(len(source_embeddings)),
        "query_observations": int(sum(len(x) for x in query_features)),
        "query_rows": rows,
        "feature_spec": "DINOv2 ViT-G/14 final-layer normalized patch tokens; mask-occupancy weighted mean",
        "assignment_spec": "maximum-cosine rectangular Hungarian independently per query frame over all reference-mask observations",
        "labels_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-relations", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = ROOT
    input_root = args.input_root if args.input_root.is_absolute() else root / args.input_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifests = sorted((input_root / "uniform20_inputs").glob("*/manifest.json"))
    if args.max_relations:
        manifests = manifests[: args.max_relations]
    if not manifests:
        raise FileNotFoundError(f"no uniform20 manifests in {input_root}")
    device = torch.device(args.device)
    model = load_model(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, manifest_path in enumerate(manifests, start=1):
        run_id = manifest_path.parent.name
        destination = output_dir / f"{run_id}.json"
        feature_path = output_dir / f"{run_id}_features.npz"
        result = run_relation(
            run_id, manifest_path, root=root, model=model, device=device,
            batch_size=args.batch_size, feature_path=feature_path,
        )
        destination.write_text(json.dumps(result, separators=(",", ":")) + "\n")
        results.append({
            "run_id": run_id,
            "status": "complete",
            "output": str(destination.resolve()),
            "features": str(feature_path.resolve()),
        })
        (output_dir / "run_manifest.json").write_text(json.dumps({
            "status": "running", "relations_requested": len(manifests), "results": results,
        }, indent=2) + "\n")
        print(f"{index}/{len(manifests)} {run_id} complete", flush=True)
    (output_dir / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "relations_requested": len(manifests), "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
