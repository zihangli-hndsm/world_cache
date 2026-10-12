#!/usr/bin/env python3
"""Label-blind CrossOver RGB instance embeddings on frozen 3RScan masks.

This adapts the public CrossOver RGB object encoder to the already-prepared
anonymous mask inputs. It uses the published 3RScan checkpoint's RGB
projection head and the model's frozen DINOv2 ViT-G/14 value/CLS features.
Identity targets are read only by the separate scoring script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as tvf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "outputs/reports/20261008_segmast3r_validation"
DEFAULT_CHECKPOINT = Path(
    "/tmp/crossover_hf/models--gradient-spaces--CrossOver/snapshots/"
    "11dfb8bbec890c60c577817c01c244ef9b3a04b1/"
    "instance_crossover_scan3r.pth/model.safetensors"
)


class RGBProjection(nn.Module):
    """The exact RGB projection head from CrossOver's public model code."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1536, 384),
            nn.ReLU(),
            nn.LayerNorm(384, eps=1e-12),
            nn.Dropout(0.0),
            nn.Linear(384, 768),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=-1)


class CrossOverRGBEncoder:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.device = device
        self.backbone = timm.create_model(
            "vit_giant_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        ).to(device).eval()
        self._qkv: torch.Tensor | None = None

        def capture_qkv(_module, _inputs, output):
            self._qkv = output

        self._hook = self.backbone.blocks[31].attn.qkv.register_forward_hook(capture_qkv)
        self.projection = RGBProjection().to(device).eval()
        checkpoint_state = load_file(str(checkpoint), device=str(device))
        state = {}
        for key in ("weight", "bias"):
            for layer in ("0", "2", "4"):
                name = f"modality_projections.rgb.{layer}.{key}"
                if name in checkpoint_state:
                    state[f"layers.{layer}.{key}"] = checkpoint_state[name]
        expected = set(self.projection.state_dict())
        if set(state) != expected:
            raise ValueError(f"RGB head checkpoint keys mismatch: {set(state) ^ expected}")
        self.projection.load_state_dict(state, strict=True)
        self.transform = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def encode_masks(self, rgb: np.ndarray, masks: np.ndarray) -> np.ndarray:
        rgb = np.rot90(rgb, k=3).copy()
        masks = np.rot90(masks, k=3, axes=(1, 2)).copy()
        image = Image.fromarray(rgb.astype(np.uint8, copy=False))
        crop_tensors: list[torch.Tensor] = []
        crop_spans: list[tuple[int, int]] = []
        for mask_value in masks:
            mask = np.asarray(mask_value, dtype=bool)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                crop_spans.append((len(crop_tensors), len(crop_tensors)))
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            start = len(crop_tensors)
            for level in range(3):
                x_exp = int((x2 - x1) * 0.2) * level
                y_exp = int((y2 - y1) * 0.2) * level
                box = (
                    max(0, x1 - x_exp),
                    max(0, y1 - y_exp),
                    min(mask.shape[1], x2 + x_exp),
                    min(mask.shape[0], y2 + y_exp),
                )
                crop = image.crop(box).resize((224, 224), Image.Resampling.BICUBIC)
                crop_tensors.append(self.transform(crop))
            crop_spans.append((start, len(crop_tensors)))

        output = np.zeros((len(masks), 768), dtype=np.float32)
        if not crop_tensors:
            return output
        batch = torch.stack(crop_tensors).to(self.device)
        features: list[torch.Tensor] = []
        for chunk in batch.split(32):
            self._qkv = None
            self.backbone(chunk)
            if self._qkv is None:
                raise RuntimeError("DINOv2 qkv hook did not capture features")
            value = self._qkv.chunk(3, dim=-1)[2][:, 0, :]
            features.append(F.normalize(value.float(), dim=-1))
        crop_features = torch.cat(features, dim=0)
        pooled = []
        valid_indices = []
        for index, (start, end) in enumerate(crop_spans):
            if end > start:
                pooled.append(crop_features[start:end].mean(dim=0))
                valid_indices.append(index)
        if pooled:
            embeddings = self.projection(torch.stack(pooled)).float().cpu().numpy()
            output[np.asarray(valid_indices, dtype=np.int64)] = embeddings
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-relations", type=int, default=0)
    parser.add_argument(
        "--one-per-reference-group",
        action="store_true",
        help="retain the first sorted relation for each distinct reference scan",
    )
    args = parser.parse_args()
    args.input_root = args.input_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()

    manifests = sorted((args.input_root / "uniform20_inputs").glob("*/manifest.json"))
    if args.one_per_reference_group:
        selected = []
        seen_reference_scans = set()
        for manifest_path in manifests:
            data = json.loads(manifest_path.read_text())
            ref_scan = Path(data["records"]["reference"][0]["input"]).parent.name
            if ref_scan not in seen_reference_scans:
                selected.append(manifest_path)
                seen_reference_scans.add(ref_scan)
        manifests = selected
    if args.max_relations:
        manifests = manifests[: args.max_relations]
    if not manifests:
        raise FileNotFoundError(f"no relation manifests under {args.input_root}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"CrossOver checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    encoder = CrossOverRGBEncoder(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_cache: dict[str, np.ndarray] = {}
    run_rows = []

    for manifest_path in manifests:
        run_id = manifest_path.parent.name
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        records = []
        for ref_record, query_record in zip(
            manifest["records"]["reference"], manifest["records"]["rescan"]
        ):
            descriptors = []
            hashes = []
            for record in (ref_record, query_record):
                input_path = ROOT / record["input"]
                input_bytes = input_path.read_bytes()
                digest = hashlib.sha256(input_bytes).hexdigest()
                if digest != record["input_sha256"]:
                    raise ValueError(f"frozen input hash mismatch: {input_path}")
                hashes.append(digest)
                if digest not in descriptor_cache:
                    with np.load(input_path) as pack:
                        rgb = pack["rgb"].copy()
                        masks = pack["masks"].astype(bool, copy=True)
                    descriptor_cache[digest] = encoder.encode_masks(rgb, masks)
                descriptors.append(descriptor_cache[digest])
            scores = descriptors[1] @ descriptors[0].T
            selected = np.full(scores.shape[0], -1, dtype=np.int64)
            mutual = np.zeros(scores.shape[0], dtype=bool)
            if min(scores.shape) > 0:
                selected = scores.argmax(axis=1)
                reverse = scores.argmax(axis=0)
                mutual = reverse[selected] == np.arange(len(selected))
            prediction_path = args.output_dir / run_id / (
                f"ref{int(ref_record['frame_id']):06d}_query{int(query_record['frame_id']):06d}_predictions.npz"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_path,
                scores=scores.astype(np.float16),
                query_to_reference=selected,
                mutual_assigned=mutual,
            )
            records.append({
                "reference_frame": int(ref_record["frame_id"]),
                "query_frame": int(query_record["frame_id"]),
                "reference_masks": int(scores.shape[1]),
                "query_masks": int(scores.shape[0]),
                "input_hashes": hashes,
                "predictions": str(prediction_path.relative_to(ROOT)),
            })
        summary = {
            "status": "complete",
            "run_id": run_id,
            "records": records,
            "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "backbone": "timm vit_giant_patch14_dinov2.lvd142m",
            "features": "CrossOver layer-31 DINOv2 value CLS; RGB and masks rotated 270 degrees; 3 mask-box crops; RGB MLP head",
            "identity_targets_read": False,
            "semantic_labels_read": False,
        }
        (args.output_dir / run_id / "inference_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        run_rows.append({"run_id": run_id, "status": "complete", "frame_pairs": len(records)})
        (args.output_dir / "inference_manifest.json").write_text(
            json.dumps({"status": "complete", "relations": run_rows}, indent=2) + "\n"
        )
        print(f"{len(run_rows)}/{len(manifests)} {run_id}: {len(records)} frame pairs", flush=True)


if __name__ == "__main__":
    main()
