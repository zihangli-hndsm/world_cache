"""Run the public REMIND tracker on frozen 3RScan RGB/mask pairs.

This is an adapted DINOv2 ViT-G variant, not the authors' DINOv3 result.
All objects share one class so semantic labels cannot constrain association.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
import timm
import yaml
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs/reports/20261009_unseen_oasis_equal_budget"
DEFAULT_VENDOR_ROOT = PROJECT_ROOT / "third_party/remind_reid"


def install_transformers_import_shim() -> None:
    """The vendored DINO extractor imports transformers, but this run uses timm."""
    if "transformers" in sys.modules:
        return

    class UnusedAutoLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("REMIND's Hugging Face loader is unused in this DINOv2 adapter")

    shim = types.ModuleType("transformers")
    shim.AutoImageProcessor = UnusedAutoLoader
    shim.AutoModel = UnusedAutoLoader
    sys.modules["transformers"] = shim


def load_remind_modules(vendor_root: Path):
    if not (vendor_root / "pipeline/reid_pipeline.py").is_file():
        raise FileNotFoundError(f"REMIND checkout not found at {vendor_root}")
    install_transformers_import_shim()
    sys.path.insert(0, str(vendor_root))

    from detection.detection import Detection
    from features.dino_extractor import DinoExtractor
    from memory.memory_store import MemoryStore
    from pipeline.initialization import RuntimeContext, set_seeds
    from pipeline.reid_pipeline import ReIDPipeline

    class TimmDinoV2G(DinoExtractor):
        def load_model(self):
            self.patch_size = 14
            self.model = timm.create_model(
                "vit_giant_patch14_dinov2.lvd142m",
                pretrained=True,
                num_classes=0,
                img_size=224,
            ).to(self.device).eval()

        @torch.inference_mode()
        def extract_patches(self, image_rgb: np.ndarray) -> np.ndarray:
            image = image_rgb.astype(np.float32, copy=False) / 255.0
            mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
            std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
            image = (image - mean[None, None, :]) / std[None, None, :]
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            tokens = self.model.forward_features(tensor)
            if isinstance(tokens, dict):
                tokens = tokens.get("x_norm_patchtokens", tokens.get("last_hidden_state"))
                if tokens is None:
                    raise RuntimeError("unrecognized timm DINOv2 feature dictionary")
            n_patches = (image.shape[0] // self.patch_size) * (image.shape[1] // self.patch_size)
            n_prefix = int(tokens.shape[1] - n_patches)
            patches = tokens[:, n_prefix:, :]
            hp = image.shape[0] // self.patch_size
            wp = image.shape[1] // self.patch_size
            return patches[0].reshape(hp, wp, -1).float().cpu().numpy()

    return Detection, TimmDinoV2G, MemoryStore, RuntimeContext, ReIDPipeline, set_seeds


class FrozenMaskSegmenter:
    """Feed the already-frozen anonymous masks into REMIND's detector interface."""

    class_id_to_name = {0: "all_objects"}

    def __init__(self, Detection):
        self.Detection = Detection
        self.active_masks: np.ndarray | None = None
        self.last_timings_seconds: dict[str, float] = {}

    def segment(self, frame, frame_id: int, timestamp: float) -> list:
        if self.active_masks is None:
            raise RuntimeError("no frozen mask array is active for this frame")
        height, width = frame.shape[:2]
        detections = []
        for index, mask in enumerate(self.active_masks):
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (height, width):
                raise ValueError(f"mask shape {mask.shape} differs from frame {(height, width)}")
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            geom = {"center": (float(xs.mean()), float(ys.mean())), "area": int(len(xs))}
            det = self.Detection(
                detection_id=int(index),
                class_id=0,
                frame_id=int(frame_id),
                timestamp=float(timestamp),
                bbox=(x1, y1, x2, y2),
                mask=mask,
                confidence=1.0,
                geom=geom,
            )
            det.class_name = "all_objects"
            det.original_class_name = "all_objects"
            detections.append(det)
        return detections


def tracking_idf1(rows: list[dict], *, gt_observations: int) -> dict:
    if not rows:
        return {"idf1": 0.0, "idtp": 0, "idfp": 0,
                "idfn": int(gt_observations), "gt_observations": int(gt_observations),
                "pred_observations": 0}
    gt_ids = sorted({int(r["gt_id"]) for r in rows})
    pred_ids = sorted({int(r["pred_id"]) for r in rows})
    gt_index = {value: i for i, value in enumerate(gt_ids)}
    pred_index = {value: i for i, value in enumerate(pred_ids)}
    counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    for row in rows:
        counts[gt_index[int(row["gt_id"])], pred_index[int(row["pred_id"])]] += 1
    rr, cc = linear_sum_assignment(-counts)
    idtp = int(counts[rr, cc].sum())
    n_pred = len(rows)
    idfn = int(gt_observations) - idtp
    idfp = n_pred - idtp
    denom = 2 * idtp + idfp + idfn
    return {"idf1": 2.0 * idtp / denom if denom else 0.0,
            "idtp": idtp, "idfp": idfp, "idfn": idfn,
            "gt_observations": int(gt_observations), "pred_observations": n_pred}


def source_track_mapping(rows: list[dict]) -> dict[int, int]:
    gt_ids = sorted({int(r["gt_id"]) for r in rows})
    track_ids = sorted({int(r["pred_id"]) for r in rows})
    if not gt_ids or not track_ids:
        return {}
    gi = {v: i for i, v in enumerate(gt_ids)}
    ti = {v: i for i, v in enumerate(track_ids)}
    counts = np.zeros((len(gt_ids), len(track_ids)), dtype=np.int64)
    for row in rows:
        counts[gi[int(row["gt_id"])], ti[int(row["pred_id"])]] += 1
    rr, cc = linear_sum_assignment(-counts)
    return {track_ids[int(col)]: gt_ids[int(row)] for row, col in zip(rr, cc) if counts[row, col] > 0}


def source_track_majority_mapping(rows: list[dict]) -> dict[int, int]:
    """Map every source tracker ID to its most frequent GT identity (many-to-one)."""
    counts: dict[int, Counter[int]] = {}
    for row in rows:
        track_id = int(row["pred_id"])
        gt_id = int(row["gt_id"])
        counts.setdefault(track_id, Counter())[gt_id] += 1
    return {
        track_id: min(gt_counts, key=lambda gt_id: (-gt_counts[gt_id], gt_id))
        for track_id, gt_counts in counts.items()
    }


def prepare_config(vendor_root: Path) -> dict:
    cfg = yaml.safe_load((vendor_root / "config/default_config.yaml").read_text())
    cfg["system"]["input_width_size"] = 224
    cfg["runtime"]["device"] = "cuda"
    cfg["timing"]["enabled"] = False
    cfg["timing"]["table"] = False
    cfg["timing"]["detail_keys"] = []
    cfg["timing"]["association_detail_keys"] = []
    cfg["timing"]["update_detail_keys"] = []
    cfg["detector"]["ignored_classes"] = []
    cfg["dino"]["default_patch_size"] = 14
    cfg["dino"]["normalize_embeddings"] = True
    cfg["perception"]["full_center_crop"] = True
    cfg["paths"]["output_dir"] = "/tmp/remind_3rscan_adapter"
    return cfg


def run_relation(run_id: str, relation_dir: Path, vendor_root: Path, output_dir: Path) -> dict:
    Detection, TimmDinoV2G, MemoryStore, RuntimeContext, ReIDPipeline, set_seeds = load_remind_modules(vendor_root)
    manifest = json.loads((relation_dir / "manifest.json").read_text())
    cfg = prepare_config(vendor_root)
    set_seeds(cfg["runtime"].get("seed", 0))
    cfg["paths"]["output_dir"] = str(output_dir / run_id)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    segmenter = FrozenMaskSegmenter(Detection)
    dino = TimmDinoV2G(cfg["dino"], "cuda")
    dino.load_model()
    memory = MemoryStore(cfg, start_object_id=int(cfg.get("memory", {}).get("start_object_id", 0)))
    runtime = RuntimeContext(cfg, "cuda", memory, segmenter, dino, Path(cfg["paths"]["output_dir"]))
    pipeline = ReIDPipeline(runtime)

    source_rows: list[dict] = []
    query_rows: list[dict] = []
    query_mask_rows: list[dict] = []
    all_rows: list[dict] = []
    total_input_sha = hashlib.sha256()
    sequence_index = 0
    source_mask_observations = 0
    query_mask_observations = 0
    query_shared_mask_observations = 0
    source_gt_ids: set[int] = set()
    source_mask_observations_by_id: Counter[int] = Counter()
    source_frames_by_id: Counter[int] = Counter()
    query_mask_observations_by_id: Counter[int] = Counter()

    for record in manifest["records"]["reference"]:
        target_path = Path(record["identity_targets"])
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
        ids = np.load(target_path).astype(np.int64)
        source_mask_observations_by_id.update(int(x) for x in ids.tolist())
        source_frames_by_id.update(int(x) for x in np.unique(ids).tolist())
    for record in manifest["records"]["rescan"]:
        target_path = Path(record["identity_targets"])
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
        ids = np.load(target_path).astype(np.int64)
        query_mask_observations_by_id.update(int(x) for x in ids.tolist())

    for role in ("reference", "rescan"):
        for record in manifest["records"][role]:
            input_path = Path(record["input"])
            target_path = Path(record["identity_targets"])
            if not input_path.is_absolute():
                input_path = PROJECT_ROOT / input_path
            if not target_path.is_absolute():
                target_path = PROJECT_ROOT / target_path
            input_bytes = input_path.read_bytes()
            got_sha = hashlib.sha256(input_bytes).hexdigest()
            if got_sha != record["input_sha256"]:
                raise ValueError(f"frozen RGB/mask input changed: {input_path}")
            total_input_sha.update(bytes.fromhex(got_sha))
            with np.load(input_path) as data:
                rgb = data["rgb"].copy()
                masks = data["masks"].astype(bool, copy=True)
            gt_ids = np.load(target_path).astype(np.int64)
            if len(masks) != len(gt_ids):
                raise ValueError(f"mask/identity target count mismatch: {input_path}")
            if role == "reference":
                source_mask_observations += len(gt_ids)
                source_gt_ids.update(int(x) for x in gt_ids.tolist())
            else:
                query_mask_observations += len(gt_ids)
                query_shared_mask_observations += sum(
                    int(x) in source_gt_ids for x in gt_ids.tolist()
                )
            segmenter.active_masks = masks
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            p_out, a_out, u_out = pipeline.process_frame(
                frame=frame_bgr,
                frame_id=sequence_index,
                timestamp=float(sequence_index),
            )
            detection_to_track: dict[int, int] = {}
            detection_state: dict[int, str] = {}
            detection_score: dict[int, float] = {}
            detection_source: dict[int, str] = {}
            for entry in getattr(u_out, "matches", []) or []:
                detection_to_track[int(entry["det_id"])] = int(entry["object_id"])
                detection_state[int(entry["det_id"])] = "matched"
                if entry.get("score_final") is not None:
                    detection_score[int(entry["det_id"])] = float(entry["score_final"])
                detection_source[int(entry["det_id"])] = str(entry.get("source", ""))
            for entry in getattr(u_out, "created", []) or []:
                detection_to_track[int(entry["det_id"])] = int(entry["object_id"])
                detection_state[int(entry["det_id"])] = "created"
            for entry in getattr(u_out, "ambiguous", []) or []:
                detection_state[int(entry["det_id"])] = "ambiguous"
                if entry.get("best_score") is not None:
                    detection_score[int(entry["det_id"])] = float(entry["best_score"])
                detection_source[int(entry["det_id"])] = str(entry.get("source", ""))
            for entry in getattr(u_out, "provisional", []) or []:
                detection_state[int(entry["det_id"])] = "provisional"
                if entry.get("best_score") is not None:
                    detection_score[int(entry["det_id"])] = float(entry["best_score"])
                detection_source[int(entry["det_id"])] = str(entry.get("context_mode", ""))
            matched_detection_ids = {
                int(x["det_id"]) for x in (getattr(u_out, "matches", []) or [])
            }
            for det_index, gt_id in enumerate(gt_ids.tolist()):
                track_id = detection_to_track.get(int(det_index))
                mask_area = int(np.count_nonzero(masks[det_index]))
                row = {
                    "frame": sequence_index,
                    "gt_id": int(gt_id),
                    "mask_area_pixels": mask_area,
                    "mask_area_fraction": float(mask_area / (rgb.shape[0] * rgb.shape[1])),
                    "tracker_state": detection_state.get(
                        int(det_index), "empty_or_unassigned" if mask_area == 0 else "unresolved"
                    ),
                    "tracker_decision_score": detection_score.get(int(det_index)),
                    "tracker_decision_source": detection_source.get(int(det_index)),
                    "pred_id": int(track_id) if track_id is not None else None,
                    "was_match": int(det_index) in matched_detection_ids,
                }
                if role == "rescan":
                    report = (getattr(a_out, "reports_by_det_id", {}) or {}).get(int(det_index))
                    candidates = getattr(report, "candidates", []) if report is not None else []
                    row["_candidate_scores"] = [
                        {
                            "object_id": int(candidate["object_id"]),
                            "score_sim": float(candidate.get("score_sim", 0.0) or 0.0),
                            "score_final": float(candidate.get("score_final", 0.0) or 0.0),
                        }
                        for candidate in candidates
                        if isinstance(candidate, dict) and candidate.get("object_id") is not None
                    ]
                    query_mask_rows.append(row)
                if track_id is None:
                    continue
                scoring_row = {**row, "pred_id": int(track_id)}
                all_rows.append(scoring_row)
                (source_rows if role == "reference" else query_rows).append(scoring_row)
            sequence_index += 1

    src_map = source_track_mapping(source_rows)
    src_majority_map = source_track_majority_mapping(source_rows)
    query_match_rows = [row for row in query_rows if row["was_match"] and int(row["pred_id"]) in src_map]
    correct_links = sum(int(src_map[int(row["pred_id"])]) == int(row["gt_id"]) for row in query_match_rows)
    query_majority_match_rows = [
        row for row in query_rows
        if row["was_match"] and int(row["pred_id"]) in src_majority_map
    ]
    majority_correct_links = sum(
        int(src_majority_map[int(row["pred_id"])] == int(row["gt_id"]))
        for row in query_majority_match_rows
    )
    for row in query_mask_rows:
        pred_id = row["pred_id"]
        source_gt_id = src_map.get(int(pred_id)) if row["was_match"] and pred_id is not None else None
        candidate_scores = row.pop("_candidate_scores", [])
        for score_key in ("score_sim", "score_final"):
            ranked_candidates = sorted(
                candidate_scores,
                key=lambda item: (-float(item[score_key]), int(item["object_id"])),
            )
            target_ranks = [
                rank
                for rank, item in enumerate(ranked_candidates, start=1)
                if src_map.get(int(item["object_id"])) == int(row["gt_id"])
            ]
            row[f"correct_source_candidate_rank_{score_key}"] = min(target_ranks) if target_ranks else None
            row[f"top_source_gt_id_{score_key}"] = (
                int(src_map[int(ranked_candidates[0]["object_id"])])
                if ranked_candidates and int(ranked_candidates[0]["object_id"]) in src_map
                else None
            )
            majority_target_ranks = [
                rank
                for rank, item in enumerate(ranked_candidates, start=1)
                if src_majority_map.get(int(item["object_id"])) == int(row["gt_id"])
            ]
            row[f"correct_source_candidate_rank_{score_key}_majority"] = (
                min(majority_target_ranks) if majority_target_ranks else None
            )
        row["memory_candidate_count"] = len(candidate_scores)
        row["source_track_candidate_count"] = sum(int(item["object_id"]) in src_map for item in candidate_scores)
        row["shared_identity"] = int(int(row["gt_id"]) in source_gt_ids)
        row["source_reference_mask_count"] = int(source_mask_observations_by_id.get(int(row["gt_id"]), 0))
        row["source_reference_frame_count"] = int(source_frames_by_id.get(int(row["gt_id"]), 0))
        row["query_identity_mask_count"] = int(query_mask_observations_by_id.get(int(row["gt_id"]), 0))
        row["source_track_gt_id"] = int(source_gt_id) if source_gt_id is not None else None
        row["predicted_source_link"] = int(source_gt_id is not None)
        row["correct_source_link"] = int(source_gt_id is not None and source_gt_id == int(row["gt_id"]))
        majority_gt_id = (
            src_majority_map.get(int(pred_id))
            if row["was_match"] and pred_id is not None else None
        )
        row["source_track_gt_id_majority"] = int(majority_gt_id) if majority_gt_id is not None else None
        row["predicted_source_link_majority"] = int(majority_gt_id is not None)
        row["correct_source_link_majority"] = int(
            majority_gt_id is not None and majority_gt_id == int(row["gt_id"])
        )
    query_tracker_assigned_observations = len(query_rows)
    sequence_metrics = tracking_idf1(
        all_rows, gt_observations=source_mask_observations + query_mask_observations
    )
    source_sequence_metrics = tracking_idf1(
        source_rows, gt_observations=source_mask_observations
    )
    query_sequence_metrics = tracking_idf1(
        query_rows, gt_observations=query_mask_observations
    )
    return {
        "run_id": run_id,
        "status": "complete",
        "source_frames": len(manifest["records"]["reference"]),
        "query_frames": len(manifest["records"]["rescan"]),
        "source_mask_observations": source_mask_observations,
        "source_tracker_assigned_observations": len(source_rows),
        "query_mask_observations": query_mask_observations,
        "query_tracker_assigned_observations": query_tracker_assigned_observations,
        "shared_query_mask_observations": query_shared_mask_observations,
        "query_links_to_source_tracks": len(query_match_rows),
        "correct_cross_scan_links": int(correct_links),
        "cross_scan_link_precision": float(correct_links / len(query_match_rows)) if query_match_rows else 0.0,
        "cross_scan_link_coverage_all_query_masks": float(correct_links / query_mask_observations) if query_mask_observations else 0.0,
        "cross_scan_link_coverage_shared_query_masks": float(correct_links / query_shared_mask_observations) if query_shared_mask_observations else 0.0,
        "source_identity_track_map": {str(k): int(v) for k, v in src_map.items()},
        "source_track_majority_identity_map": {str(k): int(v) for k, v in src_majority_map.items()},
        "cross_scan_link_precision_majority_mapping": (
            float(majority_correct_links / len(query_majority_match_rows))
            if query_majority_match_rows else 0.0
        ),
        "cross_scan_link_coverage_all_query_masks_majority_mapping": (
            float(majority_correct_links / query_mask_observations)
            if query_mask_observations else 0.0
        ),
        "query_links_to_source_tracks_majority_mapping": len(query_majority_match_rows),
        "correct_cross_scan_links_majority_mapping": int(majority_correct_links),
        "query_mask_rows": query_mask_rows,
        "sequence_idf1": sequence_metrics,
        "source_only_idf1": source_sequence_metrics,
        "query_only_idf1": query_sequence_metrics,
        "input_digest": total_input_sha.hexdigest(),
        "backbone": "DINOv2 ViT-G/14 final-layer patch tokens from timm, pretrained LVD-142M",
        "tracker": "REMIND public code at f88ea1d5d81da0a8ed28b206df6d4dab48327342",
        "random_seed": int(cfg["runtime"].get("seed", 0)),
        "mask_classes": "all detections assigned to one shared class; GT semantic labels not exposed",
        "metrics_note": "Cross-scan links count only REMIND decisions matched to a source-created identity. Identity IDs are used for scoring after inference.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-relations", type=int, default=0, help="0 runs all relation manifests in INPUT_ROOT")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("the frozen REMIND-style run requires CUDA")
    if int(timm.__version__.split(".")[0]) < 1:
        raise RuntimeError(f"unexpected timm version {timm.__version__}")

    args.input_root = args.input_root.resolve()
    args.vendor_root = args.vendor_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = sorted((args.input_root / "uniform20_inputs").glob("*/manifest.json"))
    if not manifests:
        raise ValueError(f"no frozen relation manifests found under {args.input_root / 'uniform20_inputs'}")
    available_count = len(manifests)
    if args.max_relations > 0:
        manifests = manifests[:args.max_relations]
    if args.max_relations < 0:
        raise ValueError("--max-relations must be zero or positive")

    results = []
    for manifest_path in manifests:
        run_id = manifest_path.parent.name
        result_path = args.output_dir / f"{run_id}.json"
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite {result_path}")
        result = run_relation(run_id, manifest_path.parent, args.vendor_root, args.output_dir / "internal")
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(json.dumps({"run_id": run_id,
                          "query_links_to_source_tracks": result["query_links_to_source_tracks"],
                          "cross_scan_link_precision": result["cross_scan_link_precision"],
                          "cross_scan_link_coverage_all_query_masks": result["cross_scan_link_coverage_all_query_masks"],
                          "sequence_idf1": result["sequence_idf1"]["idf1"]}), flush=True)

    manifest = {
        "status": "complete" if len(results) == available_count else "partial",
        "relations_available": available_count,
        "relations_requested": len(manifests),
        "results": [str((args.output_dir / f"{r['run_id']}.json").relative_to(PROJECT_ROOT)) for r in results],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
