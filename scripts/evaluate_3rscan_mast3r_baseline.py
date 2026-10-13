#!/usr/bin/env python3
"""Evaluate official MASt3R reciprocal matches on the WorldCache RGB-D query pool.

The adaptation uses MASt3R's published reciprocal descriptor matcher per
rescan/reference frame pair, then ranks the resulting point candidates by the
model's descriptor confidence. RGB-D geometry and 3RScan annotations are used
only for 3-D consensus and post-hoc scoring. The query and reference 3-D points
remain the 224-pixel, 14-pixel patch-center population used by the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from evaluate_3rscan_instance_correspondence import (
    load,
    patch_world_points,
    read_annotated_mesh,
)
from evaluate_3rscan_pose_free_registration import transform
from evaluate_3rscan_spatial_consensus import consensus


def _image_tensor(rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    image = Image.fromarray(rgb.astype(np.uint8)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return tensor.unsqueeze(0).to(device, non_blocking=True)


def _np(value: torch.Tensor | np.ndarray) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _dense_topk_pairs(pred_q, pred_r, grid, query_lookup, reference_lookup, top_k):
    """Return dense descriptor top-k pairs on the common 16x16 patch grid."""
    y = torch.as_tensor(grid, device=pred_q["desc"].device)
    x = y
    q_desc = pred_q["desc"][0][y[:, None], x[None, :]].reshape(-1, pred_q["desc"].shape[-1])
    r_desc = pred_r["desc"][0][y[:, None], x[None, :]].reshape(-1, pred_r["desc"].shape[-1])
    q_desc = torch.nn.functional.normalize(q_desc.float(), dim=-1)
    r_desc = torch.nn.functional.normalize(r_desc.float(), dim=-1)
    q_conf = pred_q["desc_conf"][0][y[:, None], x[None, :]].reshape(-1).float().clamp_min(0)
    r_conf = pred_r["desc_conf"][0][y[:, None], x[None, :]].reshape(-1).float().clamp_min(0)
    score = (q_desc @ r_desc.T) * torch.sqrt(q_conf[:, None] * r_conf[None, :])

    q_valid = np.flatnonzero(query_lookup >= 0)
    r_valid = np.flatnonzero(reference_lookup >= 0)
    if not len(q_valid) or not len(r_valid):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    k = min(top_k, len(r_valid))
    q_idx = torch.as_tensor(q_valid, device=score.device)
    r_idx = torch.as_tensor(r_valid, device=score.device)
    values, indices = torch.topk(score[q_idx[:, None], r_idx[None, :]], k=k, dim=1)
    q_grid = np.repeat(q_valid, k)
    r_grid = r_valid[_np(indices).reshape(-1)]
    confidence = _np(values).reshape(-1).astype(np.float32, copy=False)
    return query_lookup[q_grid], reference_lookup[r_grid], confidence


def _load_model(mast3r_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(mast3r_root.resolve()))
    from mast3r.fast_nn import fast_reciprocal_NNs, merge_corres
    from mast3r.model import AsymmetricMASt3R

    model = AsymmetricMASt3R.from_pretrained(str(checkpoint)).eval().to(device)

    @torch.no_grad()
    def symmetric_inference(model, view_q, view_r, device):
        shape_q = torch.from_numpy(view_q["true_shape"]).to(device)
        shape_r = torch.from_numpy(view_r["true_shape"]).to(device)
        img_q, img_r = view_q["img"].to(device), view_r["img"].to(device)
        feat_q, feat_r, pos_q, pos_r = model._encode_image_pairs(img_q, img_r, shape_q, shape_r)

        def decode(first, second, pos_first, pos_second, shape_first, shape_second):
            dec_first, dec_second = model._decoder(first, pos_first, second, pos_second)
            with torch.cuda.amp.autocast(enabled=False):
                pred_first = model._downstream_head(1, [x.float() for x in dec_first], shape_first)
                pred_second = model._downstream_head(2, [x.float() for x in dec_second], shape_second)
            return pred_first, pred_second

        qf, rf = decode(feat_q, feat_r, pos_q, pos_r, shape_q, shape_r)
        rr, rq = decode(feat_r, feat_q, pos_r, pos_q, shape_r, shape_q)
        return qf, rf, rr, rq

    def extract_correspondences(features, confidences, subsample=1, device="cpu"):
        paired_matches = []
        paired_confidences = []
        for A, B, QA, QB in (
            (features[0], features[1], confidences[0], confidences[1]),
            (features[3], features[2], confidences[3], confidences[2]),
        ):
            nn_ab = fast_reciprocal_NNs(
                A, B, subsample_or_initxy1=subsample, ret_xy=False,
                device=device, dist="dot", block_size=2**13,
            )
            nn_ba = fast_reciprocal_NNs(
                B, A, subsample_or_initxy1=subsample, ret_xy=False,
                device=device, dist="dot", block_size=2**13,
            )
            idx_a = np.r_[nn_ab[0], nn_ba[1]]
            idx_b = np.r_[nn_ab[1], nn_ba[0]]
            conf_a = _np(QA).reshape(-1)[idx_a]
            conf_b = _np(QB).reshape(-1)[idx_b]
            paired_matches.append((idx_a, idx_b))
            paired_confidences.append(np.sqrt(np.maximum(conf_a, 0) * np.maximum(conf_b, 0)))
        idx_a = np.concatenate([x[0] for x in paired_matches])
        idx_b = np.concatenate([x[1] for x in paired_matches])
        confidences_all = np.concatenate(paired_confidences)
        shape_a, shape_b = features[0].shape[:2], features[1].shape[:2]
        xy_a, xy_b, unique_idx = merge_corres(
            idx_a, idx_b, shape_a, shape_b, ret_xy=True, ret_index=True
        )
        return xy_a, xy_b, confidences_all[unique_idx]

    return model, symmetric_inference, extract_correspondences


def _load_relation_metadata(metadata_path: Path, reference_id: str, rescan_id: str):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    relation = next((row for row in metadata if row.get("reference") == reference_id), None)
    if relation is None:
        raise ValueError(f"reference missing from metadata: {reference_id}")
    rescan = next((row for row in relation.get("scans", []) if row.get("reference") == rescan_id), None)
    if rescan is None:
        raise ValueError(f"rescan missing from relation metadata: {rescan_id}")
    ambiguous: set[int] = set()
    for group in relation.get("ambiguity", []):
        for entry in group:
            ambiguous.update(int(entry[k]) for k in ("instance_source", "instance_target") if k in entry)
    links: dict[int, int] = {}
    for entry in rescan.get("rigid", []):
        if isinstance(entry, dict):
            query_id, reference_object_id = int(entry["instance_rescan"]), int(entry["instance_reference"])
            if query_id not in ambiguous and reference_object_id not in ambiguous:
                links[query_id] = reference_object_id
    return links


def _prepare_side(pair_run: Path, entries: list[dict], annotation_root: Path, scan_id: str):
    tree, object_ids_mesh, global_ids_mesh, semantic_ids_mesh, object_labels = read_annotated_mesh(
        annotation_root, scan_id
    )
    frame_data = []
    point_parts, object_parts, global_parts, semantic_parts, frame_ids = [], [], [], [], []
    for entry in entries:
        observation = load(pair_run / entry["path"])
        points, valid = patch_world_points(observation, patch_size=14)
        points_np = points.numpy().astype(np.float64)
        _, nearest = tree.query(points_np, k=1)
        offset = sum(len(part) for part in point_parts)
        valid_flat = np.flatnonzero(valid.numpy())
        lookup = np.full(16 * 16, -1, dtype=np.int64)
        lookup[valid_flat] = np.arange(offset, offset + len(points_np), dtype=np.int64)
        frame_data.append({
            "rgb": observation["rgb"].numpy(),
            "lookup": lookup,
            "frame_id": int(entry["frame_id"]),
        })
        point_parts.append(points_np)
        object_parts.append(object_ids_mesh[nearest])
        global_parts.append(global_ids_mesh[nearest])
        semantic_parts.append(semantic_ids_mesh[nearest])
        frame_ids.append(np.full(len(points_np), int(entry["frame_id"]), dtype=np.int64))
    return {
        "frames": frame_data,
        "points": np.concatenate(point_parts),
        "object_ids": np.concatenate(object_parts),
        "global_ids": np.concatenate(global_parts),
        "semantic_ids": np.concatenate(semantic_parts),
        "frame_ids": np.concatenate(frame_ids),
        "object_labels": object_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mast3r-root",
        type=Path,
        default=Path(os.environ.get("MAST3R_ROOT", Path.home() / "data/worldcache_models/mast3r_source")),
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "MAST3R_CHECKPOINT",
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        ),
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--match-mode",
        choices=("reciprocal", "dense_topk"),
        default="reciprocal",
        help="official sparse reciprocal matches or a budget-matched dense descriptor top-k diagnostic",
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--candidate-cache",
        type=Path,
        default=None,
        help="optional compressed cache of the exact ranked geometry candidates and scoring labels",
    )
    args = parser.parse_args()
    if args.image_size % 16 or args.image_size < 256:
        raise ValueError("MASt3R image size must be >=256 and divisible by its 16-pixel patch size")
    if args.top_k < 1 or args.iterations < 1:
        raise ValueError("top-k and iterations must be positive")
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists() and not checkpoint_path.is_dir() and not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not args.mast3r_root.is_dir():
        raise FileNotFoundError(args.mast3r_root)
    checkpoint_reference = (
        checkpoint_path.name if checkpoint_path.exists() else str(args.checkpoint)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("pair cache must retain native scan coordinates")
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    reference = _prepare_side(args.pair_run, manifest["reference"], args.annotation_root, config["reference"])
    query = _prepare_side(args.pair_run, manifest["rescan"], args.annotation_root, config["rescan"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, symmetric_inference, extract_correspondences = _load_model(args.mast3r_root, checkpoint_path if checkpoint_path.exists() else checkpoint_path, device)
    # Patch centers at 7,21,...,217 pixels in the 224 input scale to the
    # corresponding center samples in MASt3R's 512-pixel output map.
    grid = np.arange(args.image_size // 32, args.image_size, args.image_size // 16, dtype=np.int64)
    if len(grid) != 16:
        raise ValueError(f"unexpected MASt3R descriptor sampling grid: {grid}")
    candidate_lists: list[list[tuple[int, float]]] = [[] for _ in range(len(query["points"]))]
    dummy_index = len(reference["points"])

    for qframe_index, qframe in enumerate(query["frames"]):
        q_tensor = _image_tensor(qframe["rgb"], args.image_size, device)
        q_lookup = qframe["lookup"]
        for rframe in reference["frames"]:
            r_tensor = _image_tensor(rframe["rgb"], args.image_size, device)
            shape = np.asarray([[args.image_size, args.image_size]], dtype=np.int32)
            view_q = {"img": q_tensor, "true_shape": shape}
            view_r = {"img": r_tensor, "true_shape": shape}
            predictions = symmetric_inference(model, view_q, view_r, device=device)
            ys = torch.as_tensor(grid, device=device)
            xs = torch.as_tensor(grid, device=device)
            descriptors, confidences = [], []
            for pred in predictions:
                descriptors.append(pred["desc"][0][ys][:, xs].detach().cpu())
                confidences.append(pred["desc_conf"][0][ys][:, xs].detach().cpu())
            if args.match_mode == "dense_topk":
                dense_pairs = []
                for pred_q, pred_r in ((predictions[0], predictions[1]), (predictions[3], predictions[2])):
                    dense_pairs.append(_dense_topk_pairs(
                        pred_q, pred_r, grid, q_lookup, rframe["lookup"], args.top_k
                    ))
                for q_global, r_global, match_confidence in dense_pairs:
                    for qi, ri, conf in zip(q_global, r_global, match_confidence):
                        candidate_lists[int(qi)].append((int(ri), float(conf)))
            else:
                matches = extract_correspondences(
                    descriptors, confidences, subsample=1, device=device
                )
                q_xy, r_xy, match_confidence = map(_np, matches)
                q_flat = q_xy[:, 1].astype(np.int64) * 16 + q_xy[:, 0].astype(np.int64)
                r_flat = r_xy[:, 1].astype(np.int64) * 16 + r_xy[:, 0].astype(np.int64)
                q_global = q_lookup[q_flat]
                r_global = rframe["lookup"][r_flat]
                valid = (q_global >= 0) & (r_global >= 0)
                for qi, ri, conf in zip(q_global[valid], r_global[valid], match_confidence[valid]):
                    candidate_lists[int(qi)].append((int(ri), float(conf)))
            del predictions, descriptors, confidences, r_tensor
        del q_tensor
        if (qframe_index + 1) % 5 == 0:
            print(f"matched rescan frames {qframe_index + 1}/{len(query['frames'])}", flush=True)

    actual_candidates: list[list[int]] = []
    candidate_indices = np.full((len(query["points"]), args.top_k), dummy_index, dtype=np.int64)
    candidate_count = np.zeros(len(query["points"]), dtype=np.int64)
    for qi, candidates in enumerate(candidate_lists):
        # Stable confidence ordering, then deduplicate repeated proposals for
        # the same reference observation.
        ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
        unique = []
        seen = set()
        for ref_index, _confidence in ranked:
            if ref_index not in seen:
                unique.append(ref_index)
                seen.add(ref_index)
            if len(unique) == args.top_k:
                break
        actual_candidates.append(unique)
        candidate_count[qi] = len(unique)
        if unique:
            candidate_indices[qi, :len(unique)] = unique
            candidate_indices[qi, len(unique):] = np.resize(np.asarray(unique, dtype=np.int64), args.top_k - len(unique))

    reference_points = np.concatenate([reference["points"], np.array([[1e6, 1e6, 1e6]])], axis=0)
    reference_object_ids = np.concatenate([reference["object_ids"], np.array([0], dtype=np.int64)])
    reference_global_ids = np.concatenate([reference["global_ids"], np.array([0], dtype=np.int64)])
    reference_semantics = np.concatenate([reference["semantic_ids"], np.array([0], dtype=np.int64)])
    query_semantics = query["semantic_ids"]
    query_global_ids = query["global_ids"]
    query_evaluable = (query_global_ids != 0) & (query_semantics != 0)
    links = _load_relation_metadata(args.metadata, config["reference"], config["rescan"])
    # Preserve the audited shared-object-ID fallback for unchanged instances.
    ambiguous = set()
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    rel = next(row for row in meta if row.get("reference") == config["reference"])
    for group in rel.get("ambiguity", []):
        for entry in group:
            ambiguous.update(int(entry[k]) for k in ("instance_source", "instance_target") if k in entry)
    for object_id in set(reference["object_labels"]) & set(query["object_labels"]):
        if object_id not in ambiguous and reference["object_labels"][object_id] == query["object_labels"][object_id]:
            links.setdefault(object_id, object_id)
    mapped_reference_ids = np.asarray([links.get(int(x), -1) for x in query["object_ids"]], dtype=np.int64)
    physical_evaluable = query_evaluable & (mapped_reference_ids >= 0)

    if args.candidate_cache is not None:
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.candidate_cache,
            candidate_indices=candidate_indices,
            candidate_count=candidate_count,
            query_points=query["points"],
            reference_points=reference_points,
            query_frame_ids=query["frame_ids"],
            query_object_ids=query["object_ids"],
            query_global_ids=query_global_ids,
            query_semantics=query_semantics,
            query_evaluable=query_evaluable,
            physical_evaluable=physical_evaluable,
            mapped_reference_ids=mapped_reference_ids,
            reference_object_ids=reference_object_ids,
            reference_global_ids=reference_global_ids,
            reference_semantics=reference_semantics,
            ground_truth_transform=np.asarray(config["rescan_to_reference"], dtype=np.float64),
            physical_candidate_hit=np.asarray([
                any(reference_object_ids[r] == mapped_reference_ids[q] for r in actual_candidates[q])
                if mapped_reference_ids[q] >= 0 else False
                for q in range(len(actual_candidates))
            ], dtype=bool),
            semantic_candidate_hit=np.asarray([
                any(reference_semantics[r] == query_semantics[q] for r in actual_candidates[q])
                for q in range(len(actual_candidates))
            ], dtype=bool),
        )

    padded_semantics = reference_semantics[candidate_indices]
    padded_objects = reference_object_ids[candidate_indices]
    sem_hit = np.zeros(len(query["points"]), dtype=bool)
    physical_hit = np.zeros(len(query["points"]), dtype=bool)
    for qi, refs in enumerate(actual_candidates):
        if refs:
            sem_hit[qi] = np.any(reference_semantics[refs] == query_semantics[qi])
            if mapped_reference_ids[qi] >= 0:
                physical_hit[qi] = np.any(reference_object_ids[refs] == mapped_reference_ids[qi])

    consensus_eligible = candidate_count > 0
    selected = np.full(len(query["points"]), dummy_index, dtype=np.int64)
    inliers = np.zeros(len(query["points"]), dtype=bool)
    if int(consensus_eligible.sum()) >= 3:
        matrix, selected_eligible, inliers_eligible = consensus(
            query["points"][consensus_eligible], reference_points,
            candidate_indices[consensus_eligible],
            threshold=args.threshold, iterations=args.iterations, seed=args.seed,
            group_ids=query["frame_ids"][consensus_eligible],
        )
        selected[consensus_eligible] = selected_eligible
        inliers[consensus_eligible] = inliers_eligible
    else:
        matrix = np.eye(4, dtype=np.float64)
    np.save(args.output_dir / "predicted_rescan_to_reference.npy", matrix)
    residual = np.linalg.norm(transform(query["points"], matrix) - reference_points[selected], axis=1)
    gt = np.asarray(config["rescan_to_reference"], dtype=np.float64)
    rotation_error = float(np.degrees(Rotation.from_matrix(matrix[:3, :3].T @ gt[:3, :3]).magnitude()))
    translation_error = float(np.linalg.norm(matrix[:3, 3] - gt[:3, 3]))
    evaluated_inliers = inliers & query_evaluable
    physical_correct = reference_object_ids[selected] == mapped_reference_ids
    physical_correct_transfer = inliers & physical_evaluable & physical_correct
    selected_semantic_correct = reference_semantics[selected] == query_semantics
    summary = {
        "method": (
            "MASt3R dense descriptor top-k retrieval with confidence weighting"
            if args.match_mode == "dense_topk"
            else "MASt3R official reciprocal descriptor matching, confidence-ranked candidates"
        ),
        "query_points": int(len(query["points"])),
        "semantic_evaluable_queries": int(query_evaluable.sum()),
        "linked_physical_instance_queries": int(physical_evaluable.sum()),
        "mean_actual_candidates_per_query": float(candidate_count.mean()),
        "queries_with_any_candidate_fraction": float(np.mean(candidate_count > 0)),
        "queries_eligible_for_consensus": int(consensus_eligible.sum()),
        "semantic_candidate_recall_at_k": float((sem_hit & query_evaluable).sum() / query_evaluable.sum()) if query_evaluable.any() else None,
        "physical_instance_candidate_recall_at_k": float((physical_hit & physical_evaluable).sum() / physical_evaluable.sum()) if physical_evaluable.any() else None,
        "consensus_assigned_queries": int(inliers.sum()),
        "semantic_inliers": int(evaluated_inliers.sum()),
        "semantic_inlier_coverage": float(evaluated_inliers.sum() / query_evaluable.sum()) if query_evaluable.any() else None,
        "semantic_accuracy_among_inliers": float(selected_semantic_correct[evaluated_inliers].mean()) if evaluated_inliers.any() else None,
        "physical_correct_transfers": int(physical_correct_transfer.sum()),
        "physical_correct_transfer_coverage": float(physical_correct_transfer.sum() / physical_evaluable.sum()) if physical_evaluable.any() else None,
        "physical_assigned_precision": float(physical_correct_transfer.sum() / (inliers & physical_evaluable).sum()) if (inliers & physical_evaluable).any() else None,
        "scene_rotation_error_deg": rotation_error,
        "scene_translation_error_m": translation_error,
        "strict_scene_pose_success": bool(rotation_error < 5.0 and translation_error < 0.25),
        "ransac_seed": args.seed,
        "ransac_iterations": args.iterations,
        "candidate_top_k": args.top_k,
        "image_size": args.image_size,
        "match_mode": args.match_mode,
        "checkpoint_reference": checkpoint_reference,
        "pair_run": str(args.pair_run),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({
        "query_index": np.arange(len(query["points"])),
        "frame_id": query["frame_ids"],
        "query_object_id": query["object_ids"],
        "query_global_id": query_global_ids,
        "query_semantic_id": query_semantics,
        "semantic_evaluable": query_evaluable,
        "candidate_count": candidate_count,
        "semantic_candidate_hit": sem_hit & query_evaluable,
        "physical_instance_evaluable": physical_evaluable,
        "physical_instance_candidate_hit": physical_hit & physical_evaluable,
        "consensus_inlier": inliers,
        "selected_reference_object_id": reference_object_ids[selected],
        "physical_instance_correct": physical_correct,
        "physical_correct_transfer": physical_correct_transfer,
        "semantic_correct": selected_semantic_correct,
        "spatial_residual_m": residual,
    }).to_csv(args.output_dir / "query_diagnostics.csv", index=False)
    (args.output_dir / "status.json").write_text(json.dumps({
        "status": "complete",
        "method": "official MASt3R reciprocal descriptor matching adapted to the common 3RScan RGB-D patch grid",
        "known_alignment_used_for_matching": False,
        "annotations_used_for_matching": False,
        "actual_reciprocal_candidates_ranked_by": "MASt3R descriptor confidence",
        "padded_candidate_slots": "duplicates of valid candidates; no-candidate queries are excluded from consensus and retained in scoring denominators",
        "queries_eligible_for_consensus": int(consensus_eligible.sum()),
        "same_worldcache_query_population": True,
        "candidate_count_histogram": {str(k): int(v) for k, v in zip(*np.unique(candidate_count, return_counts=True))},
        **{k: summary[k] for k in ("scene_rotation_error_deg", "scene_translation_error_m", "strict_scene_pose_success")},
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
