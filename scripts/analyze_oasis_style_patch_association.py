#!/usr/bin/env python3
"""Probe OASIS-style patch-to-mask association with the available DINOv2 model.

This is an algorithm-formulation audit, not an OASIS-Map reproduction: it uses
DINOv2, annotation-derived masks, the project's fixed chronological frame
pairs, and no OASIS system frontend or shared-frame pose information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from evaluate_3rscan_instance_correspondence import preprocess
from score_segment_matcher_population import identity_links, semantics
from worldcache.backbone.dinov2 import extract_block_tokens, load_dinov2


SIMILARITY_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
MATCH_FRACTION_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80)
LAYERS = (2, 20, 40)


def anonymous_features(model, input_path: Path, expected_hash: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"anonymous input hash mismatch: {input_path}")
    with np.load(input_path) as pack:
        image = torch.from_numpy(pack["rgb"].copy())
        masks = pack["masks"].copy()
    captured = extract_block_tokens(model, preprocess(image, device), LAYERS)
    token_blocks = [
        F.normalize(captured[layer][0, model.num_prefix_tokens :], dim=1)
        for layer in LAYERS
    ]
    tokens = F.normalize(torch.cat(token_blocks, dim=1), dim=1)
    side = math.isqrt(tokens.shape[0])
    if side * side != tokens.shape[0]:
        raise ValueError(f"expected square patch grid; got {tokens.shape[0]} tokens")

    # Assign each patch by its center pixel. These anonymous instance masks
    # never expose their scan-local identity IDs to feature inference.
    centers = np.floor((np.arange(side) + 0.5) * 224 / side).astype(int)
    patch_labels = np.zeros((side, side), dtype=np.int32)
    if len(masks):
        sampled = masks[:, centers[:, None], centers[None, :]]
        overlap = sampled.sum(axis=0)
        if np.any(overlap > 1):
            raise ValueError("instance masks overlap at a patch center")
        for index, mask in enumerate(sampled, start=1):
            patch_labels[mask] = index
    return tokens.to(dtype=torch.float16, device="cpu").numpy(), patch_labels.reshape(-1)


def union_correspondence_scores(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    reference_patch_labels: np.ndarray,
    query_patch_labels: np.ndarray,
    reference_masks: int,
    query_masks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cuda")
    similarity = (
        torch.as_tensor(reference_features, device=device, dtype=torch.float32)
        @ torch.as_tensor(query_features, device=device, dtype=torch.float32).T
    ).cpu().numpy()
    forward = similarity.argmax(axis=1)  # reference -> query
    backward = similarity.argmax(axis=0)  # query -> reference
    edges = np.unique(
        np.concatenate(
            [
                np.column_stack((np.arange(len(forward)), forward)),
                np.column_stack((backward, np.arange(len(backward)))),
            ],
            axis=0,
        ),
        axis=0,
    )

    ref_labels = reference_patch_labels[edges[:, 0]]
    query_labels = query_patch_labels[edges[:, 1]]
    ref_object_edges = ref_labels > 0
    outgoing = np.bincount(ref_labels[ref_object_edges] - 1, minlength=reference_masks)
    paired = ref_object_edges & (query_labels > 0)
    counts = np.zeros((reference_masks, query_masks), dtype=np.int32)
    confidence_sum = np.zeros((reference_masks, query_masks), dtype=np.float32)
    if paired.any():
        rows = ref_labels[paired] - 1
        cols = query_labels[paired] - 1
        np.add.at(counts, (rows, cols), 1)
        np.add.at(confidence_sum, (rows, cols), similarity[edges[paired, 0], edges[paired, 1]])
    fraction = counts / np.maximum(outgoing[:, None], 1)
    confidence = confidence_sum / np.maximum(counts, 1)
    rows, cols = linear_sum_assignment(-fraction) if fraction.size else (np.array([], int), np.array([], int))
    return fraction.astype(np.float32), confidence.astype(np.float32), rows.astype(np.int32), cols.astype(np.int32)


def infer(args: argparse.Namespace, output: Path, selected: list[dict]) -> list[dict]:
    device = torch.device("cuda")
    model = load_dinov2("giant", 224, device)
    feature_cache: dict[str, np.ndarray] = {}
    patch_label_cache: dict[str, np.ndarray] = {}
    manifest = []
    for relation in selected:
        # Inputs are owned by the frozen 110-relation population. The supplied
        # input_root is recorded in the output config and used to find manifests.
        root = Path(args.input_root)
        source_manifest = json.loads((root / "uniform20_inputs" / relation["run_id"] / "manifest.json").read_text())
        prediction_dir = output / "anonymous_patch_predictions" / relation["run_id"]
        prediction_dir.mkdir(parents=True, exist_ok=True)
        relation_rows = []
        for reference, query in zip(source_manifest["records"]["reference"], source_manifest["records"]["rescan"]):
            feature_pair = []
            for record in (reference, query):
                digest = record["input_sha256"]
                if digest not in feature_cache:
                    features, patch_labels = anonymous_features(model, Path(record["input"]), digest, device)
                    feature_cache[digest] = features
                    patch_label_cache[digest] = patch_labels
                feature_pair.append(feature_cache[digest])
            fractions, confidence, ref_rows, query_cols = union_correspondence_scores(
                feature_pair[0], feature_pair[1],
                patch_label_cache[reference["input_sha256"]],
                patch_label_cache[query["input_sha256"]],
                int(reference["masks"]), int(query["masks"]),
            )
            name = f"ref{reference['frame_id']:06d}_query{query['frame_id']:06d}"
            path = prediction_dir / f"{name}_anonymous_patch_scores.npz"
            np.savez_compressed(path, match_fraction=fractions, mean_confidence=confidence,
                                assigned_reference=ref_rows, assigned_query=query_cols)
            relation_rows.append(dict(reference_frame=reference["frame_id"], query_frame=query["frame_id"],
                                       reference_masks=int(reference["masks"]), query_masks=int(query["masks"]),
                                       prediction=str(path), input_hashes=[reference["input_sha256"], query["input_sha256"]]))
        row = dict(run_id=relation["run_id"], status="complete", frame_pairs=len(relation_rows), records=relation_rows)
        manifest.append(row)
        (output / "anonymous_inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"inferred {len(manifest)}/{len(selected)} relations: {relation['run_id']}", flush=True)
    del model, feature_cache, patch_label_cache
    torch.cuda.empty_cache()
    return manifest


def score(args: argparse.Namespace, output: Path, selected: list[dict], inference_manifest: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(args.input_root)
    metadata = json.loads((Path(args.data_root) / "3RScan.json").read_text())
    relation_info = {r["run_id"]: r for r in selected}
    per_group = []
    for relation_result in inference_manifest:
        relation = relation_info[relation_result["run_id"]]
        reference_semantics, reference_names = semantics(Path(args.data_root), relation["reference_scan"])
        query_semantics, query_names = semantics(Path(args.data_root), relation["rescan_scan"])
        links, _ = identity_links(metadata, relation["reference_scan"], relation["rescan_scan"], reference_names, query_names)
        if len(set(links.values())) != len(links):
            raise ValueError("identity mapping has collisions")
        source_manifest = json.loads((root / "uniform20_inputs" / relation["run_id"] / "manifest.json").read_text())["records"]
        reference_inputs = {x["frame_id"]: x for x in source_manifest["reference"]}
        query_inputs = {x["frame_id"]: x for x in source_manifest["rescan"]}

        # Identity annotations are opened only after every anonymous prediction
        # for the selected population has been persisted and manifested.
        for record in relation_result["records"]:
            reference_record = reference_inputs[record["reference_frame"]]
            query_record = query_inputs[record["query_frame"]]
            reference_ids = np.load(reference_record["identity_targets"])
            query_ids = np.load(query_record["identity_targets"])
            qs = np.asarray([query_semantics.get(int(x), 0) for x in query_ids])
            rs = np.asarray([reference_semantics.get(int(x), 0) for x in reference_ids])
            expected = np.asarray([links.get(int(x), -1) for x in query_ids])
            semantic_evaluable = qs > 0
            linked = semantic_evaluable & (expected > 0)

            with np.load(record["prediction"]) as prediction:
                fractions = prediction["match_fraction"]
                confidence = prediction["mean_confidence"]
                ref_rows = prediction["assigned_reference"]
                query_cols = prediction["assigned_query"]
            if fractions.shape != (len(reference_ids), len(query_ids)) or confidence.shape != fractions.shape:
                raise ValueError("prediction mask dimensions differ from withheld identity targets")

            for fraction_threshold in MATCH_FRACTION_THRESHOLDS:
                for confidence_threshold in SIMILARITY_THRESHOLDS:
                    assigned = np.full(len(query_ids), -1, np.int64)
                    for ri, qi in zip(ref_rows, query_cols):
                        if (fractions[ri, qi] >= fraction_threshold
                                and confidence[ri, qi] >= confidence_threshold):
                            assigned[qi] = ri
                    has_assignment = assigned >= 0
                    mapped = np.full(len(query_ids), -1, np.int64)
                    selected_semantic = np.zeros(len(query_ids), np.int64)
                    mapped[has_assignment] = reference_ids[assigned[has_assignment]]
                    selected_semantic[has_assignment] = rs[assigned[has_assignment]]
                    identity_correct = has_assignment & linked & (mapped == expected)
                    semantic_correct = has_assignment & semantic_evaluable & (selected_semantic == qs)
                    per_group.append(dict(
                        run_id=relation["run_id"], environment_id=relation["environment_id"],
                        reference_frame=record["reference_frame"], query_frame=record["query_frame"],
                        match_fraction_threshold=fraction_threshold,
                        confidence_threshold=confidence_threshold,
                        raw_queries=len(query_ids), raw_assigned=int(has_assignment.sum()),
                        semantic_queries=int(semantic_evaluable.sum()),
                        semantic_assigned=int((has_assignment & semantic_evaluable).sum()),
                        semantic_correct=int(semantic_correct.sum()),
                        linked_queries=int(linked.sum()), linked_assigned=int((has_assignment & linked).sum()),
                        linked_correct=int(identity_correct.sum()),
                        # Match the canonical segment-comparison definition:
                        # semantic correctness on the linked-query subset is
                        # distinct from identity correctness.
                        linked_semantic_correct=int((linked & has_assignment & (selected_semantic == qs)).sum()),
                    ))
    frame = pd.DataFrame(per_group)
    count_cols = [c for c in frame.columns if c not in {"run_id", "environment_id", "reference_frame", "query_frame", "match_fraction_threshold", "confidence_threshold"}]
    env = frame.groupby(["environment_id", "match_fraction_threshold", "confidence_threshold"])[count_cols].sum().reset_index()
    env["semantic_coverage"] = env.semantic_correct / env.semantic_queries.replace(0, np.nan)
    env["semantic_precision"] = env.semantic_correct / env.semantic_assigned.replace(0, np.nan)
    env["linked_coverage"] = env.linked_correct / env.linked_queries.replace(0, np.nan)
    env["linked_precision"] = env.linked_correct / env.linked_assigned.replace(0, np.nan)
    env["linked_semantic_precision"] = env.linked_semantic_correct / env.linked_assigned.replace(0, np.nan)
    env.to_csv(output / "per_environment_thresholds.csv", index=False)
    macro = env.groupby(["match_fraction_threshold", "confidence_threshold"])[
        ["semantic_coverage", "semantic_precision", "linked_coverage", "linked_precision", "linked_semantic_precision"]
    ].mean().reset_index()
    pooled = frame.groupby(["match_fraction_threshold", "confidence_threshold"])[count_cols].sum().reset_index()
    for metric, numerator, denominator in [
        ("semantic_coverage", "semantic_correct", "semantic_queries"),
        ("semantic_precision", "semantic_correct", "semantic_assigned"),
        ("linked_coverage", "linked_correct", "linked_queries"),
        ("linked_precision", "linked_correct", "linked_assigned"),
        ("linked_semantic_precision", "linked_semantic_correct", "linked_assigned"),
    ]:
        pooled[metric] = pooled[numerator] / pooled[denominator].replace(0, np.nan)
    summary = macro.merge(pooled, on=["match_fraction_threshold", "confidence_threshold"], suffixes=("_macro", "_pooled"))
    summary.to_csv(output / "threshold_summary.csv", index=False)
    frame.to_csv(output / "per_frame_thresholds.csv", index=False)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("outputs/reports/20261008_segmast3r_validation"))
    parser.add_argument("--data-root", type=Path, default=Path.home() / "data/worldcache_datasets/3rscan")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reports/20261009_oasis_style_dinov2"))
    parser.add_argument("--max-relations", type=int, default=0,
                        help="inference-only feasibility run; no labels are opened")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((args.input_root / "frozen_segment_population.json").read_text())
    selected = frozen["population"][: args.max_relations] if args.max_relations else frozen["population"]
    config = dict(
        status="inference_only" if args.max_relations else "full_population",
        population_relations=len(selected),
        max_relations=args.max_relations,
        input_population_sha256=hashlib.sha256((args.input_root / "frozen_segment_population.json").read_bytes()).hexdigest(),
        backbone="DINOv2 ViT-Giant, concatenated normalized patch tokens from blocks 2,20,40",
        correspondence="union of forward and backward nearest patch matches, deduplicated as in the published OASIS-Map formulation",
        aggregation="one-to-one Hungarian assignment maximizing object-mask match fraction; accept if match fraction and mean cosine confidence both clear the declared sweep thresholds",
        mask_projection="annotation-derived anonymous image masks sampled at patch centers",
        caveats=[
            "Not an OASIS-Map reproduction: DINOv2 replaces DINOv3 and the OASIS system frontend is not run.",
            "Uses privileged annotation-derived masks, not SAM2 predictions.",
            "Uses the frozen chronological-rank frame pairs, not pose-selected co-visible pairs.",
            "Uses no cross-session pose or alignment at correspondence inference; identity targets are read only after anonymous predictions are manifested.",
            "Threshold surface is descriptive and not a multiplicity-corrected significance analysis.",
        ],
    )
    (output / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.score_only:
        manifest = json.loads((output / "anonymous_inference_manifest.json").read_text())
        if len(manifest) != len(frozen["population"]):
            raise ValueError("score-only requires the complete frozen population")
        _, summary = score(args, output, frozen["population"], manifest)
    else:
        inference_manifest = infer(args, output, selected)
        if args.max_relations:
            print("inference-only feasibility run completed; labels were not opened")
            return
        _, summary = score(args, output, selected, inference_manifest)
    config["status"] = "complete"
    (output / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
