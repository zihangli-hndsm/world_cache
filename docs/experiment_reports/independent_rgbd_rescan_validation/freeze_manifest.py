#!/usr/bin/env python3
"""Freeze the MultiScan pair population from public file metadata only.

This script lists gated archive metadata; it does not download archive contents.
It excludes the previously inspected scene_00069 and records every unordered
scan pair in every remaining room with at least two released scans. The earlier
scan index is the reference and the later index is the query.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
REPO_ID = "3dlg-hcvc/MultiScan"
REVISION = "c62c9aad850a8638ac3e42605926a65707600125"
EXCLUDED_SCENES = {"scene_00069"}


def main() -> None:
    entries = list(HfApi().list_repo_tree(
        REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        path_in_repo="scans",
        recursive=False,
        expand=True,
    ))
    scans_by_scene: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        match = re.fullmatch(r"scans/(scene_\d{5})_(\d{2})\.zip", entry.path)
        if not match:
            continue
        scene_id, scan_index = match.groups()
        scans_by_scene[scene_id].append({
            "scan_id": f"{scene_id}_{scan_index}",
            "scan_index": scan_index,
            "filename": entry.path,
            "size_bytes": entry.size,
            "repo_blob_id": entry.blob_id,
        })

    pairs = []
    excluded = []
    for scene_id in sorted(scans_by_scene):
        scans = sorted(scans_by_scene[scene_id], key=lambda row: row["scan_index"])
        if scene_id in EXCLUDED_SCENES:
            excluded.append({
                "scene_id": scene_id,
                "scan_ids": ";".join(row["scan_id"] for row in scans),
                "reason": "used in projection debugging/pilot; excluded from independent scoring before this evaluation",
            })
            continue
        if len(scans) < 2:
            excluded.append({
                "scene_id": scene_id,
                "scan_ids": ";".join(row["scan_id"] for row in scans),
                "reason": "no repeated scan; cannot form a rescan relation",
            })
            continue
        for reference, query in itertools.combinations(scans, 2):
            pairs.append({
                "scene_id": scene_id,
                "environment_id": scene_id,
                "reference_scan_id": reference["scan_id"],
                "query_scan_id": query["scan_id"],
                "reference_file": reference["filename"],
                "query_file": query["filename"],
                "reference_size_bytes": reference["size_bytes"],
                "query_size_bytes": query["size_bytes"],
                "reference_repo_blob_id": reference["repo_blob_id"],
                "query_repo_blob_id": query["repo_blob_id"],
                "status": "candidate_pending_archive_access_and_fixed_projection_preflight",
            })

    pair_csv = ROOT / "candidate_pairs.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    exclusion_csv = ROOT / "data_exclusions.csv"
    with exclusion_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_id", "scan_ids", "reason"])
        writer.writeheader()
        writer.writerows(excluded)

    candidate_scan_ids = {
        scan_id
        for row in pairs
        for scan_id in (row["reference_scan_id"], row["query_scan_id"])
    }
    candidate_archive_bytes = sum(
        row["size_bytes"]
        for scans in scans_by_scene.values()
        for row in scans
        if row["scan_id"] in candidate_scan_ids
    )

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {
        "status": "PAUSED_BEFORE_EVALUATION_DATA_ACCESS_UNAVAILABLE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "repository": REPO_ID,
            "revision": REVISION,
            "source": "official public repository tree metadata; archive payloads not accessed",
            "scene_count_in_listing": len(scans_by_scene),
            "scan_archive_count_in_listing": sum(map(len, scans_by_scene.values())),
            "candidate_relation_count": len(pairs),
            "candidate_environment_count": len({row["environment_id"] for row in pairs}),
            "unique_candidate_archive_count": len(candidate_scan_ids),
            "unique_candidate_archive_bytes": candidate_archive_bytes,
            "candidate_pair_rule": "all unordered pairs of released scans within each scene, except scene_00069; lower scan index is reference, higher scan index is query",
            "excluded_scene_count": len(excluded),
            "candidate_pairs_file": "candidate_pairs.csv",
            "exclusions_file": "data_exclusions.csv",
        },
        "pre_evaluation_exclusions": [
            {
                "scene_id": scene_id,
                "reason": "used for prior projection debugging and its outcomes are known; not independent held-out data",
            }
            for scene_id in sorted(EXCLUDED_SCENES)
        ],
        "projection_bug_review": {
            "root_cause": "column-major RGB intrinsic matrix was decoded correctly, but projection read principal point from K[2,0]/K[2,1] instead of K[0,2]/K[1,2]",
            "correction_basis": "official MultiScan acquired-stream schema and official reconstruction bridge; no GT, labels, retrieval results, or manually selected successful frames were used to fit a transform",
            "modified_files": [
                "scripts/audit_multiscan_rgbd_projection.py",
                "scripts/diagnose_multiscan_depth_pose_sync.py",
                "tests/test_multiscan_intrinsics.py",
            ],
            "unit_test_command": "PYTHONPATH=scripts conda run --no-capture-output -n worldcache pytest -q tests/test_multiscan_intrinsics.py",
            "unit_test_result": "1 passed (recorded in outputs/reports/rgbd_projection_diagnostic/report.md)",
            "sha256": {
                path: sha256(WORKSPACE / path)
                for path in [
                    "scripts/audit_multiscan_rgbd_projection.py",
                    "scripts/diagnose_multiscan_depth_pose_sync.py",
                    "tests/test_multiscan_intrinsics.py",
                ]
            },
        },
        "model": {
            "architecture": "DINOv2 ViT-G/14",
            "timm_name": "vit_giant_patch14_dinov2.lvd142m",
            "timm_version": "1.0.29",
            "weights_repository": "timm/vit_giant_patch14_dinov2.lvd142m",
            "weights_repository_commit": "dd22bb6485eeb29d67273f76d983061298bdc7f0",
            "weights_filename": "model.safetensors",
            "weights_bytes": 4545971222,
            "weights_sha256": "949077749c7a414f3443a007cf021d085bc11deb0b08959a6e32e170b1e8e359",
            "blocks_1_indexed": [2, 20, 40],
            "input_resolution": [224, 224],
            "weight_file_in_existing_local_cache": True,
            "loaded_or_inference_verified_this_run": False,
        },
        "environment": {
            "conda_environment": "worldcache",
            "python": "3.9.23",
            "torch": "2.5.1+cu124",
            "cuda": "12.4",
            "timm": "1.0.29",
            "numpy": "1.26.4",
            "pillow": "11.3.0",
            "gpu": "NVIDIA A100-SXM4-80GB",
            "gpu_memory_mib": 81920,
            "cuda_available": True,
            "environment_rebuilt_or_packages_installed": False,
        },
        "protocol": {
            "source": "paper/main.tex frozen primary protocol; MultiScan native RGB-D adaptation described here",
            "model_training_or_finetuning": False,
            "matching_inputs": "RGB and native depth/camera calibration only; annotations, instance IDs, annotated meshes and cross-scan GT alignment withheld until predictions are saved",
            "matching_frame_sampling": "up to 20 evenly spaced frame indices per scan, fixed before inference; include endpoints when more than one frame",
            "projection_preflight_sampling": "up to 72 deterministic uniformly spaced frame indices per scan, including endpoints: unique(linspace(0, N-1, min(72, N), dtype=int))",
            "patch_sampling": "every positive finite depth at each 14-pixel patch center after the fixed 224x224 RGB/depth resize; no cross-frame 3D deduplication",
            "descriptor": "per-block L2 normalize blocks 2/20/40, concatenate, L2 normalize",
            "top_k": 20,
            "consensus_distance_m": 0.10,
            "ransac_iterations": 1500,
            "ransac_seed": 17,
            "projection_gate_m": 0.05,
            "projection_gate_rule": "for each scan, median across the fixed 72 frames of frame-level median absolute mesh-hit/native-depth residual must be <= 0.05 m; report all frames and pixel residual fractions",
            "rgb_control": "same query/reference point pools; each 14x14 RGB input patch is represented by its mean RGB vector, L2 normalized; independent top-20 cosine candidates; deterministic, no training",
            "bootstrap_unit": "environment_id; aggregate all relations within environment before resampling",
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20261011,
        },
        "scoring": {
            "candidate_metrics": ["semantic Recall@20", "physical-instance Recall@20", "paired semantic-minus-instance difference", "query coverage"],
            "consensus_metrics": ["accepted physical-instance precision", "correct accepted-query coverage"],
            "pose_metrics": "not reported unless official within-scan camera/mesh pose provenance is independently verified; cross-scan GT alignment is never an inference input",
            "label_scoring": "post-prediction only; unmapped or missing target label is incorrect; absent query physical identity in reference pool contributes zero recall",
        },
        "preflight_blockers": [
            "Only scene_00069 archives are present locally; they are excluded from independent scoring because they were used for prior projection debugging.",
            "The remaining gated scan archives are not locally available and the current runtime has no Hugging Face access token; archive payload access is unavailable.",
            "No inference was started. Candidate projection quality and annotation qualification remain unknown until authorized archives are available.",
        ],
        "randomness": "frame and pair selection deterministic; no sampling seed for the dataset; RANSAC seed and bootstrap seed frozen above",
        "manifest_generator_sha256": sha256(ROOT / "freeze_manifest.py"),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "listing_summary.json").write_text(json.dumps({
        "revision": REVISION,
        "listing_entries": len(entries),
        "scene_count": len(scans_by_scene),
        "scan_archive_count": sum(map(len, scans_by_scene.values())),
        "candidate_relation_count": len(pairs),
        "candidate_environment_count": len({row["environment_id"] for row in pairs}),
        "unique_candidate_archive_count": len(candidate_scan_ids),
        "unique_candidate_archive_bytes": candidate_archive_bytes,
        "excluded_scene_count": len(excluded),
        "payload_downloaded": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_relations": len(pairs),
        "candidate_environments": len({row["environment_id"] for row in pairs}),
        "excluded_scenes": len(excluded),
        "status": manifest["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
