"""Audit 3RScan units and global pose transforms before descriptor evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from worldcache.datasets.threerscan import ThreeRScanPair


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def transform_report(matrix: np.ndarray) -> dict[str, object]:
    rotation = matrix[:3, :3]
    return {
        "translation": matrix[:3, 3].tolist(),
        "translation_norm": float(np.linalg.norm(matrix[:3, 3])),
        "rotation_orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "bottom_row": matrix[3].tolist(),
    }


def scan_report(sequence, global_transform=None, sample_count: int = 10) -> dict[str, object]:
    with sequence._open() as archive:
        prefix = sequence._prefix(archive)
        color_intrinsic, depth_intrinsic, depth_extrinsic, depth_shift, has_depth_intrinsic = sequence._info(archive, prefix)
    frame_ids = sequence.frame_ids()
    translations = []
    depths = []
    shapes = []
    for frame_id in frame_ids[:sample_count]:
        frame = sequence.read_frame(frame_id, global_transform=global_transform)
        translations.append(frame["camera_to_world"][:3, 3])
        valid_depth = frame["depth"][np.isfinite(frame["depth"]) & (frame["depth"] > 0)]
        if len(valid_depth):
            depths.extend(valid_depth[:: max(1, len(valid_depth) // 1000)].tolist())
        shapes.append({"rgb": list(frame["rgb"].shape), "depth": list(frame["depth"].shape)})
    translations_array = np.asarray(translations)
    depths_array = np.asarray(depths)
    return {
        "scan_id": sequence.scan_id,
        "frame_count": len(frame_ids),
        "sampled_frame_ids": frame_ids[:sample_count],
        "color_intrinsics": color_intrinsic.tolist(),
        "depth_intrinsics": depth_intrinsic.tolist(),
        "depth_extrinsic": depth_extrinsic.tolist(),
        "depth_shift_original": depth_shift,
        "depth_unit_original": "millimetres encoded by integer depth / m_depthShift",
        "depth_unit_internal": "m",
        "pose_translation_unit_original": "m (validated by metric depth/trajectory scale)",
        "pose_translation_unit_internal": "m",
        "pose_translation_norm_range_m": [float(np.min(np.linalg.norm(translations_array, axis=1))), float(np.max(np.linalg.norm(translations_array, axis=1)))],
        "valid_depth_range_m": [float(np.min(depths_array)), float(np.max(depths_array))] if len(depths_array) else [None, None],
        "color_depth_intrinsics_present": bool(has_depth_intrinsic),
        "sample_shapes": shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rescan", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()
    pair = ThreeRScanPair.from_metadata(args.root, args.metadata, args.reference, args.rescan)
    report = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset": "3RScan",
        "reference_scan": args.reference,
        "query_scan": args.rescan,
        "depth_unit": "m",
        "pose_translation_unit": "m",
        "scene_transform_translation_unit_original": "mm",
        "scene_transform_translation_unit_internal": "m",
        "internal_canonical_units": "metres",
        "rescan_to_reference": transform_report(pair.rescan_to_reference),
        "reference": scan_report(pair.reference, sample_count=args.sample_count),
        "rescan": scan_report(pair.rescan, global_transform=pair.rescan_to_reference, sample_count=args.sample_count),
        "assertions": {
            "scene_transform_bottom_row_is_homogeneous": bool(np.allclose(pair.rescan_to_reference[3], [0, 0, 0, 1], atol=1e-5)),
            "scene_transform_rotation_is_orthonormal": bool(np.linalg.norm(pair.rescan_to_reference[:3, :3].T @ pair.rescan_to_reference[:3, :3] - np.eye(3)) < 1e-3),
            "scene_transform_rotation_det_positive": bool(np.linalg.det(pair.rescan_to_reference[:3, :3]) > 0.99),
        },
    }
    if not all(report["assertions"].values()):
        raise AssertionError(json.dumps(report["assertions"], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
