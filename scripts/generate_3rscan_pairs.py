"""Convert a small official 3RScan reference/rescan pair to WorldCache NPZs."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from worldcache.datasets.threerscan import ThreeRScanPair


def resize_observation(observation: dict[str, np.ndarray], resolution: int) -> dict[str, np.ndarray]:
    color = np.asarray(Image.fromarray(observation["rgb"]).resize((resolution, resolution), Image.Resampling.BILINEAR))
    depth = np.asarray(Image.fromarray(observation["depth"]).resize((resolution, resolution), Image.Resampling.NEAREST), dtype=np.float32)
    height, width = observation["depth"].shape
    intrinsics = observation["intrinsics"].copy()
    intrinsics[0] *= resolution / width
    intrinsics[1] *= resolution / height
    return {**observation, "rgb": color, "depth": depth, "intrinsics": intrinsics}


def save(path: Path, observation: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in observation.items() if key != "frame_id"}
    np.savez_compressed(path, **arrays)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rescan", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--reference-frames", type=int, default=8)
    parser.add_argument("--rescan-frames", type=int, default=8)
    args = parser.parse_args()
    if args.reference_frames < 1 or args.rescan_frames < 1:
        raise ValueError("frame counts must be positive")
    pair = ThreeRScanPair.from_metadata(args.root, args.metadata, args.reference, args.rescan)
    ref_ids, cur_ids = pair.reference.frame_ids()[: args.reference_frames], pair.rescan.frame_ids()[: args.rescan_frames]
    if not ref_ids or not cur_ids:
        raise ValueError("selected pair has no RGB frames")

    records = {"reference": [], "rescan": []}
    for frame_id in ref_ids:
        relative = Path("reference") / f"frame_{frame_id:06d}.npz"
        save(args.output_dir / relative, resize_observation(pair.reference.read_frame(frame_id), args.resolution))
        records["reference"].append({"frame_id": frame_id, "path": str(relative)})
    for frame_id in cur_ids:
        relative = Path("rescan") / f"frame_{frame_id:06d}.npz"
        # Keep the rescan in its native scan coordinate system.  The hidden
        # scene transform belongs in config.json for post-hoc evaluation only;
        # it must not be needed to prepare points for matching.
        save(args.output_dir / relative, resize_observation(pair.rescan.read_frame(frame_id), args.resolution))
        records["rescan"].append({"frame_id": frame_id, "path": str(relative)})

    config = {
        "dataset": "3rscan",
        "reference": args.reference,
        "rescan": args.rescan,
        "resolution": args.resolution,
        "reference_frames": len(ref_ids),
        "rescan_frames": len(cur_ids),
        "rescan_to_reference": pair.rescan_to_reference.tolist(),
        "coordinate_convention": "native_scan_frames",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "complete", **config}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "reference_frames": len(ref_ids), "rescan_frames": len(cur_ids)}, indent=2))


if __name__ == "__main__":
    main()
