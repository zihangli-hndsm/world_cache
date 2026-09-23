"""Migrate cached rescan poses from reference coordinates to native frames.

Older manifests were generated with ``read_frame(..., global_transform)`` and
then every evaluator applied the hidden inverse transform before matching.
This migration makes the coordinate convention explicit and removes that
evaluation-time dependency: rescan NPZ files store the original scan-local
camera-to-world pose, while ``rescan_to_reference`` remains metadata for
post-hoc scoring only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def migrate(pair_run: Path, write: bool) -> dict[str, object]:
    config_path = pair_run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    convention = config.get("coordinate_convention", "reference_frame_rescan_stored")
    if convention == "native_scan_frames":
        return {"pair": pair_run.name, "status": "already_native", "frames": 0}
    if convention != "reference_frame_rescan_stored":
        raise ValueError(f"{pair_run}: unknown coordinate convention {convention!r}")
    inverse = np.linalg.inv(np.asarray(config["rescan_to_reference"], dtype=np.float64))
    manifest = json.loads((pair_run / "manifest.json").read_text(encoding="utf-8"))
    frame_count = 0
    for entry in manifest["rescan"]:
        path = pair_run / entry["path"]
        with np.load(path) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays["camera_to_world"] = (inverse @ arrays["camera_to_world"]).astype(np.float32)
        if write:
            np.savez_compressed(path, **arrays)
        frame_count += 1
    config["coordinate_convention"] = "native_scan_frames"
    if write:
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "pair": pair_run.name,
        "status": "migrated" if write else "would_migrate",
        "frames": frame_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, nargs="+", required=True)
    parser.add_argument("--write", action="store_true", help="apply the reversible NPZ/config migration")
    args = parser.parse_args()
    results = [migrate(path, args.write) for path in args.pair_run]
    print(json.dumps({"status": "complete", "write": args.write, "pairs": results}, indent=2))


if __name__ == "__main__":
    main()
