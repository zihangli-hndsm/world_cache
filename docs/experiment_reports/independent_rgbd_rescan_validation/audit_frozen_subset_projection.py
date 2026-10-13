#!/usr/bin/env python3
"""Run the frozen 72-frame projection gate for every downloaded subset scan."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from audit_multiscan_rgbd_projection import REVISION, audit_scan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=72)
    args = parser.parse_args()

    subset = json.loads((REPORT_DIR / "quota_subset_manifest.json").read_text())
    state = json.loads((REPORT_DIR / "download_state.json").read_text())
    output_dir = REPORT_DIR / "projection_preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {item["filename"]: item for item in subset["archives"]}
    missing = sorted(set(required) - set(state["downloaded_archives"]))
    if missing:
        raise SystemExit(f"Frozen subset archive(s) missing from verified download state: {missing[:5]}")

    scan_rows = []
    frame_rows = []
    for filename, item in required.items():
        scan_id = Path(filename).stem
        cache_path = ROOT / state["downloaded_archives"][filename]["cache_path"]
        per_scan_path = output_dir / f"{scan_id}.json"
        if per_scan_path.exists():
            cached = json.loads(per_scan_path.read_text())
            if cached.get("dataset_revision") != REVISION or cached.get("sample_count") != args.sample_count or cached.get("archive") != str(cache_path):
                raise SystemExit(f"Existing preflight output does not match frozen input/config: {per_scan_path}")
            result = cached["result"]
            elapsed = cached["wall_seconds"]
        else:
            started = time.monotonic()
            result = audit_scan(cache_path, scan_id, args.sample_count)
            elapsed = time.monotonic() - started
            # Object IDs are only used by this geometric preflight helper and are
            # not needed in its deliverable; omit per-face label summaries.
            for frame in result["frames"]:
                frame.pop("face_object_id_counts_at_overlap", None)
            wrapped = {
                "dataset_revision": REVISION,
                "sample_count": args.sample_count,
                "archive": str(cache_path),
                "archive_size_bytes": item["size_bytes"],
                "repo_blob_id": item["repo_blob_id"],
                "wall_seconds": elapsed,
                "result": result,
            }
            per_scan_path.write_text(json.dumps(wrapped, indent=2) + "\n")
            print(f"saved {scan_id}: median_abs={result['median_abs_depth_residual_m_across_frames']} m; {elapsed:.1f}s", flush=True)

        frame_count = result["rgbd_frame_count"]
        depth_w, depth_h = result["depth_resolution"]
        scan_rows.append({
            "scene_id": item["scene_id"],
            "scan_id": scan_id,
            "archive": filename,
            "archive_size_bytes": item["size_bytes"],
            "rgbd_frame_count": frame_count,
            "sampled_frame_count": len(result["sampled_frame_ids"]),
            "sampled_frame_ids": ";".join(map(str, result["sampled_frame_ids"])),
            "depth_resolution": f"{depth_w}x{depth_h}",
            "median_valid_depth_fraction": sum(f["valid_depth_pixels"] / (depth_w * depth_h) for f in result["frames"]) / max(1, len(result["frames"])),
            "median_mesh_hit_fraction_of_valid_depth": result["median_mesh_hit_fraction_of_valid_depth"],
            "median_abs_depth_residual_m_across_frames": result["median_abs_depth_residual_m_across_frames"],
            "gate_5cm_pass": result["median_abs_depth_residual_m_across_frames"] is not None and result["median_abs_depth_residual_m_across_frames"] <= 0.05,
            "wall_seconds": elapsed,
        })
        for frame in result["frames"]:
            frame_rows.append({
                "scene_id": item["scene_id"],
                "scan_id": scan_id,
                "frame_id": frame["frame_id"],
                "valid_depth_pixels": frame["valid_depth_pixels"],
                "valid_depth_fraction": frame["valid_depth_pixels"] / (depth_w * depth_h),
                "mesh_hit_pixels": frame["mesh_hit_pixels"],
                "depth_and_mesh_overlap_pixels": frame["depth_and_mesh_overlap_pixels"],
                "mesh_hit_fraction_of_valid_depth": frame["mesh_hit_fraction_of_valid_depth"],
                "median_abs_depth_residual_m": frame["median_abs_depth_residual_m"],
                "p90_abs_depth_residual_m": frame["p90_abs_depth_residual_m"],
                "p95_abs_depth_residual_m": frame["p95_abs_depth_residual_m"],
                "within_2cm_fraction": frame["within_2cm_fraction"],
                "within_5cm_fraction": frame["within_5cm_fraction"],
                "within_10cm_fraction": frame["within_10cm_fraction"],
                "within_20cm_fraction": frame["within_20cm_fraction"],
            })
        write_csv(output_dir / "per_scan.csv", scan_rows)
        write_csv(output_dir / "per_frame.csv", frame_rows)

    env_rows = []
    by_scene: dict[str, list[dict]] = {}
    for row in scan_rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    for scene_id in subset["selected_scene_ids"]:
        scans = by_scene.get(scene_id, [])
        relation = next(r for r in subset["relations"] if r["scene_id"] == scene_id)
        by_scan = {r["scan_id"]: r for r in scans}
        a, b = by_scan.get(relation["reference_scan_id"]), by_scan.get(relation["query_scan_id"])
        env_rows.append({
            "environment_id": scene_id,
            "reference_scan_id": relation["reference_scan_id"],
            "query_scan_id": relation["query_scan_id"],
            "scan_count": len(scans),
            "reference_gate_pass": a["gate_5cm_pass"] if a else False,
            "query_gate_pass": b["gate_5cm_pass"] if b else False,
            "relation_gate_pass": bool(a and b and a["gate_5cm_pass"] and b["gate_5cm_pass"]),
            "exclusion_reason": "" if a and b and a["gate_5cm_pass"] and b["gate_5cm_pass"] else "one or both scans failed frozen 5 cm projection gate",
        })
    write_csv(output_dir / "per_environment_gate.csv", env_rows)
    summary = {
        "dataset_revision": REVISION,
        "subset_manifest": "../quota_subset_manifest.json",
        "sample_count_per_scan": args.sample_count,
        "projection_gate_m": 0.05,
        "scan_count": len(scan_rows),
        "environment_count": len(env_rows),
        "passing_scan_count": sum(r["gate_5cm_pass"] for r in scan_rows),
        "passing_relation_count": sum(r["relation_gate_pass"] for r in env_rows),
        "failed_relation_count": sum(not r["relation_gate_pass"] for r in env_rows),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
