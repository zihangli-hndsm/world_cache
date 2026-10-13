#!/usr/bin/env python3
"""Fetch only archives from the frozen MultiScan candidate manifest.

Downloads are stored in the workspace Hugging Face cache and remain resumable.
Select one scene for staged preflight, or explicitly pass --all after checking
storage capacity. No labels or predictions are read by this downloader.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import get_token, hf_hub_download


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
PAIR_FILE = REPORT_DIR / "candidate_pairs.csv"
REPO_ID = "3dlg-hcvc/MultiScan"
REVISION = "c62c9aad850a8638ac3e42605926a65707600125"
CACHE_DIR = ROOT / "data" / "multiscan_independent_hf_cache"
STATE_FILE = REPORT_DIR / "download_state.json"
SUBSET_MANIFEST = REPORT_DIR / "quota_subset_manifest.json"


def frozen_archives() -> OrderedDict[str, dict]:
    archives: OrderedDict[str, dict] = OrderedDict()
    with PAIR_FILE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for filename_key, size_key, oid_key in (
                ("reference_file", "reference_size_bytes", "reference_repo_blob_id"),
                ("query_file", "query_size_bytes", "query_repo_blob_id"),
            ):
                filename = row[filename_key]
                expected = {
                    "size_bytes": int(row[size_key]),
                    "repo_blob_id": row[oid_key],
                    "scene_id": row["scene_id"],
                }
                if filename in archives and archives[filename] != expected:
                    raise RuntimeError(f"conflicting frozen metadata for {filename}")
                archives[filename] = expected
    return archives


def frozen_relations() -> list[dict]:
    with PAIR_FILE.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene-id", help="download all frozen archives for one scene")
    group.add_argument(
        "--prefix-cap-bytes",
        type=int,
        help="freeze the longest scene-sorted prefix of complete rooms that fits this byte cap",
    )
    args = parser.parse_args()

    archives = frozen_archives()
    relations = frozen_relations()
    if args.prefix_cap_bytes is not None:
        scene_ids = sorted({meta["scene_id"] for meta in archives.values()})
        selected_scenes = []
        used = 0
        for scene_id in scene_ids:
            scene_archives = [(name, meta) for name, meta in archives.items() if meta["scene_id"] == scene_id]
            scene_bytes = sum(meta["size_bytes"] for _, meta in scene_archives)
            if used + scene_bytes > args.prefix_cap_bytes:
                break
            selected_scenes.append(scene_id)
            used += scene_bytes
        if not selected_scenes:
            raise SystemExit("The byte cap is too small to include the first complete frozen room.")
        selected_set = set(selected_scenes)
        selected = [(name, meta) for name, meta in archives.items() if meta["scene_id"] in selected_set]
        selected_relations = [row for row in relations if row["scene_id"] in selected_set]
        subset = {
            "selection_rule": "Take the longest lexicographically scene-sorted prefix of complete rooms in candidate_pairs.csv whose unique archive byte total does not exceed prefix_cap_bytes; do not use preflight or model outcomes for selection.",
            "reason": "User-directed quota-limited exploratory subset; full frozen population is 359 GB while remaining user quota is approximately 50 GB before cleanup.",
            "prefix_cap_bytes": args.prefix_cap_bytes,
            "selected_scene_ids": selected_scenes,
            "scene_count": len(selected_scenes),
            "relation_count": len(selected_relations),
            "archive_count": len(selected),
            "archive_bytes": used,
            "dataset_repository": REPO_ID,
            "dataset_revision": REVISION,
            "candidate_pairs_sha256": __import__("hashlib").sha256(PAIR_FILE.read_bytes()).hexdigest(),
            "relations": [
                {"scene_id": row["scene_id"], "environment_id": row["environment_id"],
                 "reference_scan_id": row["reference_scan_id"], "query_scan_id": row["query_scan_id"]}
                for row in selected_relations
            ],
            "archives": [
                {"filename": name, **meta} for name, meta in selected
            ],
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if SUBSET_MANIFEST.exists():
            old = json.loads(SUBSET_MANIFEST.read_text())
            old_fingerprint = (old.get("prefix_cap_bytes"), old.get("selected_scene_ids"), old.get("candidate_pairs_sha256"))
            new_fingerprint = (subset["prefix_cap_bytes"], subset["selected_scene_ids"], subset["candidate_pairs_sha256"])
            if old_fingerprint != new_fingerprint:
                raise SystemExit("Existing quota subset manifest differs; refusing to silently change the frozen subset.")
        else:
            SUBSET_MANIFEST.write_text(json.dumps(subset, indent=2) + "\n")
    else:
        selected = [(name, meta) for name, meta in archives.items() if meta["scene_id"] == args.scene_id]
        if not selected:
            raise SystemExit(f"scene is not in the frozen candidate manifest: {args.scene_id}")

    if not get_token():
        raise SystemExit("No Hugging Face login is available; run `hf auth login` in worldcache.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    expected_bytes = sum(meta["size_bytes"] for _, meta in selected)
    free_bytes = shutil.disk_usage(CACHE_DIR).free
    print(f"repository={REPO_ID}@{REVISION}")
    scope = f"scene={args.scene_id}" if args.scene_id else f"prefix_cap_bytes={args.prefix_cap_bytes}"
    print(f"scope={scope}; archive_count={len(selected)}")
    if args.prefix_cap_bytes is not None:
        print(f"subset_manifest={SUBSET_MANIFEST.relative_to(ROOT)}")
    print(f"compressed_bytes={expected_bytes}; filesystem_free_bytes={free_bytes}")
    if free_bytes < int(expected_bytes * 1.05):
        raise SystemExit("Insufficient reported free space for this batch plus a 5% margin.")

    state = {
        "repository": REPO_ID,
        "revision": REVISION,
        "cache_dir": str(CACHE_DIR.relative_to(ROOT)),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_archives": {},
    }
    if STATE_FILE.exists():
        old = json.loads(STATE_FILE.read_text())
        if old.get("repository") != REPO_ID or old.get("revision") != REVISION:
            raise SystemExit("Existing download state is for a different repository revision.")
        state["downloaded_archives"] = old.get("downloaded_archives", {})

    started = time.monotonic()
    for index, (filename, meta) in enumerate(selected, 1):
        before = time.monotonic()
        resolved = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=filename,
            cache_dir=str(CACHE_DIR),
            token=True,
        ))
        size = resolved.stat().st_size
        if size != meta["size_bytes"]:
            raise RuntimeError(f"size mismatch for {filename}: expected {meta['size_bytes']}, got {size}")
        state["downloaded_archives"][filename] = {
            "scene_id": meta["scene_id"],
            "size_bytes": size,
            "repo_blob_id": meta["repo_blob_id"],
            "cache_path": str(resolved.relative_to(ROOT)),
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")
        print(f"[{index}/{len(selected)}] ready {filename} ({size} bytes; {time.monotonic()-before:.1f}s)", flush=True)

    print(f"batch_wall_seconds={time.monotonic()-started:.1f}")
    print(f"state_file={STATE_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
