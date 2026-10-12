#!/usr/bin/env python3
"""Run the published MASt3R reciprocal matcher on all official 3RScan val pairs.

The run reuses the frozen RGB-D frame/query protocol and is resumable. Model
loading is forced offline so an incomplete local cache cannot trigger a large
implicit checkpoint download.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL_ID = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
MODEL_REVISION = "06e7259f34c3060f322df5cb0c7b9094f57e41fc"
MODEL_SHA256 = "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def is_complete(path: Path, run_id: str, match_mode: str) -> bool:
    summary = path / "summary.json"
    status = path / "status.json"
    diagnostics = path / "query_diagnostics.csv"
    candidates = path / "candidate_cache.npz"
    relation = path / "relation.json"
    if not all(p.is_file() for p in (summary, status, diagnostics, candidates, relation)):
        return False
    try:
        status_data = json.loads(status.read_text(encoding="utf-8"))
        relation_data = json.loads(relation.read_text(encoding="utf-8"))
        summary_data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        status_data.get("status") == "complete"
        and relation_data.get("status") == "complete"
        and relation_data.get("run_id") == run_id
        and summary_data.get("checkpoint_reference") == MODEL_ID
        and summary_data.get("match_mode") == match_mode
        and diagnostics.stat().st_size > 0
        and candidates.stat().st_size > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--pair-manifest", type=Path,
        default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/reports/20261004_mast3r_official_validation"),
    )
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument(
        "--mast3r-root", type=Path,
        default=Path(os.environ.get("MAST3R_ROOT", Path.home() / "data/worldcache_models/mast3r_source")),
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--frames-per-scan", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--match-mode", choices=("reciprocal", "dense_topk"), default="reciprocal",
        help="published sparse reciprocal matcher or custom dense top-20 diagnostic",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    if not args.mast3r_root.is_dir():
        raise FileNotFoundError(args.mast3r_root)
    snapshot = Path.home() / ".cache/huggingface/hub/models--naver--MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/snapshots" / MODEL_REVISION
    weight = snapshot / "model.safetensors"
    if not (snapshot / "config.json").is_file() or not weight.is_file():
        raise FileNotFoundError(f"MASt3R snapshot is not completely cached: {snapshot}")
    weight_hash = sha256(weight)
    if weight_hash != MODEL_SHA256:
        raise ValueError(f"cached MASt3R weight hash mismatch: {weight_hash}")

    pairs = read_rows(args.pair_manifest)
    if len(pairs) != 110 or len({row["environment_id"] for row in pairs}) != 47:
        raise ValueError("expected the frozen official validation manifest (110 relations, 47 environments)")
    scans = sorted({row["reference_scan"] for row in pairs} | {row["rescan_scan"] for row in pairs})
    required = ("sequence.zip", "labels.instances.annotated.v2.ply", "semseg.v2.json")
    missing = [f"{scan}/{name}" for scan in scans for name in required if not (args.data_root / scan / name).is_file()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "preflight_complete" if not missing else "preflight_incomplete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "relations": len(pairs),
        "environments": len({row["environment_id"] for row in pairs}),
        "unique_scans": len(scans),
        "missing_required_files": missing,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_safetensors_sha256": weight_hash,
        "weight_bytes": weight.stat().st_size,
        "forced_offline_model_loading": True,
        "protocol": {
            "rgbd_resolution": 224,
            "frames_per_scan": args.frames_per_scan,
            "mast3r_image_size": args.image_size,
            "matcher": (
                "published MASt3R reciprocal nearest neighbours in both directions"
                if args.match_mode == "reciprocal"
                else "custom dense confidence-weighted descriptor top-k over each frame pair"
            ),
            "candidate_ranking": (
                "published descriptor confidence; top-20 cap"
                if args.match_mode == "reciprocal"
                else "cosine similarity weighted by geometric-mean MASt3R confidence; unique top-20 cap"
            ),
            "match_mode": args.match_mode,
            "ransac_threshold_m": 0.10,
            "ransac_iterations": args.iterations,
            "ransac_seed": args.seed,
        },
        "execute_requested": args.execute,
    }
    (args.output_dir / "protocol_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if missing:
        raise FileNotFoundError(f"validation inputs are incomplete: {len(missing)} required files missing")
    if not args.execute:
        print("Preflight only; pass --execute to run all 110 relations.", flush=True)
        return

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    generate_script = Path(__file__).with_name("generate_3rscan_pairs.py")
    evaluate_script = Path(__file__).with_name("evaluate_3rscan_mast3r_baseline.py")
    manifest_path = args.output_dir / "validation_run_manifest.csv"
    completed_rows: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    for index, pair in enumerate(pairs):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        output = args.output_dir / "relations" / run_id
        output.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": run_id,
            "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"],
            "rescan_scan": pair["rescan_scan"],
            "relation_index": pair["relation_index"],
            "output_dir": str(output),
        }
        if is_complete(output, run_id, args.match_mode):
            record["status"] = "complete"
            completed_rows.append(record)
            write_csv(manifest_path, completed_rows)
            print(f"skip complete {index + 1}/110: {run_id}", flush=True)
            continue

        start = time.perf_counter()
        log_path = output / "runner.log"
        with tempfile.TemporaryDirectory(prefix=f"wc_mast3r_val_{index:03d}_", dir=args.scratch_root) as scratch:
            pair_run = Path(scratch) / "pair"
            generation = [
                sys.executable, str(generate_script),
                "--root", str(args.data_root), "--metadata", str(args.metadata),
                "--reference", pair["reference_scan"], "--rescan", pair["rescan_scan"],
                "--output-dir", str(pair_run), "--resolution", "224",
                "--reference-frames", str(args.frames_per_scan),
                "--rescan-frames", str(args.frames_per_scan),
            ]
            evaluation = [
                sys.executable, str(evaluate_script),
                "--pair-run", str(pair_run), "--annotation-root", str(args.data_root),
                "--metadata", str(args.metadata), "--output-dir", str(output),
                "--mast3r-root", str(args.mast3r_root), "--checkpoint", MODEL_ID,
                "--image-size", str(args.image_size), "--top-k", "20",
                "--threshold", "0.10", "--iterations", str(args.iterations),
                "--seed", str(args.seed), "--match-mode", args.match_mode,
                "--candidate-cache", str(output / "candidate_cache.npz"),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(generation) + "\n")
                log.flush()
                subprocess.run(generation, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
                log.write("\n$ " + " ".join(evaluation) + "\n")
                log.flush()
                subprocess.run(evaluation, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
        elapsed = time.perf_counter() - start
        relation_record = {
            "status": "complete",
            "run_id": run_id,
            "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"],
            "rescan_scan": pair["rescan_scan"],
            "relation_index": int(pair["relation_index"]),
            "elapsed_seconds": elapsed,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "protocol": plan["protocol"],
        }
        (output / "relation.json").write_text(json.dumps(relation_record, indent=2) + "\n", encoding="utf-8")
        if not is_complete(output, run_id, args.match_mode):
            raise RuntimeError(f"relation output failed completeness check: {run_id}")
        record["status"] = "complete"
        record["elapsed_seconds"] = round(elapsed, 3)
        completed_rows.append(record)
        write_csv(manifest_path, completed_rows)
        print(f"completed {index + 1}/110: {run_id} ({elapsed / 60:.1f} min)", flush=True)

    status = {
        "status": "complete",
        "relations": len(completed_rows),
        "environments": len({row["environment_id"] for row in completed_rows}),
        "manifest": str(manifest_path),
        "aggregate_command": (
            f"{sys.executable} scripts/aggregate_mast3r_official_validation.py "
            f"--run-dir {args.output_dir} --dino-run-dir outputs/reports/20261003_vitg_validation "
            f"--output-dir {args.output_dir / 'aggregate'}"
        ),
    }
    (args.output_dir / "protocol_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
