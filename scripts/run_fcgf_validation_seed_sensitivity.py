#!/usr/bin/env python3
"""Run the frozen FCGF validation procedure across fixed consensus seeds."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PAIR_COLUMNS = (
    "environment_id", "reference_scan", "rescan_scan", "relation_index",
    "rigid_instances", "nonrigid_instances", "removed_instances", "has_rigid_transfer",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def complete(output: Path, seeds: list[int]) -> bool:
    summary_path = output / "summary.csv"
    status_path = output / "status.json"
    if not summary_path.is_file() or not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary = read_rows(summary_path)
    except (OSError, ValueError, csv.Error):
        return False
    return (
        status.get("status") == "complete"
        and status.get("ransac_seeds") == seeds
        and status.get("top_k") == [20]
        and len(summary) == len(seeds)
        and {int(row["ransac_seed"]) for row in summary} == set(seeds)
        and all((output / f"query_diagnostics_k20_seed{seed}.csv").is_file() for seed in seeds)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261004_fcgf_validation_seed_sensitivity"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--fcgf-checkpoint", type=Path, default=Path(os.environ.get("WORLDCACHE_FCGF_CHECKPOINT", Path.home() / "data/worldcache_models/fcgf/fcgf-3dmatch.pth")))
    parser.add_argument("--fcgf-python", type=Path, default=Path(os.environ.get("WORLDCACHE_FCGF_PYTHON", sys.executable)))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--frames-per-scan", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be a non-empty unique list")
    if not args.metadata.is_file() or not args.fcgf_checkpoint.is_file() or not args.fcgf_python.is_file():
        raise FileNotFoundError("metadata, FCGF checkpoint, and FCGF interpreter must all exist")
    pairs = read_rows(args.pair_manifest)
    environments = {row["environment_id"] for row in pairs}
    if len(pairs) != 110 or len(environments) != 47:
        raise ValueError(f"expected 110 relations / 47 environments, got {len(pairs)} / {len(environments)}")
    if any(row["reference_scan"] != row["environment_id"] for row in pairs):
        raise ValueError("manifest contains a reference/environment mismatch")

    scans = sorted({row["reference_scan"] for row in pairs} | {row["rescan_scan"] for row in pairs})
    missing = {
        scan: [name for name in ("labels.instances.annotated.v2.ply", "semseg.v2.json", "sequence.zip") if not (args.data_root / scan / name).is_file()]
        for scan in scans
    }
    missing = {scan: names for scan, names in missing.items() if names}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "planned", "posthoc": True, "relations": len(pairs), "environments": len(environments),
        "unique_scans": len(scans), "seeds": args.seeds, "top_k": 20, "frames_per_scan": args.frames_per_scan,
        "resolution": args.resolution, "voxel_size_m": 0.025, "threshold_m": 0.10, "iterations": 3000,
        "missing_scans": missing, "execution_requested": args.execute,
        "fcgf_checkpoint": str(args.fcgf_checkpoint), "fcgf_python": str(args.fcgf_python),
        "matching_vitg_seed_sensitivity_dir": "outputs/reports/20261004_vitg_validation_seed_sensitivity",
    }
    (args.output_dir / "protocol_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Dry run only; pass --execute to evaluate the frozen seed sweep.")
        return
    if missing:
        raise FileNotFoundError(f"validation inputs are missing for {len(missing)} scans; see protocol_plan.json")

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    generate_script = Path(__file__).with_name("generate_3rscan_pairs.py")
    evaluate_script = Path(__file__).with_name("evaluate_3rscan_fcgf_baseline.py")
    manifest_path = args.output_dir / "seed_sweep_manifest.csv"
    run_rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        relation_output = args.output_dir / "relations" / run_id
        output = relation_output / "fcgf"
        row: dict[str, object] = {key: pair[key] for key in PAIR_COLUMNS}
        row.update({"run_id": run_id, "output_dir": str(relation_output)})
        if complete(output, args.seeds):
            row["status"] = "complete"
            run_rows.append(row)
            write_manifest(manifest_path, run_rows)
            continue

        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"wc_fcgf_seed_{index:03d}_", dir=args.scratch_root) as scratch_name:
            pair_run = Path(scratch_name) / "pair"
            generation = [
                sys.executable, str(generate_script), "--root", str(args.data_root),
                "--metadata", str(args.metadata), "--reference", pair["reference_scan"],
                "--rescan", pair["rescan_scan"], "--output-dir", str(pair_run),
                "--resolution", str(args.resolution), "--reference-frames", str(args.frames_per_scan),
                "--rescan-frames", str(args.frames_per_scan),
            ]
            evaluation = [
                str(args.fcgf_python), str(evaluate_script), "--pair-run", str(pair_run),
                "--annotation-root", str(args.data_root), "--checkpoint", str(args.fcgf_checkpoint),
                "--output-dir", str(output), "--voxel-size", "0.025", "--top-k", "20",
                "--threshold", "0.10", "--iterations", "3000", "--seed-sweep",
                *[str(seed) for seed in args.seeds], "--save-query-diagnostics",
            ]
            relation_output.mkdir(parents=True, exist_ok=True)
            with (relation_output / "runner.log").open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(generation) + "\n")
                log.flush()
                subprocess.run(generation, check=True, stdout=log, stderr=subprocess.STDOUT)
                log.write("\n$ " + " ".join(evaluation) + "\n")
                log.flush()
                subprocess.run(evaluation, check=True, stdout=log, stderr=subprocess.STDOUT)
        if not complete(output, args.seeds):
            raise RuntimeError(f"incomplete FCGF seed sweep for {run_id}")
        status = {
            "status": "complete", "run_id": run_id, "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"], "rescan_scan": pair["rescan_scan"],
            "seeds": args.seeds, "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        (relation_output / "relation_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        row.update({"status": "complete", "elapsed_seconds": status["elapsed_seconds"]})
        run_rows.append(row)
        write_manifest(manifest_path, run_rows)
        print(f"completed {index + 1}/{len(pairs)}: {run_id}", flush=True)

    aggregate_script = Path(__file__).with_name("aggregate_fcgf_validation_seed_sensitivity.py")
    subprocess.run([
        sys.executable, str(aggregate_script), "--run-dir", str(args.output_dir),
        "--pair-manifest", str(args.pair_manifest), "--vitg-run-dir",
        "outputs/reports/20261004_vitg_validation_seed_sensitivity", "--output-dir",
        str(args.output_dir / "aggregate"),
    ], check=True)
    plan.update({"status": "complete", "completed_relations": len(pairs), "completed_environments": len(environments)})
    (args.output_dir / "protocol_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "protocol_status.json").write_text(json.dumps({
        "status": "complete", "relations": len(pairs), "environments": len(environments), "seeds": args.seeds,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
