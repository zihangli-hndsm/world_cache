#!/usr/bin/env python3
"""Run frozen ViT-G consensus seeds on all official validation relations."""
from __future__ import annotations

import argparse
import csv
import json
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
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def complete(output: Path, seeds: list[int]) -> bool:
    summary_path = output / "summary.csv"
    status_path = output / "status.json"
    diagnostics_path = output / "query_diagnostics.csv"
    if not (summary_path.is_file() and status_path.is_file() and diagnostics_path.is_file()):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary = read_rows(summary_path)
    except (OSError, ValueError, csv.Error):
        return False
    return (
        status.get("status") == "complete"
        and status.get("layers") == [2, 20, 40]
        and status.get("threshold_m") == 0.10
        and status.get("ransac_seeds") == seeds
        and status.get("physical_instance_candidate_audit_posthoc") is True
        and len(summary) == len(seeds)
        and {int(row["top_k"]) for row in summary} == {20}
        and {int(row["ransac_seed"]) for row in summary} == set(seeds)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261004_vitg_validation_seed_sensitivity"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--frames-per-scan", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("seeds must be a non-empty unique list")
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)

    pairs = read_rows(args.pair_manifest)
    environments = {row["environment_id"] for row in pairs}
    if len(pairs) != 110 or len(environments) != 47:
        raise ValueError(f"expected 110 relations and 47 environments; found {len(pairs)} and {len(environments)}")
    if any(row["reference_scan"] != row["environment_id"] for row in pairs):
        raise ValueError("manifest contains a reference/environment mismatch")
    scans = sorted({row["reference_scan"] for row in pairs} | {row["rescan_scan"] for row in pairs})
    missing: dict[str, list[str]] = {"annotations": [], "semseg": [], "sequences": []}
    for scan in scans:
        folder = args.data_root / scan
        for kind, filename in (
            ("annotations", "labels.instances.annotated.v2.ply"),
            ("semseg", "semseg.v2.json"),
            ("sequences", "sequence.zip"),
        ):
            if not (folder / filename).is_file():
                missing[kind].append(scan)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "planned",
        "posthoc": True,
        "relations": len(pairs),
        "environments": len(environments),
        "unique_scans": len(scans),
        "seeds": args.seeds,
        "model": "DINOv2 ViT-G/14, frozen",
        "layers": [2, 20, 40],
        "top_k": 20,
        "frames_per_scan": args.frames_per_scan,
        "resolution": args.resolution,
        "consensus_threshold_m": 0.10,
        "consensus_iterations": 1500,
        "missing_inputs": {key: len(value) for key, value in missing.items()},
        "execution_requested": args.execute,
    }
    (args.output_dir / "protocol_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Dry run only; pass --execute to evaluate the fixed seed sweep.")
        return
    if any(missing.values()):
        raise FileNotFoundError(f"validation inputs missing; details are in {args.output_dir / 'protocol_plan.json'}")

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "seed_sweep_manifest.csv"
    run_rows: list[dict[str, object]] = []
    generate_script = Path(__file__).with_name("generate_3rscan_pairs.py")
    evaluate_script = Path(__file__).with_name("evaluate_3rscan_spatial_consensus.py")
    for index, pair in enumerate(pairs):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        relation_output = args.output_dir / "relations" / run_id
        row: dict[str, object] = {key: pair[key] for key in PAIR_COLUMNS}
        row.update({"run_id": run_id, "output_dir": str(relation_output)})
        vitg_output = relation_output / "vitg"
        if complete(vitg_output, args.seeds):
            row["status"] = "complete"
            run_rows.append(row)
            write_manifest(manifest_path, run_rows)
            continue

        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"wc_vitg_seed_{index:03d}_", dir=args.scratch_root) as scratch:
            pair_run = Path(scratch) / "pair"
            generation = [
                sys.executable, str(generate_script), "--root", str(args.data_root),
                "--metadata", str(args.metadata), "--reference", pair["reference_scan"],
                "--rescan", pair["rescan_scan"], "--output-dir", str(pair_run),
                "--resolution", str(args.resolution), "--reference-frames", str(args.frames_per_scan),
                "--rescan-frames", str(args.frames_per_scan),
            ]
            evaluation = [
                sys.executable, str(evaluate_script), "--pair-run", str(pair_run),
                "--annotation-root", str(args.data_root), "--output-dir", str(vitg_output),
                "--model-size", "giant", "--layers", "2", "20", "40", "--top-k", "20",
                "--threshold", "0.10", "--iterations", "1500", "--seed-sweep",
                *[str(seed) for seed in args.seeds], "--save-query-diagnostics",
                "--save-physical-instance-candidate-audit",
            ]
            relation_output.mkdir(parents=True, exist_ok=True)
            log_path = relation_output / "runner.log"
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(generation) + "\n")
                log.flush()
                subprocess.run(generation, check=True, stdout=log, stderr=subprocess.STDOUT)
                log.write("\n$ " + " ".join(evaluation) + "\n")
                log.flush()
                subprocess.run(evaluation, check=True, stdout=log, stderr=subprocess.STDOUT)
        if not complete(vitg_output, args.seeds):
            raise RuntimeError(f"incomplete ViT-G seed sweep for {run_id}")
        relation = {
            "status": "complete", "run_id": run_id,
            "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"],
            "rescan_scan": pair["rescan_scan"],
            "seeds": args.seeds,
            "elapsed_seconds": round(time.monotonic() - start, 3),
        }
        (relation_output / "relation_status.json").write_text(json.dumps(relation, indent=2) + "\n", encoding="utf-8")
        row.update({"status": "complete", "elapsed_seconds": relation["elapsed_seconds"]})
        run_rows.append(row)
        write_manifest(manifest_path, run_rows)
        print(f"completed {index + 1}/{len(pairs)}: {run_id}", flush=True)

    (args.output_dir / "protocol_status.json").write_text(json.dumps({
        "status": "complete", "relations": len(pairs), "environments": len(environments),
        "seeds": args.seeds, "manifest": str(manifest_path),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
