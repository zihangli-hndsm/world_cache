#!/usr/bin/env python3
"""Run the frozen ViT-G protocol over every official 3RScan val relation.

The runner is resumable and keeps only one relation's prepared RGB-D frames in
scratch storage. It requires the official validation scan files locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


PAIR_COLUMNS = [
    "environment_id",
    "reference_scan",
    "rescan_scan",
    "relation_index",
    "rigid_instances",
    "nonrigid_instances",
    "removed_instances",
    "has_rigid_transfer",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def is_complete(path: Path, pair: dict[str, str]) -> bool:
    dino = path / "vitg"
    summary_path = dino / "summary.csv"
    query_path = dino / "query_diagnostics.csv"
    status_path = dino / "status.json"
    fpfh_status = path / "fpfh" / "status.json"
    fcgf_status = path / "fcgf" / "status.json"
    relation_status_path = path / "protocol_relation.json"
    if not (summary_path.is_file() and query_path.is_file() and status_path.is_file()
            and fpfh_status.is_file() and fcgf_status.is_file() and relation_status_path.is_file()):
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        with summary_path.open(newline="", encoding="utf-8") as stream:
            summary = list(csv.DictReader(stream))
        fpfh = json.loads(fpfh_status.read_text(encoding="utf-8"))
        fcgf = json.loads(fcgf_status.read_text(encoding="utf-8"))
        relation_status = json.loads(relation_status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, csv.Error):
        return False
    return (
        status.get("status") == "complete"
        and status.get("layers") == [2, 20, 40]
        and status.get("threshold_m") == 0.10
        and status.get("ransac_seeds") == [17]
        and status.get("physical_instance_candidate_audit_posthoc") is True
        and len(summary) == 1
        and int(summary[0].get("top_k", -1)) == 20
        and fpfh.get("status") == "complete"
        and fpfh.get("ransac_threshold_m") == 0.10
        and fcgf.get("status") == "complete"
        and fcgf.get("known_alignment_used_for_selection") is False
        and fcgf.get("top_k") == [20]
        and relation_status.get("status") == "complete"
        and relation_status.get("environment_id") == pair["environment_id"]
        and relation_status.get("reference_scan") == pair["reference_scan"]
        and relation_status.get("rescan_scan") == pair["rescan_scan"]
    )


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261002_vitg_validation"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument(
        "--fcgf-checkpoint",
        type=Path,
        default=Path(
            os.environ.get(
                "WORLDCACHE_FCGF_CHECKPOINT",
                Path.home() / "data/worldcache_models/fcgf/fcgf-3dmatch.pth",
            )
        ),
    )
    parser.add_argument(
        "--fcgf-python",
        type=Path,
        default=Path(os.environ.get("WORLDCACHE_FCGF_PYTHON", sys.executable)),
        help="Python interpreter with WarpConvNet installed for the FCGF control",
    )
    parser.add_argument("--frames-per-scan", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--execute", action="store_true", help="run all relations after validating that the full input set exists")
    args = parser.parse_args()
    if args.frames_per_scan < 1 or args.resolution < 1:
        raise ValueError("frame count and resolution must be positive")
    if not args.metadata.is_file():
        raise FileNotFoundError(f"3RScan metadata is missing: {args.metadata}")

    pairs = read_rows(args.pair_manifest)
    if len(pairs) != 110 or len({row["environment_id"] for row in pairs}) != 47:
        raise ValueError("expected the audited official validation manifest (110 relations, 47 environments)")
    if any(row["reference_scan"] != row["environment_id"] for row in pairs):
        raise ValueError("manifest has a relation whose reference differs from its environment ID")

    scan_ids = sorted({row["reference_scan"] for row in pairs} | {row["rescan_scan"] for row in pairs})
    missing: dict[str, list[str]] = {"annotations": [], "semseg": [], "sequences": []}
    for scan_id in scan_ids:
        scan_dir = args.data_root / scan_id
        if not (scan_dir / "labels.instances.annotated.v2.ply").is_file():
            missing["annotations"].append(scan_id)
        if not (scan_dir / "semseg.v2.json").is_file():
            missing["semseg"].append(scan_id)
        if not (scan_dir / "sequence.zip").is_file():
            missing["sequences"].append(scan_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "official_validation_relations": len(pairs),
        "official_validation_environments": len({row["environment_id"] for row in pairs}),
        "unique_scans_required": len(scan_ids),
        "relations_with_rigid_mapping": sum(row["has_rigid_transfer"].lower() == "true" for row in pairs),
        "frames_per_scan": args.frames_per_scan,
        "resolution": args.resolution,
        "environment_bootstrap_samples": args.bootstrap_samples,
        "protocol": {
            "model_size": "giant",
            "layers": [2, 20, 40],
            "top_k": [20],
            "threshold_m": 0.10,
            "iterations": 1500,
            "ransac_seed": 17,
            "physical_instance_candidate_audit": True,
            "query_diagnostics": True,
            "controls": {
                "fpfh_voxel_m": 0.05,
                "fpfh_normal_radius_m": 0.10,
                "fpfh_feature_radius_m": 0.25,
                "fpfh_iterations": 5000,
                "fcgf_checkpoint": str(args.fcgf_checkpoint),
                "fcgf_python": str(args.fcgf_python),
                "fcgf_voxel_m": 0.025,
                "fcgf_iterations": 3000,
            },
        },
        "missing_annotations": len(missing["annotations"]),
        "missing_semseg": len(missing["semseg"]),
        "missing_sequences": len(missing["sequences"]),
        "missing_scan_ids": sorted(
            set(missing["annotations"]) | set(missing["semseg"]) | set(missing["sequences"])
        ),
        "execution_requested": args.execute,
        "fcgf_checkpoint_available": args.fcgf_checkpoint.is_file(),
        "fcgf_python_available": args.fcgf_python.is_file(),
    }
    (args.output_dir / "protocol_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Dry run only; pass --execute after the full authorized validation input set is present.")
        return
    if missing["annotations"] or missing["semseg"] or missing["sequences"]:
        raise FileNotFoundError(
            "refusing partial validation run: the official validation set is missing "
            f"{len(missing['annotations'])} annotation PLYs, {len(missing['semseg'])} semseg JSONs, "
            f"and {len(missing['sequences'])} sequence ZIPs; "
            "see protocol_plan.json for exact scan IDs"
        )
    if not args.fcgf_checkpoint.is_file():
        raise FileNotFoundError(f"published FCGF checkpoint is missing: {args.fcgf_checkpoint}")
    if not args.fcgf_python.is_file():
        raise FileNotFoundError(f"FCGF Python interpreter is missing: {args.fcgf_python}")

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "validation_run_manifest.csv"
    run_rows: list[dict[str, object]] = []
    generate_script = Path(__file__).with_name("generate_3rscan_pairs.py")
    evaluate_script = Path(__file__).with_name("evaluate_3rscan_spatial_consensus.py")
    for index, pair in enumerate(pairs):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        relation_output = args.output_dir / "relations" / run_id
        row = {key: pair[key] for key in PAIR_COLUMNS}
        row.update({"run_id": run_id, "output_dir": str(relation_output)})
        if is_complete(relation_output, pair):
            row["status"] = "complete"
            run_rows.append(row)
            write_manifest(manifest_path, run_rows)
            continue

        with tempfile.TemporaryDirectory(prefix=f"wc_val_{index:03d}_", dir=args.scratch_root) as scratch_name:
            pair_run = Path(scratch_name) / "pair"
            generation = [
                sys.executable, str(generate_script),
                "--root", str(args.data_root),
                "--metadata", str(args.metadata),
                "--reference", pair["reference_scan"],
                "--rescan", pair["rescan_scan"],
                "--output-dir", str(pair_run),
                "--resolution", str(args.resolution),
                "--reference-frames", str(args.frames_per_scan),
                "--rescan-frames", str(args.frames_per_scan),
            ]
            evaluation = [
                sys.executable, str(evaluate_script),
                "--pair-run", str(pair_run),
                "--annotation-root", str(args.data_root),
                "--output-dir", str(relation_output / "vitg"),
                "--model-size", "giant",
                "--layers", "2", "20", "40",
                "--top-k", "20",
                "--threshold", "0.10",
                "--iterations", "1500",
                "--save-query-diagnostics",
                "--save-physical-instance-candidate-audit",
            ]
            fpfh = [
                sys.executable, str(Path(__file__).with_name("evaluate_3rscan_fpfh_baseline.py")),
                "--pair-run", str(pair_run),
                "--annotation-root", str(args.data_root),
                "--output-dir", str(relation_output / "fpfh"),
                "--voxel-size", "0.05",
                "--normal-radius", "0.10",
                "--feature-radius", "0.25",
                "--threshold", "0.10",
                "--iterations", "5000",
                "--save-query-diagnostics",
            ]
            fcgf = [
                str(args.fcgf_python), str(Path(__file__).with_name("evaluate_3rscan_fcgf_baseline.py")),
                "--pair-run", str(pair_run),
                "--annotation-root", str(args.data_root),
                "--checkpoint", str(args.fcgf_checkpoint),
                "--output-dir", str(relation_output / "fcgf"),
                "--voxel-size", "0.025",
                "--top-k", "20",
                "--threshold", "0.10",
                "--iterations", "3000",
                "--seed", "17",
                "--save-query-diagnostics",
            ]
            relation_output.mkdir(parents=True, exist_ok=True)
            log_path = relation_output / "runner.log"
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(generation) + "\n")
                log.flush()
                subprocess.run(generation, check=True, stdout=log, stderr=subprocess.STDOUT)
                for command, arm_name in ((evaluation, "vitg"), (fpfh, "fpfh"), (fcgf, "fcgf")):
                    if (relation_output / arm_name / "status.json").is_file():
                        try:
                            prior = json.loads((relation_output / arm_name / "status.json").read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            prior = {}
                        if prior.get("status") == "complete":
                            continue
                    log.write("\n$ " + " ".join(command) + "\n")
                    log.flush()
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        (relation_output / "protocol_relation.json").write_text(json.dumps({
            "status": "complete",
            "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"],
            "rescan_scan": pair["rescan_scan"],
            "relation_index": int(pair["relation_index"]),
            "protocol": plan["protocol"],
        }, indent=2) + "\n", encoding="utf-8")
        if not is_complete(relation_output, pair):
            (relation_output / "protocol_relation.json").unlink(missing_ok=True)
            raise RuntimeError(f"relation run did not produce a complete frozen-protocol result: {run_id}")
        row["status"] = "complete"
        run_rows.append(row)
        write_manifest(manifest_path, run_rows)
        print(f"completed {index + 1}/{len(pairs)}: {run_id}", flush=True)

    aggregate_command = [
        sys.executable,
        str(Path(__file__).with_name("aggregate_3rscan_validation.py")),
        "--run-dir", str(args.output_dir),
        "--pair-manifest", str(args.pair_manifest),
        "--output-dir", str(args.output_dir / "aggregate"),
        "--bootstrap-samples", str(args.bootstrap_samples),
    ]
    subprocess.run(aggregate_command, check=True)
    (args.output_dir / "protocol_status.json").write_text(json.dumps({
        "status": "complete",
        "relations": len(pairs),
        "environments": len({row["environment_id"] for row in pairs}),
        "manifest": str(manifest_path),
        "aggregate_report": str(args.output_dir / "aggregate" / "report.md"),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
