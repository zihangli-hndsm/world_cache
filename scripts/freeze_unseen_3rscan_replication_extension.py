#!/usr/bin/env python3
"""Freeze the next hash-ordered 3RScan train cohort under the prior policy.

This is a post-hoc extension of the eight-relation confirmation because their
outcomes have already been inspected. Selection itself uses only official
split metadata and current raw-sequence availability; it never scores labels or
matching outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    project = Path(__file__).resolve().parents[1]
    prior_dir = project / "outputs/reports/20261008_unseen_train_replication"
    prior = json.loads((prior_dir / "frozen_population.json").read_text())
    if prior.get("status") != "frozen" or len(prior.get("population", [])) != 8:
        raise ValueError("expected the completed eight-relation frozen population")
    if not all((prior_dir / "relations" / row["run_id"] / "complete.json").is_file()
               for row in prior["population"]):
        raise ValueError("prior eight-relation cohort is not fully complete")

    calibration = project / "outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json"
    calibration_hash = sha256(calibration)
    if calibration_hash != prior["calibration_sha256"]:
        raise ValueError("source calibration changed since the original frozen run")

    metadata = json.loads((args.data_root / "3RScan.json").read_text())
    source = pd.read_csv(project / "outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv")
    validation = pd.read_csv(project / "outputs/reports/20261008_bufferx_validation/run_manifest.csv")
    excluded_scans = set(source["reference_scan"]) | set(validation["reference_scan"])
    prior_scans = {
        row[key]
        for row in prior["population"]
        for key in ("reference_scan", "rescan_scan")
    }
    raw_local = {p.parent.name for p in args.data_root.glob("*/sequence.zip")}
    local = raw_local | prior_scans
    seed = prior["seed"]

    def order(identifier: str) -> str:
        return hashlib.sha256((seed + identifier).encode("utf-8")).hexdigest()

    eligible = [
        group for group in metadata
        if group.get("type") == "train"
        and group.get("reference") not in excluded_scans
        and group.get("reference") not in local
        and group.get("reference") not in prior_scans
        and group.get("scans")
    ]
    eligible.sort(key=lambda group: order(group["reference"]))
    if len(eligible) < args.count:
        raise ValueError(f"only {len(eligible)} unseen eligible environments remain")

    population = []
    for index, group in enumerate(eligible[:args.count], start=8):
        ref = group["reference"]
        targets = sorted(
            {
                scan["reference"] for scan in group["scans"]
                if scan["reference"] not in local
                and scan["reference"] not in prior_scans
            },
            key=order,
        )
        if not targets:
            raise ValueError(f"selected environment {ref} has no unused rescan")
        target = targets[0]
        population.append({
            "run_id": f"{index:03d}_{ref[:8]}_{target[:8]}",
            "environment_id": ref,
            "reference_scan": ref,
            "rescan_scan": target,
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "frozen",
        "population": population,
        "seed": seed,
        "selection": (
            "Next hash-ordered unused train environments under the original seed; "
            "exclude original source, official validation, all current local sequences, "
            "and the prior eight. One hash-selected unused rescan per environment."
        ),
        "selection_scope": "Post-hoc cohort extension after observing the prior eight outcomes; do not present as preregistered or independent-dataset replication.",
        "eligible_environment_count_at_freeze": len(eligible),
        "current_local_sequence_count": len(raw_local),
        "prior_population_sha256": sha256(prior_dir / "frozen_population.json"),
        "calibration_sha256": calibration_hash,
        "policy": prior["policy"],
        "inference_protocol": "Reuse the unchanged source-frozen distance/rank policies and ViT-G candidate procedure; retain all failures; no outcome-based replacements or retuning.",
    }
    (args.output_dir / "frozen_population.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pd.DataFrame(population).to_csv(args.output_dir / "relations.csv", index=False)
    scan_ids = sorted({row[key] for row in population for key in ("reference_scan", "rescan_scan")})
    pd.DataFrame({"scan_id": scan_ids}).to_csv(args.output_dir / "scan_manifest.csv", index=False)
    print(json.dumps({
        "status": manifest["status"],
        "eligible_environment_count_at_freeze": len(eligible),
        "selected_relations": population,
        "scan_count": len(scan_ids),
        "calibration_sha256": calibration_hash,
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
