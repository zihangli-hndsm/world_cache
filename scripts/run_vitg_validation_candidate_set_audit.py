#!/usr/bin/env python3
"""Post-hoc semantic vs physical-instance top-20 audit on official 3RScan val."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


PAIR_COLUMNS = (
    "environment_id", "reference_scan", "rescan_scan", "relation_index",
    "rigid_instances", "nonrigid_instances", "removed_instances", "has_rigid_transfer",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def has_audit(output: Path) -> bool:
    try:
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        with (output / "query_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
            columns = set(next(csv.reader(stream)))
    except (OSError, ValueError, StopIteration, csv.Error):
        return False
    return (
        status.get("status") == "complete"
        and status.get("layers") == [2, 20, 40]
        and status.get("threshold_m") == 0.10
        and status.get("ransac_seeds") == [17]
        and status.get("candidate_set_audit_posthoc") is True
        and status.get("physical_instance_candidate_audit_posthoc") is True
        and {"candidate_semantic_hit", "physical_instance_candidate_hit",
             "physical_instance_evaluable"}.issubset(columns)
    )


def score_outputs(run_dir: Path, manifest: list[dict[str, str]], bootstrap_samples: int, seed: int) -> dict[str, object]:
    per_relation: list[dict[str, object]] = []
    per_environment: dict[str, list[int]] = {}
    for index, pair in enumerate(manifest):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        path = run_dir / "relations" / run_id / "vitg" / "query_diagnostics.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        matched = [row for row in rows if row["physical_instance_evaluable"].lower() == "true"]
        denom = len(matched)
        sem_flags = [row["candidate_semantic_hit"].lower() == "true" for row in matched]
        phys_flags = [row["physical_instance_candidate_hit"].lower() == "true" for row in matched]
        sem_hits = sum(sem_flags)
        phys_hits = sum(phys_flags)
        both_hits = sum(s and p for s, p in zip(sem_flags, phys_flags))
        semantic_only_hits = sum(s and not p for s, p in zip(sem_flags, phys_flags))
        physical_only_hits = sum(p and not s for s, p in zip(sem_flags, phys_flags))
        neither_hits = denom - both_hits - semantic_only_hits - physical_only_hits
        sem = sem_hits / denom if denom else float("nan")
        phys = phys_hits / denom if denom else float("nan")
        rec = {
            "run_id": run_id, "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"], "rescan_scan": pair["rescan_scan"],
            "physical_evaluable_queries": denom,
            "semantic_candidate_hits_matched": sem_hits,
            "physical_candidate_hits": phys_hits,
            "both_candidate_hits": both_hits,
            "semantic_only_candidate_hits": semantic_only_hits,
            "physical_only_candidate_hits": physical_only_hits,
            "neither_candidate_hit": neither_hits,
            "semantic_candidate_recall_matched": sem,
            "physical_candidate_recall_matched": phys,
            "semantic_minus_physical_pp": 100 * (sem - phys) if denom else float("nan"),
        }
        per_relation.append(rec)
        per_environment.setdefault(pair["environment_id"], []).append(len(per_relation) - 1)

    env_rows: list[dict[str, object]] = []
    for environment, ids in sorted(per_environment.items()):
        valid = [per_relation[i] for i in ids if int(per_relation[i]["physical_evaluable_queries"]) > 0]
        n = sum(int(row["physical_evaluable_queries"]) for row in valid)
        s = sum(int(row["semantic_candidate_hits_matched"]) for row in valid)
        p = sum(int(row["physical_candidate_hits"]) for row in valid)
        both = sum(int(row["both_candidate_hits"]) for row in valid)
        semantic_only = sum(int(row["semantic_only_candidate_hits"]) for row in valid)
        physical_only = sum(int(row["physical_only_candidate_hits"]) for row in valid)
        neither = sum(int(row["neither_candidate_hit"]) for row in valid)
        sem = s / n if n else float("nan")
        phys = p / n if n else float("nan")
        env_rows.append({
            "environment_id": environment, "relations": len(ids),
            "physical_evaluable_queries": n, "semantic_candidate_hits": s,
            "physical_candidate_hits": p,
            "both_candidate_hits": both, "semantic_only_candidate_hits": semantic_only,
            "physical_only_candidate_hits": physical_only, "neither_candidate_hit": neither,
            "semantic_only_candidate_rate": semantic_only / n if n else float("nan"),
            "physical_only_candidate_rate": physical_only / n if n else float("nan"),
            "semantic_candidate_recall_matched": sem,
            "physical_candidate_recall_matched": phys,
            "semantic_minus_physical_pp": 100 * (sem - phys) if n else float("nan"),
        })

    paired = [row for row in env_rows if int(row["physical_evaluable_queries"]) > 0]
    sem_values = np.asarray([float(row["semantic_candidate_recall_matched"]) for row in paired])
    phys_values = np.asarray([float(row["physical_candidate_recall_matched"]) for row in paired])
    gaps = 100 * (sem_values - phys_values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(gaps), size=(bootstrap_samples, len(gaps)))
    boot = gaps[draws].mean(axis=1)
    pooled_n = sum(int(row["physical_evaluable_queries"]) for row in per_relation)
    pooled_sem = sum(int(row["semantic_candidate_hits_matched"]) for row in per_relation) / pooled_n
    pooled_phys = sum(int(row["physical_candidate_hits"]) for row in per_relation) / pooled_n
    pooled_semantic_only = sum(int(row["semantic_only_candidate_hits"]) for row in per_relation) / pooled_n
    pooled_physical_only = sum(int(row["physical_only_candidate_hits"]) for row in per_relation) / pooled_n
    pooled_both = sum(int(row["both_candidate_hits"]) for row in per_relation) / pooled_n
    pooled_neither = sum(int(row["neither_candidate_hit"]) for row in per_relation) / pooled_n
    env_macro_semantic_only = float(np.mean([float(row["semantic_only_candidate_rate"]) for row in paired]))
    summary = {
        "status": "complete", "posthoc": True,
        "relations": len(per_relation), "environments": len(env_rows),
        "environments_with_evaluable_queries": len(paired),
        "matched_physical_evaluable_queries": pooled_n,
        "pooled_semantic_candidate_recall": pooled_sem,
        "pooled_physical_candidate_recall": pooled_phys,
        "pooled_semantic_minus_physical_pp": 100 * (pooled_sem - pooled_phys),
        "pooled_both_candidate_rate": pooled_both,
        "pooled_semantic_only_candidate_rate": pooled_semantic_only,
        "pooled_physical_only_candidate_rate": pooled_physical_only,
        "pooled_neither_candidate_rate": pooled_neither,
        "environment_macro_semantic_candidate_recall": float(sem_values.mean()),
        "environment_macro_physical_candidate_recall": float(phys_values.mean()),
        "environment_macro_gap_pp": float(gaps.mean()),
        "environment_macro_semantic_only_candidate_rate": env_macro_semantic_only,
        "environment_bootstrap_95ci_gap_pp": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "environments_semantic_higher": int((gaps > 1e-12).sum()),
        "environments_tied": int((np.abs(gaps) <= 1e-12).sum()),
        "environments_physical_higher": int((gaps < -1e-12).sum()),
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "bootstrap_unit": "environment; all relations within an environment stay grouped",
        "matched_denominator": "queries with an unambiguous mapped physical objectId; semantic and physical hit scored on the identical top-20 candidate set",
    }
    write_csv(run_dir / "per_relation.csv", per_relation)
    write_csv(run_dir / "per_environment.csv", env_rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# ViT-G semantic vs physical-instance candidate recall",
        "",
        "Post-hoc audit of the 110 official 3RScan validation relations (47 environments). Both hit indicators are evaluated on the same descriptor-derived top-20 list. The primary denominator is the identical set of queries whose physical objectId has an unambiguous cross-scan mapping. This is a candidate-availability analysis before spatial consensus, not an independent validation split.",
        "",
        f"- Matched evaluable queries: {pooled_n}",
        f"- Pooled semantic candidate recall: {pooled_sem:.1%}",
        f"- Pooled physical-instance candidate recall: {pooled_phys:.1%}",
        f"- Environment-macro semantic recall: {sem_values.mean():.1%}",
        f"- Environment-macro physical-instance recall: {phys_values.mean():.1%}",
        f"- Paired environment-macro gap (semantic minus physical): {gaps.mean():+.1f} pp (95% cluster bootstrap CI [{np.quantile(boot, .025):+.1f}, {np.quantile(boot, .975):+.1f}] pp)",
        f"- Environments with positive / tied / negative gap: {summary['environments_semantic_higher']} / {summary['environments_tied']} / {summary['environments_physical_higher']}",
        f"- Joint candidate outcomes, pooled (both / semantic-only / physical-only / neither): {pooled_both:.1%} / {pooled_semantic_only:.1%} / {pooled_physical_only:.1%} / {pooled_neither:.1%}",
        f"- Semantic-only availability, environment-macro: {env_macro_semantic_only:.1%}; physical-only availability: {float(np.mean([float(row['physical_only_candidate_rate']) for row in paired])):.1%}",
        "",
        "The bootstrap resamples environments and keeps all relations within each sampled environment together. This post-hoc diagnostic characterizes this validation set; it does not establish cross-site generalization or a causal descriptor-specific effect.",
    ]
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261004_vitg_validation_candidate_audit"))
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=10_042_026)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pairs = read_rows(args.pair_manifest)
    environments = {row["environment_id"] for row in pairs}
    if len(pairs) != 110 or len(environments) != 47:
        raise ValueError(f"expected 110 relations and 47 environments; found {len(pairs)} and {len(environments)}")
    if any(row["reference_scan"] != row["environment_id"] for row in pairs):
        raise ValueError("manifest contains a reference/environment mismatch")
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    scans = sorted({row["reference_scan"] for row in pairs} | {row["rescan_scan"] for row in pairs})
    missing = [scan for scan in scans if any(not (args.data_root / scan / filename).is_file() for filename in (
        "labels.instances.annotated.v2.ply", "semseg.v2.json", "sequence.zip"))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "planned", "posthoc": True, "relations": len(pairs), "environments": len(environments),
        "unique_scans": len(scans), "missing_scans": missing,
        "model": "frozen DINOv2 ViT-G/14", "layers": [2, 20, 40], "top_k": 20,
        "frames_per_scan": 20, "resolution": 224, "threshold_m": 0.10,
        "iterations": 1500, "ransac_seed": 17,
        "audits": ["semantic top-20 hit", "physical-instance top-20 hit"],
        "bootstrap_unit": "47 environments", "bootstrap_samples": args.bootstrap_samples,
        "execute_requested": args.execute,
        "purpose": "measure semantic and physical candidate availability on the identical top-20 set and matched physical-evaluable query denominator",
    }
    (args.output_dir / "protocol_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Dry run only; pass --execute to evaluate the fixed post-hoc audit.")
        return
    if missing:
        raise FileNotFoundError(f"validation inputs missing for {len(missing)} scans; see protocol_plan.json")

    args.scratch_root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, object]] = []
    generate = Path(__file__).with_name("generate_3rscan_pairs.py")
    evaluate = Path(__file__).with_name("evaluate_3rscan_spatial_consensus.py")
    manifest_path = args.output_dir / "run_manifest.csv"
    for index, pair in enumerate(pairs):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        relation = args.output_dir / "relations" / run_id
        vitg = relation / "vitg"
        row: dict[str, object] = {key: pair[key] for key in PAIR_COLUMNS}
        row.update({"run_id": run_id, "output_dir": str(vitg)})
        if has_audit(vitg):
            row["status"] = "complete"
            run_rows.append(row)
            write_csv(manifest_path, run_rows)
            continue
        relation.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"wc_candidate_{index:03d}_", dir=args.scratch_root) as scratch:
            pair_run = Path(scratch) / "pair"
            generation = [
                sys.executable, str(generate), "--root", str(args.data_root), "--metadata", str(args.metadata),
                "--reference", pair["reference_scan"], "--rescan", pair["rescan_scan"], "--output-dir", str(pair_run),
                "--resolution", "224", "--reference-frames", "20", "--rescan-frames", "20",
            ]
            evaluation = [
                sys.executable, str(evaluate), "--pair-run", str(pair_run), "--annotation-root", str(args.data_root),
                "--output-dir", str(vitg), "--model-size", "giant", "--layers", "2", "20", "40",
                "--top-k", "20", "--threshold", "0.10", "--iterations", "1500",
                "--save-query-diagnostics", "--save-candidate-set-audit", "--save-physical-instance-candidate-audit",
            ]
            log_path = relation / "runner.log"
            with log_path.open("w", encoding="utf-8") as log:
                log.write("$ " + " ".join(generation) + "\n")
                log.flush()
                subprocess.run(generation, check=True, stdout=log, stderr=subprocess.STDOUT)
                log.write("\n$ " + " ".join(evaluation) + "\n")
                log.flush()
                subprocess.run(evaluation, check=True, stdout=log, stderr=subprocess.STDOUT)
        if not has_audit(vitg):
            raise RuntimeError(f"candidate audit incomplete for {run_id}; see {relation / 'runner.log'}")
        row["status"] = "complete"
        run_rows.append(row)
        write_csv(manifest_path, run_rows)
        print(f"completed {index + 1}/{len(pairs)}: {run_id}", flush=True)

    summary = score_outputs(args.output_dir, pairs, args.bootstrap_samples, args.bootstrap_seed)
    (args.output_dir / "protocol_status.json").write_text(json.dumps({
        "status": "complete", "relations": len(pairs), "environments": len(environments),
        "candidate_audit": str(args.output_dir), "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
