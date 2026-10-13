#!/usr/bin/env python3
"""Aggregate dense MASt3R RANSAC seed sensitivity against ViT-G."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def relation_bootstrap(frame: pd.DataFrame, left: str, right: str,
                       samples: int, seed: int) -> dict:
    den = frame.linked_queries.to_numpy(dtype=np.float64)
    delta = (frame[left] - frame[right]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(frame), size=(samples, len(frame)))
    boot = delta[draws].sum(axis=1) / den[draws].sum(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "point_difference": float(delta.sum() / den.sum()),
        "relation_bootstrap_95_ci": [float(lo), float(hi)],
        "relations": int(len(frame)), "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs/20261003_mast3r_dense_seed_sweep"))
    parser.add_argument("--audit-dir", type=Path, default=Path("outputs/reports/20261003_mast3r_dense_train_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261003_mast3r_dense_seed_robustness"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda p: p if p.is_absolute() else root / p
    runs = pd.read_csv(resolve(args.run_root) / "per_relation_seed.csv")
    audit = pd.read_csv(resolve(args.audit_dir) / "per_relation.csv")
    audit = audit[audit.dino_audit_available.astype(bool)].copy()
    if len(audit) != 14 or len(runs) != 14 * len(args.seeds):
        raise ValueError("expected complete 14-relation seed audit")
    if set(runs.ransac_seed.astype(int)) != set(args.seeds):
        raise ValueError("unexpected RANSAC seeds")
    if runs.duplicated(["pair_run", "ransac_seed"]).any():
        raise ValueError("duplicate relation/seed rows")
    if not (runs.groupby("ransac_seed").linked_physical_instance_queries.sum().astype(int) == 12188).all():
        raise ValueError("linked query denominator changed")
    if set(runs.candidate_top_k.astype(int)) != {20} or set(runs.ransac_iterations.astype(int)) != {1500}:
        raise ValueError("candidate budget or consensus iteration count changed")

    old_dense = dict(zip(audit.source_run.astype(str), audit.mast3r_correct_transfers.astype(int)))
    seed17 = runs[runs.ransac_seed.astype(int) == 17]
    if len(seed17) != 14:
        raise ValueError("seed 17 required for exact reproduction audit")
    for row in seed17.itertuples(index=False):
        if int(row.physical_correct_transfers) != old_dense[str(row.pair_run).split("/")[-1]]:
            raise ValueError(f"seed-17 result mismatch: {row.pair_run}")

    audit_rows = audit.set_index("source_run")
    records = []
    for row in runs.itertuples(index=False):
        source = str(row.pair_run).split("/")[-1]
        ref = audit_rows.loc[source]
        records.append({
            "source_run": source, "ransac_seed": int(row.ransac_seed),
            "linked_queries": int(row.linked_physical_instance_queries),
            "dense_candidate_recall": float(row.physical_instance_candidate_recall_at_k),
            "dense_correct_transfers": int(row.physical_correct_transfers),
            "dino_correct_transfers": int(ref.dino_correct_transfers),
            "transfer_delta_hits": int(row.physical_correct_transfers) - int(ref.dino_correct_transfers),
            "transfer_delta": (int(row.physical_correct_transfers) - int(ref.dino_correct_transfers)) / int(row.linked_physical_instance_queries),
            "dense_pose_pass": bool(row.strict_scene_pose_success),
            "rotation_error_deg": float(row.scene_rotation_error_deg),
            "translation_error_m": float(row.scene_translation_error_m),
        })
    rel = pd.DataFrame(records)
    seed_rows, cis = [], {}
    for seed in args.seeds:
        group = rel[rel.ransac_seed == seed]
        ci = relation_bootstrap(group, "dense_correct_transfers", "dino_correct_transfers",
                                args.samples, args.bootstrap_seed + seed)
        cis[str(seed)] = ci
        denom = int(group.linked_queries.sum())
        seed_rows.append({
            "ransac_seed": seed, "relations": len(group), "linked_queries": denom,
            "weighted_candidate_recall": float((group.dense_candidate_recall * group.linked_queries).sum() / denom),
            "weighted_transfer_coverage": float(group.dense_correct_transfers.sum() / denom),
            "transfer_difference_vs_vitg": ci["point_difference"],
            "transfer_ci_lower": ci["relation_bootstrap_95_ci"][0],
            "transfer_ci_upper": ci["relation_bootstrap_95_ci"][1],
            "strict_pose_successes": int(group.dense_pose_pass.sum()),
            "relations_above_vitg": int((group.transfer_delta_hits > 0).sum()),
            "relations_equal_vitg": int((group.transfer_delta_hits == 0).sum()),
            "relations_below_vitg": int((group.transfer_delta_hits < 0).sum()),
        })
    by_seed = pd.DataFrame(seed_rows)
    summary = {
        "protocol": {
            "relations": 14, "linked_queries": 12188,
            "candidate_cache": "fixed per-relation dense MASt3R top-20 candidates",
            "consensus": "1,500 iterations; 0.10 m threshold; only random seed changes",
            "seeds": [int(x) for x in args.seeds],
        },
        "seed17_exact_reproduction": True,
        "weighted_transfer_by_seed": dict(zip(by_seed.ransac_seed.astype(str), by_seed.weighted_transfer_coverage.astype(float))),
        "weighted_candidate_recall_by_seed": dict(zip(by_seed.ransac_seed.astype(str), by_seed.weighted_candidate_recall.astype(float))),
        "strict_pose_successes_by_seed": dict(zip(by_seed.ransac_seed.astype(str), by_seed.strict_pose_successes.astype(int))),
        "transfer_difference_vs_vitg": cis,
        "mean_transfer_across_seeds": float(by_seed.weighted_transfer_coverage.mean()),
        "sample_sd_transfer_across_seeds": float(by_seed.weighted_transfer_coverage.std(ddof=1)),
        "relation_direction_counts": {
            str(int(row.ransac_seed)): {
                "higher": int(row.relations_above_vitg),
                "equal": int(row.relations_equal_vitg),
                "lower": int(row.relations_below_vitg),
            } for row in by_seed.itertuples(index=False)
        },
        "caveats": [
            "RANSAC seeds are sensitivity settings on one fixed dataset and candidate set, not independent replications.",
            "All relations are availability-selected official-train evidence; physical-site independence and external validation are not established.",
            "Correct transfer and strict global-pose recovery are separate outcomes.",
        ],
    }
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out / "per_relation_seed.csv", index=False)
    by_seed.to_csv(out / "summary_by_seed.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Dense MASt3R RANSAC seed robustness", "",
        "The dense MASt3R top-20 candidates, query population, 0.10 m threshold, and 1,500 consensus iterations are fixed; only the RANSAC seed changes. Seed 17 reproduces all 14 old transfer counts exactly.", "",
        "| Seed | Candidate Recall@20 | Correct transfer | Difference vs ViT-G (95% relation-bootstrap CI) | Strict pose | Relation transfer wins/ties/losses |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_seed.itertuples(index=False):
        lines.append(
            f"| {row.ransac_seed} | {row.weighted_candidate_recall:.1%} | {row.weighted_transfer_coverage:.1%} | "
            f"{row.transfer_difference_vs_vitg*100:+.1f} pp [{row.transfer_ci_lower*100:+.1f}, {row.transfer_ci_upper*100:+.1f}] | "
            f"{row.strict_pose_successes}/14 | {row.relations_above_vitg}/{row.relations_equal_vitg}/{row.relations_below_vitg} |"
        )
    lines.extend([
        "",
        f"Across seeds, pooled transfer ranges from {by_seed.weighted_transfer_coverage.min():.1%} to {by_seed.weighted_transfer_coverage.max():.1%} (mean {summary['mean_transfer_across_seeds']:.1%}; sample SD {summary['sample_sd_transfer_across_seeds']:.1%}). Candidate recall does not change because candidates are frozen. Strict pose successes are {', '.join(f'{int(r.ransac_seed)}:{int(r.strict_pose_successes)}/14' for r in by_seed.itertuples(index=False))}.",
        "",
        "The relation-bootstrap intervals describe variation across these selected train relations conditional on this fixed snapshot. They do not make RANSAC seeds independent replications or establish external validation.",
        "",
        "Reproduce candidate extraction and consensus reruns with python scripts/run_mast3r_dense_seed_robustness.py --dataset-root PATH/3rscan --metadata PATH/3rscan/3RScan.json --mast3r-root PATH/mast3r_source.",
    ])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": summary, "output_dir": str(out.relative_to(root))}, indent=2))


if __name__ == "__main__":
    main()
