#!/usr/bin/env python3
"""Compare semantic and linked-instance candidate recall on shared 3RScan queries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--vitg-dir", type=Path, default=Path("outputs/reports/20261004_vitg_validation_candidate_audit/relations"))
    parser.add_argument("--mast3r-dir", type=Path, default=Path("outputs/reports/20261004_mast3r_dense_official_validation/relations"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261004_cross_model_candidate_semantic_physical"))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=10_042_026)
    args = parser.parse_args()

    pairs = rows(args.pair_manifest)
    envs = {pair["environment_id"] for pair in pairs}
    if len(pairs) != 110 or len(envs) != 47:
        raise ValueError(f"expected 110 relations / 47 environments, got {len(pairs)} / {len(envs)}")

    per_relation: list[dict[str, object]] = []
    env_indices: dict[str, list[int]] = {}
    matched_total = 0
    for i, pair in enumerate(pairs):
        run_id = f"{i:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        vitg_path = args.vitg_dir / run_id / "vitg" / "query_diagnostics.csv"
        cache_path = args.mast3r_dir / run_id / "candidate_cache.npz"
        vitg = rows(vitg_path)
        cache = np.load(cache_path)
        n = len(vitg)
        if n != len(cache["query_points"]):
            raise ValueError(f"query count differs for {run_id}: ViT-G={n}, MASt3R={len(cache['query_points'])}")
        if any(int(row["query_index"]) != q for q, row in enumerate(vitg)):
            raise ValueError(f"ViT-G query order is not canonical for {run_id}")
        checks = (
            (np.asarray([int(row["frame_id"]) for row in vitg]) == cache["query_frame_ids"], "frame IDs"),
            (np.asarray([int(row["query_object_id"]) for row in vitg]) == cache["query_object_ids"], "object IDs"),
            (np.asarray([int(row["query_semantic_id"]) for row in vitg]) == cache["query_semantics"], "semantic IDs"),
            (np.asarray([row["physical_instance_evaluable"].lower() == "true" for row in vitg]) == cache["physical_evaluable"], "physical-evaluable mask"),
            (np.asarray([int(row["top_k"]) == 20 for row in vitg]), "ViT-G K=20"),
        )
        for ok, label in checks:
            if not bool(np.asarray(ok).all()):
                raise ValueError(f"query population mismatch ({label}) for {run_id}")

        matched = cache["physical_evaluable"].astype(bool)
        count = int(matched.sum())
        if not np.all(cache["candidate_count"][matched] == 20):
            raise ValueError(f"MASt3R does not have 20 unique candidates for every matched query in {run_id}")
        matched_total += count
        v_sem = np.asarray([row["candidate_semantic_hit"].lower() == "true" for row in vitg])
        v_phys = np.asarray([row["physical_instance_candidate_hit"].lower() == "true" for row in vitg])
        m_sem = cache["semantic_candidate_hit"].astype(bool)
        m_phys = cache["physical_candidate_hit"].astype(bool)
        if not count:
            raise ValueError(f"no physically evaluable queries in {run_id}")
        record: dict[str, object] = {
            "run_id": run_id, "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"], "rescan_scan": pair["rescan_scan"],
            "matched_queries": count,
            "vitg_semantic_hits": int((matched & v_sem).sum()),
            "vitg_physical_hits": int((matched & v_phys).sum()),
            "mast3r_semantic_hits": int((matched & m_sem).sum()),
            "mast3r_physical_hits": int((matched & m_phys).sum()),
        }
        per_relation.append(record)
        env_indices.setdefault(pair["environment_id"], []).append(len(per_relation) - 1)

    per_environment: list[dict[str, object]] = []
    for env, indices in sorted(env_indices.items()):
        rels = [per_relation[index] for index in indices]
        denom = sum(int(row["matched_queries"]) for row in rels)
        rates = {}
        for method in ("vitg", "mast3r"):
            sem = sum(int(row[f"{method}_semantic_hits"]) for row in rels) / denom
            phys = sum(int(row[f"{method}_physical_hits"]) for row in rels) / denom
            rates[f"{method}_semantic_recall"] = sem
            rates[f"{method}_physical_recall"] = phys
            rates[f"{method}_gap"] = sem - phys
        per_environment.append({"environment_id": env, "relations": len(indices), "matched_queries": denom, **rates})

    matrix = np.asarray([[
        float(row["vitg_semantic_recall"]), float(row["vitg_physical_recall"]),
        float(row["mast3r_semantic_recall"]), float(row["mast3r_physical_recall"]),
    ] for row in per_environment], dtype=np.float64)
    vitg_gap = 100 * (matrix[:, 0] - matrix[:, 1])
    mast3r_gap = 100 * (matrix[:, 2] - matrix[:, 3])
    gap_delta = vitg_gap - mast3r_gap
    rng = np.random.default_rng(args.bootstrap_seed)
    sample = rng.integers(0, len(envs), size=(args.bootstrap_samples, len(envs)))

    def ci(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.quantile(values[sample].mean(axis=1), [0.025, 0.975])]

    method_rows = []
    for method, sem_i, phys_i, gap in (("ViT-G", 0, 1, vitg_gap), ("Dense MASt3R", 2, 3, mast3r_gap)):
        method_rows.append({
            "method": method,
            "semantic_candidate_recall_environment_macro": float(matrix[:, sem_i].mean()),
            "physical_candidate_recall_environment_macro": float(matrix[:, phys_i].mean()),
            "semantic_minus_physical_gap_pp": float(gap.mean()),
            "gap_ci95_pp": ci(gap),
        })
    pooled = {}
    for method, key in (("ViT-G", "vitg"), ("Dense MASt3R", "mast3r")):
        pooled[method] = {
            "semantic": sum(int(r[f"{key}_semantic_hits"]) for r in per_relation) / matched_total,
            "physical": sum(int(r[f"{key}_physical_hits"]) for r in per_relation) / matched_total,
        }
    for row in method_rows:
        row["semantic_candidate_recall_pooled"] = pooled[row["method"]]["semantic"]
        row["physical_candidate_recall_pooled"] = pooled[row["method"]]["physical"]

    overlap_excluded = {
        "4fbad31e-465b-2a5d-84b7-c0ddea978db4",
        "c7895f7c-339c-2d13-819f-3bb0b26c91f6",
    }
    sensitivity_env = [row for row in per_environment if row["environment_id"] not in overlap_excluded]
    if len(sensitivity_env) != 45 or {row["environment_id"] for row in per_environment if row["environment_id"] in overlap_excluded} != overlap_excluded:
        raise ValueError("expected both previously flagged overlap-target environments and 45 remaining environments")
    sensitivity_matrix = np.asarray([[
        float(row["vitg_semantic_recall"]), float(row["vitg_physical_recall"]),
        float(row["mast3r_semantic_recall"]), float(row["mast3r_physical_recall"]),
    ] for row in sensitivity_env], dtype=np.float64)
    sensitivity_vitg_gap = 100 * (sensitivity_matrix[:, 0] - sensitivity_matrix[:, 1])
    sensitivity_mast3r_gap = 100 * (sensitivity_matrix[:, 2] - sensitivity_matrix[:, 3])
    sensitivity_gap_delta = sensitivity_vitg_gap - sensitivity_mast3r_gap
    sensitivity_rng = np.random.default_rng(args.bootstrap_seed)
    sensitivity_sample = sensitivity_rng.integers(0, len(sensitivity_env), size=(args.bootstrap_samples, len(sensitivity_env)))

    def sensitivity_ci(values: np.ndarray) -> list[float]:
        return [float(x) for x in np.quantile(values[sensitivity_sample].mean(axis=1), [0.025, 0.975])]

    overlap_sensitivity = {
        "excluded_environments": sorted(overlap_excluded),
        "remaining_environments": len(sensitivity_env),
        "vitg_gap_pp": float(sensitivity_vitg_gap.mean()),
        "vitg_gap_ci95_pp": sensitivity_ci(sensitivity_vitg_gap),
        "mast3r_gap_pp": float(sensitivity_mast3r_gap.mean()),
        "mast3r_gap_ci95_pp": sensitivity_ci(sensitivity_mast3r_gap),
        "gap_difference_vitg_minus_mast3r_pp": float(sensitivity_gap_delta.mean()),
        "gap_difference_ci95_pp": sensitivity_ci(sensitivity_gap_delta),
        "vitg_positive_tied_negative": [
            int((sensitivity_vitg_gap > 1e-12).sum()), int((np.abs(sensitivity_vitg_gap) <= 1e-12).sum()),
            int((sensitivity_vitg_gap < -1e-12).sum()),
        ],
        "mast3r_positive_tied_negative": [
            int((sensitivity_mast3r_gap > 1e-12).sum()), int((np.abs(sensitivity_mast3r_gap) <= 1e-12).sum()),
            int((sensitivity_mast3r_gap < -1e-12).sum()),
        ],
    }

    summary = {
        "status": "complete", "posthoc": True, "relations": len(pairs), "environments": len(envs),
        "matched_queries": matched_total, "candidate_budget": 20,
        "query_identity_verified": True, "query_fields_verified": ["frame_id", "objectId", "semantic_id", "physical_evaluable"],
        "method_results": method_rows,
        "vitg_minus_mast3r_gap_difference_pp": float(gap_delta.mean()),
        "vitg_minus_mast3r_gap_difference_ci95_pp": ci(gap_delta),
        "overlap_target_exclusion_sensitivity": overlap_sensitivity,
        "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "environment, with all relations grouped",
        "timing_note": "exploratory comparison specified after the primary ViT-G semantic-versus-physical audit; MASt3R dense ranking was fixed on the prior train audit but applied after its sparse validation result was known",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol_plan.json").write_text(json.dumps({
        "status": "complete", "posthoc": True, "relations": 110, "environments": 47,
        "matched_queries": matched_total, "candidate_budget": 20,
        "denominator": "physical-instance-evaluable query intersection shared by both methods",
        "identity_checks": summary["query_fields_verified"], "bootstrap_unit": summary["bootstrap_unit"],
        "overlap_target_exclusion_sensitivity": overlap_sensitivity,
        "timing_note": summary["timing_note"],
    }, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "per_relation.csv", per_relation)
    write_csv(args.output_dir / "per_environment.csv", per_environment)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Matched semantic versus physical candidate availability across methods", "",
        "Post-hoc comparison of frozen ViT-G retrieval and the custom dense MASt3R top-20 procedure on the same 110 official validation relations. Both methods use the same 98,221 query points with unambiguous object links and exactly 20 candidates per matched query. Query order, frame, objectId, semantic label, and physical-evaluable mask were checked relation by relation.", "",
    ]
    for result in method_rows:
        report.append(
            f"- {result['method']}: semantic Recall@20 {result['semantic_candidate_recall_environment_macro']:.1%} environment-macro ({result['semantic_candidate_recall_pooled']:.1%} pooled), physical-instance Recall@20 {result['physical_candidate_recall_environment_macro']:.1%} ({result['physical_candidate_recall_pooled']:.1%} pooled), gap {result['semantic_minus_physical_gap_pp']:+.2f} pp (95% environment bootstrap CI [{result['gap_ci95_pp'][0]:+.2f},{result['gap_ci95_pp'][1]:+.2f}])."
        )
    report.extend([
        f"- ViT-G minus dense MASt3R in the semantic-to-physical gap: {summary['vitg_minus_mast3r_gap_difference_pp']:+.2f} pp (95% paired environment bootstrap CI [{summary['vitg_minus_mast3r_gap_difference_ci95_pp'][0]:+.2f},{summary['vitg_minus_mast3r_gap_difference_ci95_pp'][1]:+.2f}]).",
        f"- Excluding the two previously flagged overlap-target environments, the gaps remain {overlap_sensitivity['vitg_gap_pp']:+.2f} pp for ViT-G (95% CI [{overlap_sensitivity['vitg_gap_ci95_pp'][0]:+.2f},{overlap_sensitivity['vitg_gap_ci95_pp'][1]:+.2f}]) and {overlap_sensitivity['mast3r_gap_pp']:+.2f} pp for dense MASt3R (95% CI [{overlap_sensitivity['mast3r_gap_ci95_pp'][0]:+.2f},{overlap_sensitivity['mast3r_gap_ci95_pp'][1]:+.2f}]) over 45 environments.",
        "",
        "The dense MASt3R ranking rule was fixed from the prior train-split audit, but this full-validation application followed the sparse MASt3R validation result and this comparison was specified after inspecting the ViT-G audit. Treat it as exploratory cross-procedure evidence, not an independent replication, site-generalization estimate, or descriptor-only causal comparison. The published sparse reciprocal MASt3R result has fewer than 20 actual candidates per query and is not used here.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
