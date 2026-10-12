#!/usr/bin/env python3
"""Compare semantic and linked physical-instance candidate availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_vitl_vitg_candidate_audit import PAIR_TO_G_AUDIT, as_bool


QUERY_KEYS = ("query_index", "frame_id")
MATCH_FIELDS = (
    "query_evaluable",
    "query_semantic_id",
    "query_object_id",
    "mapped_reference_object_id",
    "physical_instance_evaluable",
)


def read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values(list(QUERY_KEYS)).reset_index(drop=True)
    required = {
        *QUERY_KEYS,
        *MATCH_FIELDS,
        "candidate_semantic_hit",
        "physical_instance_candidate_hit",
        "consensus_inlier",
        "physical_instance_correct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    if set(frame.top_k.unique()) != {20} or set(frame.geometry_weight.unique()) != {0.0}:
        raise ValueError(f"{path}: expected only top_k=20 and geometry_weight=0")
    return frame


def one_pair(l: pd.DataFrame, g: pd.DataFrame, pair: str) -> dict[str, object]:
    if len(l) != len(g):
        raise ValueError(f"{pair}: query counts differ")
    for col in (*QUERY_KEYS, *MATCH_FIELDS):
        if col in {"query_evaluable", "physical_instance_evaluable"}:
            equal = np.array_equal(as_bool(l[col]), as_bool(g[col]))
        else:
            equal = np.array_equal(l[col].to_numpy(), g[col].to_numpy())
        if not equal:
            raise ValueError(f"{pair}: L/G alignment differs in {col}")

    sem_eval = as_bool(l.query_evaluable)
    phys_eval = as_bool(l.physical_instance_evaluable)
    if np.any(phys_eval & ~sem_eval):
        raise ValueError(f"{pair}: physical links include semantically unevaluable queries")
    sem_l = as_bool(l.candidate_semantic_hit) & sem_eval
    sem_g = as_bool(g.candidate_semantic_hit) & sem_eval
    phys_l = as_bool(l.physical_instance_candidate_hit) & phys_eval
    phys_g = as_bool(g.physical_instance_candidate_hit) & phys_eval
    correct_l = as_bool(l.consensus_inlier) & as_bool(l.physical_instance_correct) & phys_eval
    correct_g = as_bool(g.consensus_inlier) & as_bool(g.physical_instance_correct) & phys_eval
    hits_l, hits_g = int(phys_l.sum()), int(phys_g.sum())
    linked_sem_l, linked_sem_g = int((sem_l & phys_eval).sum()), int((sem_g & phys_eval).sum())
    return {
        "pair": pair,
        "semantic_queries": int(sem_eval.sum()),
        "linked_queries": int(phys_eval.sum()),
        "linked_query_fraction": float(phys_eval.sum() / sem_eval.sum()) if sem_eval.sum() else 0.0,
        "semantic_recall_vitl": float(sem_l.sum() / sem_eval.sum()) if sem_eval.sum() else 0.0,
        "semantic_recall_vitg": float(sem_g.sum() / sem_eval.sum()) if sem_eval.sum() else 0.0,
        "semantic_recall_delta": float((sem_g.sum() - sem_l.sum()) / sem_eval.sum()) if sem_eval.sum() else 0.0,
        "linked_semantic_hits_vitl": linked_sem_l,
        "linked_semantic_hits_vitg": linked_sem_g,
        "linked_semantic_recall_vitl": float(linked_sem_l / phys_eval.sum()) if phys_eval.sum() else 0.0,
        "linked_semantic_recall_vitg": float(linked_sem_g / phys_eval.sum()) if phys_eval.sum() else 0.0,
        "physical_hits_vitl": hits_l,
        "physical_hits_vitg": hits_g,
        "physical_recall_vitl": float(hits_l / phys_eval.sum()) if phys_eval.sum() else 0.0,
        "physical_recall_vitg": float(hits_g / phys_eval.sum()) if phys_eval.sum() else 0.0,
        "physical_recall_delta": float((hits_g - hits_l) / phys_eval.sum()) if phys_eval.sum() else 0.0,
        "physical_correct_inliers_vitl": int(correct_l.sum()),
        "physical_correct_inliers_vitg": int(correct_g.sum()),
        "physical_recovery_vitl": float(correct_l.sum() / hits_l) if hits_l else 0.0,
        "physical_recovery_vitg": float(correct_g.sum() / hits_g) if hits_g else 0.0,
        "physical_recovery_delta": float((correct_g.sum() / hits_g if hits_g else 0.0) - (correct_l.sum() / hits_l if hits_l else 0.0)),
    }


def bootstrap(rows: list[dict[str, object]], field_num: str, field_den: str, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(rows)
    nums = np.asarray([float(r[field_num]) for r in rows])
    dens = np.asarray([float(r[field_den]) for r in rows])
    draws = rng.integers(0, n, size=(samples, n))
    values = nums[draws].sum(axis=1) / np.maximum(dens[draws].sum(axis=1), 1)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitl-root", type=Path, required=True)
    parser.add_argument("--vitg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair in PAIR_TO_G_AUDIT:
        l = read(args.vitl_root / pair / "query_diagnostics.csv")
        g = read(args.vitg_root / pair / "query_diagnostics.csv")
        rows.append(one_pair(l, g, pair))
    if len(rows) != 14:
        raise ValueError(f"expected 14 matched relations, got {len(rows)}")

    splits = {
        "matched_14_including_target": rows,
        "matched_13_source_only": [r for r in rows if "target" not in str(r["pair"])],
    }
    summary: dict[str, object] = {
        "protocol": {
            "candidate_set_size": 20,
            "vitl_layers": [2, 12, 24],
            "vitg_layers": [2, 20, 40],
            "bootstrap_unit": "relation; queries remain grouped within relation",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "instance_identity": (
                "scan-local objectId linked by official rigid relation metadata; "
                "unchanged shared objectIds with matching semantic labels; "
                "ambiguity-group IDs excluded"
            ),
            "labels_used_for_candidate_construction": False,
        },
        "splits": {},
    }
    for i, (name, subset) in enumerate(splits.items()):
        linked = sum(int(r["linked_queries"]) for r in subset)
        semantic_n = sum(int(r["semantic_queries"]) for r in subset)
        hits_l = sum(int(r["physical_hits_vitl"]) for r in subset)
        hits_g = sum(int(r["physical_hits_vitg"]) for r in subset)
        linked_sem_l = sum(int(r["linked_semantic_hits_vitl"]) for r in subset)
        linked_sem_g = sum(int(r["linked_semantic_hits_vitg"]) for r in subset)
        recovered_l = sum(int(r["physical_correct_inliers_vitl"]) for r in subset)
        recovered_g = sum(int(r["physical_correct_inliers_vitg"]) for r in subset)
        num_delta = hits_g - hits_l
        recovery_draws = []
        rng = np.random.default_rng(args.seed + i * 1009)
        draws = rng.integers(0, len(subset), size=(args.bootstrap_samples, len(subset)))
        gl = np.asarray([int(r["physical_hits_vitg"]) for r in subset], dtype=np.float64)
        ll = np.asarray([int(r["physical_hits_vitl"]) for r in subset], dtype=np.float64)
        gr = np.asarray([int(r["physical_correct_inliers_vitg"]) for r in subset], dtype=np.float64)
        lr = np.asarray([int(r["physical_correct_inliers_vitl"]) for r in subset], dtype=np.float64)
        rec = gr[draws].sum(axis=1) / np.maximum(gl[draws].sum(axis=1), 1) - lr[draws].sum(axis=1) / np.maximum(ll[draws].sum(axis=1), 1)
        recovery_ci = [float(x) for x in np.quantile(rec, [0.025, 0.975])]
        gap_l = linked_sem_l - hits_l
        gap_g = linked_sem_g - hits_g
        gap_rng = np.random.default_rng(args.seed + i * 1009 + 19)
        gap_draws = gap_rng.integers(0, len(subset), size=(args.bootstrap_samples, len(subset)))
        gap_l_per_pair = np.asarray([int(r["linked_semantic_hits_vitl"]) - int(r["physical_hits_vitl"]) for r in subset])
        gap_g_per_pair = np.asarray([int(r["linked_semantic_hits_vitg"]) - int(r["physical_hits_vitg"]) for r in subset])
        linked_per_pair = np.asarray([int(r["linked_queries"]) for r in subset])
        gap_l_draw = gap_l_per_pair[gap_draws].sum(axis=1) / np.maximum(linked_per_pair[gap_draws].sum(axis=1), 1)
        gap_g_draw = gap_g_per_pair[gap_draws].sum(axis=1) / np.maximum(linked_per_pair[gap_draws].sum(axis=1), 1)
        summary["splits"][name] = {
            "relations": len(subset),
            "semantic_queries": semantic_n,
            "linked_instance_queries": linked,
            "linked_instance_query_fraction": linked / semantic_n if semantic_n else 0.0,
            "physical_instance_recall_vitl": hits_l / linked if linked else 0.0,
            "physical_instance_recall_vitg": hits_g / linked if linked else 0.0,
            "linked_semantic_recall_vitl": linked_sem_l / linked if linked else 0.0,
            "linked_semantic_recall_vitg": linked_sem_g / linked if linked else 0.0,
            "physical_instance_recall_delta_g_minus_l": num_delta / linked if linked else 0.0,
            "semantic_to_physical_instance_gap_vitl": gap_l / linked if linked else 0.0,
            "semantic_to_physical_instance_gap_vitg": gap_g / linked if linked else 0.0,
            "semantic_to_physical_instance_gap_pair_bootstrap_95pct_vitl": [
                float(x) for x in np.quantile(gap_l_draw, [0.025, 0.975])
            ],
            "semantic_to_physical_instance_gap_pair_bootstrap_95pct_vitg": [
                float(x) for x in np.quantile(gap_g_draw, [0.025, 0.975])
            ],
            "recovery_given_physical_candidate_vitl": recovered_l / hits_l if hits_l else 0.0,
            "recovery_given_physical_candidate_vitg": recovered_g / hits_g if hits_g else 0.0,
            "recovery_delta_g_minus_l": recovered_g / hits_g - recovered_l / hits_l if hits_g and hits_l else 0.0,
            "recovery_delta_pair_bootstrap_95pct": recovery_ci,
        }
        # Compute the paired delta interval directly, keeping the same resampled
        # relation indices in each backbone arm.
        rng_delta = np.random.default_rng(args.seed + i * 1009 + 37)
        draw_delta = rng_delta.integers(0, len(subset), size=(args.bootstrap_samples, len(subset)))
        deltas = (gl[draw_delta].sum(axis=1) - ll[draw_delta].sum(axis=1)) / np.maximum(
            np.asarray([int(r["linked_queries"]) for r in subset])[draw_delta].sum(axis=1), 1
        )
        summary["splits"][name]["physical_instance_recall_delta_pair_bootstrap_95pct"] = [
            float(x) for x in np.quantile(deltas, [0.025, 0.975])
        ]

    per_relation = pd.DataFrame(rows)
    per_relation.to_csv(args.output_dir / "per_relation.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Semantic vs linked physical-instance candidate availability",
        "",
        "Top-20 candidates are scored after the fact. Physical-instance links use explicit 3RScan rigid mappings, unchanged shared scan-local objectIds with matching semantic labels, and exclude metadata ambiguity groups. Candidate construction and ranking never use labels.",
        "",
        "| Split | semantic queries | linked-object queries | link coverage | semantic Recall@20 L/G | physical-instance Recall@20 L/G | physical delta G−L (95% CI) | recovery given physical hit L/G | recovery delta (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["splits"].items():
        subset = splits[name]
        sem_n = sum(int(r["semantic_queries"]) for r in subset)
        sem_l = sum(int(round(float(r["semantic_recall_vitl"]) * int(r["semantic_queries"]))) for r in subset)
        sem_g = sum(int(round(float(r["semantic_recall_vitg"]) * int(r["semantic_queries"]))) for r in subset)
        p = values["physical_instance_recall_delta_pair_bootstrap_95pct"]
        r = values["recovery_delta_pair_bootstrap_95pct"]
        lines.append(
            f"| {name} | {sem_n} | {values['linked_instance_queries']} | {values['linked_instance_query_fraction']:.1%} | "
            f"{sem_l / sem_n:.1%}/{sem_g / sem_n:.1%} | "
            f"{values['physical_instance_recall_vitl']:.1%}/{values['physical_instance_recall_vitg']:.1%} | "
            f"{values['physical_instance_recall_delta_g_minus_l']:+.1%} [{p[0]:+.1%},{p[1]:+.1%}] | "
            f"{values['recovery_given_physical_candidate_vitl']:.1%}/{values['recovery_given_physical_candidate_vitg']:.1%} | "
            f"{values['recovery_delta_g_minus_l']:+.1%} [{r[0]:+.1%},{r[1]:+.1%}] |"
        )
    lines += [
        "",
        "| Split | linked-query semantic Recall@20 L/G | physical-instance Recall@20 L/G | category-to-instance drop L/G (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for name, values in summary["splits"].items():
        gl = values["semantic_to_physical_instance_gap_pair_bootstrap_95pct_vitl"]
        gg = values["semantic_to_physical_instance_gap_pair_bootstrap_95pct_vitg"]
        lines.append(
            f"| {name} | {values['linked_semantic_recall_vitl']:.1%}/{values['linked_semantic_recall_vitg']:.1%} | "
            f"{values['physical_instance_recall_vitl']:.1%}/{values['physical_instance_recall_vitg']:.1%} | "
            f"{values['semantic_to_physical_instance_gap_vitl']:.1%} [{gl[0]:.1%},{gl[1]:.1%}] / "
            f"{values['semantic_to_physical_instance_gap_vitg']:.1%} [{gg[0]:.1%},{gg[1]:.1%}] |"
        )
    lines += [
        "",
        "The physical-instance denominator is restricted to depth-valid semantic queries whose scan-local objectId has an unambiguous cross-scan link. Recovery counts only unchanged-consensus inliers whose selected reference objectId is the linked instance. Intervals resample relation pairs, not queries. These are 3RScan train-relation diagnostics and do not estimate unseen-site generalization.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
