#!/usr/bin/env python3
"""Paired pre-consensus candidate availability audit for ViT-L and ViT-G."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIR_TO_G_AUDIT = {
    "20260917_3rscan_pair02_s1": "pair02",
    "20260917_3rscan_pair03_s1": "pair03",
    "20260917_3rscan_pair04_s1": "pair04",
    "20260917_3rscan_pair05_s1": "pair05",
    "20260917_3rscan_target_pair_s1": "target",
    "20260921_3rscan_train_0cac": "0cac",
    "20260921_3rscan_train_20c993": "20c993",
    "20260921_3rscan_train_4aca_extra": "4aca",
    "20260921_3rscan_train_6bde": "6bde",
    "20260921_3rscan_train_95be": "95be",
    "20260921_3rscan_semantic_5630": "5630",
    "20260921_3rscan_semantic_751a": "751a",
    "20260921_3rscan_semantic_c670": "c670",
    "20260921_3rscan_semantic_chairs": "chairs",
}

QUERY_KEYS = ("query_index", "frame_id")
LABEL_COLUMNS = ("query_evaluable", "query_semantic_id")


def as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy()
    return values.astype(str).str.lower().eq("true").to_numpy()


def read_diagnostics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if set(frame.top_k.unique()) != {20} or set(frame.geometry_weight.unique()) != {0.0}:
        raise ValueError(f"expected only geometry_weight=0, top_k=20 in {path}")
    required = {
        *QUERY_KEYS,
        *LABEL_COLUMNS,
        "consensus_inlier",
        "semantic_correct",
        "candidate_semantic_hit",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing fields in {path}: {sorted(missing)}")
    return frame.sort_values(list(QUERY_KEYS)).reset_index(drop=True)


def one_pair(l: pd.DataFrame, g: pd.DataFrame, pair: str) -> dict[str, object]:
    if len(l) != len(g):
        raise ValueError(f"{pair}: ViT-L/ViT-G query count differs ({len(l)} vs {len(g)})")
    for column in (*QUERY_KEYS, *LABEL_COLUMNS):
        lv, gv = l[column].to_numpy(), g[column].to_numpy()
        if column == "query_evaluable":
            equal = np.array_equal(as_bool(l[column]), as_bool(g[column]))
        else:
            equal = np.array_equal(lv, gv)
        if not equal:
            raise ValueError(f"{pair}: ViT-L/ViT-G query alignment differs in {column}")

    evaluable = as_bool(l.query_evaluable)
    inlier_l, inlier_g = as_bool(l.consensus_inlier), as_bool(g.consensus_inlier)
    sem_correct_l, sem_correct_g = as_bool(l.semantic_correct), as_bool(g.semantic_correct)
    sem_hit_l, sem_hit_g = as_bool(l.candidate_semantic_hit), as_bool(g.candidate_semantic_hit)

    n = int(evaluable.sum())
    if n == 0:
        raise ValueError(f"{pair}: no evaluable queries")
    row: dict[str, object] = {"pair": pair, "queries": n}
    for metric, hits_l, hits_g, correct_l, correct_g in (
        ("semantic", sem_hit_l, sem_hit_g, inlier_l & sem_correct_l, inlier_g & sem_correct_g),
    ):
        hit_count_l = int((hits_l & evaluable).sum())
        hit_count_g = int((hits_g & evaluable).sum())
        recovered_l = int((correct_l & evaluable).sum())
        recovered_g = int((correct_g & evaluable).sum())
        row[f"{metric}_hits_vitl"] = hit_count_l
        row[f"{metric}_hits_vitg"] = hit_count_g
        row[f"{metric}_recall_vitl"] = hit_count_l / n
        row[f"{metric}_recall_vitg"] = hit_count_g / n
        row[f"{metric}_recall_delta_g_minus_l"] = (hit_count_g - hit_count_l) / n
        row[f"{metric}_correct_inliers_vitl"] = recovered_l
        row[f"{metric}_correct_inliers_vitg"] = recovered_g
        row[f"{metric}_correct_inlier_coverage_vitl"] = recovered_l / n
        row[f"{metric}_correct_inlier_coverage_vitg"] = recovered_g / n
        row[f"{metric}_recovery_given_candidate_vitl"] = recovered_l / hit_count_l if hit_count_l else 0.0
        row[f"{metric}_recovery_given_candidate_vitg"] = recovered_g / hit_count_g if hit_count_g else 0.0
    return row


def pooled(rows: list[dict[str, object]], metric: str) -> dict[str, float | int]:
    n = sum(int(row["queries"]) for row in rows)
    hits_l = sum(int(row[f"{metric}_hits_vitl"]) for row in rows)
    hits_g = sum(int(row[f"{metric}_hits_vitg"]) for row in rows)
    correct_l = sum(int(row[f"{metric}_correct_inliers_vitl"]) for row in rows)
    correct_g = sum(int(row[f"{metric}_correct_inliers_vitg"]) for row in rows)
    return {
        "queries": n,
        "candidate_recall_vitl": hits_l / n,
        "candidate_recall_vitg": hits_g / n,
        "candidate_recall_delta_g_minus_l": (hits_g - hits_l) / n,
        "candidate_hits_vitl": hits_l,
        "candidate_hits_vitg": hits_g,
        "correct_inlier_coverage_vitl": correct_l / n,
        "correct_inlier_coverage_vitg": correct_g / n,
        "recovery_given_candidate_vitl": correct_l / hits_l if hits_l else 0.0,
        "recovery_given_candidate_vitg": correct_g / hits_g if hits_g else 0.0,
    }


def bootstrap_delta(
    rows: list[dict[str, object]], metric: str, samples: int, seed: int
) -> tuple[float, float]:
    pair_count = len(rows)
    counts = np.asarray([int(row["queries"]) for row in rows], dtype=np.float64)
    l_hits = np.asarray([int(row[f"{metric}_hits_vitl"]) for row in rows], dtype=np.float64)
    g_hits = np.asarray([int(row[f"{metric}_hits_vitg"]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, pair_count, size=(samples, pair_count))
    numerator = (g_hits[draws] - l_hits[draws]).sum(axis=1)
    denominator = counts[draws].sum(axis=1)
    deltas = numerator / denominator
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_recovery_delta(
    rows: list[dict[str, object]], metric: str, samples: int, seed: int
) -> tuple[float, float]:
    pair_count = len(rows)
    g_recovered = np.asarray([int(row[f"{metric}_correct_inliers_vitg"]) for row in rows], dtype=np.float64)
    l_recovered = np.asarray([int(row[f"{metric}_correct_inliers_vitl"]) for row in rows], dtype=np.float64)
    g_hits = np.asarray([int(row[f"{metric}_hits_vitg"]) for row in rows], dtype=np.float64)
    l_hits = np.asarray([int(row[f"{metric}_hits_vitl"]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, pair_count, size=(samples, pair_count))
    g_rate = g_recovered[draws].sum(axis=1) / np.maximum(g_hits[draws].sum(axis=1), 1)
    l_rate = l_recovered[draws].sum(axis=1) / np.maximum(l_hits[draws].sum(axis=1), 1)
    low, high = np.quantile(g_rate - l_rate, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vitl-root", type=Path, required=True)
    parser.add_argument("--vitg-audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair, g_name in PAIR_TO_G_AUDIT.items():
        l = read_diagnostics(args.vitl_root / pair / "query_diagnostics.csv")
        g = read_diagnostics(args.vitg_audit_root / f"20260922_candidate_audit_{g_name}_vitg_clean" / "query_diagnostics.csv")
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
            "top_k": 20,
            "queries_and_reference_pools_verified_identical": True,
            "target_labels_used_for_candidate_construction": False,
            "bootstrap_unit": "relation; queries in a sampled relation stay together",
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "splits": {},
    }
    for index, (split, split_rows) in enumerate(splits.items()):
        summary["splits"][split] = {
            "relations": len(split_rows),
            **{
                metric: {
                    **pooled(split_rows, metric),
                    "candidate_recall_delta_pair_bootstrap_95pct": list(
                        bootstrap_delta(split_rows, metric, args.bootstrap_samples, args.seed + index * 101 + offset)
                    ),
                    "pairs_positive_delta": sum(float(r[f"{metric}_recall_delta_g_minus_l"]) > 0 for r in split_rows),
                    "pairs_negative_delta": sum(float(r[f"{metric}_recall_delta_g_minus_l"]) < 0 for r in split_rows),
                    "pairs_tied": sum(float(r[f"{metric}_recall_delta_g_minus_l"]) == 0 for r in split_rows),
                    "recovery_given_candidate_delta_g_minus_l": pooled(split_rows, metric)[f"recovery_given_candidate_vitg"] - pooled(split_rows, metric)[f"recovery_given_candidate_vitl"],
                    "recovery_given_candidate_delta_pair_bootstrap_95pct": list(
                        bootstrap_recovery_delta(split_rows, metric, args.bootstrap_samples, args.seed + index * 101 + offset + 17)
                    ),
                }
                for offset, metric in enumerate(("semantic",))
            },
        }
    pd.DataFrame(rows).to_csv(args.output_dir / "per_relation.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Paired ViT-L/ViT-G pre-consensus candidate audit",
        "",
        "Compared top-20 candidates before spatial consensus on identical cached query/reference point pools. Candidate hits are computed post hoc from annotations, which were not used to construct or rank the candidates. The 14-relation matched split includes the previously held-out target; the 13-relation source-only split excludes it.",
        "",
        "| Split | Queries | Candidate metric | ViT-L Recall@20 | ViT-G Recall@20 | Δ G−L | Pair-bootstrap 95% CI | Pairs G>L / G<L / tie | Recovery gain given candidate (G−L, 95% CI) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, values in summary["splits"].items():
        for metric, name in (("semantic", "semantic category"),):
            x = values[metric]
            ci = x["candidate_recall_delta_pair_bootstrap_95pct"]
            recovery_ci = x["recovery_given_candidate_delta_pair_bootstrap_95pct"]
            lines.append(
                f"| {split} | {x['queries']} | {name} | {x['candidate_recall_vitl']:.1%} | {x['candidate_recall_vitg']:.1%} | {x['candidate_recall_delta_g_minus_l']:+.1%} | [{ci[0]:+.1%}, {ci[1]:+.1%}] | {x['pairs_positive_delta']}/{x['pairs_negative_delta']}/{x['pairs_tied']} | {x['recovery_given_candidate_delta_g_minus_l']:+.1%} [{recovery_ci[0]:+.1%}, {recovery_ci[1]:+.1%}] |"
            )
    lines += [
        "",
        "Semantic Recall@20 measures whether at least one reference candidate with the same NYU40 semantic label is available before RANSAC. It does not measure exact physical-instance availability. Recovery given candidate measures the fraction of those semantic-hit queries that become correct semantic inliers after the unchanged consensus procedure. Physical-instance availability is reported separately by compare_physical_instance_candidate_audit.py. Both audits are post hoc on available 3RScan train relations and do not establish held-out environment generalization.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
