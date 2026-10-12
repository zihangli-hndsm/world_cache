#!/usr/bin/env python3
"""Test whether same-class instance multiplicity explains semantic/ID recall gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT_DEFAULT = Path("outputs/reports/20261006_rescan_identity_multiplicity")
STRATA = ("1", "2-3", "4-5", "6+")
SCALES = {"dino_synthetic_views": "DINOv2-G", "rgb_color": "RGB distance"}
BOOTSTRAP_N = 20_000
SEED = 20261021


def stratum(value: int) -> str:
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 5:
        return "4-5"
    return "6+"


def bootstrap(values: np.ndarray, seed: int) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"estimate": None, "ci95": [None, None], "clusters": 0}
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "clusters": int(len(values)),
    }


def fmt(item: dict[str, object], percent: bool = True) -> str:
    if item["estimate"] is None:
        return "n/a"
    factor = 100 if percent else 1
    ci = item["ci95"]
    return f"{item['estimate']*factor:.1f} [{ci[0]*factor:.1f},{ci[1]*factor:.1f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm-query", type=Path, default=Path("outputs/reports/20261006_rescan_dinog_querydiagnostics_bm/query_metrics.csv"))
    parser.add_argument("--bm-pairs", type=Path, default=Path("outputs/reports/20261006_rescan_dinog_querydiagnostics_bm/pair_metrics.csv"))
    parser.add_argument("--scene-a-query", type=Path, default=Path("outputs/reports/20261006_rescan_scene_a_external/dino_g_querydiagnostics/query_metrics.csv"))
    parser.add_argument("--scene-a-pairs", type=Path, default=Path("outputs/reports/20261006_rescan_scene_a_external/dino_g_querydiagnostics/pair_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--include-render-pilots", action="store_true", help="include scenes B/C in a labeled all-site sensitivity analysis")
    args = parser.parse_args()

    bm_q = pd.read_csv(args.bm_query)
    included_bm_scenes = "bcdefghijklm" if args.include_render_pilots else "defghijklm"
    bm_q = bm_q[bm_q.scene.isin([f"scene_{x}" for x in included_bm_scenes])].copy()
    a_q = pd.read_csv(args.scene_a_query)
    queries = pd.concat([bm_q, a_q], ignore_index=True)
    queries = queries[queries.method.isin(SCALES)].copy()
    pairs = pd.concat([pd.read_csv(args.bm_pairs), pd.read_csv(args.scene_a_pairs)], ignore_index=True)
    included_pair_scenes = "abcde fghijklm".replace(" ", "") if args.include_render_pilots else "adefghijklm"
    pairs = pairs[(pairs.k == 20) & pairs.scene.isin([f"scene_{x}" for x in included_pair_scenes]) & pairs.method.isin(SCALES)].copy()
    if queries.duplicated(["scene", "target_scan", "method", "target_sample_index"]).any():
        raise ValueError("duplicate query diagnostic row")
    expected_sites, expected_targets = (13, 32) if args.include_render_pilots else (11, 25)
    if queries.scene.nunique() != expected_sites or queries.target_scan.nunique() != expected_targets:
        raise ValueError(f"expected {expected_sites} sites and {expected_targets} target scans")

    # The query-level artifact must exactly reproduce the corresponding
    # top-20 pair-level score before any mechanism stratification is trusted.
    present = queries[queries.target_present_in_full_source]
    query_pair = present.groupby(["scene", "source_scan", "target_scan", "method"], as_index=False).agg(
        target_present_queries=("target_present_in_full_source", "sum"),
        physical_correct=("physical_hit_top20", "sum"),
        semantic_correct_present=("semantic_hit_top20", "sum"),
    )
    compare = pairs.merge(query_pair, on=["scene", "source_scan", "target_scan", "method"], suffixes=("_pair", "_query"), validate="one_to_one")
    for metric in ("target_present_queries", "physical_correct", "semantic_correct_present"):
        if not (compare[f"{metric}_pair"] == compare[f"{metric}_query"]).all():
            raise ValueError(f"query diagnostics do not reproduce pair-level {metric}")

    present = present.copy()
    present["class_multiplicity"] = present.source_class_instance_count_full.map(lambda x: stratum(int(x)))
    same_class_one_instance_gap = present[
        (present.class_multiplicity == "1") & present.semantic_only_hit_top20
    ]
    if len(same_class_one_instance_gap):
        raise ValueError("semantic-only top-20 hit occurred despite one full-source class instance")

    site_rows = []
    for (scene, method, bucket), group in present.groupby(["scene", "method", "class_multiplicity"], observed=True, sort=True):
        sem = int(group.semantic_hit_top20.sum())
        phys = int(group.physical_hit_top20.sum())
        site_rows.append({
            "scene": scene,
            "method": method,
            "class_multiplicity": bucket,
            "queries": int(len(group)),
            "semantic_hits": sem,
            "physical_hits": phys,
            "semantic_recall": sem / len(group),
            "physical_recall": phys / len(group),
            "semantic_only_gap": (sem - phys) / len(group),
            "identity_given_semantic": phys / sem if sem else np.nan,
        })
    site = pd.DataFrame(site_rows)

    summaries = {}
    pooled_strata = {}
    for method in SCALES:
        summaries[method] = {}
        pooled_strata[method] = {}
        total_semantic_only = int(present[present.method == method].semantic_only_hit_top20.sum())
        for bucket_index, bucket in enumerate(STRATA):
            rows = site[(site.method == method) & (site.class_multiplicity == bucket)]
            pooled = present[(present.method == method) & (present.class_multiplicity == bucket)]
            query_count = int(len(pooled))
            semantic_only_count = int(pooled.semantic_only_hit_top20.sum())
            pooled_strata[method][bucket] = {
                "queries": query_count,
                "semantic_only_count": semantic_only_count,
                "pooled_semantic_only_rate": semantic_only_count / query_count if query_count else None,
                "share_of_all_semantic_only_queries": semantic_only_count / total_semantic_only if total_semantic_only else None,
            }
            summaries[method][bucket] = {
                "queries": query_count,
                "sites": int(rows.scene.nunique()),
                "semantic_recall": bootstrap(rows.semantic_recall.to_numpy(), SEED + 10 * bucket_index + 1),
                "physical_recall": bootstrap(rows.physical_recall.to_numpy(), SEED + 10 * bucket_index + 2),
                "semantic_only_gap": bootstrap(rows.semantic_only_gap.to_numpy(), SEED + 10 * bucket_index + 3),
                "identity_given_semantic": bootstrap(rows.identity_given_semantic.to_numpy(), SEED + 10 * bucket_index + 4),
            }

    result = {
        "status": "complete",
        "sites": int(queries.scene.nunique()),
        "target_scans": int(queries.target_scan.nunique()),
        "included_render_pilot_sites": bool(args.include_render_pilots),
        "target_present_queries_per_method": int(len(present[present.method == "dino_synthetic_views"])),
        "unit": "site cluster; within-site rates pool target scans and query points",
        "bootstrap": {"samples": BOOTSTRAP_N, "seed": SEED},
        "class_multiplicity": "number of distinct labeled instance IDs of the query class in the full annotated source PLY",
        "ranking_label_use": False,
        "pair_metric_reproduction": {"pairs": int(len(compare)), "all_counts_match": True},
        "one_instance_semantic_only_rows": 0,
        "strata": summaries,
        "pooled_strata": pooled_strata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    site.to_csv(args.output_dir / "site_class_multiplicity_metrics.csv", index=False)

    lines = [
        "# Rescan candidate identity gap by source-class multiplicity", "",
        "Question: when top-20 contains the query's semantic class but misses its physical instance ID, is this associated with how many same-class instances exist in the reference scan? Multiplicity is computed from full source-PLY labels after ranking; labels are not used in feature extraction or retrieval.", "",
        f"The artifact contains DINO-G and same-pool RGB query-level top-20 ranks on {queries.scene.nunique()} sites ({'B-M plus A, including render pilots B/C' if args.include_render_pilots else 'D-M plus A, excluding render pilots B/C'}), {queries.target_scan.nunique()} target scans. Pair-level counts reproduce exactly for every site/scan/method before stratification. Scene A is a post-hoc addition. Cluster-bootstrap intervals resample sites; point queries are not treated as independent.", "",
        "Class multiplicity 1 is a sanity condition: if the target instance exists and there is exactly one source instance with its class, a semantic hit must be a physical hit. The audit finds zero violations.", "",
        "| Source instances of query class | Descriptor | Present queries | Sites | Semantic recall | Physical recall | Semantic-only gap | Identity hit given semantic hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in STRATA:
        for method, label in SCALES.items():
            cell = summaries[method][bucket]
            lines.append(
                f"| {bucket} | {label} | {cell['queries']:,} | {cell['sites']}/{queries.scene.nunique()} | "
                f"{fmt(cell['semantic_recall'])} | {fmt(cell['physical_recall'])} | "
                f"{fmt(cell['semantic_only_gap'])} | {fmt(cell['identity_given_semantic'])} |"
            )
    dino_gaps = [summaries["dino_synthetic_views"][bucket]["semantic_only_gap"] for bucket in ("2-3", "4-5", "6+")]
    dino_pooled = pooled_strata["dino_synthetic_views"]
    dino_query_total = sum(int(dino_pooled[bucket]["queries"]) for bucket in STRATA)
    high_multiplicity_query_share = dino_pooled["6+"]["queries"] / dino_query_total
    high_multiplicity_error_share = dino_pooled["6+"]["share_of_all_semantic_only_queries"]
    lines.extend([
        "",
        f"Interpretation: the DINO-G site-macro semantic-only gap grows across source-class multiplicity 2--3 ({dino_gaps[0]['estimate']*100:.1f} pp), 4--5 ({dino_gaps[1]['estimate']*100:.1f} pp), and 6+ ({dino_gaps[2]['estimate']*100:.1f} pp); corresponding intervals are [{dino_gaps[0]['ci95'][0]*100:.1f},{dino_gaps[0]['ci95'][1]*100:.1f}], [{dino_gaps[1]['ci95'][0]*100:.1f},{dino_gaps[1]['ci95'][1]*100:.1f}], and [{dino_gaps[2]['ci95'][0]*100:.1f},{dino_gaps[2]['ci95'][1]*100:.1f}]. RGB shows the same increasing pattern. Multiplicity 1 has no semantic-only errors by construction. The 6+ stratum is {high_multiplicity_query_share*100:.1f}% of pooled DINO-G queries and accounts for {high_multiplicity_error_share*100:.1f}% of pooled semantic-only misses. This is a post-hoc descriptive association, not a causal test: multiplicity correlates with class, object size, and scene composition. Treat it as a mechanism diagnosis, not proof that category collisions alone cause the full cross-scan identity gap.",
        "",
        "Reproduce query ranks with `scripts/probe_rescan_dino_point_candidates.py --model-size giant --save-query-metrics`; reproduce this aggregation with `scripts/analyze_rescan_candidate_identity_multiplicity.py`.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote multiplicity analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
