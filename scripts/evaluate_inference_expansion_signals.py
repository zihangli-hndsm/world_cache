#!/usr/bin/env python3
"""Compare label-free fixed-budget K expansion triggers on official rigid queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(samples, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-diagnostics", type=Path, required=True)
    parser.add_argument("--oracle-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=1 / 6)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must lie in (0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_all = pd.read_csv(args.baseline_diagnostics)
    oracle_all = pd.read_csv(args.oracle_candidates)
    oracle_by_pair = {p: d.sort_values("query_index").reset_index(drop=True)
                      for p, d in oracle_all.groupby("pair")}
    relation_rows: list[dict[str, object]] = []
    query_frames: list[pd.DataFrame] = []
    source_caches = sorted(p for p in args.candidate_root.glob("*.npz") if p.stem != "target")
    for cache_path in source_caches:
        pair = cache_path.stem
        with np.load(cache_path) as z:
            if str(z["pair"].item()) != pair:
                raise ValueError(f"{pair}: cache pair key mismatch")
            indices = z["candidate_indices"].astype(np.int64)
            scores = z["candidate_scores"].astype(np.float64)
            frames = z["query_frame_ids"].astype(np.int64)
        base = baseline_all[(baseline_all.pair == pair) & (baseline_all.arm == "global_ransac")]
        base = base.sort_values("query_index").reset_index(drop=True)
        if len(base) != len(indices) or not np.array_equal(frames, base.frame_id.to_numpy(dtype=np.int64)):
            raise ValueError(f"{pair}: candidate and baseline query pools differ")
        residual = base.spatial_residual_m.to_numpy(dtype=np.float64)
        if not np.isfinite(residual).all():
            raise ValueError(f"{pair}: non-finite baseline residual")
        margin = scores[:, 0] - scores[:, 1]
        n = len(indices)
        n_expand = int(np.floor(n * args.fraction))
        tie = np.arange(n, dtype=np.int64)
        low_margin = np.zeros(n, dtype=bool)
        low_margin[np.lexsort((tie, margin))[:n_expand]] = True
        high_residual = np.zeros(n, dtype=bool)
        high_residual[np.lexsort((tie, -residual))[:n_expand]] = True
        oracle = oracle_by_pair.get(pair)
        if oracle is None:
            linked = np.zeros(n, dtype=bool)
            new_physical = np.zeros(n, dtype=bool)
            new_valid = np.zeros(n, dtype=bool)
        else:
            qidx = oracle.query_index.to_numpy(dtype=np.int64)
            if np.any(qidx >= n) or not np.array_equal(frames[qidx], oracle.frame_id.to_numpy(dtype=np.int64)):
                raise ValueError(f"{pair}: oracle query/frame alignment mismatch")
            linked = np.zeros(n, dtype=bool)
            new_physical = np.zeros(n, dtype=bool)
            new_valid = np.zeros(n, dtype=bool)
            linked[qidx] = True
            new_physical[qidx] = (
                oracle.physical_candidate_k50.astype(bool).to_numpy()
                & ~oracle.physical_candidate_k20.astype(bool).to_numpy()
            )
            new_valid[qidx] = (
                oracle.object_valid_candidate_k50.astype(bool).to_numpy()
                & ~oracle.object_valid_candidate_k20.astype(bool).to_numpy()
            )
        row: dict[str, object] = {
            "pair": pair,
            "depth_valid_queries": n,
            "rigid_linked_queries": int(linked.sum()),
            "expanded_queries": n_expand,
            "mean_k": 20 + 30 * n_expand / max(n, 1),
            "margin_selected_rigid": int((low_margin & linked).sum()),
            "residual_selected_rigid": int((high_residual & linked).sum()),
            "new_physical_candidates": int(new_physical.sum()),
            "new_object_valid_candidates": int(new_valid.sum()),
            "margin_new_physical_hits": int((low_margin & new_physical).sum()),
            "residual_new_physical_hits": int((high_residual & new_physical).sum()),
            "margin_new_object_valid_hits": int((low_margin & new_valid).sum()),
            "residual_new_object_valid_hits": int((high_residual & new_valid).sum()),
        }
        for name, selected in (("margin", low_margin), ("residual", high_residual)):
            m = int(selected.sum())
            row[f"{name}_random_expected_physical_hits"] = m / n * int(new_physical.sum()) if n else 0.0
            row[f"{name}_random_expected_object_valid_hits"] = m / n * int(new_valid.sum()) if n else 0.0
            row[f"{name}_physical_gain_per_rigid_query"] = (
                float((selected & new_physical).sum() / linked.sum()) if linked.any() else np.nan
            )
            row[f"{name}_object_valid_gain_per_rigid_query"] = (
                float((selected & new_valid).sum() / linked.sum()) if linked.any() else np.nan
            )
            row[f"{name}_random_object_valid_gain_per_rigid_query"] = (
                float((m / n * new_valid.sum()) / linked.sum()) if linked.any() and n else np.nan
            )
        relation_rows.append(row)
        query_frames.append(pd.DataFrame({
            "pair": pair,
            "query_index": np.arange(n),
            "frame_id": frames,
            "top1_top2_margin": margin,
            "global_baseline_residual_m": residual,
            "expanded_by_margin": low_margin,
            "expanded_by_high_residual": high_residual,
            "official_rigid_linked": linked,
            "new_physical_candidate_at_k50": new_physical,
            "new_object_pose_valid_candidate_at_k50": new_valid,
        }))

    per_relation = pd.DataFrame(relation_rows)
    queries = pd.concat(query_frames, ignore_index=True)
    eligible = per_relation[per_relation.rigid_linked_queries > 0].copy()
    if eligible.empty:
        raise ValueError("no source relation has official rigid queries")
    output_summary: dict[str, object] = {
        "source_relations": len(per_relation),
        "relations_with_rigid_queries": len(eligible),
        "expansion_fraction_of_all_depth_valid_queries": args.fraction,
        "nominal_mean_k": 20 + 30 * args.fraction,
        "selector_definitions": {
            "margin": "expand the bottom 1/6 of all depth-valid queries by top1-top2 descriptor score margin",
            "residual": "expand the top 1/6 of all depth-valid queries by global-RANSAC residual",
        },
        "metrics_are_posthoc": "official object identity and transform-valid candidate flags are not used for selection",
        "relation_macro": {},
        "pooled_counts": {
            "rigid_linked_queries": int(eligible.rigid_linked_queries.sum()),
            "new_physical_candidate_queries": int(eligible.new_physical_candidates.sum()),
            "new_object_valid_candidate_queries": int(eligible.new_object_valid_candidates.sum()),
        },
    }
    rng = np.random.default_rng(args.seed)
    for selector in ("margin", "residual"):
        d = eligible[f"{selector}_object_valid_gain_per_rigid_query"].to_numpy(dtype=np.float64)
        r = eligible[f"{selector}_random_object_valid_gain_per_rigid_query"].to_numpy(dtype=np.float64)
        delta = d - r
        mean, lo, hi = paired_ci(delta, args.bootstrap_samples, args.seed + (selector == "residual"))
        output_summary["relation_macro"][selector] = {
            "object_valid_gain_per_rigid_query": float(d.mean()),
            "random_expected_gain_per_rigid_query": float(r.mean()),
            "excess_over_random_pp": 100 * mean,
            "excess_over_random_ci95_pp": [100 * lo, 100 * hi],
            "new_object_valid_hits_selected": int(eligible[f"{selector}_new_object_valid_hits"].sum()),
            "random_expected_new_object_valid_hits": float(eligible[f"{selector}_random_expected_object_valid_hits"].sum()),
        }
        p = eligible[f"{selector}_physical_gain_per_rigid_query"].to_numpy(dtype=np.float64)
        pr = (eligible[f"{selector}_random_expected_physical_hits"] / eligible.rigid_linked_queries).to_numpy(dtype=np.float64)
        pm, plo, phi = paired_ci(p - pr, args.bootstrap_samples, args.seed + 10 + (selector == "residual"))
        output_summary["relation_macro"][selector]["excess_physical_candidate_gain_over_random_pp"] = 100 * pm
        output_summary["relation_macro"][selector]["physical_gain_ci95_pp"] = [100 * plo, 100 * phi]

    per_relation.to_csv(args.output_dir / "per_relation.csv", index=False)
    queries.to_csv(args.output_dir / "query_selection.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(output_summary, indent=2) + "\n")
    lines = [
        "# Inference-time candidate expansion signal audit",
        "",
        f"Compared two fixed, label-free selectors on all {len(per_relation)} source relations: expand exactly the lowest-margin or highest-global-RANSAC-residual {args.fraction:.3f} of all depth-valid queries per relation (mean K≈{20 + 30 * args.fraction:.1f}). Annotation evaluability and official rigid-object links are used only for scoring. Candidate validity is the saved post-hoc object-transform oracle flag. The K20→K50 gain pool has {output_summary['pooled_counts']['new_object_valid_candidate_queries']} object-pose-valid queries and {output_summary['pooled_counts']['new_physical_candidate_queries']} newly retrieved physical-object queries.",
        "",
        "| Selector | New valid hits selected | Random expected | Excess over random per rigid query (95% relation bootstrap) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("margin", "residual"):
        x = output_summary["relation_macro"][name]
        ci = x["excess_over_random_ci95_pp"]
        lines.append(f"| {name} | {x['new_object_valid_hits_selected']} | {x['random_expected_new_object_valid_hits']:.1f} | {x['excess_over_random_pp']:+.2f} pp [{ci[0]:+.2f},{ci[1]:+.2f}] |")
    lines += [
        "",
        "This is an exploratory source-pool selection diagnostic, not a RANSAC system result. Positive candidate availability must be followed by mixed-K multi-motion consensus and a held-out environment evaluation. The margin selector is corrected to rank every depth-valid query; earlier reports that ranked only semantically evaluable queries are superseded.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
