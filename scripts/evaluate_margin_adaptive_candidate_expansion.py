#!/usr/bin/env python3
"""Evaluate low descriptor margin as a fixed-budget K=20->50 expansion signal."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_vitl_vitg_candidate_audit import PAIR_TO_G_AUDIT, as_bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=1 / 6)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if not (0 < args.fraction <= 1):
        raise ValueError("fraction must be in (0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, object]] = []
    query_rows: list[pd.DataFrame] = []
    for pair in PAIR_TO_G_AUDIT:
        if "target" in pair:
            continue
        path = args.audit_root / pair / "query_diagnostics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = pd.read_csv(path)
        if set(raw.top_k.unique()) != {1, 5, 10, 20, 50}:
            raise ValueError(f"{pair}: unexpected K values")
        low = raw[raw.top_k == 20].sort_values(["query_index", "frame_id"]).reset_index(drop=True)
        high = raw[raw.top_k == 50].sort_values(["query_index", "frame_id"]).reset_index(drop=True)
        key_cols = ["query_index", "frame_id", "query_object_id", "mapped_reference_object_id"]
        if not low[key_cols].equals(high[key_cols]):
            raise ValueError(f"{pair}: K=20/50 query IDs or object links differ")
        for col in ("query_evaluable", "physical_instance_evaluable"):
            if not np.array_equal(as_bool(low[col]), as_bool(high[col])):
                raise ValueError(f"{pair}: {col} differs across K")
        if not np.allclose(low.top1_top2_margin, high.top1_top2_margin, rtol=0, atol=1e-8):
            raise ValueError(f"{pair}: descriptor margins differ across K")

        sem_eval = as_bool(low.query_evaluable)
        linked = as_bool(low.physical_instance_evaluable)
        hit20 = as_bool(low.physical_instance_candidate_hit)
        hit50 = as_bool(high.physical_instance_candidate_hit)
        if np.any(hit20 & ~hit50):
            raise ValueError(f"{pair}: K=20 physical hits are not a subset of K=50")
        # Candidate expansion runs on every depth-valid query. Annotation-based
        # evaluability and object links define scoring denominators only.
        eligible = np.arange(len(low), dtype=np.int64)
        n_expand = int(np.floor(len(eligible) * args.fraction))
        order = np.lexsort((low.query_index.to_numpy()[eligible], low.top1_top2_margin.to_numpy()[eligible]))
        selected = np.zeros(len(low), dtype=bool)
        selected[eligible[order[:n_expand]]] = True
        newly_recovered = (~hit20) & hit50 & linked
        n_linked = int(linked.sum())
        adaptive_hits = int((newly_recovered & selected).sum())
        total_recoverable = int(newly_recovered.sum())
        random_expected_hits = n_expand / len(eligible) * int((newly_recovered & linked).sum()) if len(eligible) else 0.0
        margin_selected_linked = int((selected & linked).sum())
        pair_rows.append({
            "pair": pair,
            "semantic_queries": int(sem_eval.sum()),
            "linked_queries": n_linked,
            "expanded_queries": n_expand,
            "actual_mean_k": 20 + 30 * n_expand / max(len(low), 1),
            "expanded_fraction": n_expand / max(len(low), 1),
            "expanded_linked_queries": margin_selected_linked,
            "margin_new_physical_hits": adaptive_hits,
            "random_expected_new_physical_hits": random_expected_hits,
            "all_available_new_physical_hits": total_recoverable,
            "k20_physical_hits": int((hit20 & linked).sum()),
            "k50_physical_hits": int((hit50 & linked).sum()),
            "adaptive_gain_per_linked_query": adaptive_hits / n_linked if n_linked else 0.0,
            "random_gain_per_linked_query": random_expected_hits / n_linked if n_linked else 0.0,
            "adaptive_minus_random_per_linked_query": (adaptive_hits - random_expected_hits) / n_linked if n_linked else 0.0,
            "low_margin_linked_fraction": margin_selected_linked / n_linked if n_linked else 0.0,
            "margin_recall_of_available_gains": adaptive_hits / total_recoverable if total_recoverable else 0.0,
        })
        q = low[["query_index", "frame_id", "top1_top2_margin", "query_evaluable", "physical_instance_evaluable"]].copy()
        q["pair"] = pair
        q["expanded_by_margin"] = selected
        q["new_physical_hit_at_k50"] = newly_recovered
        query_rows.append(q)

    per_pair = pd.DataFrame(pair_rows)
    pairs = per_pair.pair.tolist()
    vals = per_pair.adaptive_minus_random_per_linked_query.to_numpy()
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(pairs), size=(args.bootstrap_samples, len(pairs)))
    bootstrap = vals[draws].mean(axis=1)
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    relation_mean_delta = float(vals.mean())
    sem_queries = int(per_pair.semantic_queries.sum())
    total_expanded = int(per_pair.expanded_queries.sum())
    overall_mean_k = 20 + 30 * total_expanded / sem_queries if sem_queries else 20.0
    summary = pd.DataFrame([{
        "relations": len(pairs),
        "expansion_fraction": args.fraction,
        "mean_k_over_semantic_queries": overall_mean_k,
        "relation_macro_adaptive_gain_per_linked_query": float(per_pair.adaptive_gain_per_linked_query.mean()),
        "relation_macro_random_gain_per_linked_query": float(per_pair.random_gain_per_linked_query.mean()),
        "relation_macro_adaptive_minus_random": relation_mean_delta,
        "relation_bootstrap_95pct_low": float(lo),
        "relation_bootstrap_95pct_high": float(hi),
        "linked_queries": int(per_pair.linked_queries.sum()),
        "margin_new_physical_hits": int(per_pair.margin_new_physical_hits.sum()),
        "random_expected_new_physical_hits": float(per_pair.random_expected_new_physical_hits.sum()),
        "k20_physical_hits": int(per_pair.k20_physical_hits.sum()),
        "k50_physical_hits": int(per_pair.k50_physical_hits.sum()),
    }])
    per_pair.to_csv(args.output_dir / "per_relation.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.concat(query_rows, ignore_index=True).to_csv(args.output_dir / "query_selection.csv", index=False)
    text = [
        "# Margin-guided fixed-budget candidate expansion",
        "",
        f"ViT-G/14 source-only, 13 relations. Per relation, expand the lowest-margin {args.fraction:.3f} of all depth-valid queries from K=20 to K=50 (nominal mean K=25). Selection uses descriptor margin and stable query index only; semantic evaluability, physical-instance links, and candidate hits are used only for post-hoc evaluation. The random baseline is the expected number of newly recovered linked physical instances under a uniformly selected subset of the same size. Intervals use paired relation-level bootstrap resampling.",
        "",
        f"Across relations, margin selection recovered {int(per_pair.margin_new_physical_hits.sum())} newly available physical-instance hits; random selection is expected to recover {per_pair.random_expected_new_physical_hits.sum():.1f}. Relation-macro gain over random is {relation_mean_delta * 100:.2f} pp of linked queries (95% CI [{lo * 100:.2f}, {hi * 100:.2f}]). Pooled K=20 to K=50 physical recall changed from {per_pair.k20_physical_hits.sum() / per_pair.linked_queries.sum():.1%} to {per_pair.k50_physical_hits.sum() / per_pair.linked_queries.sum():.1%}; these are candidate availability measures and do not establish pose or system improvement.",
        "",
        "The 13 source relations are not independent-site validation. This analysis tests candidate discovery only; a positive result requires a subsequent mixed-K RANSAC evaluation before claiming system benefit.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
