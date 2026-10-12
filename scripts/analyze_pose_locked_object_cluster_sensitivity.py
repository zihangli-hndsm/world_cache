#!/usr/bin/env python3
"""Check whether source-LOPO multi-motion gains depend on a few objects.

For each fixed RANSAC seed, replays the availability-expanded 18-relation
source-only LOPO selector, then summarizes held-out query outcomes by linked
physical object. No target relation is loaded or scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_3rscan_multimotion_rescue_mechanisms import (
    MARGINS_M,
    MIN_SUPPORTS,
    MODE_COUNTS,
    predict,
    read_pair,
)
from analyze_pose_locked_multimotion_extended_source import environment_groups


SEEDS = (17, 29, 41)


def choose_config(data_by_pair: dict[str, dict[str, object]], heldout: str) -> tuple[int, float, int]:
    train_pairs = [pair for pair in sorted(data_by_pair) if pair != heldout]
    scores: dict[tuple[int, float, int], int] = {}
    for config in ((m, margin, support) for m in MODE_COUNTS
                   for margin in MARGINS_M for support in MIN_SUPPORTS):
        scores[config] = sum(
            int(predict(data_by_pair[pair], *config)[0].sum()) for pair in train_pairs
        )
    best = max(scores.values())
    return min(
        (config for config, score in scores.items() if score == best),
        key=lambda config: (config[0], -config[2], -config[1]),
    )


def fixed_lopo_relation_object_bootstrap(objects: pd.DataFrame, samples: int,
                                        rng: np.random.Generator) -> list[float]:
    """Resample fixed cross-fitted outcomes; this does not refit LOPO selectors."""
    groups = [group.reset_index(drop=True) for _, group in objects.groupby("pair", sort=True)]
    within = np.empty((samples, len(groups)), dtype=np.float64)
    for j, group in enumerate(groups):
        ix = rng.integers(0, len(group), size=(samples, len(group)))
        nums = group.net.to_numpy(dtype=np.float64)[ix].sum(axis=1)
        dens = group.queries.to_numpy(dtype=np.float64)[ix].sum(axis=1)
        within[:, j] = nums / dens
    relation_ix = rng.integers(0, len(groups), size=(samples, len(groups)))
    values = np.take_along_axis(within, relation_ix, axis=1).mean(axis=1)
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261004)
    args = parser.parse_args()
    if any(seed not in SEEDS for seed in args.seeds):
        raise ValueError(f"seeds must be selected from {SEEDS}")

    pair_dirs: dict[str, Path] = {}
    for root in args.input_dir:
        for path in sorted((root / "descriptor_candidates").glob("*.npz")):
            pair = path.stem
            if pair == "target":
                continue
            if pair in pair_dirs:
                raise ValueError(f"duplicate relation {pair}")
            pair_dirs[pair] = root
    if len(pair_dirs) != 18:
        raise ValueError(f"expected 18 source relations, found {len(pair_dirs)}")
    groups = environment_groups(args.input_dir, args.metadata)
    if len({groups[pair]["environment_id"] for pair in pair_dirs}) != 18:
        raise ValueError("source relations are not environment-disjoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries: list[dict[str, object]] = []
    all_relation_rows: list[dict[str, object]] = []
    all_object_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        data = {
            pair: read_pair(root, pair, args.annotation_root, seed=seed)
            for pair, root in sorted(pair_dirs.items())
        }
        relation_rows: list[dict[str, object]] = []
        object_rows: list[dict[str, object]] = []
        for heldout in sorted(data):
            chosen = choose_config(data, heldout)
            correct, selected_mode, _, prediction = predict(data[heldout], *chosen)
            item = data[heldout]
            base = item["base"]
            moved = np.asarray(item["moved"], dtype=bool)
            baseline = np.asarray(item["baseline_correct"], dtype=bool)
            expected = np.asarray(item["expected"], dtype=np.int64)
            if len(base) != len(moved) or not np.array_equal(
                base.official_rigid_reference_instance_id.to_numpy(dtype=np.int64), expected
            ):
                raise ValueError(f"{heldout}: physical-instance labels are misaligned")
            ids = np.unique(expected[moved])
            baseline_total, method_total = int((baseline & moved).sum()), int(correct.sum())
            if moved.sum() == 0:
                relation_rows.append({
                    "seed": seed, "pair": heldout, "environment_id": groups[heldout]["environment_id"],
                    "queries": 0, "baseline_correct": 0, "method_correct": 0,
                    "delta": 0, "delta_pp": np.nan, "clusters": 0,
                    "modes": chosen[0], "margin_m": chosen[1], "min_support": chosen[2],
                })
                continue
            relation_rows.append({
                "seed": seed, "pair": heldout, "environment_id": groups[heldout]["environment_id"],
                "queries": int(moved.sum()), "baseline_correct": baseline_total,
                "method_correct": method_total, "delta": method_total - baseline_total,
                "delta_pp": 100.0 * (method_total - baseline_total) / moved.sum(),
                "clusters": len(ids), "modes": chosen[0], "margin_m": chosen[1],
                "min_support": chosen[2],
            })
            for object_id in ids:
                mask = moved & (expected == object_id)
                base_ok = baseline & mask
                method_ok = correct & mask
                object_rows.append({
                    "seed": seed, "pair": heldout,
                    "environment_id": groups[heldout]["environment_id"],
                    "reference_object_id": int(object_id), "queries": int(mask.sum()),
                    "baseline_correct": int(base_ok.sum()),
                    "method_correct": int(method_ok.sum()),
                    "rescued": int((method_ok & ~base_ok).sum()),
                    "lost": int((base_ok & ~method_ok).sum()),
                    "net": int(method_ok.sum() - base_ok.sum()),
                    "selected_secondary": int((selected_mode[mask] > 0).sum()),
                })

        rel_df = pd.DataFrame(relation_rows)
        obj_df = pd.DataFrame(object_rows)
        eligible_rel = rel_df[rel_df.queries > 0]
        total_queries = int(eligible_rel.queries.sum())
        total_delta = int(eligible_rel.delta.sum())
        leave_one_cluster = [
            (total_delta - int(row.net)) / (total_queries - int(row.queries))
            for row in obj_df.itertuples(index=False) if total_queries > int(row.queries)
        ]
        rng = np.random.default_rng(args.bootstrap_seed + seed)
        nested_ci = fixed_lopo_relation_object_bootstrap(
            obj_df[["pair", "net", "queries"]], args.bootstrap_samples, rng
        )
        dominant = obj_df.loc[obj_df.net.idxmax()]
        pooled_rate = total_delta / total_queries
        summary = {
            "seed": seed, "source_relations": len(rel_df),
            "eligible_relations": len(eligible_rel), "queries": total_queries,
            "object_clusters": len(obj_df), "baseline_correct": int(eligible_rel.baseline_correct.sum()),
            "method_correct": int(eligible_rel.method_correct.sum()),
            "net_correct": total_delta, "pooled_delta": pooled_rate,
            "clusters_with_positive_net": int((obj_df.net > 0).sum()),
            "clusters_with_negative_net": int((obj_df.net < 0).sum()),
            "clusters_tied": int((obj_df.net == 0).sum()),
            "largest_positive_cluster": {
                "pair": str(dominant.pair), "reference_object_id": int(dominant.reference_object_id),
                "queries": int(dominant.queries), "net_correct": int(dominant.net),
                "share_of_all_queries": float(dominant.queries / total_queries),
                "share_of_net_gain": float(dominant.net / total_delta) if total_delta else None,
            },
            "leave_one_object_cluster_out_pooled_delta_range": [
                float(min(leave_one_cluster)), float(max(leave_one_cluster))
            ],
            "fixed_lopo_relation_then_object_bootstrap_macro_ci95": nested_ci,
            "bootstrap_conditioning": (
                "resamples fixed out-of-fold relation/object outcomes; LOPO selector is not refit in bootstrap draws"
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed + seed,
            "selection": "one held-out relation; source-LOPO tuning re-run without that relation",
            "limitations": [
                "all 18 relations are official train and availability-selected",
                "linked points are grouped into scan-local physical-instance clusters within relation",
                "cluster intervals do not establish independent capture sites beyond relation IDs",
            ],
        }
        seed_summaries.append(summary)
        all_relation_rows.extend(relation_rows)
        all_object_rows.extend(object_rows)

    per_rel = pd.DataFrame(all_relation_rows)
    per_obj = pd.DataFrame(all_object_rows)
    per_rel.to_csv(args.output_dir / "per_relation.csv", index=False)
    per_obj.to_csv(args.output_dir / "per_object.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(seed_summaries, indent=2) + "\n")
    lines = [
        "# Pose-locked multi-motion object-cluster sensitivity", "",
        "For each seed, the extended 18-source relation-held-out selector is replayed. Held-out outcomes are grouped by linked reference object ID within relation; this prevents repeated patch observations of one instance from masquerading as many independent gains.", "",
        "| Seed | Objects (positive/negative/tied) | Largest cluster | Leave-one-cluster-out pooled delta | Relation→object bootstrap macro 95% CI* |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in seed_summaries:
        lo, hi = s["leave_one_object_cluster_out_pooled_delta_range"]
        ci = s["fixed_lopo_relation_then_object_bootstrap_macro_ci95"]
        largest = s["largest_positive_cluster"]
        lines.append(
            f"| {s['seed']} | {s['object_clusters']} ({s['clusters_with_positive_net']}/"
            f"{s['clusters_with_negative_net']}/{s['clusters_tied']}) | "
            f"{largest['queries']} queries, {largest['net_correct']:+d} net | "
            f"[{lo*100:+.2f},{hi*100:+.2f}] pp | [{ci[0]*100:+.2f},{ci[1]*100:+.2f}] pp |"
        )
    lines += [
        "", "These are influence and cluster-resampling diagnostics on the same train relations, not independent validation. The leave-one-cluster-out range asks how the pooled effect changes after removing any single linked object. *The interval resamples fixed out-of-fold outcomes, giving each relation equal macro weight and resampling object clusters within relation; it does not refit the LOPO selector and therefore omits selector-fit uncertainty.", "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
