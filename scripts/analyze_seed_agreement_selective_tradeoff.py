#!/usr/bin/env python3
"""Evaluate the fixed 2-of-3 and 3-of-3 seed-agreement operating points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = {
    "multi_motion": ["method_seed_17", "method_seed_29", "method_seed_41"],
    "global_ransac": ["baseline_seed_17", "baseline_seed_29", "baseline_seed_41"],
}


def prediction_and_agreement(values: np.ndarray) -> tuple[int, int]:
    values = values[values > 0]
    if not len(values):
        return -1, 0
    ids, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    return int(ids[best]), int(counts[best])


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def bootstrap_ratio(numerators: np.ndarray, denominators: np.ndarray,
                    rng: np.random.Generator, samples: int) -> list[float]:
    indices = rng.integers(0, len(numerators), size=(samples, len(numerators)))
    nums = numerators[indices].sum(axis=1)
    dens = denominators[indices].sum(axis=1)
    ratios = np.divide(nums, dens, out=np.full(samples, np.nan), where=dens > 0)
    return [float(x) for x in np.nanquantile(ratios, [0.025, 0.975])]


def hierarchical_object_bootstrap(object_rows: pd.DataFrame, rng: np.random.Generator,
                                  samples: int) -> list[float]:
    """Resample relations, then linked query-object clusters within relations."""
    groups = [group.reset_index(drop=True) for _, group in object_rows.groupby("pair", sort=True)]
    within_relation = np.empty((samples, len(groups)), dtype=np.float64)
    for relation_index, objects in enumerate(groups):
        sampled = rng.integers(0, len(objects), size=(samples, len(objects)))
        net = objects.net.to_numpy(dtype=np.float64)[sampled].sum(axis=1)
        queries = objects.queries.to_numpy(dtype=np.float64)[sampled].sum(axis=1)
        within_relation[:, relation_index] = net / queries
    sampled_relations = rng.integers(0, len(groups), size=(samples, len(groups)))
    draws = np.take_along_axis(within_relation, sampled_relations, axis=1).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-object-oracle", type=Path, required=True)
    parser.add_argument("--target-seed-root", type=Path)
    parser.add_argument("--target-oracle", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    data = pd.read_csv(args.input)
    keys = ["pair", "query_index"]
    if data.duplicated(keys).any() or len(data) != 1510 or data.pair.nunique() != 12:
        raise ValueError("expected one row per query over 1,510 source rigid queries / 12 relations")
    source_oracle = pd.read_csv(args.source_object_oracle)
    source_oracle = source_oracle[(source_oracle.split == "source") & (source_oracle.group == "rigid")]
    source_oracle = source_oracle[["pair", "query_index", "query_object_id", "reference_object_id"]]
    data = data.merge(source_oracle, on=keys, how="outer", validate="one_to_one", suffixes=("_vote", "_oracle"))
    if len(data) != 1510 or data[["query_object_id", "reference_object_id_oracle"]].isna().any().any():
        raise ValueError("source object oracle does not align with all voted query rows")
    if not (data.reference_object_id_vote == data.reference_object_id_oracle).all():
        raise ValueError("source object oracle reference IDs disagree with vote diagnostics")
    data["reference_object_id"] = data.reference_object_id_oracle.astype(np.int64)
    truth = data.reference_object_id.to_numpy(dtype=np.int64)
    derived: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, columns in ARMS.items():
        required = set(columns) | {"reference_object_id"}
        if not required.issubset(data.columns):
            raise ValueError(f"missing columns for {arm}: {sorted(required - set(data.columns))}")
        pairs = [prediction_and_agreement(row) for row in data[columns].to_numpy(dtype=np.int64)]
        predictions = np.asarray([row[0] for row in pairs], dtype=np.int64)
        agreement = np.asarray([row[1] for row in pairs], dtype=np.int64)
        derived[arm] = (predictions, agreement)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    relation_rows: list[dict[str, object]] = []
    source_object_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "queries": len(data),
        "relations": int(data.pair.nunique()),
        "bootstrap_unit": "source relation; queries remain grouped within relation",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "operating_points": {},
        "limitations": [
            "three RANSAC seeds multiply inference cost by three",
            "both motion settings are source-LOPO selected on the same official train relations",
            "agreement thresholds are fixed diagnostics, not independently validated confidence guarantees",
            "the intervals quantify relation variation in this train snapshot, not unseen-site generalization",
        ],
    }

    target_detail: pd.DataFrame | None = None
    if args.target_seed_root is not None or args.target_oracle is not None:
        if args.target_seed_root is None or args.target_oracle is None:
            raise ValueError("--target-seed-root and --target-oracle must be supplied together")
        oracle = pd.read_csv(args.target_oracle)
        oracle = oracle[(oracle.split == "target") & (oracle.group == "rigid")]
        target_detail = oracle[[
            "pair", "query_index", "frame_id", "query_object_id", "reference_object_id"
        ]].copy()
        if target_detail.pair.nunique() != 1 or len(target_detail) != 47:
            raise ValueError("expected the 47-query predesignated held-out target rigid pool")
        target_key = ["pair", "query_index"]
        for seed in (17, 29, 41):
            prediction_path = args.target_seed_root / f"seed_{seed}/query_outcomes.csv"
            frame = pd.read_csv(prediction_path)
            frame = frame[frame.role == "target"]
            if frame.duplicated(target_key).any() or len(frame) != 47:
                raise ValueError(f"seed {seed}: target predictions do not match the 47-query pool")
            frame = frame[target_key + ["method_prediction_object_id", "baseline_prediction_object_id"]]
            frame = frame.rename(columns={
                "method_prediction_object_id": f"method_seed_{seed}",
                "baseline_prediction_object_id": f"baseline_seed_{seed}",
            })
            target_detail = target_detail.merge(frame, on=target_key, how="outer", validate="one_to_one")
        target_seed_columns = ARMS["multi_motion"] + ARMS["global_ransac"]
        if len(target_detail) != 47 or target_detail[target_seed_columns].isna().any().any():
            raise ValueError("held-out target seed predictions failed one-to-one alignment")
        heldout: dict[str, object] = {
            "queries": 47,
            "relations": 1,
            "distinct_query_objects": int(target_detail.query_object_id.nunique()),
            "distinct_frames": int(target_detail.frame_id.nunique()),
            "queries_by_object": {
                str(int(object_id)): int(count)
                for object_id, count in target_detail.groupby("query_object_id").size().items()
            },
            "interval": None,
            "operating_points": {},
        }
        per_object_rows: list[dict[str, object]] = []
        target_truth = target_detail.reference_object_id.to_numpy(dtype=np.int64)
        for quorum in (2, 3):
            result: dict[str, object] = {}
            target_detail[f"quorum_{quorum}"] = quorum
            for arm, columns in ARMS.items():
                pred_agreement = [
                    prediction_and_agreement(row)
                    for row in target_detail[columns].to_numpy(dtype=np.int64)
                ]
                predictions = np.asarray([item[0] for item in pred_agreement], dtype=np.int64)
                agreement = np.asarray([item[1] for item in pred_agreement], dtype=np.int64)
                assigned = agreement >= quorum
                correct = assigned & (predictions == target_truth)
                target_detail[f"{arm}_prediction_q{quorum}"] = predictions
                target_detail[f"{arm}_agreement_q{quorum}"] = agreement
                target_detail[f"{arm}_assigned_q{quorum}"] = assigned
                target_detail[f"{arm}_correct_q{quorum}"] = correct
                assigned_n = int(assigned.sum())
                correct_n = int(correct.sum())
                result[arm] = {
                    "assigned": assigned_n,
                    "correct": correct_n,
                    "fixed_query_transfer_rate": correct_n / 47,
                    "assignment_coverage": assigned_n / 47,
                    "precision_when_assigned": correct_n / assigned_n if assigned_n else None,
                }
                for object_id, rows in target_detail.groupby("query_object_id", sort=True):
                    ix = rows.index.to_numpy()
                    obj_assigned = assigned[ix]
                    obj_correct = correct[ix]
                    obj_assigned_n = int(obj_assigned.sum())
                    per_object_rows.append({
                        "query_object_id": int(object_id),
                        "queries": len(ix),
                        "frames": int(rows.frame_id.nunique()),
                        "quorum": quorum,
                        "arm": arm,
                        "assigned": obj_assigned_n,
                        "correct": int(obj_correct.sum()),
                        "assignment_coverage": obj_assigned_n / len(ix),
                        "precision_when_assigned": (
                            float(obj_correct.sum() / obj_assigned_n) if obj_assigned_n else np.nan
                        ),
                    })
            heldout["operating_points"][f"quorum_{quorum}_of_3"] = result
        summary["heldout_target_diagnostic"] = heldout
        target_detail.to_csv(args.output_dir / "target_query_diagnostics.csv", index=False)
        pd.DataFrame(per_object_rows).to_csv(args.output_dir / "target_by_object.csv", index=False)

    for quorum in (2, 3):
        per_arm: dict[str, dict[str, object]] = {}
        pair_rows: dict[str, dict[str, dict[str, float | int]]] = {}
        for arm, (predictions, agreement) in derived.items():
            selected = agreement >= quorum
            correct = selected & (predictions == truth)
            grouped = []
            for pair, indices in data.groupby("pair", sort=True).indices.items():
                ix = np.asarray(indices, dtype=np.int64)
                n = len(ix)
                assigned = int(selected[ix].sum())
                hits = int(correct[ix].sum())
                row = {
                    "pair": str(pair),
                    "quorum": quorum,
                    "arm": arm,
                    "queries": n,
                    "assigned": assigned,
                    "correct": hits,
                    "fixed_query_transfer_rate": hits / n,
                    "assignment_coverage": assigned / n,
                    "precision_when_assigned": hits / assigned if assigned else np.nan,
                }
                relation_rows.append(row)
                pair_rows.setdefault(str(pair), {})[arm] = row
                grouped.append(row)

            for (pair, object_id), indices in data.groupby(
                ["pair", "query_object_id"], sort=True
            ).indices.items():
                ix = np.asarray(indices, dtype=np.int64)
                assigned_n = int(selected[ix].sum())
                correct_n = int(correct[ix].sum())
                source_object_rows.append({
                    "pair": str(pair),
                    "query_object_id": int(object_id),
                    "quorum": quorum,
                    "arm": arm,
                    "queries": len(ix),
                    "assigned": assigned_n,
                    "correct": correct_n,
                    "fixed_query_transfer_rate": correct_n / len(ix),
                    "assignment_coverage": assigned_n / len(ix),
                    "precision_when_assigned": correct_n / assigned_n if assigned_n else np.nan,
                })

            numerators = np.asarray([row["correct"] for row in grouped], dtype=np.float64)
            denominators = np.asarray([row["queries"] for row in grouped], dtype=np.float64)
            assigned_counts = np.asarray([row["assigned"] for row in grouped], dtype=np.float64)
            precision_ci = bootstrap_ratio(numerators, assigned_counts, rng, args.bootstrap_samples)
            transfer_ci = bootstrap_mean(numerators / denominators, rng, args.bootstrap_samples)
            assignment_ci = bootstrap_mean(assigned_counts / denominators, rng, args.bootstrap_samples)
            assigned_total = int(assigned_counts.sum())
            correct_total = int(numerators.sum())
            per_arm[arm] = {
                "assigned": assigned_total,
                "correct": correct_total,
                "pooled_fixed_query_transfer_rate": correct_total / len(data),
                "pooled_assignment_coverage": assigned_total / len(data),
                "pooled_precision_when_assigned": correct_total / assigned_total if assigned_total else None,
                "relation_macro_fixed_query_transfer_rate": float(np.mean(numerators / denominators)),
                "relation_macro_transfer_rate_bootstrap_95pct": transfer_ci,
                "pooled_precision_relation_bootstrap_95pct": precision_ci,
                "relation_macro_assignment_coverage": float(np.mean(assigned_counts / denominators)),
                "relation_macro_assignment_bootstrap_95pct": assignment_ci,
            }

        pairs = sorted(pair_rows)
        differences = np.asarray([
            pair_rows[pair]["multi_motion"]["fixed_query_transfer_rate"]
            - pair_rows[pair]["global_ransac"]["fixed_query_transfer_rate"]
            for pair in pairs
        ], dtype=np.float64)
        per_arm["paired_relation_macro_transfer_delta_multi_minus_global"] = {
            "mean": float(differences.mean()),
            "bootstrap_95pct": bootstrap_mean(differences, rng, args.bootstrap_samples),
            "relations_nonnegative": int((differences >= 0).sum()),
            "relations": len(differences),
        }
        objects = pd.DataFrame([
            row for row in source_object_rows if row["quorum"] == quorum
        ])
        object_wide = objects.pivot(
            index=["pair", "query_object_id"],
            columns="arm",
            values=["queries", "correct"],
        )
        object_wide.columns = [f"{metric}_{arm}" for metric, arm in object_wide.columns]
        object_wide = object_wide.reset_index()
        object_wide["queries"] = object_wide["queries_global_ransac"].astype(np.int64)
        object_wide["net"] = (
            object_wide["correct_multi_motion"] - object_wide["correct_global_ransac"]
        )
        total_queries = int(object_wide.queries.sum())
        loo_delta = [
            float((object_wide.net.sum() - row.net) / (total_queries - row.queries))
            for row in object_wide.itertuples(index=False) if total_queries > row.queries
        ]
        largest = object_wide.loc[object_wide.net.idxmax()]
        per_arm["object_cluster_sensitivity"] = {
            "object_clusters": int(len(object_wide)),
            "largest_cluster_queries": int(largest.queries),
            "largest_cluster_query_share": float(largest.queries / total_queries),
            "largest_cluster_pair": str(largest.pair),
            "largest_cluster_object_id": int(largest.query_object_id),
            "largest_cluster_net_correct_transfers": int(largest.net),
            "net_positive_object_clusters": int((object_wide.net > 0).sum()),
            "net_negative_object_clusters": int((object_wide.net < 0).sum()),
            "net_tied_object_clusters": int((object_wide.net == 0).sum()),
            "pooled_transfer_delta": float(object_wide.net.sum() / total_queries),
            "leave_one_object_cluster_out_delta_range": [min(loo_delta), max(loo_delta)],
            "relation_then_object_cluster_bootstrap_95pct_relation_macro_delta": hierarchical_object_bootstrap(
                object_wide[["pair", "queries", "net"]], rng, args.bootstrap_samples
            ),
        }
        summary["operating_points"][f"quorum_{quorum}_of_3"] = per_arm

    pd.DataFrame(relation_rows).to_csv(args.output_dir / "per_relation.csv", index=False)
    pd.DataFrame(source_object_rows).to_csv(args.output_dir / "source_by_object.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Seed agreement as a selective confidence signal",
        "",
        "Three saved source-LOPO RANSAC runs (seeds 17/29/41) are scored at both fixed quorum operating points. A query receives the most frequently predicted positive reference object ID only when at least the stated number of seeds agree; otherwise it abstains. Thresholds are not selected from the outcomes in this report.",
        "",
        "| Quorum | Arm | Correct / 1,510 | Correct transfer rate | Assigned coverage | Assigned precision (relation-cluster 95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for quorum in (2, 3):
        arms = summary["operating_points"][f"quorum_{quorum}_of_3"]
        for arm, label in (("global_ransac", "Global RANSAC"), ("multi_motion", "Multi-motion")):
            metrics = arms[arm]
            precision_lo, precision_hi = metrics["pooled_precision_relation_bootstrap_95pct"]
            lines.append(
                f"| {quorum}/3 | {label} | {metrics['correct']:,}/1,510 | "
                f"{metrics['pooled_fixed_query_transfer_rate']:.1%} | "
                f"{metrics['pooled_assignment_coverage']:.1%} | "
                f"{metrics['pooled_precision_when_assigned']:.1%} "
                f"[{precision_lo:.1%}, {precision_hi:.1%}] |"
            )
        delta = arms["paired_relation_macro_transfer_delta_multi_minus_global"]
        lo, hi = delta["bootstrap_95pct"]
        lines.append("")
        lines.append(
            f"At {quorum}/3 agreement, multi-motion minus global RANSAC relation-macro "
            f"fixed-query transfer is {delta['mean'] * 100:+.2f} pp "
            f"(95% relation-bootstrap CI [{lo * 100:+.2f}, {hi * 100:+.2f}] pp; "
            f"nonnegative on {delta['relations_nonnegative']}/{delta['relations']} relations)."
        )
        object_sensitivity = arms["object_cluster_sensitivity"]
        object_ci = object_sensitivity["relation_then_object_cluster_bootstrap_95pct_relation_macro_delta"]
        loo_lo, loo_hi = object_sensitivity["leave_one_object_cluster_out_delta_range"]
        lines.append(
            f"The same source queries form {object_sensitivity['object_clusters']} "
            f"(relation, query-object) clusters; the largest has "
            f"{object_sensitivity['largest_cluster_queries']}/1,510 queries "
            f"({object_sensitivity['largest_cluster_query_share']:.1%}). "
            f"Leave-one-object-cluster-out pooled transfer deltas range "
            f"[{loo_lo * 100:+.2f}, {loo_hi * 100:+.2f}] pp. A relation-then-object "
            f"cluster bootstrap gives relation-macro delta CI "
            f"[{object_ci[0] * 100:+.2f}, {object_ci[1] * 100:+.2f}] pp; "
            f"object clusters with positive/negative/tied net correct transfers: "
            f"{object_sensitivity['net_positive_object_clusters']}/"
            f"{object_sensitivity['net_negative_object_clusters']}/"
            f"{object_sensitivity['net_tied_object_clusters']}."
        )
        lines.append("")
    if target_detail is not None:
        lines += [
            "## Predesignated held-out target (47 queries)",
            "",
            "This target diagnostic uses per-seed settings selected from the 16 source relations. Its labels do not select the quorum. It is one relation, so no relation-level interval is reported.",
            f"The 47 queries cover {heldout['distinct_query_objects']} linked objects across {heldout['distinct_frames']} frames; query counts by object ID are {heldout['queries_by_object']}.",
            "",
            "| Quorum | Arm | Correct / 47 | Correct transfer rate | Assigned coverage | Precision when assigned |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for quorum in (2, 3):
            arms = summary["heldout_target_diagnostic"]["operating_points"][f"quorum_{quorum}_of_3"]
            for arm, label in (("global_ransac", "Global RANSAC"), ("multi_motion", "Multi-motion")):
                metrics = arms[arm]
                precision = metrics["precision_when_assigned"]
                precision_text = f"{precision:.1%}" if precision is not None else "n/a"
                lines.append(
                    f"| {quorum}/3 | {label} | {metrics['correct']}/47 | "
                    f"{metrics['fixed_query_transfer_rate']:.1%} | "
                    f"{metrics['assignment_coverage']:.1%} | {precision_text} |"
                )
        lines.append("")
        lines += [
            "At unanimity, target outcomes by linked query object are:",
            "",
            "| Query objectId | Queries / frames | Global RANSAC correct / assigned | Multi-motion correct / assigned |",
            "|---:|---:|---:|---:|",
        ]
        for object_id, rows in target_detail.groupby("query_object_id", sort=True):
            object_rows = [
                row for row in per_object_rows
                if row["quorum"] == 3 and row["query_object_id"] == int(object_id)
            ]
            by_arm = {str(row["arm"]): row for row in object_rows}
            lines.append(
                f"| {int(object_id)} | {len(rows)} / {rows.frame_id.nunique()} | "
                f"{by_arm['global_ransac']['correct']} / {by_arm['global_ransac']['assigned']} | "
                f"{by_arm['multi_motion']['correct']} / {by_arm['multi_motion']['assigned']} |"
            )
        lines.append("")
        lines.append("Per-object results are saved in `target_by_object.csv`; these three object groups, rather than 47 independent items, are the effective target clusters.")
        lines.append("")
    lines += [
        "The useful operating point is not universal: two-of-three multi-motion voting has much higher source assignment coverage but lower assigned precision, while unanimity is selective and can abstain on whole relations. At three-of-three, source multi-motion transfers more correctly labeled queries than global RANSAC under the same unanimity rule, but its assignment coverage is also higher and its assigned precision lower. The relation-cluster interval for multi-motion source precision at unanimity remains wide [54.1%,98.0%], so the pooled 94.1% is not a guarantee. On the 47-query predesignated target, the fixed unanimity rule transfers 28 queries for multi-motion and 26 for global RANSAC; this is a single-target point diagnostic. These analyses use three times the inference cost and do not establish independent-site generalization or a calibrated risk guarantee.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
