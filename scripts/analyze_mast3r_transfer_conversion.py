#!/usr/bin/env python3
"""Decompose paired dense MASt3R vs ViT-G transfer into candidate and conversion stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_mast3r_dense_train_audit import _bool, _dino_directory


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _pooled(rows: list[dict], indices: np.ndarray | None = None) -> dict:
    selected = [rows[int(i)] for i in (indices if indices is not None else np.arange(len(rows)))]
    n = sum(r["linked_queries"] for r in selected)
    dense_hits = sum(r["dense_candidate_hits"] for r in selected)
    dino_hits = sum(r["dino_candidate_hits"] for r in selected)
    dense_transfer = sum(r["dense_correct_transfers"] for r in selected)
    dino_transfer = sum(r["dino_correct_transfers"] for r in selected)
    rd, rg = dense_hits / n, dino_hits / n
    qd = _ratio(dense_transfer, dense_hits)
    qg = _ratio(dino_transfer, dino_hits)
    avail_component = (rd - rg) * (qd + qg) / 2
    conversion_component = (qd - qg) * (rd + rg) / 2
    return {
        "relations": len(selected), "linked_queries": n,
        "dense_candidate_recall": rd, "dino_candidate_recall": rg,
        "candidate_recall_difference": rd - rg,
        "dense_correct_transfer_coverage": dense_transfer / n,
        "dino_correct_transfer_coverage": dino_transfer / n,
        "transfer_coverage_difference": (dense_transfer - dino_transfer) / n,
        "dense_transfer_given_candidate_hit": qd,
        "dino_transfer_given_candidate_hit": qg,
        "conditional_conversion_difference": qd - qg,
        "shapley_candidate_availability_component": avail_component,
        "shapley_candidate_to_transfer_component": conversion_component,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--audit-dir", type=Path,
                        default=Path("outputs/reports/20261003_mast3r_dense_train_audit"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reports/20261003_mast3r_transfer_conversion"))
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = args.audit_dir if args.audit_dir.is_absolute() else root / args.audit_dir
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    relation_audit = pd.read_csv(audit / "per_relation.csv")
    paired_relations = relation_audit[relation_audit.dino_audit_available.astype(bool)].copy()
    if len(paired_relations) != 14:
        raise ValueError(f"expected 14 paired relations, found {len(paired_relations)}")

    dense_by_pair: dict[str, Path] = {}
    for summary_path in (root / "outputs/runs").glob("20261003_mast3r_dense_topk_*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dense_by_pair[Path(str(summary["pair_run"])).name] = summary_path.parent

    rows: list[dict] = []
    for item in paired_relations.itertuples(index=False):
        source_run = str(item.source_run)
        dense_dir = dense_by_pair.get(source_run)
        if dense_dir is None:
            raise FileNotFoundError(f"no dense MASt3R run for {source_run}")
        dino_dir = _dino_directory(root, source_run)
        if dino_dir is None:
            raise FileNotFoundError(f"no matched ViT-G diagnostic for {source_run}")
        dense = pd.read_csv(dense_dir / "query_diagnostics.csv")
        dino = pd.read_csv(dino_dir / "query_diagnostics.csv")
        if "top_k" in dino:
            dino = dino[dino.top_k.astype(int) == 20].copy()
        if "geometry_weight" in dino:
            dino = dino[dino.geometry_weight.astype(float) == 0.0].copy()
        required = {"query_index", "frame_id", "physical_instance_evaluable",
                    "physical_instance_candidate_hit", "consensus_inlier",
                    "physical_instance_correct"}
        for method, frame in (("dense", dense), ("ViT-G", dino)):
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{method} diagnostics missing {sorted(missing)}: {source_run}")
            if frame.query_index.duplicated().any():
                raise ValueError(f"duplicate query index for {method}: {source_run}")
        dcols = dense[["query_index", "frame_id", "physical_instance_evaluable",
                       "physical_instance_candidate_hit", "physical_correct_transfer"]].rename(columns={
            "frame_id": "frame_dense", "physical_instance_evaluable": "eval_dense",
            "physical_instance_candidate_hit": "hit_dense", "physical_correct_transfer": "transfer_dense"})
        gcols = dino[["query_index", "frame_id", "physical_instance_evaluable",
                      "physical_instance_candidate_hit", "consensus_inlier",
                      "physical_instance_correct"]].rename(columns={
            "frame_id": "frame_dino", "physical_instance_evaluable": "eval_dino",
            "physical_instance_candidate_hit": "hit_dino", "consensus_inlier": "inlier_dino",
            "physical_instance_correct": "correct_dino"})
        merged = dcols.merge(gcols, on="query_index", how="outer", indicator=True, validate="one_to_one")
        if not merged._merge.eq("both").all() or not (merged.frame_dense == merged.frame_dino).all():
            raise ValueError(f"query population/frame mismatch: {source_run}")
        if not (_bool(merged.eval_dense) == _bool(merged.eval_dino)).all():
            raise ValueError(f"physical-instance denominator mismatch: {source_run}")
        linked = merged[_bool(merged.eval_dense)].copy()
        dense_hit, dino_hit = _bool(linked.hit_dense), _bool(linked.hit_dino)
        dense_transfer = _bool(linked.transfer_dense)
        dino_transfer = _bool(linked.inlier_dino) & _bool(linked.correct_dino)
        if (dense_transfer & ~dense_hit).any() or (dino_transfer & ~dino_hit).any():
            raise ValueError(f"transfer without a physical candidate hit: {source_run}")
        n = len(linked)
        rows.append({
            "source_run": source_run, "linked_queries": n,
            "dense_candidate_hits": int(dense_hit.sum()),
            "dino_candidate_hits": int(dino_hit.sum()),
            "dense_correct_transfers": int(dense_transfer.sum()),
            "dino_correct_transfers": int(dino_transfer.sum()),
            "dense_candidate_recall": float(dense_hit.mean()),
            "dino_candidate_recall": float(dino_hit.mean()),
            "dense_transfer_coverage": float(dense_transfer.mean()),
            "dino_transfer_coverage": float(dino_transfer.mean()),
            "dense_transfer_given_candidate_hit": _ratio(dense_transfer.sum(), dense_hit.sum()),
            "dino_transfer_given_candidate_hit": _ratio(dino_transfer.sum(), dino_hit.sum()),
            "candidate_recall_delta": float(dense_hit.mean() - dino_hit.mean()),
            "transfer_coverage_delta": float(dense_transfer.mean() - dino_transfer.mean()),
            "conditional_conversion_delta": _ratio(dense_transfer.sum(), dense_hit.sum()) - _ratio(dino_transfer.sum(), dino_hit.sum()),
            "dense_only_candidate_hits": int((dense_hit & ~dino_hit).sum()),
            "dino_only_candidate_hits": int((dino_hit & ~dense_hit).sum()),
            "both_candidate_hits": int((dense_hit & dino_hit).sum()),
            "both_miss_candidate": int((~dense_hit & ~dino_hit).sum()),
            "dense_only_correct_transfers": int((dense_transfer & ~dino_transfer).sum()),
            "dino_only_correct_transfers": int((dino_transfer & ~dense_transfer).sum()),
        })

    point = _pooled(rows)
    rng = np.random.default_rng(args.seed)
    n_rel = len(rows)
    draws = rng.integers(0, n_rel, size=(args.samples, n_rel))
    boot = [_pooled(rows, ix) for ix in draws]
    ci: dict[str, list[float]] = {}
    metrics = ["candidate_recall_difference", "transfer_coverage_difference",
               "conditional_conversion_difference", "shapley_candidate_availability_component",
               "shapley_candidate_to_transfer_component"]
    for key in metrics:
        vals = np.asarray([sample[key] for sample in boot], dtype=float)
        ci[key] = [float(v) for v in np.quantile(vals, [0.025, 0.975])]
    summary = {
        "protocol": "paired physical-instance query population; dense custom MASt3R top-20 vs ViT-G top-20; relation cluster bootstrap",
        "bootstrap": {"samples": args.samples, "seed": args.seed, "unit": "relation"},
        "point_estimates": point,
        "relation_bootstrap_95_ci": ci,
        "query_level_totals": {
            key: int(sum(r[key] for r in rows)) for key in (
                "linked_queries", "dense_candidate_hits", "dino_candidate_hits",
                "dense_correct_transfers", "dino_correct_transfers",
                "dense_only_candidate_hits", "dino_only_candidate_hits",
                "both_candidate_hits", "both_miss_candidate",
                "dense_only_correct_transfers", "dino_only_correct_transfers")},
        "relation_direction_counts": {
            "dense_higher_end_to_end_transfer": int((pd.DataFrame(rows).transfer_coverage_delta > 0).sum()),
            "dense_equal_end_to_end_transfer": int((pd.DataFrame(rows).transfer_coverage_delta == 0).sum()),
            "dense_lower_end_to_end_transfer": int((pd.DataFrame(rows).transfer_coverage_delta < 0).sum()),
            "conversion_defined_relations": int(pd.DataFrame(rows).conditional_conversion_delta.notna().sum()),
            "dense_higher_conditional_conversion": int((pd.DataFrame(rows).conditional_conversion_delta > 0).sum()),
            "dense_equal_conditional_conversion": int((pd.DataFrame(rows).conditional_conversion_delta == 0).sum()),
            "dense_lower_conditional_conversion": int((pd.DataFrame(rows).conditional_conversion_delta < 0).sum()),
        },
        "caveats": [
            "The conditional conversion rates divide correct transfers by each method's own candidate-hit set; the sets differ and conditioning is descriptive, not causal.",
            "Relation resampling does not establish independent physical sites; all selected relations are official train split and selected by local data availability.",
            "This decomposition does not address strict scene-pose success or external validation.",
        ],
    }
    pd.DataFrame(rows).to_csv(out / "per_relation.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    p, c = point, ci
    report = f"""# Candidate to transfer decomposition: dense MASt3R vs ViT-G

This post-hoc analysis uses the identical query/frame population in {p['relations']} locally available train relations ({p['linked_queries']:,} linked physical-instance queries). Candidate hit means the physical cross-scan instance appears in the method's top-20 list. Correct transfer additionally requires that the method's consensus prediction assigns that query to the correct physical instance.

| Metric | Dense MASt3R | ViT-G | Difference (95% relation-bootstrap CI) |
|---|---:|---:|---:|
| Physical candidate Recall@20 | {p['dense_candidate_recall']:.1%} | {p['dino_candidate_recall']:.1%} | {p['candidate_recall_difference']*100:+.1f} pp [{c['candidate_recall_difference'][0]*100:+.1f}, {c['candidate_recall_difference'][1]*100:+.1f}] |
| Correct physical transfer / all linked queries | {p['dense_correct_transfer_coverage']:.1%} | {p['dino_correct_transfer_coverage']:.1%} | {p['transfer_coverage_difference']*100:+.1f} pp [{c['transfer_coverage_difference'][0]*100:+.1f}, {c['transfer_coverage_difference'][1]*100:+.1f}] |
| Correct transfer / own candidate-hit queries | {p['dense_transfer_given_candidate_hit']:.1%} | {p['dino_transfer_given_candidate_hit']:.1%} | {p['conditional_conversion_difference']*100:+.1f} pp [{c['conditional_conversion_difference'][0]*100:+.1f}, {c['conditional_conversion_difference'][1]*100:+.1f}] |

A symmetric Shapley decomposition of the pooled end-to-end transfer difference attributes {p['shapley_candidate_availability_component']*100:+.1f} pp to candidate availability and {p['shapley_candidate_to_transfer_component']*100:+.1f} pp to candidate-hit-to-transfer conversion. The two terms sum exactly to the {p['transfer_coverage_difference']*100:+.1f} pp overall difference. This is an arithmetic decomposition, not a causal attribution: each method has its own candidate-hit set and its own independently estimated consensus pose.

At the query level, dense MASt3R finds {summary['query_level_totals']['dense_only_candidate_hits']:,} physical candidate hits missed by ViT-G, while ViT-G finds {summary['query_level_totals']['dino_only_candidate_hits']:,} missed by dense MASt3R. Correct transfer is dense-only on {summary['query_level_totals']['dense_only_correct_transfers']:,} queries and ViT-G-only on {summary['query_level_totals']['dino_only_correct_transfers']:,}. These paired transitions retain query matching but are clustered by relation for uncertainty.

The conditional conversion rates condition on different method-specific candidate sets, so they are descriptive stage diagnostics rather than controlled post-retrieval treatment effects. The relation bootstrap does not establish site independence; this remains availability-selected official-train evidence, not external validation. It also does not overturn the low strict scene-pose success rates (3/14 dense MASt3R versus 1/14 ViT-G).

The relation-level transfer difference favors dense MASt3R in {summary['relation_direction_counts']['dense_higher_end_to_end_transfer']}/14 relations; conditional conversion favors it in {summary['relation_direction_counts']['dense_higher_conditional_conversion']}/{summary['relation_direction_counts']['conversion_defined_relations']} relations where either method has candidate hits.

Reproduce with `python scripts/analyze_mast3r_transfer_conversion.py --root .`.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "pass", "summary": summary["point_estimates"],
                      "ci": ci, "output_dir": str(out.relative_to(root))}, indent=2))


if __name__ == "__main__":
    main()
