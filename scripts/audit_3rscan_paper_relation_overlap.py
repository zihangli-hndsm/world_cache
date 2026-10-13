#!/usr/bin/env python3
"""Intersect the complete split-overlap screen with the paper's actual relations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
FIELDS = ("NYU40", "Eigen13", "RIO27")


def read_paper_relations(path: Path) -> tuple[set[str], set[str]]:
    text = path.read_text(encoding="utf-8")
    label = text.index(r"\label{tab:relation-inclusion}")
    start = text.rindex(r"\midrule", 0, label) + len(r"\midrule")
    end = text.rindex(r"\bottomrule", start, label)
    source_refs: set[str] = set()
    all_refs: set[str] = set()
    for line in text[start:end].splitlines():
        if "\\texttt{" not in line:
            continue
        ids = UUID.findall(line)
        if len(ids) != 2:
            continue
        all_refs.add(ids[0])
        if "& Source &" in line:
            source_refs.add(ids[0])
    if len(source_refs) != 16 or len(all_refs) != 17:
        raise ValueError(f"expected 16 source and 17 total relations; found {len(source_refs)} and {len(all_refs)}")
    return source_refs, all_refs


def read_validation_relations(root: Path) -> tuple[set[str], dict[str, int], int]:
    refs: set[str] = set()
    counts: dict[str, int] = {}
    relation_count = 0
    for path in sorted(root.glob("*/protocol_relation.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        ref = row.get("reference_scan")
        if not ref:
            raise ValueError(f"missing reference_scan in {path}")
        refs.add(ref)
        counts[ref] = counts.get(ref, 0) + 1
        relation_count += 1
    if relation_count != 110 or len(refs) != 47:
        raise ValueError(f"expected 110 relations in 47 environments; found {relation_count} and {len(refs)}")
    return refs, counts, relation_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-table", type=Path, required=True)
    parser.add_argument("--validation-relations", type=Path, required=True)
    parser.add_argument("--cross-split-csv", type=Path, required=True)
    parser.add_argument("--annotation-summary", type=Path, required=True)
    parser.add_argument("--spatial-report", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    paper_sources, paper_refs = read_paper_relations(args.relation_table)
    validation_refs, validation_relation_counts, validation_relation_count = read_validation_relations(args.validation_relations)
    annotation = json.loads(args.annotation_summary.read_text(encoding="utf-8"))
    annotation_rows = {(row["source_id"], row["target_id"]): row for row in annotation["candidates"]}

    candidates: list[dict[str, object]] = []
    source_target_rows: list[dict[str, str]] = []
    with args.cross_split_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source, target = row["source_id"], row["target_id"]
            if source not in paper_sources or target not in validation_refs:
                continue
            source_target_rows.append(row)
            fitness = float(row["icp_symmetric_fitness"])
            rmse = float(row["icp_symmetric_rmse_m"])
            if fitness < 0.40 or rmse > 0.08:
                continue
            ann = annotation_rows.get((source, target))
            if ann is None:
                raise ValueError(f"geometry candidate lacks annotation row: {source} -> {target}")
            coverage = min(ann["source_fraction_with_match"], ann["target_fraction_with_match"])
            metrics = ann["field_agreement"]
            class_values = {
                field: {
                    "nonzero_exact_agreement": metrics[field]["nonzero_both_exact_agreement"],
                    "nonzero_marginal_reference": metrics[field]["nonzero_both_marginal_reference"],
                    "lift": metrics[field]["nonzero_both_exact_agreement"] - metrics[field]["nonzero_both_marginal_reference"],
                }
                for field in FIELDS
            }
            coherent = coverage >= 0.10 and all(
                class_values[field]["nonzero_exact_agreement"] >= 0.70 and class_values[field]["lift"] >= 0.30
                for field in FIELDS
            )
            candidates.append({
                "source_id": source,
                "validation_reference_id": target,
                "validation_relations_evaluated": validation_relation_counts[target],
                "symmetric_fitness": fitness,
                "symmetric_rmse_m": rmse,
                "source_coverage": ann["source_fraction_with_match"],
                "target_coverage": ann["target_fraction_with_match"],
                "matched_vertices": ann["matched_vertices"],
                "semantic_fields": class_values,
                "meets_fixed_semantic_coherence_rule": coherent,
            })

    summary = {
        "status": "complete",
        "paper_source_relation_count": len(paper_sources),
        "paper_source_reference_ids": sorted(paper_sources),
        "paper_total_relation_scan_ids": len(paper_refs),
        "official_validation_relation_count": validation_relation_count,
        "official_validation_environment_count": len(validation_refs),
        "paper_source_refs_with_validation_geometry_candidates": len({row["source_id"] for row in candidates}),
        "validation_refs_with_paper_source_geometry_candidates": len({row["validation_reference_id"] for row in candidates}),
        "screened_paper_source_by_validation_pairs": len(source_target_rows),
        "geometry_candidate_count_at_fitness_0_40_rmse_0_08": len(candidates),
        "fixed_semantic_coherence_candidate_count": sum(row["meets_fixed_semantic_coherence_rule"] for row in candidates),
        "candidate_pairs": candidates,
        "note": (
            "This is an intersection with the 16 source relations listed in the paper and the 110 official-validation relations. "
            "It does not imply other official-train references were used by the paper's method, and registration outcomes are post hoc."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Paper relation intersection with the complete split-overlap screen", "",
        f"The paper lists {len(paper_sources)} source reference/rescan relations. The official validation evaluation contains {validation_relation_count} relations across {len(validation_refs)} reference environments.",
        f"The source-by-validation audit covers {len(source_target_rows)} relevant train-reference/validation-reference pairs. At the fixed geometric screen, {len(candidates)} candidates intersect the paper's actual source references and evaluated validation references; {summary['fixed_semantic_coherence_candidate_count']} meet the fixed semantic-coherence rule.",
        "", "| Paper source reference | Evaluated validation reference | Eval relations | Fitness | RMSE (m) | Source/target coverage | Matched vertices | Fixed semantic rule |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        lines.append(
            f"| `{row['source_id']}` | `{row['validation_reference_id']}` | {row['validation_relations_evaluated']} | "
            f"{row['symmetric_fitness']:.3f} | {row['symmetric_rmse_m']:.3f} | "
            f"{row['source_coverage']:.1%}/{row['target_coverage']:.1%} | {row['matched_vertices']} | "
            f"{'yes' if row['meets_fixed_semantic_coherence_rule'] else 'no'} |"
        )
    lines += ["", summary["note"]]
    if args.spatial_report is not None:
        lines += ["", f"Spatial-block sensitivity for the direct overlap pair: `{args.spatial_report}`."]
    lines.append("")
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "paper_source_relation_count", "official_validation_relation_count",
        "official_validation_environment_count", "screened_paper_source_by_validation_pairs",
        "geometry_candidate_count_at_fitness_0_40_rmse_0_08", "fixed_semantic_coherence_candidate_count",
    )}, indent=2))


if __name__ == "__main__":
    main()
