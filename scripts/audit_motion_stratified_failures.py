"""DEPRECATED: the original 3RScan motion audit used PLY globalId as identity.

The helpers below are retained for the corrected object-motion oracle script.
Do not use this module's original report-generation path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAIR_ROW = re.compile(
    r"\\texttt\{([^}]+)\} & (?:Source|Held out) & "
    r"\\texttt\{([^}]+)\} & \\texttt\{([^}]+)\}"
)


def read_pair_table(path: Path) -> list[tuple[str, str, str]]:
    rows = PAIR_ROW.findall(path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"no reference/rescan rows found in {path}")
    return rows


def object_to_global_id(annotation_root: Path, scan_id: str) -> tuple[dict[int, int], int]:
    """Read local object ID -> shared global ID from the official PLY columns."""
    payload = (annotation_root / scan_id / "labels.instances.annotated.v2.ply").read_bytes()
    marker = b"end_header\n"
    header_end = payload.index(marker) + len(marker)
    header = payload[:header_end].decode("ascii")
    vertex_count_match = re.search(r"element vertex (\d+)", header)
    if vertex_count_match is None:
        raise ValueError(f"PLY header has no vertex count: {scan_id}")
    vertex_count = int(vertex_count_match.group(1))
    values = np.fromstring(payload[header_end:].decode("ascii"), sep=" ", dtype=np.float64)
    expected_vertex_values = vertex_count * 11
    if values.size < expected_vertex_values:
        raise ValueError(f"truncated PLY vertex data for {scan_id}: {values.size}/{expected_vertex_values}")
    vertices = values[:expected_vertex_values].reshape(vertex_count, 11)
    object_ids = vertices[:, 6].astype(np.int64)
    global_ids = vertices[:, 7].astype(np.int64)
    mapping: dict[int, int] = {}
    ambiguous = 0
    for object_id in np.unique(object_ids):
        if object_id == 0:
            continue
        labels, counts = np.unique(global_ids[object_ids == object_id], return_counts=True)
        nonzero = labels != 0
        labels, counts = labels[nonzero], counts[nonzero]
        if not len(labels):
            continue
        if len(labels) > 1:
            ambiguous += 1
        mapping[int(object_id)] = int(labels[np.argmax(counts)])
    return mapping, ambiguous


def object_ids(entries: Any, *, side: str) -> set[int]:
    result: set[int] = set()
    for entry in entries or []:
        if isinstance(entry, dict):
            key = "instance_reference" if side == "reference" else "instance_rescan"
            value = entry.get(key)
            if value is None:
                value = entry.get("id")
            if value is not None:
                result.add(int(value))
        elif isinstance(entry, int):
            result.add(entry)
    return result


def group_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    n = len(frame)
    inlier = frame.consensus_inlier.astype(bool)
    semantic_correct = frame.semantic_correct.astype(bool)
    instance_correct = frame.instance_correct.astype(bool)
    semantic_candidate = frame.candidate_semantic_hit.astype(bool)
    instance_candidate = frame.candidate_instance_hit.astype(bool)
    semantic_inliers = int((inlier & semantic_correct).sum())
    instance_inliers = int((inlier & instance_correct).sum())
    return {
        "queries": n,
        "semantic_candidate_recall": float(semantic_candidate.mean()) if n else float("nan"),
        "instance_candidate_recall": float(instance_candidate.mean()) if n else float("nan"),
        "inlier_coverage": float(inlier.mean()) if n else float("nan"),
        "correct_semantic_inlier_coverage": semantic_inliers / n if n else float("nan"),
        "correct_instance_inlier_coverage": instance_inliers / n if n else float("nan"),
        "retained_semantic_accuracy": semantic_inliers / int(inlier.sum()) if inlier.any() else float("nan"),
        "retained_instance_accuracy": instance_inliers / int(inlier.sum()) if inlier.any() else float("nan"),
        "semantic_recovery_given_candidate": semantic_inliers / int(semantic_candidate.sum()) if semantic_candidate.any() else float("nan"),
        "instance_recovery_given_candidate": instance_inliers / int(instance_candidate.sum()) if instance_candidate.any() else float("nan"),
    }


def main() -> None:
    raise SystemExit(
        "Deprecated: this audit treated PLY globalId as physical-instance identity. "
        "Use scripts/audit_object_motion_oracle.py, which groups by local objectId "
        "and official relation annotations."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-template", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-table-output", type=Path, default=Path("paper/motion_stratified_table.tex"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    relations = json.loads(args.metadata.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for pair, reference_id, rescan_id in read_pair_table(args.pair_table):
        relation = next(
            (item for item in relations
             if item.get("reference") == reference_id
             and any(scan.get("reference") == rescan_id for scan in item.get("scans", []))),
            None,
        )
        if relation is None:
            raise ValueError(f"metadata relation not found for {pair}")
        scan = next(item for item in relation["scans"] if item.get("reference") == rescan_id)

        changed_reference = object_ids(scan.get("rigid"), side="reference")
        changed_rescan = object_ids(scan.get("rigid"), side="rescan")
        # The released metadata encodes nonrigid and removed entries as local
        # object IDs for the affected side in some relations and as records in
        # others, so retain both forms for post-hoc grouping.
        changed_rescan |= object_ids(scan.get("nonrigid"), side="rescan")
        changed_reference |= object_ids(scan.get("removed"), side="reference")

        ref_map, ref_ambiguous = object_to_global_id(args.annotation_root, reference_id)
        rescan_map, rescan_ambiguous = object_to_global_id(args.annotation_root, rescan_id)
        changed_ref_global = {ref_map[oid] for oid in changed_reference if oid in ref_map}
        changed_rescan_global = {rescan_map[oid] for oid in changed_rescan if oid in rescan_map}

        run_dir = args.runs_root / args.run_template.format(pair=pair)
        diagnostics_path = run_dir / "query_diagnostics.csv"
        summary_path = run_dir / "summary.csv"
        if not diagnostics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"missing diagnostics or summary for {pair}: {run_dir}")
        diagnostics = pd.read_csv(diagnostics_path)
        required = {
            "query_evaluable", "query_instance_id", "selected_reference_instance_id",
            "candidate_semantic_hit", "candidate_instance_hit", "consensus_inlier",
            "semantic_correct", "instance_correct",
        }
        missing = required.difference(diagnostics.columns)
        if missing:
            raise ValueError(f"{diagnostics_path} misses fields {sorted(missing)}")
        diagnostics = diagnostics[diagnostics.query_evaluable.astype(bool)].copy()
        diagnostics["query_on_annotated_change"] = diagnostics.query_instance_id.isin(changed_rescan_global)
        diagnostics["selected_reference_on_annotated_change"] = diagnostics.selected_reference_instance_id.isin(changed_ref_global)

        all_metrics = group_metrics(diagnostics)
        changed_metrics = group_metrics(diagnostics[diagnostics.query_on_annotated_change])
        other_metrics = group_metrics(diagnostics[~diagnostics.query_on_annotated_change])
        summary = pd.read_csv(summary_path).iloc[0].to_dict()
        rows.append({
            "pair": pair,
            "reference_id": reference_id,
            "rescan_id": rescan_id,
            "annotated_rigid_objects": len(scan.get("rigid", [])),
            "annotated_nonrigid_objects": len(scan.get("nonrigid", [])),
            "annotated_removed_objects": len(scan.get("removed", [])),
            "ambiguity_groups": len(relation.get("ambiguity", [])),
            "changed_global_ids_reference": len(changed_ref_global),
            "changed_global_ids_rescan": len(changed_rescan_global),
            "query_instances_with_multiple_global_ids_reference": ref_ambiguous,
            "query_instances_with_multiple_global_ids_rescan": rescan_ambiguous,
            "query_fraction_on_annotated_change": len(diagnostics[diagnostics.query_on_annotated_change]) / len(diagnostics) if len(diagnostics) else float("nan"),
            "inlier_fraction_on_annotated_change": float(diagnostics.loc[diagnostics.consensus_inlier.astype(bool), "query_on_annotated_change"].mean()) if diagnostics.consensus_inlier.astype(bool).any() else float("nan"),
            "selected_reference_fraction_on_annotated_change": float(diagnostics.loc[diagnostics.consensus_inlier.astype(bool), "selected_reference_on_annotated_change"].mean()) if diagnostics.consensus_inlier.astype(bool).any() else float("nan"),
            **{f"all_{key}": value for key, value in all_metrics.items()},
            **{f"annotated_change_{key}": value for key, value in changed_metrics.items()},
            **{f"other_{key}": value for key, value in other_metrics.items()},
            **{f"pose_{key}": value for key, value in summary.items() if key in {"rotation_error_deg", "translation_error_m", "inlier_coverage", "inlier_semantic_accuracy", "inlier_instance_accuracy"}},
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "motion_stratified_pairs.csv", index=False)
    source = frame[frame.pair != "target"].copy()
    target = frame[frame.pair == "target"].copy()

    def aggregate_group(group_name: str, subset: pd.DataFrame) -> dict[str, float | int]:
        query_column = f"{group_name}_queries"
        query_count = int(subset[query_column].sum())
        inlier_count = int(round((subset[query_column] * subset[f"{group_name}_inlier_coverage"]).sum()))
        semantic_correct = int(round((subset[query_column] * subset[f"{group_name}_correct_semantic_inlier_coverage"]).sum()))
        instance_correct = int(round((subset[query_column] * subset[f"{group_name}_correct_instance_inlier_coverage"]).sum()))
        total_queries = int(subset.all_queries.sum())
        total_inliers = int(round((subset.all_queries * subset.all_inlier_coverage).sum()))
        semantic_candidates = int(round((subset[query_column] * subset[f"{group_name}_semantic_candidate_recall"]).sum()))
        instance_candidates = int(round((subset[query_column] * subset[f"{group_name}_instance_candidate_recall"]).sum()))
        return {
            "queries": query_count,
            "query_share": query_count / total_queries if total_queries else float("nan"),
            "inliers": inlier_count,
            "inlier_share": inlier_count / total_inliers if total_inliers else float("nan"),
            "semantic_correct_inliers": semantic_correct,
            "semantic_accuracy": semantic_correct / inlier_count if inlier_count else float("nan"),
            "instance_correct_inliers": instance_correct,
            "instance_accuracy": instance_correct / inlier_count if inlier_count else float("nan"),
            "semantic_candidate_recall": semantic_candidates / query_count if query_count else float("nan"),
            "instance_candidate_recall": instance_candidates / query_count if query_count else float("nan"),
        }

    aggregate = {
        "source_annotated_change": aggregate_group("annotated_change", source),
        "source_other": aggregate_group("other", source),
        "target_annotated_change": aggregate_group("annotated_change", target),
        "target_other": aggregate_group("other", target),
    }
    (args.output_dir / "motion_stratified_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )

    def pct(value: object) -> str:
        if pd.isna(value):
            return "--"
        return f"{100 * float(value):.1f}\\%"

    def num(value: object) -> str:
        if pd.isna(value):
            return "--"
        return f"{float(value):.1f}"

    table_rows = []
    for row in frame.itertuples(index=False):
        change_accuracy = f"{pct(row.annotated_change_retained_semantic_accuracy)} / {pct(row.other_retained_semantic_accuracy)}"
        change_instance = f"{pct(row.annotated_change_retained_instance_accuracy)} / {pct(row.other_retained_instance_accuracy)}"
        change_support = f"{pct(row.query_fraction_on_annotated_change)} / {pct(row.inlier_fraction_on_annotated_change)}"
        table_rows.append(
            f"\\texttt{{{row.pair}}} & {int(row.annotated_rigid_objects)}/{int(row.annotated_nonrigid_objects)} & "
            f"{change_support} & {change_accuracy} & {change_instance} & "
            f"{num(row.pose_rotation_error_deg)}$^\\circ$ & {num(row.pose_translation_error_m)} "
            + r"\\"
        )
    tex_table = "\n".join([
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Pair & Rigid/nonrigid objs. & Query/inlier share & Sem. acc. change/other & Inst. acc. change/other & Rot. err. & Trans. err. (m) \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    args.paper_table_output.parent.mkdir(parents=True, exist_ok=True)
    args.paper_table_output.write_text(tex_table, encoding="utf-8")

    markdown_rows = [
        "| Pair | Rigid / nonrigid objects | Query / inlier share | Semantic accuracy, changed / other | Instance accuracy, changed / other | Rotation | Translation (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        markdown_rows.append(
            f"| {row.pair} | {int(row.annotated_rigid_objects)} / {int(row.annotated_nonrigid_objects)} | "
            f"{num(100 * row.query_fraction_on_annotated_change)}% / {num(100 * row.inlier_fraction_on_annotated_change)}% | "
            f"{num(100 * row.annotated_change_retained_semantic_accuracy)}% / {num(100 * row.other_retained_semantic_accuracy)}% | "
            f"{num(100 * row.annotated_change_retained_instance_accuracy)}% / {num(100 * row.other_retained_instance_accuracy)}% | "
            f"{num(row.pose_rotation_error_deg)}° | {num(row.pose_translation_error_m)} |"
        )
    report = [
        "# Motion-stratified correspondence audit",
        "",
        "3RScan `rigid` and `nonrigid` instance annotations define the query stratum. Local PLY `objectId` values are mapped to shared `globalId`; labels are used only after the saved candidate and consensus outputs are loaded. Matching is unchanged.",
        "",
        "The changed-object queries are 15.9% of source queries and contribute 21.1% of source consensus inliers. Their retained semantic accuracy is 91.5% (885/967), compared with 83.2% (3,001/3,605) for other queries. On the held-out target, the corresponding figures are 4.4% of queries, 4.5% of inliers, and 90.9% versus 93.7% retained semantic accuracy.",
        "",
        "Pair behavior is heterogeneous. 20c993 has no `rigid` or `nonrigid` object entries and still has a 145.9°/2.05 m pose error. pair07 has seven annotated changed objects, but only 0.2% of sampled queries belong to them and none of the consensus inliers do; its pose error is 149.5°/3.00 m. Conversely, pair03 has 30.7% changed-object queries and 78.0% changed-object inlier support, with 97.4% semantic accuracy on those retained queries despite a 60.5°/0.31 m pose error. Thus annotated object motion is an additional diagnostic axis, but the current audit does not support attributing wrong poses to motion alone.",
        "",
        "The changed-object group is defined by official relation annotations and observed shared instance IDs. Missing motion entries do not prove a relation is physically static. These pooled comparisons are descriptive and do not establish that motion caused retrieval, consensus, or pose errors.",
        "",
        f"The query diagnostics were read from run directories matching {args.run_template}. The date label in this prefix is later than the 2026-09-24 review date, while the inspected CSV modification times predate it; reconcile this path/date provenance before packaging the audit for external release.",
        "",
        *markdown_rows,
        "",
    ]
    (args.output_dir / "motion_stratified_failures.md").write_text("\n".join(report), encoding="utf-8")

    protocol = {
        "status": "complete",
        "source": "3RScan.json rigid/nonrigid/removed annotations and saved post-hoc query diagnostics",
        "run_template": args.run_template,
        "pair_table": str(args.pair_table),
        "annotation_root": str(args.annotation_root),
        "metadata": str(args.metadata),
        "matching_changed": False,
        "group_definition": "query globalId is mapped from the rescan-side local instance IDs listed as rigid or nonrigid in the official relation metadata",
        "caveat": "the metadata stratum is an annotation-defined grouping; it does not prove that motion caused a correspondence or pose failure",
        "pairs": len(rows),
    }
    (args.output_dir / "status.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(frame[["pair", "annotated_rigid_objects", "annotated_nonrigid_objects", "ambiguity_groups", "query_fraction_on_annotated_change", "inlier_fraction_on_annotated_change", "all_retained_semantic_accuracy", "annotated_change_retained_semantic_accuracy", "other_retained_semantic_accuracy", "pose_rotation_error_deg", "pose_translation_error_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
