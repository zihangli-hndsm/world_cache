#!/usr/bin/env python3
"""Compare two train references after each was registered to one validation scan.

This post-hoc diagnostic asks whether source references independently aligned
to a shared validation target also overlap each other in that target frame.
It is exploratory evidence only and does not establish physical-site identity.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from audit_3rscan_cross_split_site_overlap import read_annotated_vertices


ANNOTATION_COLUMNS = {
    "objectId": 6,
    "globalId": 7,
    "NYU40": 8,
    "Eigen13": 9,
    "RIO27": 10,
}


def agreement_stats(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float | None]:
    if not len(values_a):
        return {
            "exact_agreement": None,
            "marginal_reference": None,
            "lift_over_marginal": None,
        }
    unique_a, counts_a = np.unique(values_a, return_counts=True)
    unique_b, counts_b = np.unique(values_b, return_counts=True)
    counts_b_by_value = dict(zip(unique_b.tolist(), counts_b.tolist()))
    marginal = sum(
        (count_a / len(values_a)) * (counts_b_by_value.get(int(value), 0) / len(values_b))
        for value, count_a in zip(unique_a, counts_a)
    )
    observed = float(np.mean(values_a == values_b))
    return {
        "exact_agreement": observed,
        "marginal_reference": float(marginal),
        "lift_over_marginal": observed - float(marginal),
    }


def find_pair(rows: list[dict[str, str]], source_id: str, target_id: str) -> dict[str, str]:
    for row in rows:
        if row["source_id"] == source_id and row["target_id"] == target_id:
            return row
    raise ValueError(f"pair is missing from registration CSV: {source_id} -> {target_id}")


def align(path: Path, transform_text: str) -> np.ndarray:
    points = read_annotated_vertices(path)
    transform = np.asarray(json.loads(transform_text), dtype=float)
    points[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    return points


def directed_metrics(source: np.ndarray, target: np.ndarray, threshold_m: float) -> dict[str, object]:
    distances, target_index = cKDTree(target[:, :3]).query(source[:, :3], k=1)
    keep = distances < threshold_m
    source_matched = source[keep]
    target_matched = target[target_index[keep]]
    fields: dict[str, object] = {}
    for name, column in ANNOTATION_COLUMNS.items():
        values_a = source_matched[:, column].astype(np.int64)
        values_b = target_matched[:, column].astype(np.int64)
        nonzero = (values_a != 0) & (values_b != 0)
        all_metrics = agreement_stats(values_a, values_b)
        nonzero_metrics = agreement_stats(values_a[nonzero], values_b[nonzero])
        fields[name] = {
            "exact_agreement_all_matches": all_metrics["exact_agreement"],
            "marginal_reference_all_matches": all_metrics["marginal_reference"],
            "lift_over_marginal_all_matches": all_metrics["lift_over_marginal"],
            "both_nonzero_count": int(nonzero.sum()),
            "exact_agreement_both_nonzero": nonzero_metrics["exact_agreement"],
            "marginal_reference_both_nonzero": nonzero_metrics["marginal_reference"],
            "lift_over_marginal_both_nonzero": nonzero_metrics["lift_over_marginal"],
        }
    rgb_equal = np.all(source_matched[:, 3:6] == target_matched[:, 3:6], axis=1)
    return {
        "source_point_count": int(len(source)),
        "target_point_count": int(len(target)),
        "matched_point_count": int(keep.sum()),
        "matched_distance_threshold_m": threshold_m,
        "source_coverage": float(keep.mean()),
        "mean_matched_distance_m": float(np.mean(distances[keep])) if np.any(keep) else None,
        "rgb_exact_agreement": float(np.mean(rgb_equal)) if len(rgb_equal) else None,
        "annotation_field_agreement": fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-a", required=True)
    parser.add_argument("--source-b", required=True)
    parser.add_argument("--shared-target", required=True)
    parser.add_argument("--distance-max-m", type=float, default=0.02)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    with args.pairs_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pair_a = find_pair(rows, args.source_a, args.shared_target)
    pair_b = find_pair(rows, args.source_b, args.shared_target)
    cloud_a = align(args.dataset_root / args.source_a / "labels.instances.annotated.v2.ply", pair_a["transform"])
    cloud_b = align(args.dataset_root / args.source_b / "labels.instances.annotated.v2.ply", pair_b["transform"])
    a_to_b = directed_metrics(cloud_a, cloud_b, args.distance_max_m)
    b_to_a = directed_metrics(cloud_b, cloud_a, args.distance_max_m)
    result = {
        "status": "exploratory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_a": args.source_a,
        "source_b": args.source_b,
        "shared_validation_target": args.shared_target,
        "source_a_to_source_b": a_to_b,
        "source_b_to_source_a": b_to_a,
        "interpretation_limit": (
            "Both source clouds were registered independently to the same validation target. "
            "Their induced overlap and annotation agreement are descriptive and do not prove "
            "that the scans depict the same physical site."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Shared-target source-to-source overlap (exploratory)",
        "",
        f"Both `{args.source_a}` and `{args.source_b}` were independently registered to validation reference `{args.shared_target}`.",
        f"Nearest-neighbor matches use a {args.distance_max_m:.2f} m threshold in that shared target frame.",
        "",
        "| Direction | Coverage | Matched points | RGB exact | objectId exact | globalId exact / ref. | NYU40 exact / ref. | Eigen13 exact / ref. | RIO27 exact / ref. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (("A → B", a_to_b), ("B → A", b_to_a)):
        fields = metrics["annotation_field_agreement"]
        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.1%}"
        def exact_ref(field: dict[str, float | int | None]) -> str:
            return f"{pct(field['exact_agreement_both_nonzero'])} / {pct(field['marginal_reference_both_nonzero'])}"
        lines.append(
            f"| {label} | {metrics['source_coverage']:.1%} | {metrics['matched_point_count']} | "
            f"{pct(metrics['rgb_exact_agreement'])} | "
            f"{pct(fields['objectId']['exact_agreement_all_matches'])} | "
            + " | ".join(exact_ref(fields[name]) for name in ("globalId", "NYU40", "Eigen13", "RIO27"))
            + " |"
        )
    lines += ["", result["interpretation_limit"], ""]
    lines += [
        "Nonzero categorical entries show observed exact agreement / the expected agreement from the two matched subsets' label marginals.",
        "The marginal comparison is descriptive and does not account for spatial autocorrelation or the post-hoc selection of these scan groups.",
        "",
    ]
    args.output_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "a_to_b_coverage": a_to_b["source_coverage"],
        "b_to_a_coverage": b_to_a["source_coverage"],
        "a_to_b_class_agreement": {name: a_to_b["annotation_field_agreement"][name]["exact_agreement_both_nonzero"] for name in ("NYU40", "Eigen13", "RIO27")},
        "b_to_a_class_agreement": {name: b_to_a["annotation_field_agreement"][name]["exact_agreement_both_nonzero"] for name in ("NYU40", "Eigen13", "RIO27")},
    }, indent=2))


if __name__ == "__main__":
    main()
