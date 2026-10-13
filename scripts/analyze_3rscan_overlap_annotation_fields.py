#!/usr/bin/env python3
"""Compare individual 3RScan annotation fields on geometric overlap candidates.

The existing overlap audit reports joint agreement over all instance/semantic
fields. This analysis separates individual fields and compares each observed
match rate with a shuffled-label reference that preserves the matched-point
marginals. The latter is descriptive and is not a site-identity test.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ANNOTATION_PLY = "labels.instances.annotated.v2.ply"
FIELDS = {
    "red": 3,
    "green": 4,
    "blue": 5,
    "objectId": 6,
    "globalId": 7,
    "NYU40": 8,
    "Eigen13": 9,
    "RIO27": 10,
}


def read_vertices(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        vertex_count = None
        for line in stream:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex declaration: {path}")
        return np.loadtxt(stream, max_rows=vertex_count)


def field_agreement(source: np.ndarray, target: np.ndarray, target_idx: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for name, column in FIELDS.items():
        a = source[:, column].astype(np.int64)
        b = target[target_idx, column].astype(np.int64)
        values_a, counts_a = np.unique(a, return_counts=True)
        values_b, counts_b = np.unique(b, return_counts=True)
        b_counts = dict(zip(values_b.tolist(), counts_b.tolist()))
        marginal_match = sum(
            count / len(a) * b_counts.get(int(value), 0) / len(b)
            for value, count in zip(values_a, counts_a)
        )
        observed = float(np.mean(a == b))
        result[name] = {
            "observed_exact_agreement": observed,
            "matched_subset_marginal_reference": float(marginal_match),
            "agreement_lift_over_marginal": observed - float(marginal_match),
        }
        if name in {"objectId", "globalId", "NYU40", "Eigen13", "RIO27"}:
            nonzero = (a != 0) & (b != 0)
            if np.any(nonzero):
                nz_a, nz_b = a[nonzero], b[nonzero]
                values_a, counts_a = np.unique(nz_a, return_counts=True)
                values_b, counts_b = np.unique(nz_b, return_counts=True)
                nz_b_counts = dict(zip(values_b.tolist(), counts_b.tolist()))
                nz_marginal = sum(
                    count / len(nz_a) * nz_b_counts.get(int(value), 0) / len(nz_b)
                    for value, count in zip(values_a, counts_a)
                )
                result[name].update(
                    {
                        "nonzero_both_count": int(np.sum(nonzero)),
                        "nonzero_both_exact_agreement": float(np.mean(nz_a == nz_b)),
                        "nonzero_both_marginal_reference": float(nz_marginal),
                    }
                )
            else:
                result[name].update(
                    {
                        "nonzero_both_count": 0,
                        "nonzero_both_exact_agreement": None,
                        "nonzero_both_marginal_reference": None,
                    }
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fitness-min", type=float, default=0.40)
    parser.add_argument("--rmse-max-m", type=float, default=0.08)
    parser.add_argument("--distance-max-m", type=float, default=0.02)
    parser.add_argument("--expected-pairs", type=int, default=18_095)
    parser.add_argument("--pair-label", default="cross-split")
    args = parser.parse_args()

    with args.pairs_csv.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    candidates = [
        row for row in all_rows
        if float(row["icp_symmetric_fitness"]) >= args.fitness_min
        and float(row["icp_symmetric_rmse_m"]) <= args.rmse_max_m
    ]
    results = []
    for row in candidates:
        source_path = args.dataset_root / row["source_id"] / ANNOTATION_PLY
        target_path = args.dataset_root / row["target_id"] / ANNOTATION_PLY
        source = read_vertices(source_path)
        target = read_vertices(target_path)
        transform = np.asarray(json.loads(row["transform"]))
        aligned = source[:, :3] @ transform[:3, :3].T + transform[:3, 3]
        distances, target_idx = cKDTree(target[:, :3]).query(aligned, k=1)
        keep = distances < args.distance_max_m
        if not np.any(keep):
            field_metrics = {}
        else:
            field_metrics = field_agreement(source[keep], target, target_idx[keep])
        results.append(
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "symmetric_fitness": float(row["icp_symmetric_fitness"]),
                "symmetric_rmse_m": float(row["icp_symmetric_rmse_m"]),
                "matched_vertices": int(np.sum(keep)),
                "source_fraction_with_match": float(np.mean(keep)),
                "target_fraction_with_match": float(np.mean(cKDTree(aligned).query(target[:, :3], k=1)[0] < args.distance_max_m)),
                "field_agreement": field_metrics,
            }
        )

    summary = {
        "status": "complete" if len(all_rows) == args.expected_pairs else "interim",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pair_label": args.pair_label,
        "pair_count": len(all_rows),
        "candidate_count": len(candidates),
        "fitness_min": args.fitness_min,
        "symmetric_rmse_max_m": args.rmse_max_m,
        "distance_max_m": args.distance_max_m,
        "field_names": list(FIELDS),
        "marginal_reference_definition": (
            "Expected exact agreement after independently pairing source and target labels "
            "within the geometrically matched point subsets; descriptive only."
        ),
        "candidates": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Candidate annotation fields ({summary['status']})",
        "",
        f"Current {args.pair_label} CSV: {len(all_rows):,} pairs; {len(candidates)} geometric candidates.",
        f"Candidate gate: symmetric fitness >= {args.fitness_min:.2f}, symmetric RMSE <= {args.rmse_max_m:.2f} m.",
        f"Point match: nearest neighbor within {args.distance_max_m:.2f} m after alignment.",
        "Marginal reference is the expected exact agreement after shuffling labels within the matched subsets; it is descriptive, not a site-identity test.",
        "",
        "| Train reference | Validation reference | Fitness | Matched coverage src/tgt | objectId | globalId | NYU40 | Eigen13 | RIO27 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        fields = item["field_agreement"]
        vals = []
        for name in ("objectId", "globalId", "NYU40", "Eigen13", "RIO27"):
            if name not in fields:
                vals.append("n/a")
            else:
                observed = fields[name]["observed_exact_agreement"]
                reference = fields[name]["matched_subset_marginal_reference"]
                vals.append(f"{observed:.1%} ({reference:.1%} ref.)")
        lines.append(
            f"| `{item['source_id']}` | `{item['target_id']}` | {item['symmetric_fitness']:.3f} | "
            f"{item['source_fraction_with_match']:.1%}/{item['target_fraction_with_match']:.1%} | "
            + " | ".join(vals)
            + " |"
        )
    lines += [
        "",
        "## Nonzero categorical labels",
        "",
        "For ID/class fields, this table excludes vertices where either scan has label 0. Values are observed exact agreement / matched-subset marginal reference.",
        "",
        "| Train reference | Validation reference | objectId | globalId | NYU40 | Eigen13 | RIO27 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        metrics = item["field_agreement"]
        vals = []
        for name in ("objectId", "globalId", "NYU40", "Eigen13", "RIO27"):
            field = metrics.get(name, {})
            observed = field.get("nonzero_both_exact_agreement")
            reference = field.get("nonzero_both_marginal_reference")
            if observed is None or reference is None:
                vals.append("n/a")
            else:
                vals.append(f"{observed:.1%} / {reference:.1%}")
        lines.append(
            f"| `{item['source_id']}` | `{item['target_id']}` | "
            + " | ".join(vals)
            + " |"
        )
    lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "pair_label", "pair_count", "candidate_count")}, indent=2))


if __name__ == "__main__":
    main()
