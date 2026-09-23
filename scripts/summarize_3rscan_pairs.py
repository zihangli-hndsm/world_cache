"""Summarize fixed-protocol 3RScan audit runs without retuning per scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for audit_dir in args.audit_dirs:
        stats = json.loads((audit_dir / "audit_stats.json").read_text(encoding="utf-8"))
        summary = pd.read_csv(audit_dir / "summary.csv").set_index("method")
        local_descriptor = summary.loc["Geometry + Descriptor (fixed local gate)"]
        local_geometry = summary.loc["Geometry-only (fixed local gate)"]
        local_random = summary.loc["Random (fixed local gate)"]
        full_descriptor = summary.loc["Descriptor-only (full scene)"]
        rows.append({
            "scene_id": f"{stats['reference_scan']}__{stats['query_scan']}",
            "reference_scan": stats["reference_scan"],
            "rescan_scan": stats["query_scan"],
            "num_query_tokens": stats["query_valid"],
            "geometry_hit_rate": stats["geometry_gate_hit_rate"],
            "candidate_count_mean": stats["candidate_count_mean"],
            "candidate_count_median": stats["candidate_count_median"],
            "candidate_count_p10": stats["candidate_count_p10"],
            "candidate_count_p50": stats["candidate_count_p50"],
            "candidate_count_p90": stats["candidate_count_p90"],
            "random_R1": local_random["recall_at_1"],
            "geometry_R1": local_geometry["recall_at_1"],
            "descriptor_R1": full_descriptor["recall_at_1"],
            "geometry_descriptor_R1": local_descriptor["recall_at_1"],
            "geometry_descriptor_R5": local_descriptor["recall_at_5"],
            "random_median_3d_error": local_random["error_median"],
            "geometry_median_3d_error": local_geometry["error_median"],
            "geometry_descriptor_median_3d_error": local_descriptor["error_median"],
            "geometry_descriptor_mean_3d_error": local_descriptor["error_mean"],
            "geometry_descriptor_error_cdf_10cm": local_descriptor["error_cdf_10cm"],
            "geometry_only_error_cdf_10cm": local_geometry["error_cdf_10cm"],
            "full_descriptor_median_3d_error": full_descriptor["error_median"],
        })
    frame = pd.DataFrame(rows).sort_values("scene_id")
    frame.to_csv(args.output_dir / "per_pair.csv", index=False)
    numeric = frame.select_dtypes(include="number")
    aggregate = {
        "pairs": int(len(frame)),
        "mean": numeric.mean().to_dict(),
        "std": numeric.std(ddof=1).to_dict(),
        "median": numeric.median().to_dict(),
        "per_pair": frame.to_dict(orient="records"),
        "decision": {
            "geometry_descriptor_beats_geometry_only_median_error_in_all_pairs": bool((frame["geometry_descriptor_median_3d_error"] < frame["geometry_median_3d_error"]).all()),
            "geometry_descriptor_beats_geometry_only_error_cdf_10cm_in_all_pairs": bool((frame["geometry_descriptor_error_cdf_10cm"] > frame["geometry_only_error_cdf_10cm"]).all()),
            "geometry_descriptor_beats_random_R1_in_all_pairs": bool((frame["geometry_descriptor_R1"] > frame["random_R1"]).all()),
        },
    }
    (args.output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, default=float) + "\n", encoding="utf-8")
    (args.output_dir / "status.json").write_text(json.dumps({"status": "complete", "pairs": len(frame)}, indent=2) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(aggregate["decision"], indent=2))


if __name__ == "__main__":
    main()
