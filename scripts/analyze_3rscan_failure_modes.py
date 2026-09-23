"""Analyze why 3RScan descriptor ranking fails to improve geometry-only.

This is a deterministic, CPU-only post-hoc analysis over completed artifacts.
It intentionally does not rerun backbone extraction or search new thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PAIR_COLUMNS = [
    "scene_id",
    "reference_scan",
    "rescan_scan",
    "num_query_tokens",
    "geometry_hit_rate",
    "candidate_count_mean",
    "candidate_count_median",
    "candidate_count_p90",
    "random_R1",
    "geometry_descriptor_R1",
    "geometry_median_3d_error",
    "geometry_descriptor_median_3d_error",
    "geometry_descriptor_mean_3d_error",
    "geometry_descriptor_error_cdf_10cm",
    "geometry_only_error_cdf_10cm",
    "full_descriptor_median_3d_error",
]


def pair_label(row: pd.Series) -> str:
    return f"{row['reference_scan'][:8]}→{row['rescan_scan'][:8]}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def load_pair_diagnostics(per_pair: Path, audit_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(per_pair)
    frame = frame[PAIR_COLUMNS].copy()
    records = []
    for row in frame.to_dict(orient="records"):
        scene_id = row["scene_id"]
        if scene_id.startswith("02b33dfb"):
            audit_name = "20260917_3rscan_target_pair_audit_s1"
        elif scene_id.startswith("02b33e01"):
            audit_name = "20260917_3rscan_pair02_audit_s1"
        elif scene_id.startswith("0958220d"):
            audit_name = "20260917_3rscan_pair03_audit_s1"
        elif scene_id.startswith("09582212"):
            audit_name = "20260917_3rscan_pair04_audit_s1"
        elif scene_id.startswith("09582225"):
            audit_name = "20260917_3rscan_pair05_audit_s1"
        else:
            raise ValueError(f"No audit mapping for scene_id={scene_id}")

        audit_dir = audit_root / audit_name
        stats = json.loads((audit_dir / "audit_stats.json").read_text(encoding="utf-8"))
        query = pd.read_parquet(audit_dir / "query_metrics.parquet")
        gated = query[query["local_has_candidate"] == 1]
        summary = pd.read_csv(audit_dir / "summary.csv")
        local_geometry = summary.loc[summary["method"] == "Geometry-only (fixed local gate)"].iloc[0]
        local_descriptor = summary.loc[summary["method"] == "Geometry + Descriptor (fixed local gate)"].iloc[0]
        local_random = summary.loc[summary["method"] == "Random (fixed local gate)"].iloc[0]

        row.update(
            {
                "pair_label": pair_label(pd.Series(row)),
                "gated_queries": int(stats["gated_query_count"]),
                "candidate_count_one_fraction": float((gated["local_candidate_count"] == 1).mean()),
                "candidate_count_multi_fraction": float((gated["local_candidate_count"] > 1).mean()),
                "correct_candidate_fraction_10cm": float(stats["correct_candidate_fraction_mean"]["10cm"]),
                "descriptor_minus_geometry_median_m": float(
                    row["geometry_descriptor_median_3d_error"] - row["geometry_median_3d_error"]
                ),
                "descriptor_minus_geometry_cdf10": float(
                    row["geometry_descriptor_error_cdf_10cm"] - row["geometry_only_error_cdf_10cm"]
                ),
                "descriptor_minus_random_local_r1": float(
                    row["geometry_descriptor_R1"] - row["random_R1"]
                ),
                "full_descriptor_minus_geometry_median_m": float(
                    row["full_descriptor_median_3d_error"] - row["geometry_median_3d_error"]
                ),
                "local_geometry_median_m": float(local_geometry["error_median"]),
                "local_descriptor_median_m": float(local_descriptor["error_median"]),
                "local_random_median_m": float(local_random["error_median"]),
                "local_descriptor_minus_geometry_median_m": float(
                    local_descriptor["error_median"] - local_geometry["error_median"]
                ),
                "local_descriptor_minus_geometry_cdf10": float(
                    local_descriptor["error_cdf_10cm"] - local_geometry["error_cdf_10cm"]
                ),
                "local_descriptor_minus_random_r1": float(
                    local_descriptor["recall_at_1"] - local_random["recall_at_1"]
                ),
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def write_report(frame: pd.DataFrame, aggregate: dict, output_dir: Path) -> None:
    weighted = {
        "query_weighted_geometry_hit_rate": float(
            frame["gated_queries"].sum() / frame["num_query_tokens"].sum()
        ),
        "mean_pair_descriptor_minus_geometry_median_m": float(frame["descriptor_minus_geometry_median_m"].mean()),
        "mean_pair_descriptor_minus_geometry_cdf10": float(frame["descriptor_minus_geometry_cdf10"].mean()),
        "pairs_descriptor_worse_median": int((frame["descriptor_minus_geometry_median_m"] > 0).sum()),
        "pairs_descriptor_worse_cdf10": int((frame["descriptor_minus_geometry_cdf10"] < 0).sum()),
        "pairs_descriptor_beats_random_local_r1": int((frame["descriptor_minus_random_local_r1"] > 0).sum()),
    }
    correlations = {}
    for metric in (
        "geometry_hit_rate",
        "candidate_count_mean",
        "candidate_count_multi_fraction",
        "correct_candidate_fraction_10cm",
    ):
        correlations[metric] = {
            "spearman_vs_median_penalty": float(
                frame[metric].corr(frame["descriptor_minus_geometry_median_m"], method="spearman")
            ),
            "spearman_vs_cdf10_penalty": float(
                frame[metric].corr(frame["descriptor_minus_geometry_cdf10"], method="spearman")
            ),
        }

    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(
            {
                "pairs": int(len(frame)),
                "weighted": weighted,
                "correlations_descriptive_only": correlations,
                "aggregate_source": aggregate,
                "interpretation": [
                    "Descriptor ranking is worse than geometry-only on every pair for median 3-D error and 10 cm CDF.",
                    "Most gated queries have one candidate, so descriptor ranking has little opportunity to correct geometry.",
                    "The full-scene descriptor baseline is substantially worse than geometry-only, indicating weak global spatial discrimination under domain shift.",
                    "Pair-level correlations are descriptive only because there are five pairs.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = frame[
        [
            "pair_label",
            "num_query_tokens",
            "geometry_hit_rate",
            "candidate_count_mean",
            "candidate_count_one_fraction",
            "correct_candidate_fraction_10cm",
            "descriptor_minus_geometry_median_m",
            "descriptor_minus_geometry_cdf10",
            "descriptor_minus_random_local_r1",
            "full_descriptor_minus_geometry_median_m",
        ]
    ].copy()
    rows.to_csv(output_dir / "failure_modes.csv", index=False)

    report = [
        "# 3RScan descriptor failure-mode analysis",
        "",
        "This report is a deterministic post-hoc analysis of completed five-pair artifacts. It does not rerun DINOv2 or search new parameters.",
        "",
        "## Findings",
        "",
        f"- Geometry + Descriptor has a higher median 3-D error than Geometry-only in {weighted['pairs_descriptor_worse_median']}/{len(frame)} pairs.",
        f"- Geometry + Descriptor has a lower 10 cm error CDF in {weighted['pairs_descriptor_worse_cdf10']}/{len(frame)} pairs.",
        f"- Descriptor ranking beats the deterministic random local R1 control in {weighted['pairs_descriptor_beats_random_local_r1']}/{len(frame)} pairs, but this does not beat the geometry-only target.",
        f"- The gated query rate is {weighted['query_weighted_geometry_hit_rate']:.1%} when weighted by valid query count.",
        f"- Mean pair-level descriptor penalty is {weighted['mean_pair_descriptor_minus_geometry_median_m']:.3f} m in median error and {weighted['mean_pair_descriptor_minus_geometry_cdf10']:+.3f} in the 10 cm CDF.",
        "- The fixed local gate is usually very small: the table reports the fraction of gated queries with exactly one candidate. A one-candidate set makes geometry-only the unavoidable selection and limits descriptor correction opportunities.",
        "- Full-scene descriptor retrieval is much worse than geometry-only on all five pairs, consistent with weak global descriptor discrimination under real RGB-D/domain shift.",
        "",
        "## Pair-level table",
        "",
        markdown_table(rows),
        "",
        "## Interpretation and limits",
        "",
        "The evidence supports a failure decomposition rather than a single bug: sparse overlap reduces gated coverage; most retained local candidate sets contain one point; and when multiple candidates exist, the descriptor ranking is not robust enough to improve metric correspondence. Pair-level correlations are descriptive only (n=5). The experiment does not separate scene change, depth noise, and appearance/domain shift at object-instance level.",
        "",
        "## Decision",
        "",
        "Keep Gate D NO-GO. Do not promote the current world-memory descriptor as a real-data retrieval method or use it to justify compute reuse. A future continuation must define a narrower task or a new representation and a new evaluation protocol.",
    ]
    (output_dir / "failure_modes.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.scatter(frame["geometry_hit_rate"], frame["descriptor_minus_geometry_median_m"], s=55)
    for _, row in frame.iterrows():
        plt.annotate(row["pair_label"], (row["geometry_hit_rate"], row["descriptor_minus_geometry_median_m"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Geometry-gated coverage")
    plt.ylabel("Descriptor − geometry median error (m)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_vs_descriptor_penalty.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.scatter(frame["candidate_count_mean"], frame["descriptor_minus_geometry_cdf10"], s=55)
    for _, row in frame.iterrows():
        plt.annotate(row["pair_label"], (row["candidate_count_mean"], row["descriptor_minus_geometry_cdf10"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Mean local candidate count")
    plt.ylabel("Descriptor − geometry 10 cm CDF")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "candidate_count_vs_descriptor_cdf_delta.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-pair",
        type=Path,
        default=Path("outputs/runs/20260917_3rscan_multi_pair_summary_s1/per_pair.csv"),
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("outputs/runs/20260917_3rscan_multi_pair_summary_s1/aggregate.json"),
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("outputs/runs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports/20260920_3rscan_failure_modes_s1"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_pair_diagnostics(args.per_pair, args.audit_root)
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    frame.to_csv(args.output_dir / "failure_modes_raw.csv", index=False)
    write_report(frame, aggregate, args.output_dir)
    write_plots(frame, args.output_dir)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
