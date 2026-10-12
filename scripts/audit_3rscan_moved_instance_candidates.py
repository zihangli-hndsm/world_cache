"""Measure exact moved-rigid reference-object availability in top-K sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_3rscan_instance_correspondence import read_annotated_mesh


def validate_candidate_query_alignment(
    query_indices: np.ndarray,
    frame_ids: np.ndarray,
    candidate_indices: np.ndarray,
    cached_frame_ids: np.ndarray,
) -> None:
    query_indices = np.asarray(query_indices, dtype=np.int64)
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if len(query_indices) != len(candidate_indices):
        raise ValueError("candidate/query count mismatch")
    if not np.array_equal(query_indices, np.arange(len(candidate_indices), dtype=np.int64)):
        raise ValueError("query diagnostics are incomplete or misordered")
    if len(frame_ids) != len(candidate_indices) or not np.array_equal(frame_ids, cached_frame_ids):
        raise ValueError("candidate/query frame order mismatch")


def candidate_hit_counts(
    candidate_indices: np.ndarray,
    reference_object_ids: np.ndarray,
    expected_reference_object_ids: np.ndarray,
    query_evaluable: np.ndarray,
    k_values: tuple[int, ...] = (1, 5, 10, 20, 50),
) -> dict[str, int | float]:
    if candidate_indices.ndim != 2 or len(candidate_indices) != len(expected_reference_object_ids):
        raise ValueError("candidate and expected-object arrays must share the query dimension")
    if len(query_evaluable) != len(expected_reference_object_ids):
        raise ValueError("query evaluation mask has the wrong length")
    eligible = query_evaluable.astype(bool) & (expected_reference_object_ids > 0)
    denominator = int(eligible.sum())
    result: dict[str, int | float] = {"moved_rigid_queries": denominator}
    for k in k_values:
        if k > candidate_indices.shape[1]:
            raise ValueError(f"K={k} exceeds saved candidate width {candidate_indices.shape[1]}")
        candidate_objects = reference_object_ids[candidate_indices[:, :k]]
        hit = (candidate_objects == expected_reference_object_ids[:, None]).any(axis=1) & eligible
        result[f"top{k}_hits"] = int(hit.sum())
        result[f"top{k}_recall"] = float(hit.sum() / denominator) if denominator else float("nan")
    return result


def run_audit(output_dir: Path, annotation_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(output_dir / "query_diagnostics.csv")
    rows: list[dict[str, object]] = []
    for cache_path in sorted((output_dir / "descriptor_candidates").glob("*.npz")):
        pair = cache_path.stem
        frame = diagnostics[(diagnostics.pair == pair) & (diagnostics.arm == "global_ransac")]
        frame = frame.sort_values("query_index").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"missing fixed-query diagnostics for {pair}")
        with np.load(cache_path) as cache:
            if str(cache["pair"].item()) != pair:
                raise ValueError(f"descriptor cache pair mismatch: {cache_path}")
            reference_id = str(cache["reference_id"].item())
            candidate_indices = cache["candidate_indices"].copy()
            reference_points = cache["reference_points"].copy()
            query_frame_ids = cache["query_frame_ids"].copy()
        if len(frame) != len(candidate_indices):
            raise ValueError(f"candidate/query count mismatch for {pair}")
        try:
            validate_candidate_query_alignment(
                frame.query_index.to_numpy(),
                frame.frame_id.to_numpy(),
                candidate_indices,
                query_frame_ids,
            )
        except ValueError as error:
            raise ValueError(f"query alignment check failed for {pair}: {error}") from error
        reference_tree, local_object_ids, _, _, _ = read_annotated_mesh(annotation_root, reference_id)
        _, nearest = reference_tree.query(reference_points.astype(np.float32), k=1)
        reference_point_object_ids = local_object_ids[nearest]
        expected = frame.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
        evaluable = frame.query_evaluable.astype(bool).to_numpy()
        metrics = candidate_hit_counts(
            candidate_indices,
            reference_point_object_ids,
            expected,
            evaluable,
        )
        rows.append({"pair": pair, "role": "target" if pair == "target" else "source", **metrics})
    by_pair = pd.DataFrame(rows)
    by_pair.to_csv(output_dir / "moved_instance_candidate_recall_by_pair.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    for role, frame in by_pair.groupby("role", sort=False):
        denominator = int(frame.moved_rigid_queries.sum())
        row: dict[str, object] = {
            "role": role,
            "pairs": int(len(frame)),
            "moved_rigid_queries": denominator,
        }
        for k in (1, 5, 10, 20, 50):
            hits = int(frame[f"top{k}_hits"].sum())
            row[f"top{k}_hits"] = hits
            row[f"top{k}_recall"] = hits / denominator if denominator else float("nan")
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "moved_instance_candidate_recall_summary.csv", index=False)
    report_path = output_dir / "multimotion_report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else "# 3RScan multi-motion pilot\n"
    marker = "## Exact physical-instance candidate availability"
    report = report.split(marker)[0].rstrip()
    lines = [
        "## Exact physical-instance candidate availability",
        "",
        "For each officially mapped moved-rigid query, this post-hoc audit checks whether any top-K reference patch belongs to that query's officially mapped local objectId. It measures the candidate-set ceiling; annotations do not enter retrieval or motion inference.",
        "",
        "| Split | Moved queries | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.role} | {row.moved_rigid_queries} | {100 * row.top1_recall:.1f}% | "
            f"{100 * row.top5_recall:.1f}% | {100 * row.top10_recall:.1f}% | "
            f"{100 * row.top20_recall:.1f}% | {100 * row.top50_recall:.1f}% |"
        )
    report_path.write_text(report + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return by_pair, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    args = parser.parse_args()
    _, summary = run_audit(args.output_dir, args.annotation_root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
