#!/usr/bin/env python3
"""Aggregate the common-protocol MASt3R/ViT-G official validation comparison."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


OVERLAP_TARGETS = {
    "4fbad31e-465b-2a5d-84b7-c0ddea978db4",
    "c7895f7c-339c-2d13-819f-3bb0b26c91f6",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def query_metrics(rows: list[dict[str, str]], summary: dict[str, object]) -> dict[str, float | int | bool]:
    if not rows:
        raise ValueError("empty query diagnostics")
    semantic_eval = np.asarray([truth(row.get("semantic_evaluable", row.get("query_evaluable", "false"))) for row in rows])
    inlier = np.asarray([truth(row["consensus_inlier"]) for row in rows])
    semantic_correct = np.asarray([truth(row["semantic_correct"]) for row in rows])
    physical_eval = np.asarray([truth(row.get("physical_instance_evaluable", "false")) for row in rows])
    physical_hit = np.asarray([truth(row.get("physical_instance_candidate_hit", "false")) for row in rows])
    physical_correct = np.asarray([truth(row.get("physical_instance_correct", "false")) for row in rows])
    semantic_hit_field = next(
        (key for key in ("semantic_candidate_hit", "candidate_semantic_hit") if key in rows[0]),
        None,
    )
    semantic_hit_count = (
        int((semantic_eval & np.asarray([truth(row[semantic_hit_field]) for row in rows])).sum())
        if semantic_hit_field is not None else None
    )
    candidate_count = (
        np.asarray([int(row["candidate_count"]) for row in rows])
        if "candidate_count" in rows[0] else None
    )
    semantic_n = int(semantic_eval.sum())
    semantic_inliers = semantic_eval & inlier
    correct_inliers = semantic_inliers & semantic_correct
    physical_inliers = physical_eval & inlier
    physical_correct_inliers = physical_inliers & physical_correct
    return {
        "query_points": len(rows),
        "semantic_queries": semantic_n,
        "semantic_candidate_hits": semantic_hit_count,
        "semantic_correct_inliers": int(correct_inliers.sum()),
        "semantic_inliers": int(semantic_inliers.sum()),
        "semantic_coverage": float(correct_inliers.sum() / semantic_n) if semantic_n else float("nan"),
        "semantic_precision": float(correct_inliers.sum() / semantic_inliers.sum()) if semantic_inliers.any() else float("nan"),
        "physical_queries": int(physical_eval.sum()),
        "physical_candidate_hits": int((physical_eval & physical_hit).sum()),
        "physical_correct_inliers": int(physical_correct_inliers.sum()),
        "physical_inliers": int(physical_inliers.sum()),
        "physical_candidate_recall": float((physical_eval & physical_hit).sum() / physical_eval.sum()) if physical_eval.any() else float("nan"),
        "physical_transfer_coverage": float(physical_correct_inliers.sum() / physical_eval.sum()) if physical_eval.any() else float("nan"),
        "physical_precision": float(physical_correct_inliers.sum() / physical_inliers.sum()) if physical_inliers.any() else float("nan"),
        "mean_candidates_per_query": (
            float(candidate_count.sum() / len(rows)) if candidate_count is not None else None
        ),
        "rotation_error_deg": float(summary["scene_rotation_error_deg"]),
        "translation_error_m": float(summary["scene_translation_error_m"]),
        "strict_pose_success": bool(summary["strict_scene_pose_success"]),
    }


def env_aggregate(relation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in relation_rows:
        grouped[str(row["environment_id"])].append(row)
    out: list[dict[str, object]] = []
    count_fields = (
        "query_points", "semantic_queries", "semantic_correct_inliers",
        "semantic_inliers", "physical_queries", "physical_candidate_hits", "physical_correct_inliers",
        "physical_inliers",
    )
    for env, relations in sorted(grouped.items()):
        row: dict[str, object] = {"environment_id": env, "relations": len(relations)}
        for key in count_fields:
            row[f"MASt3R_{key}"] = sum(int(rel[f"MASt3R_{key}"]) for rel in relations)
            row[f"ViT-G_{key}"] = sum(int(rel[f"ViT-G_{key}"]) for rel in relations)
        row["MASt3R_semantic_candidate_hits"] = sum(
            int(rel["MASt3R_semantic_candidate_hits"]) for rel in relations
            if rel["MASt3R_semantic_candidate_hits"] is not None
        )
        row["ViT-G_semantic_candidate_hits"] = None
        for arm in ("MASt3R", "ViT-G"):
            sem_n = int(row[f"{arm}_semantic_queries"])
            sem_inliers = int(row[f"{arm}_semantic_inliers"])
            sem_correct = int(row[f"{arm}_semantic_correct_inliers"])
            physical_n = int(row[f"{arm}_physical_queries"])
            physical_hits = int(row[f"{arm}_physical_candidate_hits"])
            physical_correct = int(row[f"{arm}_physical_correct_inliers"])
            physical_inliers = int(row[f"{arm}_physical_inliers"])
            row[f"{arm}_semantic_coverage_env"] = sem_correct / sem_n if sem_n else float("nan")
            row[f"{arm}_semantic_precision_env"] = sem_correct / sem_inliers if sem_inliers else float("nan")
            row[f"{arm}_semantic_candidate_recall_env"] = (
                int(row[f"{arm}_semantic_candidate_hits"]) / sem_n
                if sem_n and row[f"{arm}_semantic_candidate_hits"] is not None else None
            )
            row[f"{arm}_physical_candidate_recall_env"] = physical_hits / physical_n if physical_n else float("nan")
            row[f"{arm}_physical_transfer_coverage_env"] = physical_correct / physical_n if physical_n else float("nan")
            row[f"{arm}_physical_precision_env"] = physical_correct / physical_inliers if physical_inliers else float("nan")
            row[f"{arm}_strict_pose_success_env"] = sum(bool(rel[f"{arm}_strict_pose_success"]) for rel in relations) / len(relations)
        row["MASt3R_mean_candidates_per_query_env"] = sum(
            float(rel["MASt3R_mean_candidates_per_query"]) * int(rel["MASt3R_query_points"]) for rel in relations
        ) / int(row["MASt3R_query_points"])
        row["MASt3R_strict_pose_successes"] = sum(bool(rel["MASt3R_strict_pose_success"]) for rel in relations)
        row["ViT-G_strict_pose_successes"] = sum(bool(rel["ViT-G_strict_pose_success"]) for rel in relations)
        out.append(row)
    return out


def bootstrap_delta(values: np.ndarray, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(samples, len(values)))
    return np.mean(values[indexes], axis=1).tolist()


def compare(
    env_rows: list[dict[str, object]], method: str, control: str, samples: int, seed: int
) -> dict[str, object]:
    delta = np.asarray([
        float(row[f"{method}_semantic_coverage_env"]) - float(row[f"{control}_semantic_coverage_env"])
        for row in env_rows
    ])
    draws = np.asarray(bootstrap_delta(delta, samples, seed))
    return {
        "environments": len(env_rows),
        "method": method,
        "control": control,
        "method_macro_coverage": float(np.mean([float(row[f"{method}_semantic_coverage_env"]) for row in env_rows])),
        "control_macro_coverage": float(np.mean([float(row[f"{control}_semantic_coverage_env"]) for row in env_rows])),
        "difference": float(delta.mean()),
        "difference_ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dino-run-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, default=Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv"))
    parser.add_argument("--environment-table", type=Path, default=Path("outputs/reports/20261003_vitg_validation/aggregate/per_environment.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    manifest = read_csv(args.pair_manifest)
    environment_vitg = {row["environment_id"]: float(row["ViT-G_semantic_inlier_coverage_env"]) for row in read_csv(args.environment_table)}
    environment_fcgf = {row["environment_id"]: float(row["FCGF_semantic_inlier_coverage_env"]) for row in read_csv(args.environment_table)}
    relation_rows: list[dict[str, object]] = []
    match_modes: set[str] = set()
    for index, pair in enumerate(manifest):
        run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
        mast_root = args.run_dir / "relations" / run_id
        dino_root = args.dino_run_dir / "relations" / run_id / "vitg"
        relation_meta = json.loads((mast_root / "relation.json").read_text(encoding="utf-8"))
        mast_summary = json.loads((mast_root / "summary.json").read_text(encoding="utf-8"))
        match_modes.add(str(mast_summary.get("match_mode", "unknown")))
        mast_diag = read_csv(mast_root / "query_diagnostics.csv")
        dino_diag = read_csv(dino_root / "query_diagnostics.csv")
        if relation_meta.get("environment_id") != pair["environment_id"] or relation_meta.get("rescan_scan") != pair["rescan_scan"]:
            raise ValueError(f"MASt3R relation ID mismatch: {run_id}")
        identity_fields = ("query_index", "frame_id", "query_global_id", "query_semantic_id", "query_object_id")
        if len(mast_diag) != len(dino_diag):
            raise ValueError(f"query denominator mismatch for {run_id}: MASt3R={len(mast_diag)}, ViT-G={len(dino_diag)}")
        for field in identity_fields:
            if [row[field] for row in mast_diag] != [row[field] for row in dino_diag]:
                raise ValueError(f"query identity mismatch in {run_id}: {field}")
        mast = query_metrics(mast_diag, mast_summary)
        dino = query_metrics(dino_diag, {
            "scene_rotation_error_deg": read_csv(dino_root / "summary.csv")[0]["rotation_error_deg"],
            "scene_translation_error_m": read_csv(dino_root / "summary.csv")[0]["translation_error_m"],
            "strict_scene_pose_success": (
                float(read_csv(dino_root / "summary.csv")[0]["rotation_error_deg"]) < 5
                and float(read_csv(dino_root / "summary.csv")[0]["translation_error_m"]) < 0.25
            ),
        })
        if mast["semantic_queries"] != dino["semantic_queries"] or mast["physical_queries"] != dino["physical_queries"]:
            raise ValueError(f"evaluable-label denominator mismatch for {run_id}")
        row = {
            "run_id": run_id,
            "environment_id": pair["environment_id"],
            "reference_scan": pair["reference_scan"],
            "rescan_scan": pair["rescan_scan"],
            **{f"MASt3R_{key}": value for key, value in mast.items()},
            **{f"ViT-G_{key}": value for key, value in dino.items()},
        }
        relation_rows.append(row)

    if len(relation_rows) != 110:
        raise ValueError(f"expected 110 completed validation relations, found {len(relation_rows)}")
    if len(match_modes) != 1 or not match_modes <= {"reciprocal", "dense_topk"}:
        raise ValueError(f"expected one recognized MASt3R matcher mode, found {sorted(match_modes)}")
    match_mode = next(iter(match_modes))
    env_rows = env_aggregate(relation_rows)
    if len(env_rows) != 47:
        raise ValueError(f"expected 47 validation environments, found {len(env_rows)}")
    for row in env_rows:
        env = str(row["environment_id"])
        if env not in environment_fcgf or env not in environment_vitg:
            raise ValueError(f"validation control table lacks environment {env}")
        row["FCGF_semantic_coverage_env"] = environment_fcgf[env]
        # Replace macro ViT-G values with the authoritative saved environment
        # aggregate after the query-level identity/denominator checks above.
        if not np.isclose(float(row["ViT-G_semantic_coverage_env"]), environment_vitg[env], atol=1e-12):
            raise ValueError(f"recomputed ViT-G environment score mismatch: {env}")

    overlap_rows = [row for row in env_rows if row["environment_id"] not in OVERLAP_TARGETS]
    comparisons = {
        "MASt3R_minus_ViT-G": compare(env_rows, "MASt3R", "ViT-G", args.bootstrap_samples, args.seed),
        "MASt3R_minus_FCGF": compare(env_rows, "MASt3R", "FCGF", args.bootstrap_samples, args.seed + 1),
        "MASt3R_minus_ViT-G_excluding_overlap_targets": compare(overlap_rows, "MASt3R", "ViT-G", args.bootstrap_samples, args.seed + 2),
    }
    strict = {
        "MASt3R": sum(bool(row["MASt3R_strict_pose_success"]) for row in relation_rows),
        "ViT-G": sum(bool(row["ViT-G_strict_pose_success"]) for row in relation_rows),
    }
    summary = {
        "status": "complete",
        "relations": len(relation_rows),
        "environments": len(env_rows),
        "query_identity_checks": "all query_index/frame_id/globalId/semanticId/objectId arrays match ViT-G for every relation",
        "model_id": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "match_mode": match_mode,
        "method": (
            "published reciprocal matcher with confidence-ranked top-20 candidates"
            if match_mode == "reciprocal"
            else "custom confidence-weighted dense descriptor retrieval with a unique top-20 candidate cap"
        ) + "; same 512px/20x20 RGB frame pairs, depth-valid query points, 10cm consensus, 1500 iterations, seed 17",
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "MASt3R_pooled_physical_candidate_recall": sum(int(row["MASt3R_physical_candidate_hits"]) for row in relation_rows) / sum(int(row["MASt3R_physical_queries"]) for row in relation_rows),
        "ViT-G_pooled_physical_candidate_recall": sum(int(row["ViT-G_physical_candidate_hits"]) for row in relation_rows) / sum(int(row["ViT-G_physical_queries"]) for row in relation_rows),
        "MASt3R_environment_macro_physical_candidate_recall": float(np.mean([float(row["MASt3R_physical_candidate_recall_env"]) for row in env_rows])),
        "ViT-G_environment_macro_physical_candidate_recall": float(np.mean([float(row["ViT-G_physical_candidate_recall_env"]) for row in env_rows])),
        "MASt3R_pooled_physical_transfer_coverage": sum(int(row["MASt3R_physical_correct_inliers"]) for row in relation_rows) / sum(int(row["MASt3R_physical_queries"]) for row in relation_rows),
        "ViT-G_pooled_physical_transfer_coverage": sum(int(row["ViT-G_physical_correct_inliers"]) for row in relation_rows) / sum(int(row["ViT-G_physical_queries"]) for row in relation_rows),
        "MASt3R_environment_macro_physical_transfer_coverage": float(np.mean([float(row["MASt3R_physical_transfer_coverage_env"]) for row in env_rows])),
        "ViT-G_environment_macro_physical_transfer_coverage": float(np.mean([float(row["ViT-G_physical_transfer_coverage_env"]) for row in env_rows])),
        "MASt3R_pooled_physical_assigned_precision": sum(int(row["MASt3R_physical_correct_inliers"]) for row in relation_rows) / sum(int(row["MASt3R_physical_inliers"]) for row in relation_rows),
        "ViT-G_pooled_physical_assigned_precision": sum(int(row["ViT-G_physical_correct_inliers"]) for row in relation_rows) / sum(int(row["ViT-G_physical_inliers"]) for row in relation_rows),
        "MASt3R_environment_macro_physical_assigned_precision": float(np.mean([float(row["MASt3R_physical_precision_env"]) for row in env_rows])),
        "ViT-G_environment_macro_physical_assigned_precision": float(np.mean([float(row["ViT-G_physical_precision_env"]) for row in env_rows])),
        "MASt3R_pooled_semantic_candidate_recall": sum(int(row["MASt3R_semantic_candidate_hits"]) for row in relation_rows) / sum(int(row["MASt3R_semantic_queries"]) for row in relation_rows),
        "MASt3R_mean_actual_candidates_per_query": sum(float(row["MASt3R_mean_candidates_per_query"]) * int(row["MASt3R_query_points"]) for row in relation_rows) / sum(int(row["MASt3R_query_points"]) for row in relation_rows),
        "strict_scene_pose_successes": strict,
        "strict_scene_pose_environment_macro": {
            "MASt3R": float(np.mean([float(row["MASt3R_strict_pose_success_env"]) for row in env_rows])),
            "ViT-G": float(np.mean([float(row["ViT-G_strict_pose_success_env"]) for row in env_rows])),
        },
        "comparisons": comparisons,
        "posthoc_note": "MASt3R and ViT-G use identical RGB-D queries and label denominators. Environment bootstrap quantifies official environment-ID variation; cross-split scan-content overlap means it does not establish physical-site independence. Exclusion of the two overlap-target environments is post hoc.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_relation.csv", relation_rows)
    write_csv(args.output_dir / "per_environment.csv", env_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    full = comparisons["MASt3R_minus_ViT-G"]
    against_fcgf = comparisons["MASt3R_minus_FCGF"]
    overlap = comparisons["MASt3R_minus_ViT-G_excluding_overlap_targets"]
    method_title = "official reciprocal matcher" if match_mode == "reciprocal" else "custom dense top-20 adaptation"
    method_description = (
        "the published reciprocal matcher and its confidence-ranked sparse candidate pool"
        if match_mode == "reciprocal"
        else "a custom dense confidence-weighted descriptor ranking with a unique top-20 candidate pool"
    )
    matcher_caveat = (
        "The native reciprocal matcher is sparse, averaging fewer than 20 candidates per query; its full-validation estimate therefore describes the published matching procedure, not a candidate-count-matched descriptor comparison."
        if match_mode == "reciprocal"
        else "This dense adaptation was specified from the prior train-split audit and run after the sparse validation result was available; treat its validation metrics as exploratory. It uses a dense candidate budget and is not the published reciprocal procedure."
    )
    candidate_count_note = (
        f"Actual reciprocal candidate count: {summary['MASt3R_mean_actual_candidates_per_query']:.2f} per query on average, below the top-20 cap."
        if match_mode == "reciprocal"
        else f"Actual unique candidate count: {summary['MASt3R_mean_actual_candidates_per_query']:.2f} per query on average, reaching the top-20 cap by construction."
    )
    report = f"""# MASt3R {method_title} on full 3RScan validation

This {('frozen' if match_mode == 'reciprocal' else 'exploratory')} comparison evaluates all {len(relation_rows)} official validation relations across {len(env_rows)} environment IDs. MASt3R uses {method_description}; ViT-G and MASt3R use the same RGB-D frame pairs, depth-valid query records, 10 cm spatial-consensus threshold, 1,500 RANSAC iterations, and seed 17. The aggregator verified query index, frame ID, global label, semantic label, and object ID equality for every relation before comparing outcomes.

## Environment-macro semantic inlier coverage

- MASt3R: {full['method_macro_coverage']:.1%}
- ViT-G: {full['control_macro_coverage']:.1%}
- MASt3R minus ViT-G: {full['difference'] * 100:+.1f} percentage points (95% paired environment-bootstrap CI [{full['difference_ci95'][0] * 100:+.1f}, {full['difference_ci95'][1] * 100:+.1f}]); environment wins/ties/losses {full['wins_ties_losses']}.
- MASt3R minus FCGF: {against_fcgf['difference'] * 100:+.1f} points (95% CI [{against_fcgf['difference_ci95'][0] * 100:+.1f}, {against_fcgf['difference_ci95'][1] * 100:+.1f}]).
- Excluding both post-hoc overlap-target environments, MASt3R minus ViT-G is {overlap['difference'] * 100:+.1f} points (95% CI [{overlap['difference_ci95'][0] * 100:+.1f}, {overlap['difference_ci95'][1] * 100:+.1f}]) across {overlap['environments']} environments.

## Candidate availability and pose

- MASt3R physical-instance candidate Recall@20: {summary['MASt3R_pooled_physical_candidate_recall']:.1%} pooled; {summary['MASt3R_environment_macro_physical_candidate_recall']:.1%} environment macro.
- ViT-G physical-instance candidate Recall@20: {summary['ViT-G_pooled_physical_candidate_recall']:.1%} pooled; {summary['ViT-G_environment_macro_physical_candidate_recall']:.1%} environment macro.
- Physical correct-transfer coverage: MASt3R {summary['MASt3R_pooled_physical_transfer_coverage']:.1%} pooled ({summary['MASt3R_environment_macro_physical_transfer_coverage']:.1%} environment macro); ViT-G {summary['ViT-G_pooled_physical_transfer_coverage']:.1%} pooled ({summary['ViT-G_environment_macro_physical_transfer_coverage']:.1%} environment macro).
- Physical assigned precision: MASt3R {summary['MASt3R_pooled_physical_assigned_precision']:.1%} pooled ({summary['MASt3R_environment_macro_physical_assigned_precision']:.1%} environment macro); ViT-G {summary['ViT-G_pooled_physical_assigned_precision']:.1%} pooled ({summary['ViT-G_environment_macro_physical_assigned_precision']:.1%} environment macro).
- MASt3R semantic candidate Recall@20: {summary['MASt3R_pooled_semantic_candidate_recall']:.1%} pooled.
- {candidate_count_note}
- Strict scene-pose successes: MASt3R {strict['MASt3R']}/110; ViT-G {strict['ViT-G']}/110.

{matcher_caveat} MASt3R weights and their dataset notices are external inputs and are not included in the output bundle. The held-out split has previously been shown to contain clustered scan-content overlap, so environment intervals do not certify independent physical sites. The overlap-excluded comparison is post hoc.

Machine-readable relation/environment tables and summary are saved beside this report. Reproduce with `scripts/run_mast3r_official_validation.py` followed by `scripts/aggregate_mast3r_official_validation.py`.
"""
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
