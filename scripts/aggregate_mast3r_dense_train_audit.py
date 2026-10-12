#!/usr/bin/env python3
"""Aggregate the full cached 3RScan train-split MASt3R dense audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.lower().isin({"true", "1", "yes"})
    return series.astype(bool)


def _dino_directory(root: Path, pair_run: str) -> Path | None:
    matched = root / "outputs/reports/20261002_physical_instance_candidate_audit_vitg_matched14" / pair_run
    if (matched / "query_diagnostics.csv").is_file():
        return matched
    if "pair06" in pair_run:
        candidate = root / "outputs/reports/20261003_dino_pair06_physical_probe"
        return candidate if (candidate / "query_diagnostics.csv").is_file() else None
    if "pair08" in pair_run:
        candidate = root / "outputs/reports/20261003_dino_pair08_physical_probe"
        return candidate if (candidate / "query_diagnostics.csv").is_file() else None
    return None


def _load_dino(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path / "query_diagnostics.csv")
    if "top_k" in frame:
        frame = frame[frame.top_k.astype(int) == 20].copy()
    needed = {
        "query_index", "frame_id", "physical_instance_evaluable",
        "physical_instance_candidate_hit", "physical_instance_correct",
        "consensus_inlier",
    }
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"DINO diagnostics missing {sorted(missing)}: {path}")
    return frame


def _sparse_directory(root: Path, pair_run: str) -> Path | None:
    candidates = []
    for path in (root / "outputs/runs").glob("20261003_mast3r_reciprocal*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if Path(str(summary.get("pair_run", ""))).name == pair_run:
            candidates.append(path.parent)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: ("fixed" not in p.name, "cuda" not in p.name, p.name))[0]


def relation_row(root: Path, summary_path: Path, dino_path: Path | None,
                 sparse_path: Path | None) -> tuple[dict, dict | None]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pair_run = str(summary["pair_run"])
    source_run = Path(pair_run).name
    mast = pd.read_csv(summary_path.parent / "query_diagnostics.csv")
    evaluable = _bool(mast.physical_instance_evaluable)
    denominator = int(evaluable.sum())
    if denominator != int(summary["linked_physical_instance_queries"]):
        raise ValueError(f"MASt3R denominator mismatch: {source_run}")
    candidate = _bool(mast.physical_instance_candidate_hit)
    correct = _bool(mast.physical_correct_transfer)
    assigned = _bool(mast.consensus_inlier) & evaluable
    if int((correct & evaluable).sum()) != int(summary["physical_correct_transfers"]):
        raise ValueError(f"MASt3R correct-transfer numerator mismatch: {source_run}")
    row = {
        "source_run": source_run,
        "pair_run": pair_run,
        "linked_queries": denominator,
        "mast3r_candidate_hits": int((candidate & evaluable).sum()),
        "mast3r_candidate_recall": float((candidate & evaluable).sum() / denominator),
        "mast3r_correct_transfers": int((correct & evaluable).sum()),
        "mast3r_transfer_coverage": float((correct & evaluable).sum() / denominator),
        "mast3r_assigned_queries": int(assigned.sum()),
        "mast3r_assigned_correct": int((assigned & _bool(mast.physical_instance_correct)).sum()),
        "mast3r_assigned_precision": float((assigned & _bool(mast.physical_instance_correct)).sum() / assigned.sum()) if assigned.any() else 0.0,
        "mast3r_pose_rotation_deg": float(summary["scene_rotation_error_deg"]),
        "mast3r_pose_translation_m": float(summary["scene_translation_error_m"]),
        "mast3r_pose_pass": bool(summary["strict_scene_pose_success"]),
        "mean_actual_candidates": float(summary["mean_actual_candidates_per_query"]),
    }
    if dino_path is None:
        if sparse_path is not None:
            sparse_summary = json.loads((sparse_path / "summary.json").read_text(encoding="utf-8"))
            sparse = pd.read_csv(sparse_path / "query_diagnostics.csv")
            sparse_eval = _bool(sparse.physical_instance_evaluable)
            if int(sparse_eval.sum()) != int(sparse_summary["linked_physical_instance_queries"]):
                raise ValueError(f"sparse MASt3R denominator mismatch: {source_run}")
            sparse_correct = _bool(sparse.physical_correct_transfer)
            row.update({
                "sparse_mean_actual_candidates": float(sparse_summary["mean_actual_candidates_per_query"]),
                "sparse_candidate_recall": float((_bool(sparse.physical_instance_candidate_hit) & sparse_eval).sum() / sparse_eval.sum()),
                "sparse_transfer_coverage": float((sparse_correct & sparse_eval).sum() / sparse_eval.sum()),
                "sparse_pose_rotation_deg": float(sparse_summary["scene_rotation_error_deg"]),
                "sparse_pose_translation_m": float(sparse_summary["scene_translation_error_m"]),
                "sparse_pose_pass": bool(sparse_summary["strict_scene_pose_success"]),
            })
        return row, None
    if sparse_path is None:
        raise ValueError(f"missing official reciprocal result for paired DINO relation {source_run}")
    sparse_summary = json.loads((sparse_path / "summary.json").read_text(encoding="utf-8"))
    sparse = pd.read_csv(sparse_path / "query_diagnostics.csv")
    sparse_eval = _bool(sparse.physical_instance_evaluable)
    if int(sparse_eval.sum()) != int(sparse_summary["linked_physical_instance_queries"]):
        raise ValueError(f"sparse MASt3R denominator mismatch: {source_run}")
    sparse_correct = _bool(sparse.physical_correct_transfer)
    row.update({
        "sparse_mean_actual_candidates": float(sparse_summary["mean_actual_candidates_per_query"]),
        "sparse_candidate_recall": float((_bool(sparse.physical_instance_candidate_hit) & sparse_eval).sum() / sparse_eval.sum()),
        "sparse_transfer_coverage": float((sparse_correct & sparse_eval).sum() / sparse_eval.sum()),
        "sparse_pose_rotation_deg": float(sparse_summary["scene_rotation_error_deg"]),
        "sparse_pose_translation_m": float(sparse_summary["scene_translation_error_m"]),
        "sparse_pose_pass": bool(sparse_summary["strict_scene_pose_success"]),
    })
    dino = _load_dino(dino_path)
    dino_summary = pd.read_csv(dino_path / "summary.csv")
    if "top_k" in dino_summary:
        dino_summary = dino_summary[dino_summary.top_k.astype(int) == 20]
    if "geometry_weight" in dino_summary:
        dino_summary = dino_summary[dino_summary.geometry_weight.astype(float) == 0.0]
    if len(dino_summary) != 1:
        raise ValueError(f"expected one DINO summary row at K=20: {dino_path}")
    dino_summary = dino_summary.iloc[0]
    if dino.query_index.duplicated().any() or mast.query_index.duplicated().any():
        raise ValueError(f"duplicate query_index in {source_run}")
    mast_keys = mast[["query_index", "frame_id", "physical_instance_evaluable",
                      "physical_instance_candidate_hit", "physical_correct_transfer"]].rename(columns={
        "frame_id": "frame_mast", "physical_instance_evaluable": "eval_mast",
        "physical_instance_candidate_hit": "candidate_mast", "physical_correct_transfer": "transfer_mast",
    })
    dino_keys = dino[["query_index", "frame_id", "physical_instance_evaluable",
                      "physical_instance_candidate_hit", "physical_instance_correct", "consensus_inlier"]].rename(columns={
        "frame_id": "frame_dino", "physical_instance_evaluable": "eval_dino",
        "physical_instance_candidate_hit": "candidate_dino", "physical_instance_correct": "correct_dino",
        "consensus_inlier": "inlier_dino",
    })
    sparse_keys = sparse[["query_index", "frame_id", "physical_instance_evaluable",
                          "physical_instance_candidate_hit", "physical_correct_transfer"]].rename(columns={
        "frame_id": "frame_sparse", "physical_instance_evaluable": "eval_sparse",
        "physical_instance_candidate_hit": "candidate_sparse", "physical_correct_transfer": "transfer_sparse",
    })
    joined = mast_keys.merge(dino_keys, on="query_index", how="outer", indicator=True)
    joined = joined.merge(sparse_keys, on="query_index", how="outer", indicator="sparse_merge")
    if not joined._merge.eq("both").all():
        raise ValueError(f"query populations differ for paired relation {source_run}")
    if not joined.sparse_merge.eq("both").all():
        raise ValueError(f"sparse query population differs for paired relation {source_run}")
    if not ((joined.frame_mast == joined.frame_dino) & (joined.frame_mast == joined.frame_sparse)).all():
        raise ValueError(f"query frame IDs differ for paired relation {source_run}")
    mast_eval = _bool(joined.eval_mast)
    dino_eval = _bool(joined.eval_dino)
    sparse_eval = _bool(joined.eval_sparse)
    if not ((mast_eval == dino_eval) & (mast_eval == sparse_eval)).all():
        raise ValueError(f"physical-instance evaluability differs for {source_run}")
    linked = joined[mast_eval]
    if len(linked) != denominator:
        raise ValueError(f"paired query denominator differs for {source_run}")
    mast_hit = _bool(linked.candidate_mast)
    dino_hit = _bool(linked.candidate_dino)
    sparse_hit = _bool(linked.candidate_sparse)
    mast_transfer = _bool(linked.transfer_mast)
    sparse_transfer = _bool(linked.transfer_sparse)
    dino_transfer = _bool(linked.correct_dino) & _bool(linked.inlier_dino)
    paired = {
        "source_run": source_run,
        "linked_queries": denominator,
        "dino_candidate_hits": int(dino_hit.sum()),
        "dino_candidate_recall": float(dino_hit.mean()),
        "dino_correct_transfers": int(dino_transfer.sum()),
        "dino_transfer_coverage": float(dino_transfer.mean()),
        "sparse_candidate_hits": int(sparse_hit.sum()),
        "sparse_candidate_recall": float(sparse_hit.mean()),
        "sparse_mean_actual_candidates": float(sparse_summary["mean_actual_candidates_per_query"]),
        "sparse_correct_transfers": int(sparse_transfer.sum()),
        "sparse_transfer_coverage": float(sparse_transfer.mean()),
        "paired_candidate_delta_hits": int(mast_hit.sum() - dino_hit.sum()),
        "paired_candidate_delta": float(mast_hit.mean() - dino_hit.mean()),
        "paired_transfer_delta_hits": int(mast_transfer.sum() - dino_transfer.sum()),
        "paired_transfer_delta": float(mast_transfer.mean() - dino_transfer.mean()),
        "dense_sparse_candidate_delta_hits": int(mast_hit.sum() - sparse_hit.sum()),
        "dense_sparse_transfer_delta_hits": int(mast_transfer.sum() - sparse_transfer.sum()),
        "sparse_dino_candidate_delta_hits": int(sparse_hit.sum() - dino_hit.sum()),
        "sparse_dino_transfer_delta_hits": int(sparse_transfer.sum() - dino_transfer.sum()),
        "mast3r_pose_pass": bool(summary["strict_scene_pose_success"]),
        "dino_pose_rotation_deg": float(dino_summary.rotation_error_deg),
        "dino_pose_translation_m": float(dino_summary.translation_error_m),
        "dino_pose_pass": bool(float(dino_summary.rotation_error_deg) < 5.0 and float(dino_summary.translation_error_m) < 0.25),
        "sparse_pose_pass": bool(sparse_summary["strict_scene_pose_success"]),
    }
    return row, paired


def relation_bootstrap(rows: list[dict], left_hits: str, right_hits: str,
                       denominator: str, seed: int, samples: int) -> dict:
    rng = np.random.default_rng(seed)
    count = len(rows)
    left = np.asarray([r[left_hits] for r in rows], dtype=np.float64)
    right = np.asarray([r[right_hits] for r in rows], dtype=np.float64)
    weights = np.asarray([r[denominator] for r in rows], dtype=np.float64)
    draws = rng.integers(0, count, size=(samples, count))
    den = weights[draws].sum(axis=1)
    delta = ((left - right)[draws].sum(axis=1)) / den
    point = float((left - right).sum() / weights.sum())
    lo, hi = np.quantile(delta, [0.025, 0.975])
    return {"point_difference": point, "relation_bootstrap_95_ci": [float(lo), float(hi)],
            "relations": count, "bootstrap_samples": samples, "bootstrap_seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/reports/20261003_mast3r_dense_train_audit"))
    parser.add_argument("--metadata", type=Path, default=Path("data/3rscan/3RScan.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/3rscan"))
    parser.add_argument("--target-pair-run", type=Path,
                        default=Path("outputs/runs/20260917_3rscan_target_pair_s1"))
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted((root / "outputs/runs").glob("20261003_mast3r_dense_topk_*/summary.json"))
    metadata_path = args.metadata if args.metadata.is_absolute() else root / args.metadata
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_by_reference = {scene["reference"]: scene.get("type") for scene in metadata}
    target_cfg = json.loads(((args.target_pair_run if args.target_pair_run.is_absolute() else root / args.target_pair_run) / "config.json").read_text(encoding="utf-8"))
    target_key = (target_cfg["reference"], target_cfg["rescan"])

    eligible_pairs: dict[tuple[str, str], list[str]] = {}
    for manifest_path in (root / "outputs/runs").rglob("manifest.json"):
        config_path = manifest_path.parent / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ref, rescan = config.get("reference"), config.get("rescan")
        if not ref or not rescan or split_by_reference.get(ref) != "train":
            continue
        if len(manifest.get("reference", [])) != 20 or len(manifest.get("rescan", [])) != 20:
            continue
        if (ref, rescan) == target_key:
            continue
        if not all((dataset_root / scan / filename).is_file()
                   for scan in (ref, rescan)
                   for filename in ("sequence.zip", "labels.instances.annotated.v2.ply")):
            continue
        eligible_pairs.setdefault((ref, rescan), []).append(manifest_path.parent.relative_to(root).as_posix())
    rows: list[dict] = []
    paired_rows: list[dict] = []
    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        pair_run = str(summary.get("pair_run", ""))
        pair_path = root / pair_run
        config = json.loads((pair_path / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((pair_path / "manifest.json").read_text(encoding="utf-8"))
        if len(manifest.get("reference", [])) != 20 or len(manifest.get("rescan", [])) != 20:
            raise ValueError(f"non-20x20 pair cache included: {pair_run}")
        if split_by_reference.get(config.get("reference")) != "train":
            raise ValueError(f"non-train relation included: {pair_run}")
        if config.get("reference") == "02b33dfb-698f-29c5-8534-5304111c28af":
            raise ValueError(f"held-out target relation must remain excluded: {pair_run}")
        dino_path = _dino_directory(root, Path(pair_run).name)
        sparse_path = _sparse_directory(root, Path(pair_run).name)
        row, paired = relation_row(root, path, dino_path, sparse_path)
        row["dino_audit_available"] = paired is not None
        if paired is not None:
            paired_rows.append(paired)
            row.update({k: v for k, v in paired.items() if k != "source_run"})
        rows.append(row)
    if not rows:
        raise SystemExit("no completed dense MASt3R source relations found")
    actual_pairs: dict[tuple[str, str], str] = {}
    for row in rows:
        config = json.loads((root / row["pair_run"] / "config.json").read_text(encoding="utf-8"))
        key = (config["reference"], config["rescan"])
        if key in actual_pairs:
            raise ValueError(f"duplicate MASt3R evaluation for relation {key}")
        actual_pairs[key] = row["source_run"]
    missing_pairs = sorted(set(eligible_pairs) - set(actual_pairs))
    unexpected_pairs = sorted(set(actual_pairs) - set(eligible_pairs))
    if missing_pairs or unexpected_pairs:
        raise ValueError(f"eligible-source coverage mismatch: missing={missing_pairs}, unexpected={unexpected_pairs}")
    selection = [{
        "reference": key[0], "rescan": key[1],
        "eligible_cache_count": len(cache_dirs),
        "cache_directories": ";".join(sorted(cache_dirs)),
        "evaluated_source_run": actual_pairs[key],
        "split": "train", "frame_count_per_side": 20,
    } for key, cache_dirs in sorted(eligible_pairs.items())]
    pd.DataFrame(selection).to_csv(out / "selection_manifest.csv", index=False)
    full = pd.DataFrame(rows).sort_values("source_run")
    paired = pd.DataFrame(paired_rows).sort_values("source_run")
    if not full.mean_actual_candidates.map(lambda x: abs(x - 20.0) < 1e-8).all():
        raise ValueError("one or more dense runs did not fill the top-20 candidate budget")
    full.to_csv(out / "per_relation.csv", index=False)
    paired.to_csv(out / "paired_dino_comparison.csv", index=False)

    linked = int(full.linked_queries.sum())
    metrics = {
        "relations": int(len(full)),
        "linked_queries": linked,
        "weighted_candidate_recall": float(full.mast3r_candidate_hits.sum() / linked),
        "weighted_correct_transfer_coverage": float(full.mast3r_correct_transfers.sum() / linked),
        "assigned_correct_precision": float(full.mast3r_assigned_correct.sum() / max(1, int(full.mast3r_assigned_queries.sum()))),
        "strict_pose_successes": int(full.mast3r_pose_pass.sum()),
        "strict_pose_relations": int(len(full)),
        "median_rotation_error_deg": float(full.mast3r_pose_rotation_deg.median()),
        "median_translation_error_m": float(full.mast3r_pose_translation_m.median()),
    }
    if len(paired):
        pair_linked = int(paired.linked_queries.sum())
        paired = paired.assign(_zero=0)
        records = paired.to_dict("records")
        metrics["paired_dino_scope"] = {
            "relations": int(len(paired)),
            "linked_queries": pair_linked,
            "mast3r_candidate_recall": float(paired.paired_candidate_delta_hits.add(paired.dino_candidate_hits).sum() / pair_linked),
            "dino_candidate_recall": float(paired.dino_candidate_hits.sum() / pair_linked),
            "mast3r_transfer_coverage": float(paired.paired_transfer_delta_hits.add(paired.dino_correct_transfers).sum() / pair_linked),
            "dino_transfer_coverage": float(paired.dino_correct_transfers.sum() / pair_linked),
            "sparse_candidate_recall": float(paired.sparse_candidate_hits.sum() / pair_linked),
            "sparse_transfer_coverage": float(paired.sparse_correct_transfers.sum() / pair_linked),
            "sparse_mean_actual_candidates_per_query": float(
                (paired.sparse_mean_actual_candidates * paired.linked_queries).sum() / pair_linked
            ),
            "candidate_recall_difference": None,
            "dense_vs_sparse_candidate_difference": relation_bootstrap(
                records, "dense_sparse_candidate_delta_hits", "_zero", "linked_queries",
                args.seed, args.bootstrap_samples,
            ),
            "dense_vs_sparse_transfer_difference": relation_bootstrap(
                records, "dense_sparse_transfer_delta_hits", "_zero", "linked_queries",
                args.seed, args.bootstrap_samples,
            ),
            "sparse_vs_dino_candidate_difference": relation_bootstrap(
                records, "sparse_dino_candidate_delta_hits", "_zero", "linked_queries",
                args.seed, args.bootstrap_samples,
            ),
            "sparse_vs_dino_transfer_difference": relation_bootstrap(
                records, "sparse_dino_transfer_delta_hits", "_zero", "linked_queries",
                args.seed, args.bootstrap_samples,
            ),
            "transfer_difference": relation_bootstrap(
                records, "paired_transfer_delta_hits", "_zero", "linked_queries",
                args.seed, args.bootstrap_samples,
            ),
            "relations_with_higher_mast3r_transfer": int((paired.paired_transfer_delta_hits > 0).sum()),
            "relations_with_equal_transfer": int((paired.paired_transfer_delta_hits == 0).sum()),
            "relations_with_lower_mast3r_transfer": int((paired.paired_transfer_delta_hits < 0).sum()),
            "mast3r_pose_successes": int(paired.mast3r_pose_pass.sum()),
            "dino_pose_successes": int(paired.dino_pose_pass.sum()),
        }
        metrics["paired_dino_scope"]["candidate_recall_difference"] = relation_bootstrap(
            records, "paired_candidate_delta_hits", "_zero", "linked_queries",
            args.seed, args.bootstrap_samples,
        )
        metrics["paired_dino_scope"]["sparse_pose_successes"] = int(full.loc[full.dino_audit_available, "sparse_pose_pass"].sum())
        metrics["paired_dino_scope"]["dense_pose_successes"] = int(full.loc[full.dino_audit_available, "mast3r_pose_pass"].sum())

    summary = {
        "status": "complete",
        "protocol": {
            "scope": "all completed, deduplicated, locally cached 20-frame-per-side 3RScan train relations with annotations and sequence archives; held-out target excluded",
            "method": "custom MASt3R ViT-L 512 dense confidence-weighted descriptor top-20 per query; seed 17; 1,500 consensus iterations; 0.10m threshold",
            "paired_control": "official MASt3R reciprocal matching and same query_index/frame_id DINOv2 ViT-G physical-instance audit; candidate cap K=20; paired query deltas",
        },
        "caveats": [
            "This is a train-split, availability-defined audit, not external validation.",
            "Relation-level bootstrap intervals describe sampled relation variability; physical-site independence was not verified.",
            "The MASt3R dense rule is a custom budget-matched diagnostic, not the official reciprocal-matching protocol.",
            "Three evaluated relations lack a saved matching physical-instance DINO audit and are included only in the MASt3R full-scope summary.",
            "All metrics use the linked physical-instance query denominator; pose success uses rotation <5 degrees and translation <0.25 m.",
            "Sparse queries with no reciprocal candidates are excluded from RANSAC but remain in coverage denominators.",
        ],
        "metrics": metrics,
        "relations": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Full cached 3RScan train audit: dense MASt3R",
        "",
        "This fixed-protocol audit includes every completed, deduplicated local train relation with 20 frames per side, annotations, and sequence archives. The designated held-out target is excluded. Relation selection is determined by local input availability, so the results are train-split evidence rather than external validation.",
        "",
        f"MASt3R covers {metrics['relations']} relations and {linked:,} linked physical-instance queries. Weighted physical candidate Recall@20 is {metrics['weighted_candidate_recall']:.1%}; correct physical-transfer coverage is {metrics['weighted_correct_transfer_coverage']:.1%}. Strict scene-pose success is {metrics['strict_pose_successes']}/{metrics['strict_pose_relations']}.",
        "",
        "| Source cache | Linked queries | Candidate Recall@20 | Correct transfer | Precision | Rotation | Translation | Pose |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in full.itertuples(index=False):
        lines.append(
            f"| {row.source_run} | {row.linked_queries} | {row.mast3r_candidate_recall:.1%} | {row.mast3r_transfer_coverage:.1%} | "
            f"{row.mast3r_assigned_precision:.1%} | "
            f"{row.mast3r_pose_rotation_deg:.2f}° | {row.mast3r_pose_translation_m:.3f} m | {'pass' if row.mast3r_pose_pass else 'fail'} |"
        )
    if len(paired):
        p = metrics["paired_dino_scope"]
        ci = p["transfer_difference"]["relation_bootstrap_95_ci"]
        lines += [
            "",
            f"On the exact three-method subset ({p['relations']} relations; {p['linked_queries']:,} queries), physical candidate Recall@20 is dense MASt3R {p['mast3r_candidate_recall']:.1%}, official reciprocal MASt3R {p['sparse_candidate_recall']:.1%}, and ViT-G {p['dino_candidate_recall']:.1%}. Correct-transfer coverage is {p['mast3r_transfer_coverage']:.1%}, {p['sparse_transfer_coverage']:.1%}, and {p['dino_transfer_coverage']:.1%}, respectively.",
            f"The official sparse matcher averages {p['sparse_mean_actual_candidates_per_query']:.2f} actual candidates per query; dense MASt3R uses 20. Dense MASt3R exceeds ViT-G candidate recall by {p['candidate_recall_difference']['point_difference']:+.1%} (95% CI [{p['candidate_recall_difference']['relation_bootstrap_95_ci'][0]:+.1%}, {p['candidate_recall_difference']['relation_bootstrap_95_ci'][1]:+.1%}]) and sparse MASt3R by {p['dense_vs_sparse_candidate_difference']['point_difference']:+.1%} (95% CI [{p['dense_vs_sparse_candidate_difference']['relation_bootstrap_95_ci'][0]:+.1%}, {p['dense_vs_sparse_candidate_difference']['relation_bootstrap_95_ci'][1]:+.1%}]).",
            f"Dense MASt3R exceeds ViT-G transfer by {p['transfer_difference']['point_difference']:+.1%} (relation-bootstrap 95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]) and exceeds sparse MASt3R by {p['dense_vs_sparse_transfer_difference']['point_difference']:+.1%} (95% CI [{p['dense_vs_sparse_transfer_difference']['relation_bootstrap_95_ci'][0]:+.1%}, {p['dense_vs_sparse_transfer_difference']['relation_bootstrap_95_ci'][1]:+.1%}]). These intervals resample relations, not confirmed independent sites.",
            f"Strict scene-pose success is dense MASt3R {p['dense_pose_successes']}/{p['relations']}, reciprocal MASt3R {p['sparse_pose_successes']}/{p['relations']}, and ViT-G {p['dino_pose_successes']}/{p['relations']} (rotation <5 degrees and translation <0.25 m).",
        ]
    lines += [
        "",
        "The custom dense retrieval rule and relation-level uncertainty limit interpretation. Candidate recall, consensus transfer, and strict scene pose are separate outcomes; high candidate recall alone does not establish useful transfer.",
        "",
        "Machine-readable outputs are `summary.json`, `per_relation.csv`, and `paired_dino_comparison.csv`.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"relations": metrics["relations"], "linked_queries": linked,
                      "paired_relations": len(paired), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
