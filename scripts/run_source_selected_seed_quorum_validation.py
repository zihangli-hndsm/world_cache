#!/usr/bin/env python3
"""Apply the source-selected 2/3 and 3/3 motion-vote rules to full 3RScan val.

Each RANSAC seed uses the motion parameters selected on all 16 training
sources for that seed. Saved descriptor candidates and annotations are reused;
validation labels score votes only after inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from audit_pose_locked_motion_validation_seed_sensitivity import fit_seed
from evaluate_3rscan_instance_correspondence import read_annotated_mesh
from evaluate_3rscan_multimotion import load_feature_cache


SEEDS = (17, 29, 41)
INPUT = Path("outputs/reports/20261006_vitg_pose_locked_motion_validation")
OUTPUT = Path("outputs/reports/20261004_source_seed_quorum_validation")
PAIR_MANIFEST = Path("outputs/reports/20260929_validation_access/validation_pair_manifest.csv")
SELECTION_FILES = {
    17: Path("outputs/reports/20260929_pose_locked_multimotion/source_lopo_selection.csv"),
    29: Path("outputs/reports/20261003_pose_locked_multimotion_seed29/source_lopo_selection.csv"),
    41: Path("outputs/reports/20261003_pose_locked_multimotion_seed41/source_lopo_selection.csv"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_frozen_configs(root: Path) -> dict[int, dict[str, object]]:
    result = {}
    for seed, path in SELECTION_FILES.items():
        frame = pd.read_csv(root / path)
        row = frame[frame.pair == "target_once"]
        if len(row) != 1:
            raise ValueError(f"seed {seed}: expected one source-only target_once selector row")
        row = row.iloc[0]
        config = {
            "modes": int(row.modes),
            "margin_m": float(row.margin_m),
            "min_independent_support": int(row.min_independent_support),
            "training_relations": int(row.training_relations),
            "seed": seed,
            "selection_file": str(path),
        }
        if config["modes"] != 4 or not np.isclose(config["margin_m"], 0.15) or config["min_independent_support"] not in (16, 32):
            raise ValueError(f"unexpected source-selected parameters for seed {seed}: {config}")
        result[seed] = config
    return result


def process_relation(task: tuple) -> tuple[dict[str, object], list[dict[str, object]]]:
    index, pair, root, input_dir, annotation_root, configs = task
    root, input_dir, annotation_root = Path(root), Path(input_dir), Path(annotation_root)
    run_id = f"{index:03d}_{pair['environment_id'][:8]}_{pair['rescan_scan'][:8]}"
    relation_dir = input_dir / "relations" / run_id
    saved = pd.read_csv(relation_dir / "query_diagnostics.csv")
    base = saved[saved.arm == "global_ransac"].sort_values("query_index").reset_index(drop=True)
    method_saved = saved[saved.arm == "combined"].sort_values("query_index").reset_index(drop=True)
    if not np.array_equal(base.query_index, method_saved.query_index) or not np.array_equal(base.frame_id, method_saved.frame_id):
        raise ValueError(f"saved global/motion rows do not align: {run_id}")
    cache_path = input_dir / "descriptor_candidates" / f"{run_id}.npz"
    features = load_feature_cache(
        cache_path, run_id, pair["reference_scan"], pair["rescan_scan"],
        relation_dir, "giant", (2, 20, 40), 224,
    )
    if features is None or len(features.query_points) != len(base):
        raise ValueError(f"missing or misaligned cached candidates: {run_id}")
    if not np.array_equal(np.asarray(features.query_frame_ids), base.frame_id.to_numpy(dtype=np.int64)):
        raise ValueError(f"cached query frame IDs do not match saved validation rows: {run_id}")
    tree, object_ids, _, _, _ = read_annotated_mesh(annotation_root, pair["reference_scan"])
    _, nearest = tree.query(features.reference_points.astype(np.float32), k=1)
    candidate_point_object = object_ids[nearest]
    expected = base.official_rigid_reference_instance_id.to_numpy(dtype=np.int64)
    moved = base.moved_rigid.astype(bool).to_numpy() & (expected > 0)
    candidate_hit = np.zeros(len(base), dtype=bool)
    for qi in np.flatnonzero(moved):
        candidate_hit[qi] = np.any(candidate_point_object[features.candidate_indices[qi, :20]] == expected[qi])

    pred: dict[str, np.ndarray] = {}
    supports: dict[int, int] = {}
    exact_seed17 = False
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            config = configs[seed]
            base_selected, base_assigned, method_selected, method_assigned = fit_seed(
                features, seed, min_independent_support=int(config["min_independent_support"])
            )
            baseline_object = candidate_point_object[base_selected].astype(np.int64)
            method_object = candidate_point_object[method_selected].astype(np.int64)
            baseline_prediction = np.where(base_assigned & (baseline_object > 0), baseline_object, -1)
            method_prediction = np.where(method_assigned & (method_object > 0), method_object, -1)
            pred[f"baseline_seed_{seed}"] = baseline_prediction
            pred[f"method_seed_{seed}"] = method_prediction

            # Independent support is the count of secondary mode 1 inliers not
            # explained by the global mode (all available seed-17 rows are saved).
            status = json.loads((relation_dir / "status.json").read_text())
            if seed == 17:
                support = int(status["selector_diagnostics"]["secondary_independent_support"][0])
                if support != int(((base.spatial_residual_m.to_numpy(dtype=float) > 0.10) &
                                   (method_saved.hypothesis_1_residual_m.to_numpy(dtype=float) <= 0.10)).sum()):
                    raise ValueError(f"seed-17 secondary support failed saved validation reproduction: {run_id}")
                saved_base_idx = base.selected_reference_index.to_numpy(dtype=np.int64)
                saved_method_idx = method_saved.selected_reference_index.to_numpy(dtype=np.int64)
                if not np.array_equal(base_selected, saved_base_idx) or not np.array_equal(method_selected, saved_method_idx):
                    raise ValueError(f"seed-17 selected references failed exact reproduction: {run_id}")
                if not np.array_equal(base_assigned, base.consensus_inlier.astype(bool).to_numpy()):
                    raise ValueError(f"seed-17 baseline assignment mask failed exact reproduction: {run_id}")
                if not np.array_equal(method_assigned, method_saved.consensus_inlier.astype(bool).to_numpy()):
                    raise ValueError(f"seed-17 assignment mask failed exact reproduction: {run_id}")
                expected_base_correct = (
                    base.consensus_inlier.astype(bool).to_numpy()
                    & base.official_rigid_instance_correct.astype(bool).to_numpy()
                )
                expected_method_correct = (
                    method_saved.consensus_inlier.astype(bool).to_numpy()
                    & method_saved.official_rigid_instance_correct.astype(bool).to_numpy()
                )
                reproduced_base_correct = base_assigned & (expected > 0) & (baseline_prediction == expected)
                reproduced_method_correct = method_assigned & (expected > 0) & (method_prediction == expected)
                if not np.array_equal(reproduced_base_correct, expected_base_correct):
                    raise ValueError(f"seed-17 baseline correctness failed exact reproduction: {run_id}")
                if not np.array_equal(reproduced_method_correct, expected_method_correct):
                    raise ValueError(f"seed-17 method correctness failed exact reproduction: {run_id}")
                exact_seed17 = True
            # The first secondary mode support is fully derivable from cached
            # residuals inside fit_seed; record its eligibility from selected
            # target counts as a reproducibility diagnostic.
            supports[seed] = int((method_assigned & ~base_assigned).sum())

    rows = []
    for qi in np.flatnonzero(moved):
        row = {
            "run_id": run_id,
            "environment_id": pair["environment_id"],
            "query_index": int(qi),
            "frame_id": int(base.frame_id.iloc[qi]),
            "expected_reference_object_id": int(expected[qi]),
            "official_target_in_top20": bool(candidate_hit[qi]),
        }
        row.update({key: int(values[qi]) for key, values in pred.items()})
        rows.append(row)
    relation = {
        "run_id": run_id,
        "environment_id": pair["environment_id"],
        "moved_queries": int(moved.sum()),
        "candidate_hit_queries": int((moved & candidate_hit).sum()),
        "seed17_exact_reproduction": exact_seed17,
        **{f"seed_{seed}_secondary_assigned": supports[seed] for seed in SEEDS},
    }
    return relation, rows


def quorum_vote(values: np.ndarray) -> tuple[int, int]:
    values = values[values > 0]
    if not len(values):
        return -1, 0
    ids, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    return int(ids[best]), int(counts[best])


def cluster_ratio_ci(env_counts: pd.DataFrame, num: str, den: str, samples: int, seed: int) -> list[float]:
    counts = env_counts[[num, den]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(counts), size=(samples, len(counts)))
    n, d = counts[draw, 0].sum(axis=1), counts[draw, 1].sum(axis=1)
    values = np.divide(n, d, out=np.full(samples, np.nan), where=d > 0)
    return [float(x) for x in np.nanquantile(values, [0.025, 0.975])]


def macro_delta_ci(env: pd.DataFrame, candidate_col: str, baseline_col: str, samples: int, seed: int) -> dict[str, object]:
    delta = env[candidate_col].to_numpy(dtype=float) - env[baseline_col].to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = delta[draws].mean(axis=1)
    return {"estimate": float(delta.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])], "environments": int(len(delta))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-dir", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--pair-manifest", type=Path, default=PAIR_MANIFEST)
    parser.add_argument("--annotation-root", type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=10_042_029)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    pairs = read_rows(args.root / args.pair_manifest)
    if len(pairs) != 110 or len({p["environment_id"] for p in pairs}) != 47:
        raise ValueError("expected 110 validation relations across 47 environments")
    configs = load_frozen_configs(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(i, pair, str(args.root), str(args.input_dir), str(args.annotation_root), configs) for i, pair in enumerate(pairs)]
    relation_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_relation, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            relation, queries = future.result()
            relation_rows.append(relation)
            query_rows.extend(queries)
            if done % 10 == 0 or done == len(futures):
                print(f"processed {done}/110 relations", flush=True)
    relation_frame = pd.DataFrame(relation_rows)
    query_frame = pd.DataFrame(query_rows)
    keys = ["run_id", "query_index"]
    if relation_frame.run_id.nunique() != 110 or query_frame.duplicated(keys).any():
        raise ValueError("duplicate or incomplete relation/query outputs")
    relation_frame.to_csv(args.output_dir / "per_relation_reproduction.csv", index=False)

    seed_columns = {arm: [f"{arm}_seed_{s}" for s in SEEDS] for arm in ("baseline", "method")}
    for arm, columns in seed_columns.items():
        votes = [quorum_vote(row) for row in query_frame[columns].to_numpy(dtype=np.int64)]
        query_frame[f"{arm}_vote_prediction"] = [x[0] for x in votes]
        query_frame[f"{arm}_vote_agreement"] = [x[1] for x in votes]
    query_frame.to_csv(args.output_dir / "per_query_seed_predictions.csv", index=False)

    summary: dict[str, object] = {
        "status": "complete", "relations": 110, "environments": 47,
        "moved_rigid_queries": int(len(query_frame)),
        "source_selected_configs": {str(s): configs[s] for s in SEEDS},
        "seed17_exact_reproduction_relations": int(relation_frame.seed17_exact_reproduction.sum()),
        "validation_labels_used_for_matching_or_voting": False,
        "quorum_thresholds_selected_from_validation": False,
        "candidate_availability_source": "official rigid reference objectId checked against cached descriptor-ranked top-20 points and annotated reference mesh",
        "bootstrap": {"unit": "47 validation environment clusters", "samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "operating_points": {},
    }
    for quorum in (2, 3):
        result: dict[str, object] = {}
        env_metrics: dict[str, pd.DataFrame] = {}
        for arm in ("baseline", "method"):
            assigned = query_frame[f"{arm}_vote_agreement"].to_numpy() >= quorum
            prediction = query_frame[f"{arm}_vote_prediction"].to_numpy(dtype=np.int64)
            correct = assigned & (prediction == query_frame.expected_reference_object_id.to_numpy(dtype=np.int64))
            query_frame[f"{arm}_q{quorum}_assigned"] = assigned
            query_frame[f"{arm}_q{quorum}_correct"] = correct
            temp = query_frame[["environment_id", "official_target_in_top20"]].copy()
            temp["moved"] = 1
            temp["assigned"] = assigned.astype(int)
            temp["correct"] = correct.astype(int)
            env = temp.groupby("environment_id", sort=True).agg(moved=("moved", "sum"), assigned=("assigned", "sum"), correct=("correct", "sum")).reset_index()
            env["coverage"] = env.correct / env.moved
            env["precision"] = np.divide(env.correct, env.assigned, out=np.full(len(env), np.nan), where=env.assigned.to_numpy() > 0)
            env_metrics[arm] = env
            pooled_assigned = int(assigned.sum())
            pooled_correct = int(correct.sum())
            result[arm] = {
                "assigned": pooled_assigned,
                "correct": pooled_correct,
                "moved_queries": int(len(query_frame)),
                "pooled_correct_coverage": pooled_correct / len(query_frame),
                "pooled_assignment_coverage": pooled_assigned / len(query_frame),
                "pooled_precision": pooled_correct / pooled_assigned if pooled_assigned else None,
                "environment_macro_correct_coverage": float(env.coverage.mean()),
                "environment_macro_assignment_precision": float(env.precision.mean()),
                "pooled_precision_cluster_ci95": cluster_ratio_ci(env, "correct", "assigned", args.bootstrap_samples, args.bootstrap_seed + quorum + (arm == "method") * 20),
            }
            by_availability = {}
            for avail, label in ((True, "candidate_present"), (False, "candidate_absent")):
                mask = query_frame.official_target_in_top20.astype(bool).to_numpy() == avail
                count = int(mask.sum())
                n_assigned = int((assigned & mask).sum())
                n_correct = int((correct & mask).sum())
                by_availability[label] = {"queries": count, "assigned": n_assigned, "correct": n_correct, "coverage": n_correct / count if count else None, "precision": n_correct / n_assigned if n_assigned else None}
            result[arm]["by_candidate_availability"] = by_availability
        paired = env_metrics["method"].merge(env_metrics["baseline"], on="environment_id", suffixes=("_method", "_baseline"))
        for metric in ("coverage", "precision"):
            result[f"paired_environment_macro_delta_{metric}_method_minus_baseline"] = macro_delta_ci(
                paired, f"{metric}_method", f"{metric}_baseline", args.bootstrap_samples,
                args.bootstrap_seed + quorum * 100 + (metric == "precision"),
            )
        summary["operating_points"][f"quorum_{quorum}_of_3"] = result
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# Source-selected seed quorum on full 3RScan validation",
        "",
        "Each of seeds 17/29/41 uses the all-source train configuration from its own frozen LOPO selector; the minimum independent secondary-mode support is 32/16/16, respectively. The vote threshold is fixed at two-of-three or unanimity from the source-only study. Validation labels are used only for post-hoc scoring and candidate-availability stratification.",
        "",
        "| Quorum | Arm | Correct / moved | Pooled correct coverage | Pooled precision |",
        "|---:|---|---:|---:|---:|",
    ]
    for quorum in (2, 3):
        result = summary["operating_points"][f"quorum_{quorum}_of_3"]
        for arm in ("baseline", "method"):
            item = result[arm]
            lines.append(f"| {quorum}/3 | {arm} | {item['correct']:,}/{item['moved_queries']:,} | {item['pooled_correct_coverage']:.1%} | {item['pooled_precision']:.1%} |")
    for quorum in (2, 3):
        result = summary["operating_points"][f"quorum_{quorum}_of_3"]
        cov = result["paired_environment_macro_delta_coverage_method_minus_baseline"]
        prec = result["paired_environment_macro_delta_precision_method_minus_baseline"]
        lines.extend([
            f"",
            f"Quorum {quorum}/3 paired environment-macro method-minus-baseline change: correct coverage {cov['estimate']*100:+.1f} pp (95% CI [{cov['ci95'][0]*100:+.1f},{cov['ci95'][1]*100:+.1f}], {cov['environments']} environments); precision {prec['estimate']*100:+.1f} pp (95% CI [{prec['ci95'][0]*100:+.1f},{prec['ci95'][1]*100:+.1f}], {prec['environments']} environments). Baseline here is seed-voted global RANSAC, not the original single-seed headline baseline.",
        ])
    lines.extend(["", "## Candidate availability", "", "At each quorum, the same vote is scored separately for queries where the official target object is present or absent from the descriptor top-20. This diagnoses whether seed agreement can reject assignments that are impossible to correct with the fixed candidate pool; it does not feed availability labels to the vote.", ""])
    for quorum in (2, 3):
        result = summary["operating_points"][f"quorum_{quorum}_of_3"]
        lines.extend([f"### Quorum {quorum}/3", "", "| Arm | Candidate status | Queries | Assigned | Correct | Precision |", "|---|---|---:|---:|---:|---:|"])
        for arm in ("baseline", "method"):
            for status in ("candidate_present", "candidate_absent"):
                item = result[arm]["by_candidate_availability"][status]
                precision = "—" if item["precision"] is None else f"{item['precision']:.1%}"
                lines.append(f"| {arm} | {status.replace('_',' ')} | {item['queries']:,} | {item['assigned']:,} | {item['correct']:,} | {precision} |")
        lines.append("")
    lines.extend(["All 110 relation runs are seed-17 exact replays; seed-specific support settings come from official-train-only selector artifacts. These seeds and quorums share the same validation scenes, so the environment bootstrap quantifies within-split variation rather than independent-site uncertainty. Three-seed inference also triples matching cost.", "", "Reproduce with `scripts/run_source_selected_seed_quorum_validation.py`. Machine-readable per-query votes, per-relation reproduction, and summary are stored beside this report."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote source-selected quorum validation to {args.output_dir}")


if __name__ == "__main__":
    main()
