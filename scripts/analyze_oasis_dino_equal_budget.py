#!/usr/bin/env python3
"""Compare OASIS-style patch aggregation with DINO mutual at exact frame budgets.

Both procedures use the same frozen RGB-D frame pairs, annotation-derived masks,
and DINOv2 ViT-G tokens. For each frame pair, the OASIS assignment is truncated
to exactly the number of assignments made by native DINO mutual. No identity or
semantic labels determine either assignment or truncation order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_segment_matcher_population import identity_links, semantics


SEED = 20261009
BOOTSTRAP_DRAWS = 100_000
METRICS = {
    "semantic_coverage": ("semantic_correct", "semantic_queries"),
    "semantic_precision": ("semantic_correct", "semantic_assigned"),
    "linked_identity_coverage": ("linked_correct", "linked_queries"),
    "linked_identity_precision": ("linked_correct", "linked_assigned"),
    "linked_semantic_precision": ("linked_semantic_correct", "linked_assigned"),
}


def _counts(query_ids: np.ndarray, ref_ids: np.ndarray, query_sem: np.ndarray,
            ref_sem: np.ndarray, links: dict[int, int], selected: np.ndarray) -> dict[str, int]:
    semantic_eval = query_sem > 0
    expected = np.asarray([links.get(int(x), -1) for x in query_ids], dtype=np.int64)
    linked = semantic_eval & (expected > 0)
    assigned = selected >= 0
    mapped = np.full(len(query_ids), -1, dtype=np.int64)
    pred_sem = np.zeros(len(query_ids), dtype=np.int64)
    mapped[assigned] = ref_ids[selected[assigned]]
    pred_sem[assigned] = ref_sem[selected[assigned]]
    physical_correct = assigned & linked & (mapped == expected)
    semantic_correct = assigned & semantic_eval & (pred_sem == query_sem)
    return {
        "raw_queries": int(len(query_ids)),
        "raw_assigned": int(assigned.sum()),
        "semantic_queries": int(semantic_eval.sum()),
        "semantic_assigned": int((assigned & semantic_eval).sum()),
        "semantic_correct": int(semantic_correct.sum()),
        "linked_queries": int(linked.sum()),
        "linked_assigned": int((assigned & linked).sum()),
        "linked_correct": int(physical_correct.sum()),
        "linked_semantic_correct": int((linked & semantic_correct).sum()),
    }


def _metric_table(frame: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    count_cols = [c for c in frame.columns if c not in {"run_id", "environment_id", "reference_frame", "query_frame", "arm"}]
    groups = frame.groupby(group_col, sort=True)[count_cols].sum().reset_index()
    out: dict[str, float] = {}
    for name, (numerator, denominator) in METRICS.items():
        values = groups[numerator] / groups[denominator].replace(0, np.nan)
        if values.isna().any():
            raise ValueError(f"undefined {name} for at least one {group_col}")
        groups[name] = values
        out[name] = float(values.mean())
    return groups, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-root", type=Path, default=Path("outputs/reports/20261008_segmast3r_validation"))
    parser.add_argument("--oasis-root", type=Path, default=Path("outputs/reports/20261009_oasis_style_dinov2"))
    parser.add_argument("--data-root", type=Path, default=Path("data/3rscan"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_oasis_equal_budget"))
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    root = args.root.resolve()
    input_root = args.input_root if args.input_root.is_absolute() else root / args.input_root
    oasis_root = args.oasis_root if args.oasis_root.is_absolute() else root / args.oasis_root
    data_root = args.data_root
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    frozen = json.loads((input_root / "frozen_segment_population.json").read_text())
    population = frozen["population"]
    oasis_manifest = json.loads((oasis_root / "anonymous_inference_manifest.json").read_text())
    if len(population) != 110 or len(oasis_manifest) != 110:
        raise ValueError("expected complete 110-relation frozen population and OASIS-style manifest")
    expected_runs = {r["run_id"] for r in population}
    if {r["run_id"] for r in oasis_manifest} != expected_runs or any(r["status"] != "complete" for r in oasis_manifest):
        raise ValueError("OASIS-style anonymous manifest differs from the frozen relation population")

    dino = pd.read_csv(input_root / "segment_comparison_dataset/per_frame.csv")
    dino = dino[dino.arm.eq("dino_mutual")].copy()
    key = ["run_id", "reference_frame", "query_frame"]
    if len(dino) != 2181 or dino.duplicated(key).any():
        raise ValueError("expected 2,181 unique DINO-mutual frame-pair rows")
    dino = dino.drop(columns="arm")

    metadata = json.loads((data_root / "3RScan.json").read_text())
    relation_by_id = {r["run_id"]: r for r in population}
    dino_by_key = dino.set_index(key)
    rows: list[dict[str, object]] = []
    ranking_modes = ("match_fraction_then_confidence", "confidence_then_match_fraction")

    for relation_result in oasis_manifest:
        relation = relation_by_id[relation_result["run_id"]]
        ref_semantics, ref_names = semantics(data_root, relation["reference_scan"])
        query_semantics, query_names = semantics(data_root, relation["rescan_scan"])
        links, _ = identity_links(metadata, relation["reference_scan"], relation["rescan_scan"], ref_names, query_names)
        if len(set(links.values())) != len(links):
            raise ValueError(f"identity link collision in {relation['run_id']}")

        source_manifest = json.loads(
            (input_root / "uniform20_inputs" / relation["run_id"] / "manifest.json").read_text()
        )["records"]
        reference_inputs = {int(x["frame_id"]): x for x in source_manifest["reference"]}
        query_inputs = {int(x["frame_id"]): x for x in source_manifest["rescan"]}
        for record in relation_result["records"]:
            rf, qf = int(record["reference_frame"]), int(record["query_frame"])
            row_key = (relation["run_id"], rf, qf)
            if row_key not in dino_by_key.index:
                raise ValueError(f"missing DINO mutual frame pair {row_key}")
            dino_row = dino_by_key.loc[row_key]
            budget = int(dino_row.raw_assigned)
            query_ids = np.load(query_inputs[qf]["identity_targets"])
            ref_ids = np.load(reference_inputs[rf]["identity_targets"])
            query_sem = np.asarray([query_semantics.get(int(x), 0) for x in query_ids], dtype=np.int64)
            ref_sem = np.asarray([ref_semantics.get(int(x), 0) for x in ref_ids], dtype=np.int64)
            with np.load(root / record["prediction"]) as pred:
                fraction_matrix = pred["match_fraction"].astype(np.float64, copy=True)
                confidence_matrix = pred["mean_confidence"].astype(np.float64, copy=True)
                ref_rows = pred["assigned_reference"].astype(np.int64, copy=True)
                query_cols = pred["assigned_query"].astype(np.int64, copy=True)
            if fraction_matrix.shape != (len(ref_ids), len(query_ids)) or confidence_matrix.shape != fraction_matrix.shape:
                raise ValueError(f"mask shape mismatch in {row_key}")
            if len(ref_rows) != len(query_cols) or len(np.unique(query_cols)) != len(query_cols):
                raise ValueError(f"OASIS assignment is not one-to-one in {row_key}")
            if budget > len(query_cols):
                raise ValueError(f"cannot meet DINO raw budget {budget} with {len(query_cols)} OASIS assignments in {row_key}")
            fractions = fraction_matrix[ref_rows, query_cols]
            confidences = confidence_matrix[ref_rows, query_cols]
            if not np.isfinite(fractions).all() or not np.isfinite(confidences).all():
                raise ValueError(f"non-finite OASIS score in {row_key}")

            baseline = dino_row.to_dict()
            for ranking in ranking_modes:
                if ranking == "match_fraction_then_confidence":
                    order = np.lexsort((query_cols, -confidences, -fractions))
                else:
                    order = np.lexsort((query_cols, -fractions, -confidences))
                selected = np.full(len(query_ids), -1, dtype=np.int64)
                chosen = order[:budget]
                selected[query_cols[chosen]] = ref_rows[chosen]
                counts = _counts(query_ids, ref_ids, query_sem, ref_sem, links, selected)
                if counts["raw_assigned"] != budget:
                    raise ValueError(f"exact frame budget mismatch for {ranking} in {row_key}")
                rows.append({
                    "run_id": relation["run_id"], "environment_id": relation["environment_id"],
                    "reference_frame": rf, "query_frame": qf, "arm": ranking, **counts,
                })
            rows.append({
                "run_id": relation["run_id"], "environment_id": relation["environment_id"],
                "reference_frame": rf, "query_frame": qf, "arm": "dino_mutual", **{
                    col: int(baseline[col]) for col in counts.keys()
                },
            })

    frame = pd.DataFrame(rows)
    if len(frame) != 2181 * 3:
        raise ValueError(f"unexpected matched cohort size: {len(frame)} rows")
    frame.to_csv(output / "per_frame_equal_budget.csv", index=False)
    env_frames = []
    macro_summaries = {}
    for arm, subset in frame.groupby("arm", sort=False):
        env, means = _metric_table(subset, "environment_id")
        env["arm"] = arm
        env_frames.append(env)
        macro_summaries[arm] = means
    env_all = pd.concat(env_frames, ignore_index=True)
    env_all.to_csv(output / "per_environment_equal_budget.csv", index=False)

    rng = np.random.default_rng(SEED)
    groups = sorted(frame.environment_id.unique())
    if len(groups) != 47:
        raise ValueError(f"expected 47 reference groups, found {len(groups)}")
    metrics_by_arm = {
        arm: env_all[env_all.arm.eq(arm)].set_index("environment_id").loc[groups]
        for arm in macro_summaries
    }
    family_size = len(ranking_modes) * len(METRICS)
    intervals: dict[str, dict[str, list[float]]] = {}
    boot_draws: dict[str, dict[str, np.ndarray]] = {}
    sampled_indices = rng.integers(0, len(groups), size=(args.draws, len(groups)))
    alpha_tail = 0.05 / (2 * family_size)
    for arm in ranking_modes:
        intervals[arm] = {}
        boot_draws[arm] = {}
        for metric in METRICS:
            diff = metrics_by_arm[arm][metric].to_numpy() - metrics_by_arm["dino_mutual"][metric].to_numpy()
            draws = diff[sampled_indices].mean(axis=1)
            intervals[arm][metric] = [float(x) for x in np.quantile(draws, [alpha_tail, 1 - alpha_tail])]
            boot_draws[arm][metric] = draws
    pd.DataFrame({f"{arm}__{metric}": vals for arm, metrics in boot_draws.items() for metric, vals in metrics.items()}).to_csv(
        output / "paired_group_bootstrap.csv.gz", index=False, compression="gzip"
    )

    pooled = {}
    for arm, subset in frame.groupby("arm", sort=False):
        sums = subset.drop(columns=["run_id", "environment_id", "reference_frame", "query_frame", "arm"]).sum(numeric_only=True)
        pooled[arm] = {
            metric: float(sums[num] / sums[den]) if sums[den] else None
            for metric, (num, den) in METRICS.items()
        }

    summary = {
        "cohort": "110 official 3RScan validation relations, 2,181 frozen frame pairs, 47 released reference groups",
        "input_population_sha256": json.loads((oasis_root / "protocol.json").read_text())["input_population_sha256"],
        "comparison": "OASIS-style DINOv2 patch-to-mask/Hungarian assignment truncated to exactly the per-frame raw acceptance count of native DINO mutual; same frames, masks, and query population",
        "ranking_modes": list(ranking_modes),
        "per_frame_acceptance_budget_exact": True,
        "total_raw_accepts_dino_mutual": int(frame[frame.arm.eq("dino_mutual")].raw_assigned.sum()),
        "total_raw_accepts_each_oasis_arm": {arm: int(frame[frame.arm.eq(arm)].raw_assigned.sum()) for arm in ranking_modes},
        "bootstrap": {
            "cluster": "47 released reference groups, paired resampling",
            "draws": args.draws,
            "seed": SEED,
            "family_size": family_size,
            "interval": "Bonferroni-adjusted paired percentile interval across 2 ranking modes x 5 metrics",
            "interval_quantiles": [alpha_tail, 1 - alpha_tail],
        },
        "macro_metrics": macro_summaries,
        "pooled_metrics": pooled,
        "oasis_minus_dino_mutual_macro_effects": {
            arm: {metric: macro_summaries[arm][metric] - macro_summaries["dino_mutual"][metric] for metric in METRICS}
            for arm in ranking_modes
        },
        "oasis_minus_dino_mutual_adjusted_intervals": intervals,
        "limitations": [
            "This is a controlled DINOv2 formulation audit, not an OASIS-Map system reproduction: the official system uses DINOv3, SAM2 masks, pose-selected co-visibility, and mapping front-end inputs.",
            "Masks are privileged annotation-derived inputs shared by the compared procedures.",
            "DINO mutual supplies a label-free per-frame acceptance budget; OASIS assignments are ranked by their own match fraction/confidence, with both lexicographic orderings reported.",
            "The population and backbone outputs were previously explored; this is not a preregistered independent confirmation.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table_lines = [
        "| Arm | Semantic coverage | Semantic precision | Linked identity coverage | Linked identity precision | Linked semantic precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["dino_mutual", *ranking_modes]
    for arm in order:
        values = macro_summaries[arm]
        table_lines.append("| " + arm.replace("_", " ") + " | " + " | ".join(f"{values[m]*100:.2f}%" for m in METRICS) + " |")
    effects = ["", "| OASIS ranking | Outcome | Difference vs DINO mutual | Family adjusted 95% interval |", "|---|---|---:|---:|"]
    for arm in ranking_modes:
        for metric in METRICS:
            delta = macro_summaries[arm][metric] - macro_summaries["dino_mutual"][metric]
            lo, hi = intervals[arm][metric]
            effects.append(f"| {arm.replace('_', ' ')} | {metric.replace('_', ' ')} | {delta*100:+.2f} pp | [{lo*100:+.2f}, {hi*100:+.2f}] pp |")
    report = f"""# OASIS-style patch association at an exact DINO-mutual acceptance budget\n\nThe complete frozen cohort contains 110 official validation relations, 2,181 frame pairs and 47 released reference groups. For every frame pair, we truncate the OASIS-style one-to-one Hungarian assignment to exactly the raw accepted-mask count made by native DINO mutual. Both procedures use identical images, annotation-derived masks, candidate query masks, and frozen DINOv2 ViT-G features. The budget itself is label-free and comes from the DINO-mutual assignment count.\n\nThe OASIS-style procedure yields a pair-level match fraction and mean patch cosine confidence. Since its published operating rule uses both as thresholds and provides no ranking for equal-budget truncation, we report both possible lexicographic rankings, with deterministic query-index tie-breaking. Neither ordering uses evaluation labels. Metrics follow the canonical segment comparison; identity and linked-query semantic correctness remain separate.\n\n## 47-group macro results\n\n{chr(10).join(table_lines)}\n\nThe OASIS-minus-DINO effects below use paired reference-group bootstrap draws with a Bonferroni correction across 10 outcomes (2 rankings by 5 metrics).\n\n{chr(10).join(effects)}\n\n## Scope\n\nThis is a controlled DINOv2 formulation comparison, not a reproduction of OASIS-Map: the published system uses DINOv3, SAM2 masks, pose-selected co-visible views and a mapping frontend. The shared annotation-derived masks are privileged inputs, and the complete cohort was previously explored. The equal-budget diagnostic can reveal whether OASIS-style patch aggregation changes the correspondence/acceptance tradeoff under fixed inputs; it cannot establish general superiority, independent-site validity, or novelty by itself.\n\nDetailed frame/group rows, 100,000 paired bootstrap draws and exact summary are stored beside this report. Reproduce with `python scripts/analyze_oasis_dino_equal_budget.py`.\n"""
    (output / "report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
