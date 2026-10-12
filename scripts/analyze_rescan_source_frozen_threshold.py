#!/usr/bin/env python3
"""Transfer a 3RScan source-only DINO score threshold to rendered Rescan points.

The 75th-percentile gate is recalibrated only from the 16 frozen 3RScan
source relations. Rescan labels are used only to score the frozen gate.
This tests transfer across both dataset and input protocol (native RGB-D
patches -> synthetic renders of labeled point clouds), so results are
descriptive external evidence rather than a replication of the RGB-D method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_GLOB = "20260922_3rscan_consensus_*_vitg_diag_clean/query_diagnostics.csv"
RESCAN_DEFAULT = Path("outputs/reports/20261006_rescan_dinog_scoreconfidence_corrected/query_metrics.csv")
LOPO_DEFAULT = Path("outputs/reports/20261006_rescan_score_margin_selectivity_allqueries/lopo_selectivity_predictions.csv")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261009


def load_source(root: Path) -> tuple[list[np.ndarray], int]:
    paths = sorted((root / "outputs/runs").glob(SOURCE_GLOB))
    paths = [p for p in paths if p.parent.name != "20260922_3rscan_consensus_target_vitg_diag_clean"]
    if len(paths) != 16:
        raise ValueError(f"expected 16 frozen source relations, found {len(paths)}")
    relation_scores = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[
            frame.geometry_weight.eq(0.0)
            & frame.top_k.eq(20)
            & frame.query_evaluable.astype(bool)
        ]
        scores = frame.top1_descriptor_score.to_numpy(dtype=np.float64)
        if len(scores) == 0 or not np.isfinite(scores).all():
            raise ValueError(f"invalid source score rows: {path}")
        relation_scores.append(scores)
    return relation_scores, len(paths)


def bool_series(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def load_rescan(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame.method.eq("dino_synthetic_views")].copy()
    frame["present"] = bool_series(frame.target_present_in_full_source)
    frame["physical_top1"] = pd.to_numeric(frame.physical_rank_top20, errors="coerce").eq(1).to_numpy()
    frame["semantic_top1"] = pd.to_numeric(frame.semantic_rank_top20, errors="coerce").eq(1).to_numpy()
    frame["score"] = pd.to_numeric(frame.top1_similarity, errors="coerce")
    if frame.score.isna().any() or not np.isfinite(frame.score).all():
        raise ValueError("Rescan rows contain missing/nonfinite top-1 scores")
    if frame.scene.nunique() != 13 or frame.target_scan.nunique() != 32:
        raise ValueError(f"expected 13 scenes / 32 target scans, got {frame.scene.nunique()} / {frame.target_scan.nunique()}")
    return frame


def site_metrics(group: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    keep = group.score.to_numpy() >= threshold
    physical = group.physical_top1.to_numpy(dtype=bool)
    semantic = group.semantic_top1.to_numpy(dtype=bool)
    present = group.present.to_numpy(dtype=bool)
    accepted = int(keep.sum())
    n_present = int(present.sum())
    accepted_present = int((keep & present).sum())
    return {
        "queries": len(group),
        "accepted": accepted,
        "baseline_physical_top1": float(physical.mean()),
        "accepted_physical_precision": float(physical[keep].mean()) if accepted else float("nan"),
        "accepted_physical_precision_delta": float(physical[keep].mean() - physical.mean()) if accepted else float("nan"),
        "all_query_coverage": float(keep.mean()),
        "target_present_queries": n_present,
        "target_present_acceptance": float(accepted_present / n_present) if n_present else float("nan"),
        "target_present_physical_recall": float((keep & physical & present).sum() / n_present) if n_present else float("nan"),
        "accepted_semantic_precision": float(semantic[keep].mean()) if accepted else float("nan"),
    }


def pooled_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    keep = frame.score.to_numpy() >= threshold
    physical = frame.physical_top1.to_numpy(dtype=bool)
    semantic = frame.semantic_top1.to_numpy(dtype=bool)
    present = frame.present.to_numpy(dtype=bool)
    return {
        "queries": int(len(frame)),
        "accepted": int(keep.sum()),
        "all_query_coverage": float(keep.mean()),
        "baseline_physical_top1": float(physical.mean()),
        "accepted_physical_precision": float(physical[keep].mean()) if keep.any() else float("nan"),
        "accepted_physical_precision_delta": float(physical[keep].mean() - physical.mean()) if keep.any() else float("nan"),
        "accepted_semantic_precision": float(semantic[keep].mean()) if keep.any() else float("nan"),
        "target_present_queries": int(present.sum()),
        "target_present_acceptance": float((keep & present).sum() / present.sum()) if present.any() else float("nan"),
        "target_present_physical_recall": float((keep & physical & present).sum() / present.sum()) if present.any() else float("nan"),
    }


def bootstrap(source_relations: list[np.ndarray], sites: dict[str, pd.DataFrame], reps: int) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = np.asarray(sorted(sites))
    arrays = {}
    for name, frame in sites.items():
        score = frame.score.to_numpy(dtype=np.float64)
        physical = frame.physical_top1.to_numpy(dtype=bool)
        arrays[name] = (score, physical)

    rows = []
    nr = len(source_relations)
    for _ in range(reps):
        sampled_rel = rng.integers(0, nr, size=nr)
        threshold = float(np.quantile(np.concatenate([source_relations[i] for i in sampled_rel]), 0.75))
        sampled_sites = rng.integers(0, len(names), size=len(names))
        covs, precisions, deltas = [], [], []
        pooled_n = pooled_accept = pooled_correct = pooled_base_correct = 0
        for idx in sampled_sites:
            score, physical = arrays[names[idx]]
            keep = score >= threshold
            covs.append(float(keep.mean()))
            if keep.any():
                p = float(physical[keep].mean())
                precisions.append(p)
                deltas.append(p - float(physical.mean()))
            pooled_n += len(score)
            pooled_accept += int(keep.sum())
            pooled_correct += int((keep & physical).sum())
            pooled_base_correct += int(physical.sum())
        rows.append({
            "threshold": threshold,
            "scene_macro_coverage": float(np.mean(covs)),
            "scene_macro_accepted_precision": float(np.mean(precisions)) if precisions else float("nan"),
            "scene_macro_precision_delta": float(np.mean(deltas)) if deltas else float("nan"),
            "pooled_coverage": pooled_accept / pooled_n,
            "pooled_accepted_precision": pooled_correct / pooled_accept if pooled_accept else float("nan"),
            "pooled_precision_delta": pooled_correct / pooled_accept - pooled_base_correct / pooled_n if pooled_accept else float("nan"),
        })
    return pd.DataFrame(rows)


def load_lopo(path: Path, rescan: pd.DataFrame) -> pd.DataFrame:
    keys = ["scene", "source_scan", "target_scan", "target_instance_id", "target_sample_index"]
    columns = keys + ["population", "signal", "train_acceptance_target", "method_fold", "threshold", "accepted", "is_target_present", "physical_top1"]
    lopo = pd.read_csv(path, usecols=columns)
    lopo = lopo[
        lopo.population.eq("all_queries")
        & lopo.signal.eq("top1_similarity")
        & lopo.train_acceptance_target.eq(0.25)
        & lopo.method_fold.eq("dino_synthetic_views")
        & ~lopo.scene.isin(["scene_b", "scene_c"])
    ].copy()
    if lopo.duplicated(keys).any():
        raise ValueError("duplicate Rescan LOPO rows for frozen query IDs")
    target = rescan[~rescan.scene.isin(["scene_b", "scene_c"])].copy()
    merged = target.merge(lopo[keys + ["threshold", "accepted", "is_target_present", "physical_top1"]], on=keys, how="outer", suffixes=("_query", "_lopo"), indicator=True, validate="one_to_one")
    if len(merged) != len(target) or not merged._merge.eq("both").all():
        counts = merged._merge.value_counts().to_dict()
        raise ValueError(f"fixed/LOPO query supports differ: {counts}")
    if not np.array_equal(merged.present.to_numpy(), bool_series(merged.is_target_present)):
        raise ValueError("target-present labels differ between frozen query and LOPO artifacts")
    if not np.array_equal(merged.physical_top1_query.astype(bool).to_numpy(), bool_series(merged.physical_top1_lopo)):
        raise ValueError("top-1 identity outcomes differ between artifacts")
    return merged


def interval(series: pd.Series) -> dict[str, float]:
    values = series.dropna().to_numpy(dtype=np.float64)
    return {"q025": float(np.quantile(values, 0.025)), "median": float(np.quantile(values, 0.5)), "q975": float(np.quantile(values, 0.975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rescan-query-metrics", type=Path, default=RESCAN_DEFAULT)
    parser.add_argument("--lopo-predictions", type=Path, default=LOPO_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/20261009_rescan_source_frozen_threshold"))
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    root = args.root.resolve()
    query_path = args.rescan_query_metrics if args.rescan_query_metrics.is_absolute() else root / args.rescan_query_metrics
    lopo_path = args.lopo_predictions if args.lopo_predictions.is_absolute() else root / args.lopo_predictions
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    source_relations, source_n = load_source(root)
    source_scores = np.concatenate(source_relations)
    threshold = float(np.quantile(source_scores, 0.75))
    if not 0.6983 <= threshold <= 0.6987:
        raise ValueError(f"source threshold failed reproduction: {threshold}")
    rescan = load_rescan(query_path)
    lopo = load_lopo(lopo_path, rescan)
    sites = {name: frame.copy() for name, frame in rescan.groupby("scene", sort=True)}
    site_rows = []
    for name, group in sites.items():
        row = {"scene": name, **site_metrics(group, threshold)}
        site_rows.append(row)
    per_site = pd.DataFrame(site_rows)
    per_site.to_csv(out / "per_scene_metrics.csv", index=False)

    comparison_rows = []
    for scene, group in lopo.groupby("scene", sort=True):
        group = group.copy()
        source_keep = group.score.to_numpy() >= threshold
        lopo_keep = bool_series(group.accepted)
        correct = group.physical_top1_query.astype(bool).to_numpy()
        row = {"scene": scene, "queries": len(group), "baseline_physical_top1": float(correct.mean())}
        for name, keep in (("source_frozen", source_keep), ("rescan_lopo", lopo_keep)):
            row[f"{name}_threshold"] = threshold if name == "source_frozen" else float(group.threshold.iloc[0])
            row[f"{name}_coverage"] = float(keep.mean())
            row[f"{name}_accepted"] = int(keep.sum())
            row[f"{name}_physical_precision"] = float(correct[keep].mean()) if keep.any() else float("nan")
            row[f"{name}_precision_delta"] = row[f"{name}_physical_precision"] - row["baseline_physical_top1"] if keep.any() else float("nan")
        comparison_rows.append(row)
    policy_comparison = pd.DataFrame(comparison_rows)
    policy_comparison.to_csv(out / "policy_comparison_by_scene.csv", index=False)
    policy_comparison_summary = {
        "query_ids_exactly_matched": int(len(lopo)),
        "scenes": int(lopo.scene.nunique()),
        "target_scans": int(lopo.target_scan.nunique()),
        "baseline_physical_top1_macro": float(policy_comparison.baseline_physical_top1.mean()),
        "source_frozen": {
            "scene_macro_coverage": float(policy_comparison.source_frozen_coverage.mean()),
            "scene_macro_accepted_precision": float(policy_comparison.source_frozen_physical_precision.mean()),
            "scene_macro_precision_delta": float(policy_comparison.source_frozen_precision_delta.mean()),
            "sites_with_positive_delta": int((policy_comparison.source_frozen_precision_delta > 0).sum()),
        },
        "rescan_lopo_q25": {
            "scene_macro_coverage": float(policy_comparison.rescan_lopo_coverage.mean()),
            "scene_macro_accepted_precision": float(policy_comparison.rescan_lopo_physical_precision.mean()),
            "scene_macro_precision_delta": float(policy_comparison.rescan_lopo_precision_delta.mean()),
            "sites_with_positive_delta": int((policy_comparison.rescan_lopo_precision_delta > 0).sum()),
        },
        "pooled_source_frozen": {
            "coverage": float(bool_series(lopo.score.ge(threshold)).mean()),
            "accepted_precision": float(lopo.loc[lopo.score.ge(threshold), "physical_top1_query"].astype(bool).mean()),
            "baseline_precision": float(lopo.physical_top1_query.astype(bool).mean()),
            "precision_delta": float(lopo.loc[lopo.score.ge(threshold), "physical_top1_query"].astype(bool).mean() - lopo.physical_top1_query.astype(bool).mean()),
        },
        "pooled_rescan_lopo_q25": {
            "coverage": float(bool_series(lopo.accepted).mean()),
            "accepted_precision": float(lopo.loc[bool_series(lopo.accepted), "physical_top1_query"].astype(bool).mean()),
            "baseline_precision": float(lopo.physical_top1_query.astype(bool).mean()),
            "precision_delta": float(lopo.loc[bool_series(lopo.accepted), "physical_top1_query"].astype(bool).mean() - lopo.physical_top1_query.astype(bool).mean()),
        },
        "interpretation": "descriptive same-query comparison; Rescan LOPO thresholds target 25% acceptance using other Rescan sites, while the source-frozen threshold is a single 3RScan-derived q75",
    }

    summaries = {}
    for label, frame in {
        "all_13_scenes": rescan,
        "nonpilot_11_scenes_excluding_b_c": rescan[~rescan.scene.isin(["scene_b", "scene_c"])],
    }.items():
        summaries[label] = {
            "pooled": pooled_metrics(frame, threshold),
            "scene_macro": {
                col: float(per_site.loc[per_site.scene.isin(frame.scene.unique()), col].mean())
                for col in ["all_query_coverage", "baseline_physical_top1", "accepted_physical_precision", "accepted_physical_precision_delta", "target_present_acceptance", "target_present_physical_recall", "accepted_semantic_precision"]
            },
            "scenes": int(frame.scene.nunique()),
            "target_scans": int(frame.target_scan.nunique()),
            "queries": int(len(frame)),
        }

    draws = bootstrap(source_relations, sites, args.bootstrap_replicates)
    draws.to_csv(out / "joint_source_site_bootstrap.csv.gz", index=False, compression="gzip")
    nonpilot_sites = {name: frame for name, frame in sites.items() if name not in {"scene_b", "scene_c"}}
    nonpilot_draws = bootstrap(source_relations, nonpilot_sites, args.bootstrap_replicates)
    nonpilot_draws.to_csv(out / "joint_source_site_bootstrap_nonpilot.csv.gz", index=False, compression="gzip")
    summary = {
        "status": "complete",
        "source_dataset": "3RScan official source/train set",
        "source_relations": source_n,
        "source_queries": int(len(source_scores)),
        "source_quantile": 0.75,
        "frozen_threshold": threshold,
        "rescan_model": "DINOv2 ViT-Giant, blocks 2/20/40, synthetic views from colored annotated point clouds",
        "rescan_query_rows": int(len(rescan)),
        "rescan_scenes": int(rescan.scene.nunique()),
        "rescan_target_scans": int(rescan.target_scan.nunique()),
        "rescan_labels_used_for_threshold": False,
        "score_definition": "maximum cosine similarity of a target point descriptor to the 5 cm-sampled source-point candidate pool; accept if score >= source q75",
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
            "units": "resample 16 3RScan source relations to refit q75; independently resample Rescan scenes; both dataset and threshold-fit variability included",
            "pooled_intervals": {key: interval(draws[key]) for key in ["threshold", "pooled_coverage", "pooled_accepted_precision", "pooled_precision_delta"]},
            "scene_macro_intervals": {key: interval(draws[key]) for key in ["threshold", "scene_macro_coverage", "scene_macro_accepted_precision", "scene_macro_precision_delta"]},
            "nonpilot_11_scene_pooled_intervals": {key: interval(nonpilot_draws[key]) for key in ["threshold", "pooled_coverage", "pooled_accepted_precision", "pooled_precision_delta"]},
            "nonpilot_11_scene_macro_intervals": {key: interval(nonpilot_draws[key]) for key in ["threshold", "scene_macro_coverage", "scene_macro_accepted_precision", "scene_macro_precision_delta"]},
        },
        "subsets": summaries,
        "paired_policy_comparison_on_exact_nonpilot_query_ids": policy_comparison_summary,
        "caveats": [
            "This is cross-dataset and cross-input-protocol transfer: native RGB-D patch scores are applied to synthetic-view point descriptors.",
            "Rescan retrieval outputs and labels were extensively explored before this threshold-transfer audit; the source threshold itself is not tuned on Rescan.",
            "The comparison with Rescan LOPO thresholds is descriptive and at near, not exactly equal, accept coverage; its threshold-training folds overlap.",
            "Queries are annotation-defined valid-class point samples; accepted physical precision counts target-absent queries as errors.",
            "This does not validate raw RGB-D frame matching, calibrated poses, or a deployable object association system.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    all13 = summaries["all_13_scenes"]["scene_macro"]
    nonpilot = summaries["nonpilot_11_scenes_excluding_b_c"]["scene_macro"]
    ci = summary["bootstrap"]["nonpilot_11_scene_macro_intervals"]
    paired = policy_comparison_summary
    report = f"""# 3RScan-frozen DINO confidence transferred to Rescan

## Protocol

The confidence gate was frozen from the 75th percentile of top-1 DINO cosine
scores on 15,048 query-evaluable observations in 16 official 3RScan source/train
relations. The reproduced threshold is {threshold:.6f}. Rescan scores come from
DINOv2 ViT-Giant block 2/20/40 descriptors pooled from eight synthetic views of
colored point-cloud annotations. Each target-point descriptor is scored by its
maximum cosine to the 5 cm source-point candidate pool. Rescan labels do not
set the threshold or candidate ranking. They define the eligible valid-class
point-query population and score physical-instance top-1 correctness. The
target set includes both source-present and source-absent instances, so
accepted physical precision counts absent-target accepts as errors.

This is a cross-dataset **and cross-input-protocol** threshold transfer. The
source score is from native RGB-D patches; the Rescan score is from synthetic
rendered point descriptors. It does not validate native Rescan RGB-D frames,
camera calibration, or pose estimation.

## Results

| Rescan subset | Scenes / target scans | Scene-macro coverage | Baseline physical top-1 precision | Accepted physical precision | Precision gain |
|---|---:|---:|---:|---:|---:|
| All scenes | 13 / 32 | {all13['all_query_coverage']:.1%} | {all13['baseline_physical_top1']:.1%} | {all13['accepted_physical_precision']:.1%} | {all13['accepted_physical_precision_delta']*100:+.1f} pp |
| Excluding render pilots B/C | 11 / 25 | {nonpilot['all_query_coverage']:.1%} | {nonpilot['baseline_physical_top1']:.1%} | {nonpilot['accepted_physical_precision']:.1%} | {nonpilot['accepted_physical_precision_delta']*100:+.1f} pp |

The source-frozen gate improves the per-scene precision point estimate in
{paired['source_frozen']['sites_with_positive_delta']}/11 pilot-excluded scenes.
For this 11-scene subset, a joint bootstrap resampling 16 source relations to
refit the threshold and 11 Rescan scenes gives macro precision gain
{ci['scene_macro_precision_delta']['median']*100:+.1f} pp [{ci['scene_macro_precision_delta']['q025']*100:+.1f},{ci['scene_macro_precision_delta']['q975']*100:+.1f}] and macro coverage
{nonpilot['all_query_coverage']:.1%} [{ci['scene_macro_coverage']['q025']:.1%},{ci['scene_macro_coverage']['q975']:.1%}]. The wide coverage interval reflects
strong site-to-site score-distribution differences: the fixed threshold's
per-scene coverage ranges from
{per_site.all_query_coverage.min():.1%} to
{per_site.all_query_coverage.max():.1%} across all 13 scenes.

As a descriptive reference, on the exact same 41,955 query IDs from the 11
pilot-excluded scenes, a leave-one-Rescan-scene-out threshold targeting 25%
acceptance gives {paired['rescan_lopo_q25']['scene_macro_coverage']:.1%}
macro coverage and {paired['rescan_lopo_q25']['scene_macro_accepted_precision']:.1%}
macro accepted precision, versus {paired['source_frozen']['scene_macro_coverage']:.1%}
and {paired['source_frozen']['scene_macro_accepted_precision']:.1%} for the
single 3RScan-frozen threshold. Precision gains are positive in 11/11 scenes for
both policies. This is a descriptive comparison without a separate interval;
the leave-one-out folds overlap and its threshold uses other Rescan sites.
On the exact same queries, pooled coverage/precision are
{paired['pooled_source_frozen']['coverage']:.1%}/
{paired['pooled_source_frozen']['accepted_precision']:.1%} for the source-frozen
gate and {paired['pooled_rescan_lopo_q25']['coverage']:.1%}/
{paired['pooled_rescan_lopo_q25']['accepted_precision']:.1%} for the Rescan
LOPO gate.

## Interpretation and limits

The result supports a narrow claim: a source-only similarity threshold can
retain a precision gain after transfer to point descriptors from an independent
rescan dataset, even under a rendering-protocol change. It does **not** show
that the native RGB-D score is calibrated across datasets: the fixed threshold
produces highly variable site coverage, and the point-cloud rendering changes
the feature-score distribution. The Rescan query population and all prior
Rescan retrieval analyses are annotation-defined and exploratory. Treat this
as external stress-test evidence, not a method-superiority or deployment claim.

Reproduce with `scripts/analyze_rescan_source_frozen_threshold.py`; all 13-site
metrics, exact query-paired local-threshold comparison, and joint bootstrap
draws are stored beside this report.
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
