#!/usr/bin/env python3
"""Test whether the source-frozen gate selects identity-correct matches
among semantically correct Rescan top-1 queries.

The threshold is refit from the 16 frozen 3RScan source relations in every
source-cluster bootstrap draw. Rescan scenes are resampled jointly. A second
estimand conditions on target instances known to occur in the source, which
isolates same-category instance ranking from absent-target rejection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_rescan_source_frozen_threshold import load_rescan, load_source  # noqa: E402


DEFAULT_OUT = Path("outputs/reports/20261009_rescan_conditional_identity_selectivity")
SEED = 20261011
DEFAULT_REPLICATES = 10_000


def as_bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    keep = frame.score.to_numpy(dtype=np.float64) >= threshold
    semantic = frame.semantic_top1.to_numpy(dtype=bool)
    physical = frame.physical_top1.to_numpy(dtype=bool)
    present = frame.present.to_numpy(dtype=bool)
    out: dict[str, float | int] = {"queries": int(len(frame)), "accepted": int(keep.sum())}
    for name, subset in (("all", np.ones(len(frame), dtype=bool)), ("present", present)):
        sem_base = subset & semantic
        sem_kept = subset & semantic & keep
        out[f"{name}_semantic_correct"] = int(sem_base.sum())
        out[f"{name}_accepted_semantic_correct"] = int(sem_kept.sum())
        out[f"{name}_semantic_correct_coverage"] = (
            float(sem_kept.sum() / sem_base.sum()) if sem_base.any() else float("nan")
        )
        out[f"{name}_identity_given_semantic_baseline"] = (
            float(physical[sem_base].mean()) if sem_base.any() else float("nan")
        )
        out[f"{name}_identity_given_semantic_accepted"] = (
            float(physical[sem_kept].mean()) if sem_kept.any() else float("nan")
        )
        out[f"{name}_identity_given_semantic_lift"] = (
            out[f"{name}_identity_given_semantic_accepted"]
            - out[f"{name}_identity_given_semantic_baseline"]
        )
    return out


def object_metrics(
    object_code: np.ndarray,
    n_objects: int,
    semantic: np.ndarray,
    physical: np.ndarray,
    present: np.ndarray,
    keep: np.ndarray,
) -> dict[str, float]:
    """Equal-weight target identities; points remain clustered by object."""
    out: dict[str, float] = {}
    for name, subset in (("all", np.ones(len(semantic), dtype=bool)), ("present", present)):
        sem = semantic & subset
        sem_keep = sem & keep
        sem_count = np.bincount(object_code, weights=sem.astype(float), minlength=n_objects)
        phys_count = np.bincount(object_code, weights=(physical & sem).astype(float), minlength=n_objects)
        keep_count = np.bincount(object_code, weights=sem_keep.astype(float), minlength=n_objects)
        keep_phys = np.bincount(object_code, weights=(physical & sem_keep).astype(float), minlength=n_objects)
        valid = sem_count > 0
        eligible = keep_count > 0
        if not valid.any() or not eligible.any():
            for metric in ("baseline", "accepted", "lift", "coverage", "instance_acceptance_share"):
                out[f"{name}_{metric}"] = float("nan")
            continue
        baseline = phys_count[valid] / sem_count[valid]
        base_paired = phys_count[eligible] / sem_count[eligible]
        accepted = keep_phys[eligible] / keep_count[eligible]
        out[f"{name}_baseline"] = float(baseline.mean())
        out[f"{name}_accepted"] = float(accepted.mean())
        out[f"{name}_lift"] = float((accepted - base_paired).mean())
        out[f"{name}_coverage"] = float(np.mean(keep_count[valid] / sem_count[valid]))
        out[f"{name}_instance_acceptance_share"] = float(eligible.sum() / valid.sum())
    return out


def object_bootstrap_metrics(
    object_code: np.ndarray,
    n_objects: int,
    semantic: np.ndarray,
    physical: np.ndarray,
    present: np.ndarray,
    keep: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Resample target identities within a scene for nested cluster intervals."""
    out: dict[str, float] = {}
    for name, subset in (("all", np.ones(len(semantic), dtype=bool)), ("present", present)):
        sem = semantic & subset
        sem_keep = sem & keep
        sem_count = np.bincount(object_code, weights=sem.astype(float), minlength=n_objects)
        phys_count = np.bincount(object_code, weights=(physical & sem).astype(float), minlength=n_objects)
        keep_count = np.bincount(object_code, weights=sem_keep.astype(float), minlength=n_objects)
        keep_phys = np.bincount(object_code, weights=(physical & sem_keep).astype(float), minlength=n_objects)
        valid = np.flatnonzero(sem_count > 0)
        if not len(valid):
            continue
        draw = rng.choice(valid, size=len(valid), replace=True)
        eligible = keep_count[draw] > 0
        if not eligible.any():
            continue
        base = phys_count[draw] / sem_count[draw]
        accepted = keep_phys[draw[eligible]] / keep_count[draw[eligible]]
        paired_base = base[eligible]
        out[f"{name}_baseline"] = float(base.mean())
        out[f"{name}_accepted"] = float(accepted.mean())
        out[f"{name}_lift"] = float((accepted - paired_base).mean())
        out[f"{name}_coverage"] = float(np.mean(keep_count[draw] / sem_count[draw]))
        out[f"{name}_instance_acceptance_share"] = float(eligible.mean())
    return out


def bootstrap(
    source_relations: list[np.ndarray],
    sites: dict[str, pd.DataFrame],
    *,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    names = np.asarray(sorted(sites))
    arrays = {}
    for name, frame in sites.items():
        object_code, object_ids = pd.factorize(frame.target_instance_id, sort=True)
        arrays[name] = (
            frame.score.to_numpy(dtype=np.float64),
            frame.semantic_top1.to_numpy(dtype=bool),
            frame.physical_top1.to_numpy(dtype=bool),
            frame.present.to_numpy(dtype=bool),
            object_code.astype(np.int64),
            len(object_ids),
        )
    rng = np.random.default_rng(seed)
    rows = []
    n_rel, n_site = len(source_relations), len(names)
    for _ in range(reps):
        draw_rel = rng.integers(0, n_rel, size=n_rel)
        threshold = float(np.quantile(np.concatenate([source_relations[i] for i in draw_rel]), .75))
        draw_site = rng.integers(0, n_site, size=n_site)
        per_site: dict[str, list[float]] = {
            key: [] for key in (
                "all_identity_given_semantic_baseline",
                "all_identity_given_semantic_accepted",
                "all_identity_given_semantic_lift",
                "all_semantic_correct_coverage",
                "present_identity_given_semantic_baseline",
                "present_identity_given_semantic_accepted",
                "present_identity_given_semantic_lift",
                "present_semantic_correct_coverage",
                "object_all_lift",
                "object_all_baseline",
                "object_all_accepted",
                "object_all_coverage",
                "object_all_instance_acceptance_share",
                "object_present_lift",
                "object_present_baseline",
                "object_present_accepted",
                "object_present_coverage",
                "object_present_instance_acceptance_share",
            )
        }
        for idx in draw_site:
            score, semantic, physical, present, object_code, n_objects = arrays[names[idx]]
            keep = score >= threshold
            for name, subset in (("all", np.ones(len(score), dtype=bool)), ("present", present)):
                sem_base = subset & semantic
                sem_keep = sem_base & keep
                if sem_base.any() and sem_keep.any():
                    base = float(physical[sem_base].mean())
                    accepted = float(physical[sem_keep].mean())
                    per_site[f"{name}_identity_given_semantic_baseline"].append(base)
                    per_site[f"{name}_identity_given_semantic_accepted"].append(accepted)
                    per_site[f"{name}_identity_given_semantic_lift"].append(accepted - base)
                    per_site[f"{name}_semantic_correct_coverage"].append(float(sem_keep.sum() / sem_base.sum()))
            object_result = object_bootstrap_metrics(
                object_code, n_objects, semantic, physical, present, keep, rng
            )
            for key, value in object_result.items():
                if np.isfinite(value):
                    per_site[f"object_{key}"].append(value)
        rows.append({
            "threshold": threshold,
            **{key: float(np.mean(values)) if values else float("nan") for key, values in per_site.items()},
        })
    return pd.DataFrame(rows)


def interval(values: pd.Series) -> dict[str, float]:
    arr = values.dropna().to_numpy(dtype=np.float64)
    return {"q025": float(np.quantile(arr, .025)), "median": float(np.quantile(arr, .5)), "q975": float(np.quantile(arr, .975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    source_relations, _ = load_source(root)
    threshold = float(np.quantile(np.concatenate(source_relations), .75))
    query_path = root / "outputs/reports/20261006_rescan_dinog_scoreconfidence_corrected/query_metrics.csv"
    rescan = load_rescan(query_path)
    rescan["present"] = as_bool(rescan.target_present_in_full_source)
    subsets = {
        "all_13_scenes": rescan,
        "nonpilot_11_scenes": rescan[~rescan.scene.isin(["scene_b", "scene_c"])].copy(),
    }
    summaries = {}
    site_rows = []
    bootstrap_rows = []
    for subset_name, frame in subsets.items():
        sites = {str(k): v.copy() for k, v in frame.groupby("scene", sort=True)}
        site_summaries = []
        for scene, group in sites.items():
            object_code, object_ids = pd.factorize(group.target_instance_id, sort=True)
            row = {
                "subset": subset_name,
                "scene": scene,
                **metrics(group, threshold),
                **{
                    f"object_{key}": value
                    for key, value in object_metrics(
                        object_code.astype(np.int64),
                        len(object_ids),
                        group.semantic_top1.to_numpy(dtype=bool),
                        group.physical_top1.to_numpy(dtype=bool),
                        group.present.to_numpy(dtype=bool),
                        group.score.to_numpy(dtype=np.float64) >= threshold,
                    ).items()
                },
            }
            site_rows.append(row)
            site_summaries.append(row)
        boot = bootstrap(source_relations, sites, reps=args.bootstrap_replicates, seed=args.seed)
        boot.insert(0, "subset", subset_name)
        bootstrap_rows.append(boot)
        summary_frame = pd.DataFrame(site_summaries)
        keys = [c for c in summary_frame.columns if c not in {"subset", "scene", "queries", "accepted"}]
        summaries[subset_name] = {
            "scenes": len(sites),
            "queries": int(len(frame)),
            "target_present_queries": int(frame.present.sum()),
            "macro": {key: float(summary_frame[key].mean()) for key in keys},
            "joint_bootstrap_95": {
                key: interval(boot[key]) for key in (
                    "all_identity_given_semantic_lift",
                    "all_semantic_correct_coverage",
                    "present_identity_given_semantic_lift",
                    "present_semantic_correct_coverage",
                )
            },
            "object_balanced_macro": {
                key: float(summary_frame[f"object_{key}"].mean())
                for key in (
                    "all_baseline", "all_accepted", "all_lift", "all_coverage",
                    "all_instance_acceptance_share", "present_baseline", "present_accepted",
                    "present_lift", "present_coverage", "present_instance_acceptance_share",
                )
            },
            "object_balanced_joint_bootstrap_95": {
                key: interval(boot[f"object_{key}"])
                for key in (
                    "all_lift", "all_coverage", "all_instance_acceptance_share",
                    "present_lift", "present_coverage", "present_instance_acceptance_share",
                )
            },
            "positive_scenes": {
                "all_identity_given_semantic_lift": int((summary_frame.all_identity_given_semantic_lift > 0).sum()),
                "present_identity_given_semantic_lift": int((summary_frame.present_identity_given_semantic_lift > 0).sum()),
                "object_present_lift": int((summary_frame.object_present_lift > 0).sum()),
            },
        }
    pd.DataFrame(site_rows).to_csv(out / "per_scene_metrics.csv", index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(out / "joint_source_scene_object_bootstrap.csv.gz", index=False, compression="gzip")
    summary = {
        "status": "complete",
        "source_frozen_q75_threshold": threshold,
        "source_relations": len(source_relations),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "bootstrap_units": "refit the threshold from resampled 3RScan source relations, resample Rescan scenes, and additionally resample target object identities within each selected scene for object-balanced intervals",
        "estimand": "physical top-1 correctness conditional on semantic top-1 correctness; present-only subset additionally excludes target instances absent from the source",
        "physical_correct_is_subset_of_semantic_correct": bool((rescan.physical_top1 & ~rescan.semantic_top1).sum() == 0),
        "subsets": summaries,
        "interpretation": "A positive present-only conditional lift indicates enrichment of exact-instance top-1 success among same-category, source-present queries; this is descriptive and post-hoc, not causal or independent confirmation.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Conditional identity selectivity under the source-frozen Rescan gate",
        "",
        f"The q75 score threshold is frozen from {len(source_relations)} 3RScan source relations ({threshold:.6f}). This audit asks whether the gate selects the correct physical instance among queries whose top-1 candidate is already semantically correct. The present-only estimand further restricts to instances known to occur in the source, removing absent-target rejection as an explanation.",
        "",
        "| Subset | Scenes | Point-weighted present-only lift | 95% source/scene interval | Object-balanced lift* | 95% source/scene/object interval | Objects with accepted semantic-correct points |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summaries.items():
        m = result["macro"]
        ci = result["joint_bootstrap_95"]["present_identity_given_semantic_lift"]
        object_ci = result["object_balanced_joint_bootstrap_95"]["present_lift"]
        obj = result["object_balanced_macro"]
        accepted_share = obj["present_instance_acceptance_share"]
        lines.append(
            f"| {name.replace('_', ' ')} | {result['scenes']} | {m['present_identity_given_semantic_lift']*100:+.1f} pp | [{ci['q025']*100:+.1f},{ci['q975']*100:+.1f}] pp | {obj['present_lift']*100:+.1f} pp | [{object_ci['q025']*100:+.1f},{object_ci['q975']*100:+.1f}] pp | {accepted_share:.1%} |"
        )
    lines += [
        "",
        "The conditional decomposition is descriptive: selected queries can differ in object, visibility, or scene difficulty, and the Rescan RGB-D stream is not used. The target-present subset is the more specific instance-ranking diagnostic; the all-query estimate additionally reflects rejection of absent targets. The source threshold was not fitted on Rescan, but Rescan data and related outcomes were explored previously.",
        "",
        f"*Object-balanced estimates first average within each scene/object identity, then macro-average scenes. The object-level paired lift compares accepted and baseline identity accuracy on the same objects that have at least one accepted semantically correct sample; the accompanying object-acceptance share discloses how many objects contribute. Positive object-balanced lift occurs in {summaries['nonpilot_11_scenes']['positive_scenes']['object_present_lift']}/11 non-pilot scenes.",
        "",
        "Per-scene rows, source-relation/scene/object bootstrap draws, and machine-readable results are stored beside this report. Reproduce with `scripts/analyze_rescan_conditional_identity_selectivity.py`.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
