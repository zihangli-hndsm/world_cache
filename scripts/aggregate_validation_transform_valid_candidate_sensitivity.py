#!/usr/bin/env python3
"""Join and validate saved validation candidate-oracle distance sensitivities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PREFIX = "outputs/reports/20261008_validation_transform_valid_candidate_availability"
KEYS = ["run_id", "query_index"]
KS = (1, 5, 10, 20, 50)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float, message: str) -> None:
    require(bool(np.isclose(actual, expected, atol=1e-12, rtol=0)), message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input-prefix", default=PREFIX,
                        help="three audit directories sharing this prefix")
    parser.add_argument("--output-dir", type=Path, default=Path(PREFIX + "_sensitivity"))
    parser.add_argument("--paper-table", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    previous = None
    for threshold, suffix in ((0.05, "_5cm"), (0.10, ""), (0.20, "_20cm")):
        directory = root / (args.input_prefix + suffix)
        s = json.loads((directory / "summary.json").read_text())
        q = pd.read_csv(directory / "query_diagnostics.csv.gz").sort_values(KEYS).reset_index(drop=True)
        require(s["status"] == "complete" and s["candidate_archives"] == 110, "incomplete relation audit")
        close(s["threshold_m"], threshold, "threshold mismatch")
        require(not q.duplicated(KEYS).any(), "duplicate query keys")
        require(len(q) == s["moved_rigid_queries"] == 14088 and
                q.run_id.nunique() == s["relations"] == 82 and
                q.environment_id.nunique() == s["environments_with_moved_rigid_queries"] == 42,
                "unexpected moved-rigid scope")
        varying = [c for c in q if c.startswith(("scene_valid_", "object_valid_"))] + [
            "scene_surface_support", "object_surface_support"]
        for column in varying + [f"physical_hit_k{k}" for k in KS] + ["global_correct", "motion_correct"]:
            require(pd.api.types.is_bool_dtype(q[column]), f"invalid boolean field: {column}")
        if previous is not None:
            invariant = [c for c in q if c not in varying]
            pd.testing.assert_frame_equal(q[invariant], previous[invariant])
            for column in varying:
                require(not (previous[column] & ~q[column]).any(), f"nonmonotone tolerance: {column}")
        previous = q
        for metric in ("physical_hit", "scene_valid", "object_valid"):
            for k in KS:
                column = f"{metric}_k{k}"
                require(int(q[column].sum()) == s["pooled_counts"][column], f"count mismatch: {column}")
                close(q[column].mean(), s["pooled_rates"][column], f"rate mismatch: {column}")
            for left, right in zip(KS[:-1], KS[1:]):
                require(not (q[f"{metric}_k{left}"] & ~q[f"{metric}_k{right}"]).any(), "non-nested K sets")
        for k in KS:
            require(not ((q[f"scene_valid_k{k}"] | q[f"object_valid_k{k}"]) &
                         ~q[f"physical_hit_k{k}"]).any(), "valid candidate lacks linked identity")
        for column, count in (("physical_hit_k20", 7626), ("physical_hit_k50", 7929),
                              ("global_correct", 2876), ("motion_correct", 3808)):
            require(int(q[column].sum()) == count, f"saved validation count differs: {column}")
        require(not ((q.global_correct | q.motion_correct) & ~q.physical_hit_k20).any(),
                "correct identity prediction lacks linked candidate")
        macro = {r["metric"]: r["macro"] for r in s["environment_macro"]}
        for column, estimate in macro.items():
            close(q.groupby("environment_id")[column].mean().mean(), estimate, "macro differs")
        d = s["environment_bootstrap_object_vs_scene"]["object_minus_scene_valid_k20"]
        close(macro["object_valid_k20"] - macro["scene_valid_k20"], d["estimate"], "availability delta differs")
        valid = q[q.object_valid_k20]
        env = valid.groupby("environment_id")[["global_correct", "motion_correct"]].mean()
        c = s["conditional_recovery_given_object_valid_k20"]
        require(len(valid) == c["object_valid_k20_queries"] and
                len(env) == c["environment_count_with_object_valid_k20"], "conditional scope differs")
        for arm in ("global", "motion"):
            require(int(valid[f"{arm}_correct"].sum()) == c[f"{arm}_correct_count"], "conditional count differs")
            close(valid[f"{arm}_correct"].mean(), c[f"pooled_{arm}_correct_rate"], "conditional pooled rate differs")
            close(env[f"{arm}_correct"].mean(), c[f"environment_macro_{arm}_correct_rate"], "conditional macro differs")
        delta = (env.motion_correct - env.global_correct).mean()
        close(delta, c["environment_macro_motion_minus_global_delta"], "conditional delta differs")
        new = q.object_valid_k50 & ~q.object_valid_k20
        rows.append({
            "threshold_cm": round(threshold * 100), "queries": len(q), "environments": 42,
            "object_valid_k20_queries": len(valid), "object_valid_k20_pooled": float(q.object_valid_k20.mean()),
            "scene_valid_k20_macro": macro["scene_valid_k20"], "object_valid_k20_macro": macro["object_valid_k20"],
            "availability_delta": d["estimate"], "availability_ci_low": d["ci95"][0], "availability_ci_high": d["ci95"][1],
            "conditional_environments": len(env), "conditional_delta": float(delta),
            "conditional_ci_low": c["environment_macro_motion_minus_global_delta_ci95"][0],
            "conditional_ci_high": c["environment_macro_motion_minus_global_delta_ci95"][1],
            "new_valid_k50": int(new.sum()), "new_valid_k50_with_physical_hit_k20": int((new & q.physical_hit_k20).sum()),
            "new_physical_hits_k50": int((q.physical_hit_k50 & ~q.physical_hit_k20).sum()),
            "global_identity_correct_without_valid_k20": int(q.loc[~q.object_valid_k20, "global_correct"].sum()),
            "motion_identity_correct_without_valid_k20": int(q.loc[~q.object_valid_k20, "motion_correct"].sum()),
        })
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "sensitivity.csv", index=False)
    (output / "summary.json").write_text(json.dumps({"status": "complete", "query_alignment": "exact",
        "checks": "scope, query alignment, K/tolerance nesting, pooled/macro/conditional estimates", "rows": rows}, indent=2) + "\n")
    tex = [r"\begin{tabular}{rrrrrr}", r"\toprule",
           r"$\tau$ (cm) & Scene (\%) & Object (\%) & $\Delta$ (pp, 95\% CI) & Cond. $\Delta$ (pp, 95\% CI) & $n_{\rm cond}$ \\", r"\midrule"]
    report = "# Validation candidate-oracle distance sensitivity\n\nAll tolerances use exactly the same 14,088 moved-rigid queries across 82 relations and 42 environments from 110 official validation caches. K/tolerance nesting and pooled, macro, and conditional estimates reproduce from saved query rows. Intervals are retained from the original 20,000-draw environment bootstraps.\n\n| Tolerance (cm) | Object-valid pooled | Scene-valid macro | Object-valid macro | Object minus scene (pp, 95% CI) | Conditional motion minus baseline (pp, 95% CI) | Conditional environments |\n|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in rows:
        availability = f"{100*r['availability_delta']:+.1f} [{100*r['availability_ci_low']:+.1f},{100*r['availability_ci_high']:+.1f}]"
        conditional = f"{100*r['conditional_delta']:+.1f} [{100*r['conditional_ci_low']:+.1f},{100*r['conditional_ci_high']:+.1f}]"
        report += f"| {r['threshold_cm']} | {r['object_valid_k20_pooled']:.1%} | {r['scene_valid_k20_macro']:.1%} | {r['object_valid_k20_macro']:.1%} | {availability} | {conditional} | {r['conditional_environments']} |\n"
        tex.append(f"{r['threshold_cm']} & {100*r['scene_valid_k20_macro']:.1f} & {100*r['object_valid_k20_macro']:.1f} & {availability} & {conditional} & {r['conditional_environments']} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    report += "\nAt 10 cm, K50 adds 677 object-valid queries, 653 of which already had a physical-object hit in K20; K50 adds 303 physical-object hits.\n\nConditional correctness scores physical-instance identity, not geometric point correspondence or estimated object pose. At 10 cm, 1,117 motion and 577 baseline predictions are identity-correct without an object-valid top-20 candidate. Geometric availability therefore is not an upper bound on identity correctness. The conditional populations vary with tolerance (26/29/30 environments); these are not common-population trend estimates. Annotation-defined oracle subsets on an explored validation split do not establish label-free method gains or independent generalization.\n\nReproduce: `conda run -n worldcache python scripts/aggregate_validation_transform_valid_candidate_sensitivity.py`. Underlying audits: `scripts/audit_validation_transform_valid_candidates.py --distance-threshold {0.05,0.10,0.20}` with separate output directories.\n"
    report += f"\nInput prefix: `{args.input_prefix}`. Reproduce these inputs with `--input-prefix {args.input_prefix} --output-dir {args.output_dir}`.\n"
    (output / "report.md").write_text(report)
    (output / "table.tex").write_text("\n".join(tex) + "\n")
    if args.paper_table is not None:
        destination = args.paper_table if args.paper_table.is_absolute() else root / args.paper_table
        destination.write_text("\n".join(tex) + "\n")
    print(report)


if __name__ == "__main__":
    main()
