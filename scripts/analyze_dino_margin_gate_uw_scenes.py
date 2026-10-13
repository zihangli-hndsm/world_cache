#!/usr/bin/env python3
"""Transfer the frozen 3RScan top-3 object-margin gate to UW RGB-D Scenes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/reports/20261009_rgbd_scenes_temporal_retrieval"
DEFAULT_GATE = ROOT / "outputs/reports/20261009_dino_object_margin_gate_transfer/summary.json"
DEFAULT_OUT = ROOT / "outputs/reports/20261009_rgbd_scenes_margin_gate_transfer"


def summarize(rows: list[dict], threshold: float, draws: int, seed: int) -> dict:
    if not rows:
        raise ValueError("no eligible query rows")
    required = {"dino_top3mean_margin", "dino_top3mean_correct", "object_id", "scene"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("query rows lack top-3 margin fields; rerun the frozen evaluator")
    accepted = np.asarray([r["dino_top3mean_margin"] >= threshold for r in rows])
    correct = np.asarray([bool(r["dino_top3mean_correct"]) for r in rows])
    unfiltered_correct = np.asarray([bool(r["dino_top3mean_correct"]) for r in rows])
    total_a, total_c = int(accepted.sum()), int((accepted & correct).sum())
    object_groups: dict[str, list[int]] = defaultdict(list)
    scene_groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        object_groups[row["object_id"]].append(i)
        scene_groups[row["scene"]].append(i)

    def cluster_boot(groups: dict[str, list[int]], cluster_name: str) -> dict:
        keys = sorted(groups)
        a = np.asarray([int(accepted[groups[k]].sum()) for k in keys], dtype=float)
        c = np.asarray([int((accepted[groups[k]] & correct[groups[k]]).sum()) for k in keys], dtype=float)
        q = np.asarray([len(groups[k]) for k in keys], dtype=float)
        cov = c / q
        prec = np.divide(c, a, out=np.full_like(c, np.nan), where=a > 0)
        rng = np.random.default_rng(seed)
        ix = rng.integers(0, len(keys), size=(draws, len(keys)))
        pooled_p = c[ix].sum(1) / np.maximum(a[ix].sum(1), 1)
        pooled_cov = c[ix].sum(1) / np.maximum(q[ix].sum(1), 1)
        macro_cov = cov[ix].mean(1)
        sampled_prec = prec[ix]
        support_count = np.isfinite(sampled_prec).sum(1)
        macro_p = np.divide(
            np.nansum(sampled_prec, axis=1), support_count,
            out=np.full(draws, np.nan, dtype=float), where=support_count > 0,
        )
        return {
            "clusters": len(keys), "unit": cluster_name, "draws": draws, "seed": seed,
            "pooled_precision_95": [float(x) for x in np.quantile(pooled_p, [.025, .975])],
            "correct_coverage_95": [float(x) for x in np.quantile(pooled_cov, [.025, .975])],
            "cluster_macro_correct_coverage_95": [float(x) for x in np.quantile(macro_cov, [.025, .975])],
            "supported_cluster_macro_precision_95": [float(x) for x in np.nanquantile(macro_p, [.025, .975])],
        }

    objects = sorted(object_groups)
    per_object = []
    for oid in objects:
        ix = object_groups[oid]
        nacc = int(accepted[ix].sum())
        ncorr = int((accepted[ix] & correct[ix]).sum())
        per_object.append({"object_id": oid, "queries": len(ix), "accepted": nacc,
                           "correct": ncorr, "precision": ncorr / nacc if nacc else None,
                           "correct_coverage": ncorr / len(ix)})
    per_scene = []
    for scene in sorted(scene_groups):
        ix = scene_groups[scene]
        nacc = int(accepted[ix].sum())
        ncorr = int((accepted[ix] & correct[ix]).sum())
        per_scene.append({"scene": scene, "queries": len(ix), "accepted": nacc,
                          "correct": ncorr, "precision": ncorr / nacc if nacc else None,
                          "correct_coverage": ncorr / len(ix),
                          "unfiltered_top3mean_accuracy": float(unfiltered_correct[ix].mean())})
    supported = [r for r in per_object if r["accepted"]]
    return {
        "queries": len(rows), "objects": len(objects), "scenes": len(scene_groups),
        "threshold": threshold, "accepted": total_a, "correct": total_c,
        "incorrect_accepted": total_a - total_c,
        "acceptance_rate": total_a / len(rows),
        "pooled_precision": total_c / total_a if total_a else None,
        "correct_coverage": total_c / len(rows),
        "object_macro_correct_coverage": float(np.mean([r["correct_coverage"] for r in per_object])),
        "objects_with_accepts": len(supported),
        "supported_object_zero_error_fraction_one_sided_95_lower": 0.05 ** (1.0 / len(supported)) if supported else None,
        "supported_object_macro_precision": float(np.mean([r["precision"] for r in supported])) if supported else None,
        "unfiltered_top3mean_object_macro_accuracy": float(np.mean([
            np.mean([rows[i]["dino_top3mean_correct"] for i in object_groups[oid]]) for oid in objects
        ])),
        "object_cluster_bootstrap": cluster_boot(object_groups, "global physical category_instance ID"),
        "scene_cluster_bootstrap": cluster_boot(scene_groups, "scene/video"),
        "per_object": per_object, "per_scene": per_scene,
    }


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--gate-summary", type=Path, default=DEFAULT_GATE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bootstrap-draws", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=20261009)
    a = p.parse_args()
    gate = json.loads(a.gate_summary.read_text())
    if gate.get("selection", {}).get("views_per_object") != 3:
        raise ValueError("expected the source-frozen top-3-view-mean gate")
    threshold = float(gate["selection"]["threshold"])
    probes = {
        "within_scene_temporal": read_rows(a.input_dir / "query_rows.jsonl"),
        "cross_scene_balanced": read_rows(a.input_dir / "cross_scene_query_rows.jsonl"),
        "cross_scene_all_refs": read_rows(a.input_dir / "cross_scene_all_refs_query_rows.jsonl"),
    }
    metrics = {name: summarize(rows, threshold, a.bootstrap_draws, a.seed + i)
               for i, (name, rows) in enumerate(probes.items())}
    result = {
        "status": "complete", "threshold": threshold,
        "source_fit": {"groups": gate["selection"]["source_groups"],
                       "macro_precision": gate["selection"]["source_fit"]["group_macro_precision"],
                       "macro_correct_coverage": gate["selection"]["source_fit"]["group_macro_correct_coverage"],
                       "source_leave_one_group_out": gate["source_leave_one_group_out"]},
        "target": "UW RGB-D Scenes Dataset; same family as UW RGB-D Object; privileged official 2D boxes; RGB-only retrieval",
        "protocols": {
            "within_scene_temporal": "first temporal third reference, last third query, middle third gap; 2-12 candidates; candidate score is mean of top three reference-view cosines",
            "cross_scene_balanced": "each held-out scene queries global object IDs in the other seven scenes; 12 balanced references per candidate; closed set; absent IDs excluded",
            "cross_scene_all_refs": "same leave-one-scene-out query rows using all references per candidate; unequal gallery support sensitivity",
        },
        "probes": metrics,
        "limits": [
            "The frozen threshold was selected on eight explored 3RScan source groups; the target scenes are post-hoc and are from the same UW family as the turntable pilot.",
            "The 2D boxes are privileged annotations; depth, camera pose, and predicted detections are not used.",
            "Cross-scene retrieval is closed set and excludes queries whose global object ID is absent from the gallery.",
            "The target has only eight videos and 17-21 eligible physical IDs, so uncertainty is limited at the object/site level.",
        ],
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Frozen 3RScan object-margin gate on UW RGB-D Scenes", "",
        f"The top-3-view-mean score-margin cutoff `{threshold:.8f}` was selected on eight 3RScan source groups and transferred unchanged. No UW target labels were used to select it.", "",
        "| Probe | Accepted | Precision | Correct coverage | Objects with accepts | Unfiltered top-3 accuracy | Object-cluster coverage interval | Scene-cluster coverage interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("within_scene_temporal", "Within-scene temporal"),
                        ("cross_scene_balanced", "Cross-scene, 12 refs/ID"),
                        ("cross_scene_all_refs", "Cross-scene, all refs")):
        m = metrics[name]
        ob = m["object_cluster_bootstrap"]["correct_coverage_95"]
        sc = m["scene_cluster_bootstrap"]["correct_coverage_95"]
        lines.append(f"| {label} | {m['accepted']:,}/{m['queries']:,} ({m['acceptance_rate']:.1%}) | {m['pooled_precision']:.1%} | {m['correct_coverage']:.1%} | {m['objects_with_accepts']}/{m['objects']} | {m['unfiltered_top3mean_object_macro_accuracy']:.1%} | [{ob[0]:.1%},{ob[1]:.1%}] | [{sc[0]:.1%},{sc[1]:.1%}] |")
    lines.extend(["", "No incorrect accepts occur in these probes, so query precision bootstrap intervals are degenerate and should not be read as guarantees. The supported-object one-sided lower bounds for the fraction of objects with no accepted error are: " + "; ".join(f"{name}: {m['supported_object_zero_error_fraction_one_sided_95_lower']:.1%} over {m['objects_with_accepts']} supported objects" for name, m in metrics.items()) + ". These exact binomial bounds assume independent supported objects.", "", "These are source-frozen selective-retrieval diagnostics on a small companion dataset, not independent dataset replication or room-rescan validation. The gallery is closed set, queries use official 2D boxes, the dataset family is shared with the turntable evaluation, and DINOv2 pretraining exposure is unknown. Coverage intervals resample physical-object IDs or scenes; they do not include source-threshold fitting uncertainty.", ""])
    (a.output_dir / "report.md").write_text("\n".join(lines))
    print((a.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
