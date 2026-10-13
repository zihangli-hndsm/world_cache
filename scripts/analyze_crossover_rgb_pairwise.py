#!/usr/bin/env python3
"""Score label-blind CrossOver RGB mutual assignments on frozen 3RScan masks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _arm_counts(
    pairwise_root: Path,
    crossover_root: Path,
    run_id: str,
    *,
    budget: bool = False,
) -> dict:
    manifest_path = pairwise_root / "uniform20_inputs" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dino_path = pairwise_root / "dino_segment_predictions" / run_id / "inference_summary.json"
    dino = json.loads(dino_path.read_text())
    cross_path = crossover_root / run_id / "inference_summary.json"
    cross = json.loads(cross_path.read_text())
    if dino.get("identity_targets_read") or cross.get("identity_targets_read"):
        raise ValueError(f"inference consumed identity targets for {run_id}")
    refs = manifest["records"]["reference"]
    queries = manifest["records"]["rescan"]
    env_id = Path(refs[0]["input"]).parent.name
    query_count = dino_accepts = dino_correct = cross_accepts = cross_correct = 0
    dino_selected: list[tuple[float, int, bool]] = []
    cross_selected: list[tuple[float, int, bool]] = []
    overlap = {"both_correct": 0, "dino_only_correct": 0, "crossover_only_correct": 0, "neither_correct": 0}
    row_base = 0
    if len(dino["records"]) != len(queries) or len(cross["records"]) != len(queries):
        raise ValueError(f"frame-pair count mismatch for {run_id}")

    for dino_record, cross_record, ref_record, query_record in zip(
        dino["records"], cross["records"], refs, queries
    ):
        if int(dino_record["query_frame"]) != int(query_record["frame_id"]):
            raise ValueError(f"DINO frame order mismatch for {run_id}")
        if int(cross_record["query_frame"]) != int(query_record["frame_id"]):
            raise ValueError(f"CrossOver frame order mismatch for {run_id}")
        ref_ids = np.load(ROOT / ref_record["identity_targets"]).astype(np.int64)
        query_ids = np.load(ROOT / query_record["identity_targets"]).astype(np.int64)
        with np.load(ROOT / dino_record["predictions"]) as d:
            ds = d["scores"].astype(np.float32)
            da = d["mutual_assigned"].astype(bool)
            dl = d["query_to_reference"].astype(np.int64)
        with np.load(ROOT / cross_record["predictions"]) as c:
            cs = c["scores"].astype(np.float32)
            ca = c["mutual_assigned"].astype(bool)
            cl = c["query_to_reference"].astype(np.int64)
        if ds.shape != cs.shape or len(query_ids) != len(da) or len(query_ids) != len(ca):
            raise ValueError(f"prediction shape mismatch for {run_id}/{query_record['frame_id']}")
        if ds.shape[1] != len(ref_ids):
            raise ValueError(f"reference mask count mismatch for {run_id}/{ref_record['frame_id']}")

        d_ok = np.zeros(len(query_ids), dtype=bool)
        c_ok = np.zeros(len(query_ids), dtype=bool)
        for q_idx, qid in enumerate(query_ids.tolist()):
            query_count += 1
            if da[q_idx]:
                d_ok[q_idx] = qid > 0 and int(ref_ids[dl[q_idx]]) == qid
                dino_accepts += 1
                dino_correct += int(d_ok[q_idx])
                dino_selected.append((float(ds[q_idx, dl[q_idx]]), row_base + q_idx, bool(d_ok[q_idx])))
            if ca[q_idx]:
                c_ok[q_idx] = qid > 0 and int(ref_ids[cl[q_idx]]) == qid
                cross_accepts += 1
                cross_correct += int(c_ok[q_idx])
                cross_selected.append((float(cs[q_idx, cl[q_idx]]), row_base + q_idx, bool(c_ok[q_idx])))
            if d_ok[q_idx] and c_ok[q_idx]:
                overlap["both_correct"] += 1
            elif d_ok[q_idx]:
                overlap["dino_only_correct"] += 1
            elif c_ok[q_idx]:
                overlap["crossover_only_correct"] += 1
            else:
                overlap["neither_correct"] += 1
        row_base += len(query_ids)

    if budget:
        accepted_budget = min(dino_accepts, cross_accepts)
        dino_correct = sum(x[2] for x in sorted(dino_selected, key=lambda x: (-x[0], x[1]))[:accepted_budget])
        cross_correct = sum(x[2] for x in sorted(cross_selected, key=lambda x: (-x[0], x[1]))[:accepted_budget])
        dino_accepts = cross_accepts = accepted_budget
    return {
        "run_id": run_id,
        "environment_id": env_id,
        "frame_pairs": len(queries),
        "query_masks": query_count,
        "dino": {"accepted": dino_accepts, "correct": dino_correct},
        "crossover": {"accepted": cross_accepts, "correct": cross_correct},
        "native_counts": {
            "dino": {"accepted": dino_accepts if not budget else None, "correct": None},
            "crossover": {"accepted": cross_accepts if not budget else None, "correct": None},
        },
        "overlap": overlap,
    }


def rates(row: dict, arm: str) -> dict:
    counts = row[arm]
    return {
        "accepted": counts["accepted"],
        "correct": counts["correct"],
        "precision": counts["correct"] / counts["accepted"] if counts["accepted"] else None,
        "coverage": counts["correct"] / row["query_masks"] if row["query_masks"] else None,
    }


def summarize(rows: list[dict], draws: int, seed: int) -> dict:
    groups: dict[str, dict] = {}
    for row in rows:
        group = groups.setdefault(row["environment_id"], {
            "frame_pairs": 0,
            "query_masks": 0,
            "dino": {"accepted": 0, "correct": 0},
            "crossover": {"accepted": 0, "correct": 0},
        })
        group["frame_pairs"] += row["frame_pairs"]
        group["query_masks"] += row["query_masks"]
        for arm in ("dino", "crossover"):
            for key in ("accepted", "correct"):
                group[arm][key] += row[arm][key]
    group_rows = [{"environment_id": k, **v} for k, v in sorted(groups.items())]
    group_rates = [{"environment_id": r["environment_id"], **{a: rates(r, a) for a in ("dino", "crossover")}} for r in group_rows]

    rng = np.random.default_rng(seed)
    bootstrap_index = rng.integers(0, len(group_rows), size=(draws, len(group_rows)))
    effects = {}
    for metric in ("precision", "coverage"):
        valid = np.asarray([
            r["dino"][metric] is not None and r["crossover"][metric] is not None
            for r in group_rates
        ])
        vals_d = np.asarray([r["dino"][metric] if r["dino"][metric] is not None else np.nan for r in group_rates])
        vals_c = np.asarray([r["crossover"][metric] if r["crossover"][metric] is not None else np.nan for r in group_rates])
        if metric == "coverage":
            differences = vals_d - vals_c
            boot = differences[bootstrap_index].mean(axis=1)
        else:
            valid_indices = np.flatnonzero(valid)
            valid_boot = rng.integers(0, len(valid_indices), size=(draws, len(valid_indices)))
            differences = vals_d[valid] - vals_c[valid]
            boot = differences[valid_boot].mean(axis=1) if len(valid_indices) else np.array([])
        effects[metric] = {
            "dino_minus_crossover": float(np.nanmean(differences)) if len(differences) else None,
            "paired_groups": int(valid.sum()) if metric == "precision" else len(group_rates),
            "percentile_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])] if len(boot) else None,
        }

    leave_one_group_out = {}
    single_pair_groups = [
        r["environment_id"] for r in group_rows if r["frame_pairs"] == 1
    ]
    for metric in ("precision", "coverage"):
        differences = np.asarray([
            (r["dino"][metric] - r["crossover"][metric])
            if r["dino"][metric] is not None and r["crossover"][metric] is not None
            else np.nan
            for r in group_rates
        ])
        keep = np.isfinite(differences)
        vals = []
        for index in range(len(group_rates)):
            retained = keep.copy()
            retained[index] = False
            vals.append(float(np.mean(differences[retained])))
        one_pair_effects = {}
        for group_id in single_pair_groups:
            index = next(i for i, row in enumerate(group_rows) if row["environment_id"] == group_id)
            retained = keep.copy()
            retained[index] = False
            one_pair_effects[group_id] = float(np.mean(differences[retained]))
        leave_one_group_out[metric] = {
            "min": min(vals),
            "max": max(vals),
            "omitting_single_pair_groups": one_pair_effects,
        }

    pooled = {}
    total_queries = sum(r["query_masks"] for r in rows)
    for arm in ("dino", "crossover"):
        accepts = sum(r[arm]["accepted"] for r in rows)
        correct = sum(r[arm]["correct"] for r in rows)
        pooled[arm] = {"accepted": accepts, "correct": correct,
                       "precision": correct / accepts if accepts else None,
                       "coverage": correct / total_queries if total_queries else None}
    overlap = {k: sum(r["overlap"][k] for r in rows) for k in rows[0]["overlap"]}
    return {
        "status": "complete",
        "scope": "same-mask, per-frame RGB object-crop comparison on official 3RScan validation; CrossOver projection head is evaluated with a single-view three-crop adaptation, not the authors' multi-view retrieval protocol",
        "query_masks": total_queries,
        "relations": rows,
        "reference_groups": group_rates,
        "macro_bootstrap": effects,
        "leave_one_group_out": leave_one_group_out,
        "pooled": pooled,
        "correctness_overlap": overlap,
        "bootstrap": {"draws": draws, "seed": seed, "unit": "released reference group"},
        "limits": [
            "The checkpoint is advertised as 3RScan-trained; public training instructions use official train/validation splits, and all 47 sampled reference groups are official validation. The checkpoint file itself does not record its sample-level training provenance.",
            "The adaptation supplies one anonymous frame-level mask with three nested RGB crops; the released object-retrieval pipeline aggregates multiple views per object.",
            "All relations use already explored validation data and reference groups are not certified physical sites.",
            "Masks are shared with the DINO comparator; segmentation is therefore privileged and is not evaluated.",
            "The exact-budget diagnostic uses the smaller of the two native mutual-accept counts per relation and ranks only mutual links by label-free cosine.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-root", type=Path, default=ROOT / "outputs/reports/20261008_segmast3r_validation")
    parser.add_argument("--crossover-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=9041)
    args = parser.parse_args()
    manifest = json.loads((args.crossover_root / "inference_manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("CrossOver inference manifest is not complete")
    run_ids = [r["run_id"] for r in manifest["relations"]]
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("CrossOver inference manifest has no unique relation set")
    native_rows = [_arm_counts(args.pairwise_root, args.crossover_root, rid) for rid in run_ids]
    budget_rows = [_arm_counts(args.pairwise_root, args.crossover_root, rid, budget=True) for rid in run_ids]
    native = summarize(native_rows, args.bootstrap_draws, args.seed)
    budget = summarize(budget_rows, args.bootstrap_draws, args.seed)
    first_inference = json.loads(
        (args.crossover_root / run_ids[0] / "inference_summary.json").read_text()
    )
    revision_proc = subprocess.run(
        ["git", "-C", str(ROOT / "third_party/crossover"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    result = {
        "status": "complete",
        "metadata": {
            "checkpoint_sha256": first_inference["checkpoint_sha256"],
            "crossover_code_revision": revision_proc.stdout.strip() if revision_proc.returncode == 0 else None,
            "relation_selection": "first sorted relation per distinct reference scan UUID; selected before CrossOver inference",
        },
        "native": native,
        "equal_budget": budget,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = [
        "# CrossOver RGB pairwise correspondence evaluation",
        "",
        f"Relations: {len(run_ids)}; masks: {native['query_masks']}; reference groups: {len(native['reference_groups'])}.",
        f"Checkpoint SHA-256: `{result['metadata']['checkpoint_sha256']}`.",
        f"CrossOver code revision: `{result['metadata']['crossover_code_revision']}`.",
        "",
        "| Method | Accepted | Correct | Pooled precision | Pooled correct coverage | Macro precision | Macro coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, result_scope in (("Native mutual", native), ("Exact per-relation budget", budget)):
        for arm in ("dino", "crossover"):
            pooled = result_scope["pooled"][arm]
            rates_by_group = [g[arm] for g in result_scope["reference_groups"]]
            precisions = [r["precision"] for r in rates_by_group if r["precision"] is not None]
            coverages = [r["coverage"] for r in rates_by_group]
            report.append(
                f"| {scope}: {arm} | {pooled['accepted']:,} | {pooled['correct']:,} | "
                f"{pooled['precision']*100:.1f}% | {pooled['coverage']*100:.1f}% | "
                f"{np.mean(precisions)*100:.1f}% | {np.mean(coverages)*100:.1f}% |"
            )
        effect = result_scope["macro_bootstrap"]
        report.append("")
        p_ci = effect["precision"]["percentile_95"]
        c_ci = effect["coverage"]["percentile_95"]
        report.append(
            f"Paired group bootstrap, DINO minus CrossOver: precision "
            f"{effect['precision']['dino_minus_crossover']*100:+.1f} pp "
            f"[{p_ci[0]*100:+.1f},{p_ci[1]*100:+.1f}]; coverage "
            f"{effect['coverage']['dino_minus_crossover']*100:+.1f} pp "
            f"[{c_ci[0]*100:+.1f},{c_ci[1]*100:+.1f}]."
        )
        sensitivity = result_scope["leave_one_group_out"]
        single_pair_ids = sorted(sensitivity["precision"]["omitting_single_pair_groups"])
        report.append(
            f"Leave-one-group-out effect ranges (macro DINO minus CrossOver): "
            f"precision [{sensitivity['precision']['min']*100:+.1f},"
            f"{sensitivity['precision']['max']*100:+.1f}] pp; coverage "
            f"[{sensitivity['coverage']['min']*100:+.1f},"
            f"{sensitivity['coverage']['max']*100:+.1f}] pp. "
            f"Omitting the single-frame-pair group(s) {', '.join(single_pair_ids) or 'none'} "
            f"gives precision {', '.join(f'{v*100:+.1f}' for v in sensitivity['precision']['omitting_single_pair_groups'].values()) or 'n/a'} pp "
            f"and coverage {', '.join(f'{v*100:+.1f}' for v in sensitivity['coverage']['omitting_single_pair_groups'].values()) or 'n/a'} pp."
        )
    report.extend(["", "## Scope", "", *[f"- {x}" for x in native["limits"]], ""])
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(json.dumps({"status": "complete", "relations": len(run_ids),
                      "query_masks": native["query_masks"], "pooled": native["pooled"],
                      "equal_budget_pooled": budget["pooled"],
                      "effects": budget["macro_bootstrap"]}, indent=2))


if __name__ == "__main__":
    main()
