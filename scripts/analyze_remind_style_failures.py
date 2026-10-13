"""Describe where the frozen REMIND-style 3RScan adapter loses query links."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/reports/20261109_remind_style_3rscan_failure_audit"


def relation_bootstrap(values: list[float], *, draws: int = 100_000, seed: int = 20261109) -> list[float]:
    values_array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values_array), size=(draws, len(values_array)))
    samples = values_array[indexes].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(input_dir: Path) -> dict:
    result_files = sorted(input_dir.glob("[0-9]*.json"))
    if len(result_files) != 8:
        raise ValueError(f"expected 8 relation outputs, found {len(result_files)}")
    relation_rows = [json.loads(path.read_text()) for path in result_files]
    rows = [
        {**row, "run_id": relation["run_id"]}
        for relation in relation_rows
        for row in relation["query_mask_rows"]
    ]
    if len(rows) != sum(int(x["query_mask_observations"]) for x in relation_rows):
        raise ValueError("per-mask rows do not cover all frozen query masks")

    shared = [row for row in rows if row["shared_identity"]]
    if not shared:
        raise ValueError("no query identities are shared with their reference")

    areas = np.asarray([int(row["mask_area_pixels"]) for row in shared], dtype=np.float64)
    cuts = [float(x) for x in np.quantile(areas, [0.25, 0.50, 0.75])]
    for row in shared:
        area = float(row["mask_area_pixels"])
        row["area_bin"] = min(int(np.searchsorted(cuts, area, side="left")), 3)
        frames = int(row["source_reference_frame_count"])
        row["reference_exposure_bin"] = "1" if frames == 1 else "2-4" if frames <= 4 else "5+"

    state_summary = {}
    state_counts = Counter(row["tracker_state"] for row in rows)
    total = len(rows)
    for state, count in sorted(state_counts.items()):
        state_rows = [row for row in rows if row["tracker_state"] == state]
        state_summary[state] = {
            "mask_observations": count,
            "fraction_all_query_masks": count / total,
            "source_links_predicted": sum(int(row["predicted_source_link"]) for row in state_rows),
            "correct_source_links": sum(int(row["correct_source_link"]) for row in state_rows),
        }

    def summarize_bins(field: str, bins: list[str | int]) -> dict:
        output = {}
        for bin_value in bins:
            per_relation = {}
            pooled_n = pooled_links = pooled_correct = 0
            unique_ids: set[tuple[str, int]] = set()
            for relation in relation_rows:
                rel = relation["run_id"]
                rel_rows = [row for row in shared if row["run_id"] == rel and row[field] == bin_value]
                n = len(rel_rows)
                links = sum(int(row["predicted_source_link"]) for row in rel_rows)
                correct = sum(int(row["correct_source_link"]) for row in rel_rows)
                if n:
                    per_relation[rel] = correct / n
                pooled_n += n
                pooled_links += links
                pooled_correct += correct
                unique_ids.update((rel, int(row["gt_id"])) for row in rel_rows)
            macro = float(np.mean(list(per_relation.values()))) if per_relation else 0.0
            output[str(bin_value)] = {
                "query_mask_observations": pooled_n,
                "unique_relation_identity_pairs": len(unique_ids),
                "predicted_source_links": pooled_links,
                "correct_source_links": pooled_correct,
                "pooled_correct_coverage": pooled_correct / pooled_n if pooled_n else 0.0,
                "pooled_link_precision": pooled_correct / pooled_links if pooled_links else 0.0,
                "macro_relation_coverage": macro,
                "macro_relation_bootstrap_95": relation_bootstrap(list(per_relation.values())) if len(per_relation) == 8 else None,
                "per_relation_coverage": per_relation,
            }
        return output

    size_summary = summarize_bins("area_bin", [0, 1, 2, 3])
    exposure_summary = summarize_bins("reference_exposure_bin", ["1", "2-4", "5+"])

    candidate_summary = {}
    candidate_rank1_states = {}
    candidate_rank_matches = []
    for score_key in ("score_sim", "score_final", "score_sim_majority", "score_final_majority"):
        majority_map = score_key.endswith("_majority")
        correct_key = "correct_source_link_majority" if majority_map else "correct_source_link"
        predicted_key = "predicted_source_link_majority" if majority_map else "predicted_source_link"
        if score_key in ("score_sim", "score_final"):
            candidate_rank_matches.append([
                row.get(f"correct_source_candidate_rank_{score_key}") for row in shared
            ])
        candidate_summary[score_key] = {}
        for label, keep in [
            ("missing", lambda row: row.get(f"correct_source_candidate_rank_{score_key}") is None),
            ("rank_1", lambda row: row.get(f"correct_source_candidate_rank_{score_key}") == 1),
            ("rank_2_5", lambda row: row.get(f"correct_source_candidate_rank_{score_key}") is not None and 2 <= row[f"correct_source_candidate_rank_{score_key}"] <= 5),
            ("rank_6_plus", lambda row: row.get(f"correct_source_candidate_rank_{score_key}") is not None and row[f"correct_source_candidate_rank_{score_key}"] > 5),
        ]:
            selected = [row for row in shared if keep(row)]
            per_relation = {}
            for relation in relation_rows:
                rel_rows = [row for row in selected if row["run_id"] == relation["run_id"]]
                if rel_rows:
                    per_relation[relation["run_id"]] = float(np.mean([int(row[correct_key]) for row in rel_rows]))
            candidate_summary[score_key][label] = {
                "query_mask_observations": len(selected),
                "correct_source_links": sum(int(row[correct_key]) for row in selected),
                "predicted_source_links": sum(int(row[predicted_key]) for row in selected),
                "pooled_correct_coverage": float(np.mean([int(row[correct_key]) for row in selected])) if selected else 0.0,
                "macro_relation_coverage": float(np.mean(list(per_relation.values()))) if per_relation else 0.0,
                "macro_relation_coverage_by_relation": per_relation,
            }
        rank1_rows = [row for row in shared if row.get(f"correct_source_candidate_rank_{score_key}") == 1]
        candidate_rank1_states[score_key] = {
            "query_masks": len(rank1_rows),
            "tracker_states": dict(Counter(row["tracker_state"] for row in rank1_rows)),
            "correct_source_links": sum(int(row[correct_key]) for row in rank1_rows),
            "predicted_source_links": sum(int(row[predicted_key]) for row in rank1_rows),
        }

    by_identity: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in shared:
        by_identity[(row["run_id"], int(row["gt_id"]))].append(row)
    within_identity = []
    within_relation: dict[str, list[float]] = defaultdict(list)
    for (run_id, gt_id), object_rows in by_identity.items():
        object_areas = [int(row["mask_area_pixels"]) for row in object_rows]
        if len(object_rows) < 2 or len(set(object_areas)) < 2:
            continue
        median_area = float(np.median(object_areas))
        smaller = [row for row in object_rows if int(row["mask_area_pixels"]) < median_area]
        larger = [row for row in object_rows if int(row["mask_area_pixels"]) >= median_area]
        if not smaller or not larger:
            continue
        smaller_coverage = float(np.mean([int(row["correct_source_link"]) for row in smaller]))
        larger_coverage = float(np.mean([int(row["correct_source_link"]) for row in larger]))
        difference = larger_coverage - smaller_coverage
        within_relation[run_id].append(difference)
        within_identity.append({
            "run_id": run_id,
            "gt_id": gt_id,
            "query_mask_observations": len(object_rows),
            "smaller_mask_observations": len(smaller),
            "larger_mask_observations": len(larger),
            "smaller_coverage": smaller_coverage,
            "larger_coverage": larger_coverage,
            "coverage_difference": difference,
        })
    within_relation_means = {key: float(np.mean(values)) for key, values in within_relation.items()}
    within_relation_values = list(within_relation_means.values())

    relation_summary = []
    candidate_relation_metrics = {}
    for relation in relation_rows:
        rel_rows = [row for row in rows if row["run_id"] == relation["run_id"]]
        counts = Counter(row["tracker_state"] for row in rel_rows)
        rel_shared = [row for row in rel_rows if row["shared_identity"]]
        majority_rank_key = "correct_source_candidate_rank_score_sim_majority"
        rank1 = [row for row in rel_shared if row.get(majority_rank_key) == 1]
        top5 = [row for row in rel_shared if row.get(majority_rank_key) is not None and row[majority_rank_key] <= 5]
        candidate_relation_metrics[relation["run_id"]] = {
            "shared_query_masks": len(rel_shared),
            "rank1_count": len(rank1),
            "rank1_recall": len(rank1) / len(rel_shared) if rel_shared else 0.0,
            "top5_count": len(top5),
            "top5_recall": len(top5) / len(rel_shared) if rel_shared else 0.0,
            "correct_source_link_coverage": (
                sum(int(row["correct_source_link_majority"]) for row in rank1) / len(rel_shared)
                if rel_shared else 0.0
            ),
        }
        relation_summary.append({
            "run_id": relation["run_id"],
            "query_masks": len(rel_rows),
            "mean_area_pixels": float(np.mean([row["mask_area_pixels"] for row in rel_rows])) if rel_rows else 0.0,
            "correct_links": int(relation["correct_cross_scan_links"]),
            "source_links": int(relation["query_links_to_source_tracks"]),
            "correct_coverage": float(relation["cross_scan_link_coverage_all_query_masks"]),
            "tracker_states": dict(counts),
        })

    q1 = size_summary["0"]["per_relation_coverage"]
    q4 = size_summary["3"]["per_relation_coverage"]
    size_diffs = [q4[key] - q1[key] for key in sorted(q1.keys() & q4.keys())]
    ex1 = exposure_summary["1"]["per_relation_coverage"]
    ex5 = exposure_summary["5+"]["per_relation_coverage"]
    exposure_diffs = [ex5[key] - ex1[key] for key in sorted(ex1.keys() & ex5.keys())]
    majority_precision_by_relation = {
        row["run_id"]: float(row["cross_scan_link_precision_majority_mapping"])
        for row in relation_rows
    }
    majority_coverage_by_relation = {
        row["run_id"]: float(row["cross_scan_link_coverage_all_query_masks_majority_mapping"])
        for row in relation_rows
    }
    rank1_recall_by_relation = {key: item["rank1_recall"] for key, item in candidate_relation_metrics.items()}
    top5_recall_by_relation = {key: item["top5_recall"] for key, item in candidate_relation_metrics.items()}

    return {
        "cohort": "eight frozen 3RScan relations; 929 query masks; anonymous frozen masks; identity targets used post-inference",
        "stratification": "Shared query identities only for area and reference-exposure strata. Area quartiles are pooled over shared-identity query-mask observations; repeated observations within objects are descriptive, not independent samples.",
        "area_quartile_cut_pixels": cuts,
        "tracker_state_counts": state_summary,
        "area_quartiles": size_summary,
        "reference_exposure": exposure_summary,
        "correct_source_candidate_rank": candidate_summary,
        "correct_source_candidate_rank1_decisions": candidate_rank1_states,
        "candidate_rank_recall_by_relation": candidate_relation_metrics,
        "score_sim_final_rank_agreement": {
            "query_mask_observations": len(shared),
            "identical_rank_count": sum(a == b for a, b in zip(*candidate_rank_matches)),
            "different_rank_count": sum(a != b for a, b in zip(*candidate_rank_matches)),
        },
        "majority_mapping_sensitivity": {
            "macro_link_precision": float(np.mean([row["cross_scan_link_precision_majority_mapping"] for row in relation_rows])),
            "macro_all_mask_coverage": float(np.mean([row["cross_scan_link_coverage_all_query_masks_majority_mapping"] for row in relation_rows])),
            "pooled_link_precision": (
                sum(row["correct_cross_scan_links_majority_mapping"] for row in relation_rows)
                / max(1, sum(row["query_links_to_source_tracks_majority_mapping"] for row in relation_rows))
            ),
            "pooled_all_mask_coverage": (
                sum(row["correct_cross_scan_links_majority_mapping"] for row in relation_rows)
                / max(1, sum(row["query_mask_observations"] for row in relation_rows))
            ),
            "source_tracker_tracks": sum(len(row["source_track_majority_identity_map"]) for row in relation_rows),
            "source_gt_identity_count": sum(len(set(row["source_track_majority_identity_map"].values())) for row in relation_rows),
            "relations_with_fragmented_source_identities": sum(
                len(set(row["source_track_majority_identity_map"].values()))
                < len(row["source_track_majority_identity_map"])
                for row in relation_rows
            ),
            "macro_precision_bootstrap_95": relation_bootstrap(list(majority_precision_by_relation.values())),
            "macro_coverage_bootstrap_95": relation_bootstrap(list(majority_coverage_by_relation.values())),
            "precision_by_relation": majority_precision_by_relation,
            "coverage_by_relation": majority_coverage_by_relation,
            "rank1_recall_macro": float(np.mean(list(rank1_recall_by_relation.values()))),
            "rank1_recall_bootstrap_95": relation_bootstrap(list(rank1_recall_by_relation.values())),
            "top5_recall_macro": float(np.mean(list(top5_recall_by_relation.values()))),
            "top5_recall_bootstrap_95": relation_bootstrap(list(top5_recall_by_relation.values())),
        },
        "macro_relation_difference_q4_minus_q1_coverage": {
            "mean": float(np.mean(size_diffs)) if size_diffs else None,
            "relations_paired": len(size_diffs),
            "bootstrap_95": relation_bootstrap(size_diffs) if len(size_diffs) >= 2 else None,
            "per_relation": {key: q4[key] - q1[key] for key in q1.keys() & q4.keys()},
        },
        "macro_relation_difference_5plus_minus_1frame_coverage": {
            "mean": float(np.mean(exposure_diffs)) if exposure_diffs else None,
            "relations_paired": len(exposure_diffs),
            "bootstrap_95": relation_bootstrap(exposure_diffs) if len(exposure_diffs) >= 2 else None,
            "per_relation": {key: ex5[key] - ex1[key] for key in ex1.keys() & ex5.keys()},
        },
        "within_identity_area_contrast": {
            "definition": "For each shared identity with at least two query masks and nonconstant area, compare correct-link coverage above versus below its own median query-mask area; then average identity differences within each relation and relation means across groups.",
            "identities": len(within_identity),
            "identities_larger_better": sum(row["coverage_difference"] > 0 for row in within_identity),
            "identities_smaller_better": sum(row["coverage_difference"] < 0 for row in within_identity),
            "identities_tied": sum(row["coverage_difference"] == 0 for row in within_identity),
            "macro_relation_difference": float(np.mean(within_relation_values)) if within_relation_values else None,
            "relations": len(within_relation_values),
            "relation_bootstrap_95": relation_bootstrap(within_relation_values) if len(within_relation_values) >= 2 else None,
            "per_relation_difference": within_relation_means,
            "per_identity": within_identity,
        },
        "relations": relation_summary,
        "limitations": [
            "Eight related groups only; site independence is unknown.",
            "Mask-size and reference-exposure associations are post-hoc descriptive results, not causal effects.",
            "Repeated query observations from one identity are correlated; percentile bins pool observations and cluster intervals resample relations only.",
        ],
    }


def to_markdown(summary: dict) -> str:
    lines = [
        "# REMIND-style failure stratification",
        "",
        "## Tracker decisions",
        "",
        "| Query-mask state | Count | Share | Source links | Correct links |",
        "|---|---:|---:|---:|---:|",
    ]
    for state, item in summary["tracker_state_counts"].items():
        lines.append(
            f"| {state} | {item['mask_observations']} | {100*item['fraction_all_query_masks']:.1f}% | "
            f"{item['source_links_predicted']} | {item['correct_source_links']} |"
        )
    lines += [
        "",
        "## Query mask area",
        "",
        "Area quartiles are set from shared-identity query masks (20 frames per stream; 224x224 RGB). Coverage counts a correct link over all masks in that bin.",
        "",
        "| Shared query-mask area quartile | Pixel area | Masks | Unique relation/identity pairs | Correct / predicted source links | Pooled coverage | Macro relation coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cuts = summary["area_quartile_cut_pixels"]
    area_ranges = [f"≤{cuts[0]:.0f}", f"{cuts[0]:.0f}–{cuts[1]:.0f}", f"{cuts[1]:.0f}–{cuts[2]:.0f}", f">{cuts[2]:.0f}"]
    for idx, (key, item) in enumerate(summary["area_quartiles"].items()):
        lines.append(
            f"| Q{int(key)+1} | {area_ranges[int(key)]} | {item['query_mask_observations']} | "
            f"{item['unique_relation_identity_pairs']} | {item['correct_source_links']} / {item['predicted_source_links']} | "
            f"{100*item['pooled_correct_coverage']:.1f}% | {100*item['macro_relation_coverage']:.1f}% |"
        )
    lines += [
        "",
        "## Reference exposure",
        "",
        "| Query identity appearances in reference frames | Masks | Unique relation/identity pairs | Correct / predicted source links | Pooled coverage | Macro relation coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["reference_exposure"].items():
        lines.append(
            f"| {key} | {item['query_mask_observations']} | {item['unique_relation_identity_pairs']} | "
            f"{item['correct_source_links']} / {item['predicted_source_links']} | "
            f"{100*item['pooled_correct_coverage']:.1f}% | {100*item['macro_relation_coverage']:.1f}% |"
        )
    lines += [
        "",
        "## Correct source candidate rank",
        "",
        "The correct reference track is mapped from reference predictions to GT identity after inference. Candidate rank therefore measures retrieval availability/ranking, not a matching decision given to the tracker.",
        "",
        "| Score order / map | Correct source track rank | Masks | Source links (predicted/correct) | Correct coverage | Macro relation coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for score_key, values in summary["correct_source_candidate_rank"].items():
        for rank_bin, item in values.items():
            mapping_label = "majority-map" if score_key.endswith("_majority") else "one-to-one"
            ordering_label = score_key.replace("_majority", "")
            lines.append(
                f"| {ordering_label} ({mapping_label}) | {rank_bin} | {item['query_mask_observations']} | {item['predicted_source_links']}/{item['correct_source_links']} | "
                f"{100*item['pooled_correct_coverage']:.1f}% | {100*item['macro_relation_coverage']:.1f}% |"
            )
    lines += [
        "",
        "| Relation | Majority-map precision | Majority-map coverage | Candidate rank 1 | Candidate top 5 |",
        "|---|---:|---:|---:|---:|",
    ]
    relation_mapping = summary["majority_mapping_sensitivity"]
    for run_id in sorted(relation_mapping["coverage_by_relation"]):
        candidate = summary["candidate_rank_recall_by_relation"][run_id]
        lines.append(
            f"| {run_id} | {100*relation_mapping['precision_by_relation'][run_id]:.1f}% | "
            f"{100*relation_mapping['coverage_by_relation'][run_id]:.1f}% | "
            f"{100*candidate['rank1_recall']:.1f}% | {100*candidate['top5_recall']:.1f}% |"
        )
    strict_bins = summary["correct_source_candidate_rank"]["score_sim"]
    rank_bins = summary["correct_source_candidate_rank"]["score_sim_majority"]
    rank1 = rank_bins["rank_1"]
    rank2_5 = rank_bins["rank_2_5"]
    missing = rank_bins["missing"]
    rank1_states = summary["correct_source_candidate_rank1_decisions"]["score_sim_majority"]
    rank_agreement = summary["score_sim_final_rank_agreement"]
    shared_count = sum(item["query_mask_observations"] for item in strict_bins.values())
    majority = summary["majority_mapping_sensitivity"]
    strict_rank1 = strict_bins["rank_1"]
    strict_rank2_5 = strict_bins["rank_2_5"]
    strict_missing = strict_bins["missing"]
    nonrank_false_links = sum(
        rank_bins[key]["predicted_source_links"] - rank_bins[key]["correct_source_links"]
        for key in ("missing", "rank_2_5", "rank_6_plus")
    )
    lines += [
        "",
        f"Under the strict one-to-one reference-track map, the correct source candidate is missing for {strict_missing['query_mask_observations']} shared masks, ranks first for {strict_rank1['query_mask_observations']} ({100*strict_rank1['query_mask_observations']/shared_count:.1f}%), and ranks in the top five for {strict_rank1['query_mask_observations']+strict_rank2_5['query_mask_observations']} ({100*(strict_rank1['query_mask_observations']+strict_rank2_5['query_mask_observations'])/shared_count:.1f}%).",
        f"A sensitivity map assigning each source track to its most frequent GT identity allows source fragments to share an identity. This changes macro link precision/coverage to {100*majority['macro_link_precision']:.1f}%/{100*majority['macro_all_mask_coverage']:.1f}% (relation bootstrap intervals {100*majority['macro_precision_bootstrap_95'][0]:.1f}–{100*majority['macro_precision_bootstrap_95'][1]:.1f}% and {100*majority['macro_coverage_bootstrap_95'][0]:.1f}–{100*majority['macro_coverage_bootstrap_95'][1]:.1f}%); repeated source tracks occur in {majority['relations_with_fragmented_source_identities']}/8 relations. Under this map, the target is missing for {missing['query_mask_observations']} masks, ranks first for {rank1['query_mask_observations']} ({100*rank1['query_mask_observations']/shared_count:.1f}%), and top five for {rank1['query_mask_observations']+rank2_5['query_mask_observations']} ({100*(rank1['query_mask_observations']+rank2_5['query_mask_observations'])/shared_count:.1f}%).",
        f"In the many-to-one rank-1 group, the tracker predicts {rank1['predicted_source_links']} source links, {rank1['correct_source_links']} correct ({100*rank1['correct_source_links']/max(1,rank1['predicted_source_links']):.1f}% precision). There are {rank1_states['query_masks']-rank1_states['predicted_source_links']} rank-1 cases without a source link; state counts are {rank1_states['tracker_states']}. Another {nonrank_false_links} predicted links are outside rank 1 and all are incorrect. The score_sim and score_final rank arrays agree on {rank_agreement['identical_rank_count']}/{rank_agreement['query_mask_observations']} shared masks; this is not a contextual-association ablation.",
        f"Macro rank-one and top-five candidate recall across relations are {100*majority['rank1_recall_macro']:.1f}% (interval {100*majority['rank1_recall_bootstrap_95'][0]:.1f}–{100*majority['rank1_recall_bootstrap_95'][1]:.1f}%) and {100*majority['top5_recall_macro']:.1f}% ({100*majority['top5_recall_bootstrap_95'][0]:.1f}–{100*majority['top5_recall_bootstrap_95'][1]:.1f}%).",
        "",
    ]
    size_diff = summary["macro_relation_difference_q4_minus_q1_coverage"]
    exposure_diff = summary["macro_relation_difference_5plus_minus_1frame_coverage"]
    within = summary["within_identity_area_contrast"]
    lines += [
        "",
        f"Across {size_diff['relations_paired']} paired relations, Q4 minus Q1 macro coverage is {100*size_diff['mean']:.1f} pp "
        f"(relation bootstrap 95% interval {100*size_diff['bootstrap_95'][0]:.1f} to {100*size_diff['bootstrap_95'][1]:.1f} pp).",
        f"Across {exposure_diff['relations_paired']} paired relations, five-or-more reference frames minus one frame is {100*exposure_diff['mean']:.1f} pp "
        f"({100*exposure_diff['bootstrap_95'][0]:.1f} to {100*exposure_diff['bootstrap_95'][1]:.1f} pp).",
        f"Within identity, larger-than-median masks improve coverage for {within['identities_larger_better']} identities and reduce it for {within['identities_smaller_better']} ({within['identities_tied']} tie); the macro-relation contrast is {100*within['macro_relation_difference']:.1f} pp "
        f"(95% interval {100*within['relation_bootstrap_95'][0]:.1f} to {100*within['relation_bootstrap_95'][1]:.1f} pp across {within['relations']} relations).",
        "",
        "These are post-hoc associations. The quartiles pool repeated masks, and eight relation groups do not establish independent-site generalization or causality. Full per-mask rows and machine-readable summaries accompany this report.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    summary = summarize(args.input_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failure_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "failure_analysis.md").write_text(to_markdown(summary))
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
