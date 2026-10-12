"""Generate LaTeX tables for the current 17-pair supplementary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ORDER = [
    "0cac", "20c993", "4aca", "5630", "6bde", "751a", "95be", "c670",
    "chairs", "pair02", "pair03", "pair04", "pair05", "pair06", "pair07",
    "pair08", "target",
]


def short_pair_name(value: str) -> str:
    if "target" in value:
        return "target"
    for name in ORDER:
        if name != "target" and (value.endswith(name) or f"_{name}" in value):
            return name
    return value


def consensus_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clean_aggregate = root / "outputs/runs/20260922_3rscan_consensus_vitg_clean_v2/summary_all.csv"
    if clean_aggregate.exists():
        for raw in pd.read_csv(clean_aggregate).to_dict("records"):
            rows.append({
                "pair": short_pair_name(str(raw["pair"])),
                "queries": int(raw["queries"]),
                "inliers": int(raw["inliers"]),
                "coverage": 100 * float(raw["inlier_coverage"]),
                "semantic": 100 * float(raw["inlier_semantic_accuracy"]),
                "rotation": float(raw["rotation_error_deg"]),
                "translation": float(raw["translation_error_m"]),
                "pose": float(raw["rotation_error_deg"]) < 5 and float(raw["translation_error_m"]) < 0.25,
            })
    else:
        for path in sorted((root / "outputs/runs").glob("*_consensus_*_vitg_diag/summary.csv")):
            match = re.search(r"_consensus_(.+)_vitg_diag$", path.parent.name)
            if match is None:
                continue
            row = pd.read_csv(path).iloc[0]
            rows.append({
                "pair": match.group(1),
                "queries": int(row.queries),
                "inliers": int(row.inliers),
                "coverage": 100 * float(row.inlier_coverage),
                "semantic": 100 * float(row.inlier_semantic_accuracy),
                "rotation": float(row.rotation_error_deg),
                "translation": float(row.translation_error_m),
                "pose": float(row.rotation_error_deg) < 5 and float(row.translation_error_m) < 0.25,
            })
    frame = pd.DataFrame(rows)
    corrected = pd.read_csv(root / "outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv")
    if len(corrected) != 17 or not corrected.selected_and_inlier_arrays_reproduced.all():
        raise ValueError("complete reproduced source pose corrections required")
    corrected["pair"] = corrected.run_id.map(short_pair_name)
    if corrected.pair.duplicated().any():
        raise ValueError("source pose relation mapping is ambiguous")
    corrected = corrected.set_index("pair")
    for index, row in frame.iterrows():
        pose_row = corrected.loc[row["pair"]]
        frame.loc[index, "rotation"] = pose_row.rotation_error_degrees
        frame.loc[index, "translation"] = pose_row.translation_error_m
        frame.loc[index, "pose"] = pose_row.strict_pose_success
    frame["order"] = frame.pair.map({name: i for i, name in enumerate(ORDER)})
    return frame.sort_values("order")


def generate_pair_table(frame: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Pair & $Q$ & Inl. & Cov. & Sem. acc. & Rot. & Trans. & Pose \\",
        r" & & & (\%) & (\%) & (deg.) & (m) & $<5^\circ/0.25$m \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        pose = r"\checkmark" if row.pose else r"--"
        lines.append(
            f"\\texttt{{{row.pair}}} & {row.queries:d} & {row.inliers:d} & "
            f"{row.coverage:.1f} & {row.semantic:.1f} & {row.rotation:.1f} & "
            f"{row.translation:.2f} & {pose} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_transfer_table(root: Path) -> str:
    # row.instance_mIoU is a legacy column name for the PLY globalId label.
    frame = pd.read_csv(root / "outputs/runs/20260922_semantic_label_transfer_vitg_clean_calibrated_v2/summary.csv")
    labels = {
        ("source", "all_correspondences"): "Source / all",
        ("source", "consensus_inlier"): "Source / consensus",
        ("source", "source_calibrated_residual"): "Source / source-fit",
        ("target", "all_correspondences"): "Target / all",
        ("target", "consensus_inlier"): "Target / consensus",
        ("target", "source_calibrated_residual"): "Target / source-fit",
    }
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Dataset / selector & Queries & Cov. & Sem. mIoU & Global-label mIoU \\",
             r"\midrule"]
    for row in frame.itertuples(index=False):
        label = labels[(row.dataset, row.selector)]
        lines.append(
            f"{label} & {row.queries:d} & {100 * row.coverage:.1f}\\% & "
            f"{100 * row.semantic_mIoU:.1f}\\% & {100 * row.instance_mIoU:.1f}\\% \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def fixed_abstention_miou(frame: pd.DataFrame, mask: pd.Series, label_column: str,
                          prediction_column: str) -> float:
    """Macro IoU on a fixed query-label universe, with rejection as void."""
    labels = sorted(int(value) for value in frame[label_column].unique() if int(value) != 0)
    ground_truth = frame[label_column].to_numpy()
    prediction = frame[prediction_column].to_numpy()
    prediction = prediction.copy()
    prediction[~mask.to_numpy(bool)] = -1
    scores = []
    for label in labels:
        actual = ground_truth == label
        guessed = prediction == label
        union = np.logical_or(actual, guessed).sum()
        scores.append(float(np.logical_and(actual, guessed).sum() / union) if union else 0.0)
    return float(sum(scores) / len(scores)) if scores else float("nan")


def generate_abstention_miou_table(root: Path) -> str:
    """Generate full-query mIoU where rejected queries count as false negatives."""
    paths = sorted((root / "outputs/runs").glob(
        "20260922_3rscan_consensus_*_vitg_diag_clean/query_diagnostics.csv"
    ))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame[frame.query_evaluable.astype(bool)].copy()
        frame["split"] = "target" if "target" in path.parent.name else "source"
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    threshold = 0.040931
    lines = [
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Split & Selector & Semantic mIoU & Global-label mIoU \\",
        r" & & (\%) & (\%) \\", r"\midrule",
    ]
    for split, label in (("source", "Source, 16 pairs"), ("target", "Target, held out")):
        frame = all_rows[all_rows.split == split].reset_index(drop=True)
        selectors = [
            ("all", pd.Series(True, index=frame.index)),
            ("consensus", frame.consensus_inlier.astype(bool)),
            ("source-fit gate", frame.spatial_residual_m <= threshold),
        ]
        for selector, mask in selectors:
            semantic = fixed_abstention_miou(
                frame, mask, "query_semantic_id", "selected_reference_semantic_id"
            )
            instance = fixed_abstention_miou(
                frame, mask, "query_instance_id", "selected_reference_instance_id"
            )
            lines.append(f"{label} & {selector} & {100 * semantic:.1f} & {100 * instance:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_calibration_table(root: Path) -> str:
    frame = pd.read_csv(root / "outputs/runs/20260923_selective_calibration_vitg_clean_v2/leave_one_pair_out.csv")
    frame = frame[(frame.precision_target == 0.9) & (frame.score == "low_spatial_residual")]
    selected = frame[frame.held_pair.isin(["target_source_only_fit", "4aca", "95be", "pair06"])]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Fit / held-out pair & Threshold (m) & Cov. & Sem. acc. & Global-label acc. \\",
             r"\midrule"]
    for row in selected.itertuples(index=False):
        label = r"all source $\rightarrow$ target" if row.held_pair == "target_source_only_fit" else f"LOO {row.held_pair}"
        coverage = row.target_coverage if row.held_pair == "target_source_only_fit" else row.held_coverage
        semantic = row.target_semantic_precision if row.held_pair == "target_source_only_fit" else row.held_semantic_precision
        instance = row.target_instance_precision if row.held_pair == "target_source_only_fit" else row.held_instance_precision
        lines.append(
            f"{label} & {(-row.threshold):.5f} & {100 * coverage:.1f}\\% & "
            f"{100 * semantic:.1f}\\% & {100 * instance:.1f}\\% \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_risk_controlled_table(root: Path) -> str:
    audit_path = root / "outputs/runs/20260922_risk_control_selection_audit_vitg_clean/summary.csv"
    if audit_path.exists():
        frame = pd.read_csv(audit_path)
        lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
                 r"Operating point & Split & Threshold & Cov. & Sem. acc. & LCB \\",
                 r" & & (m) & (\%) & (\%) & (\%) \\", r"\midrule"]
        labels = {
            "source_fit_descriptive": "source-fit (raw)",
            "fixed_2cm_audit": "fixed 2 cm",
        }
        for row in frame.itertuples(index=False):
            lower = "--" if pd.isna(row.semantic_lower_bound) else f"{100 * row.semantic_lower_bound:.1f}"
            lines.append(
                f"{labels[row.operating_point]} & {row.dataset} & {row.threshold:.5f} & "
                f"{100 * row.coverage:.1f} & {100 * row.semantic_precision:.1f} & {lower} \\\\"
            )
    else:
        frame = pd.read_csv(root / "outputs/runs/20260922_risk_controlled_calibration_vitg_clean/summary.csv")
        lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
                 r"Split & Threshold & Cov. & Sem. acc. & LCB & Sem. mIoU \\",
                 r" & (m) & (\%) & (\%) & (\%) & (\%) \\", r"\midrule"]
        for row in frame.itertuples(index=False):
            lines.append(
                f"{row.dataset} & {row.threshold:.5f} & {100 * row.coverage:.1f} & "
                f"{100 * row.semantic_precision:.1f} & {100 * row.semantic_lower_bound:.1f} & "
                f"{100 * row.semantic_mIoU:.1f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_fcgf_table(root: Path) -> str:
    control_root = root / "outputs/reports/20261008_source_geometry_controls_corrected_units"
    manifest = pd.read_csv(control_root / "run_manifest.csv")
    if len(manifest) != 17 or not manifest.status.eq("complete").all():
        raise ValueError("all 17 corrected source controls required")
    frame = pd.concat([pd.read_csv(control_root / pair / "fcgf/summary.csv").assign(pair=pair)
                       for pair in manifest.pair], ignore_index=True)
    frame = frame[frame.top_k == 20].copy()
    frame["order"] = frame.pair.map({name: i for i, name in enumerate(ORDER)})
    frame = frame.sort_values("order")
    lines = [
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Pair & Cov. & Sem. acc. & Rot. & Trans. & Pose \\ ",
        r" & (\%) & (\%) & (deg.) & (m) & $<5^\circ/0.25$m \\ ", r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        pose = r"\checkmark" if row.pose_success_lt5deg_lt025m else r"--"
        label_match = re.search(
            r"(0cac|20c993|4aca|5630|6bde|751a|95be|c670|chairs|pair0[2-8])|target",
            short_pair_name(str(row.pair)),
        )
        label = label_match.group(0) if label_match else row.pair
        lines.append(
            f"\\texttt{{{label}}} & {100 * row.inlier_coverage:.1f} & "
            f"{100 * row.inlier_semantic_accuracy:.1f} & {row.rotation_error_deg:.1f} & "
            f"{row.translation_error_m:.2f} & {pose} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_scale_table(root: Path) -> str:
    vitl = pd.read_csv(root / "outputs/runs/20260922_3rscan_consensus_vitl_clean/summary_all.csv")
    vitg = pd.read_csv(root / "outputs/runs/20260922_3rscan_consensus_vitg_clean/summary_all.csv")
    corrected_l = pd.read_csv(root / "outputs/reports/20261008_vitl_source_corrected_units/per_relation.csv")
    corrected_g = pd.read_csv(root / "outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv")
    for frame, corrected, key in [(vitl, corrected_l, "pair"), (vitg, corrected_g, "run_id")]:
        for index, row in frame.iterrows():
            match = corrected[corrected[key].eq(short_pair_name(str(row.pair)))]
            if len(match) != 1:
                raise ValueError(f"missing corrected scale pose: {row.pair}")
            frame.loc[index, "translation_error_m"] = match.iloc[0].translation_error_m
    original_pairs = set(vitl.pair)
    rows = [
        ("ViT-L/14, original 14", vitl),
        ("ViT-G/14, original 14", vitg[vitg.pair.isin(original_pairs)]),
        ("ViT-L/14, target", vitl[vitl.pair.astype(str).str.contains("target")]),
        ("ViT-G/14, target", vitg[vitg.pair.astype(str).str.contains("target")]),
    ]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Setting & Cov. & Sem. acc. & Rot. & Pose \\ ",
             r" & (\%) & (\%) & (deg.) & $<5^\circ/0.25$m \\ ", r"\midrule"]
    for setting, frame in rows:
        coverage = frame.inliers.sum() / frame.queries.sum()
        semantic = (frame.inliers * frame.inlier_semantic_accuracy).sum() / frame.inliers.sum()
        pose = int(((frame.rotation_error_deg < 5) & (frame.translation_error_m < 0.25)).sum())
        rotation = f"{float(frame.rotation_error_deg.iloc[0]):.2f}" if len(frame) == 1 else "--"
        lines.append(
            f"{setting} & {100 * coverage:.1f} & {100 * semantic:.1f} & "
            f"{rotation} & {pose}/{len(frame)} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_k_sweep_table(root: Path) -> str:
    frame = pd.read_csv(root / "outputs/runs/20260922_k_sweep_vitg_clean/summary.csv")
    corrected = pd.read_csv(root / "outputs/reports/20261008_source_k_sweep_corrected_units/corrected_pose_counts.csv")
    for index, row in frame.iterrows():
        new = corrected[(corrected.split == row["split"]) & (corrected.top_k == row["top_k"])]
        if len(new) != 1:
            raise ValueError("unique corrected K-sweep pose count required")
        frame.loc[index, "strict_pose_successes"] = new.iloc[0].strict_pose_success
    source = frame[frame.split == "source"].set_index("top_k")
    target = frame[frame.split == "target"].set_index("top_k")
    lines = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"$K$ & Src. cov. & Src. sem. acc. & Src. pose & Tgt. cov. & Tgt. sem. acc. & Tgt. pose \\",
        r" & (\%) & (\%) & $<5^\circ/0.25$m & (\%) & (\%) & $<5^\circ/0.25$m \\", r"\midrule",
    ]
    for top_k in sorted(source.index):
        src = source.loc[top_k]
        tgt = target.loc[top_k]
        lines.append(
            f"{int(top_k)} & {100 * src.weighted_coverage:.1f} & "
            f"{100 * src.weighted_semantic_precision:.1f} & "
            f"{int(src.strict_pose_successes)}/{int(src.pairs)} & "
            f"{100 * tgt.weighted_coverage:.1f} & "
            f"{100 * tgt.weighted_semantic_precision:.1f} & "
            f"{int(tgt.strict_pose_successes)}/{int(tgt.pairs)} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_candidate_set_audit_table(root: Path) -> str:
    frame = pd.read_csv(root / "outputs/runs/20260922_candidate_set_audit_vitg_clean/per_pair.csv")
    source = frame[frame.split == "source"]
    target = frame[frame.split == "target"].iloc[0]
    rows = [
        (
            "Source weighted",
            source.candidate_semantic_hits.sum() / source.queries.sum(),
            source.semantic_correct_inliers.sum() / source.queries.sum(),
            source.semantic_correct_inliers.sum() / source.candidate_semantic_hits.sum(),
            int(source.semantic_miss_inliers.sum()),
        ),
        (
            "Target held out",
            target.candidate_semantic_recall,
            target.semantic_coverage,
            target.semantic_recovery_given_candidate,
            int(target.semantic_miss_inliers),
        ),
    ]
    for pair in ("4aca", "95be"):
        row = frame[frame.pair == pair].iloc[0]
        rows.append(
            (
                pair,
                row.candidate_semantic_recall,
                row.semantic_coverage,
                row.semantic_recovery_given_candidate,
                int(row.semantic_miss_inliers),
            )
        )
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Split / pair & Cand. sem. recall & Correct-inlier cov. & Recovery $|$ cand. & Miss inliers \\ ",
        r" & (\%) & (\%) & (\%) & \\ ", r"\midrule",
    ]
    for label, recall, coverage, recovery, misses in rows:
        lines.append(
            f"{label} & {100 * recall:.1f} & {100 * coverage:.1f} & "
            f"{100 * recovery:.1f} & {misses:d} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_seed_sweep_table(root: Path) -> str:
    frame = pd.read_csv(root / "outputs/runs/20260922_seed_sweep_vitg_clean/summary.csv")
    corrected = pd.read_csv(root / "outputs/reports/20261008_source_seed_sweep_corrected_units/corrected_pose_counts.csv")
    for index, row in frame.iterrows():
        match = corrected[(corrected.split == row["split"]) & (corrected.ransac_seed == row.ransac_seed)]
        if len(match) != 1:
            raise ValueError("missing corrected seed pose")
        frame.loc[index, "strict_pose_successes"] = match.iloc[0].strict_pose_success
    lines = [
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Split & Seed & Cov. & Sem. acc. & Global-label acc. & Pose \\ ",
        r" & & (\%) & (\%) & (\%) & $<5^\circ/0.25$m \\ ", r"\midrule",
    ]
    labels = {"source": "Source, 16 pairs", "target": "Target, held out"}
    for row in frame.itertuples(index=False):
        lines.append(
            f"{labels[row.split]} & {int(row.ransac_seed)} & {100 * row.weighted_coverage:.1f} & "
            f"{100 * row.weighted_semantic_precision:.1f} & "
            f"{100 * row.weighted_instance_precision:.1f} & "
            f"{int(row.strict_pose_successes)}/{int(row.pairs)} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_negative_control_table(root: Path) -> str:
    frame = pd.read_csv(root / "outputs/runs/20260922_negative_control_vitg/summary_all.csv")
    source = frame[frame.pair != "target"]
    target = frame[frame.pair == "target"].iloc[0]
    source_coverage = source.inliers.sum() / source.queries.sum()
    source_semantic = (source.inliers * source.inlier_semantic_accuracy).sum() / source.inliers.sum()
    if not (source.rotation_error_deg >= 5).all():
        raise ValueError("negative-control pose counts require corrected translations")
    source_pose = 0  # Rotation alone rejects all 16; translation units cannot change this.
    rows = [
        ("Source, 16 pairs", source_coverage, source_semantic, f"{source_pose}/16"),
        ("Target, held out", float(target.inlier_coverage), float(target.inlier_semantic_accuracy), "--"),
    ]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Split & Coverage & Semantic & Strict pose \\",
             r" & (\%) & (\%) & $<5^\circ/0.25$m \\", r"\midrule"]
    for label, coverage, semantic, pose in rows:
        lines.append(f"{label} & {100 * coverage:.1f} & {100 * semantic:.1f} & {pose} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_framelevel_table(root: Path) -> str:
    aggregate = pd.read_csv(root / "outputs/runs/20260922_framelevel_selection_clean/aggregate.csv")
    target = pd.read_csv(root / "outputs/runs/20260922_framelevel_vitg_clean_target/summary.csv")
    selection = json.loads(
        (root / "outputs/runs/20260922_framelevel_selection_clean/summary.json").read_text(encoding="utf-8")
    )
    feature = selection["source_best_feature"]
    method = selection["source_best_method"]
    source_one_way = aggregate[(aggregate.feature == feature) & (aggregate.method == "one_way")].iloc[0]
    source_selected = aggregate[(aggregate.feature == feature) & (aggregate.method == method)].iloc[0]
    target_selected = target[(target.feature == feature) & (target.method == method)].iloc[0]
    rows = [
        ("Source macro", "one-way", source_one_way),
        ("Source macro", "source-selected", source_selected),
        ("Target", "source-selected", target_selected),
    ]
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Split & Filter & Cov. & Sem. acc. & RANSAC & Rot. \\ ",
             r" & & (\%) & (\%) & inlier & (deg.) \\ ", r"\midrule"]
    for split, filter_name, row in rows:
        lines.append(
            f"{split} & {filter_name} & {100 * row.coverage:.1f} & "
            f"{100 * row.semantic_accuracy:.1f} & {100 * row.ransac_inlier_rate:.1f} & "
            f"{row.rotation_error_deg:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_motion_stratified_table(root: Path) -> str:
    """Use local objectId groups; take only pose/count metadata from legacy audit."""
    query_path = root / "outputs/reports/20261008_object_motion_oracle_corrected_units/object_motion_oracle_queries.csv"
    pair_path = root / "outputs/reports/20260924_motion_stratified_audit/motion_stratified_pairs.csv"
    queries = pd.read_csv(query_path)
    legacy_pairs = pd.read_csv(pair_path).set_index("pair")
    corrected_poses = pd.read_csv(root / "outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv").set_index("run_id")
    rows = []

    def pct(numerator: int, denominator: int) -> str:
        return "--" if denominator == 0 else f"{100 * numerator / denominator:.1f}\\%"

    for pair in ORDER:
        group = queries[queries.pair == pair]
        if group.empty:
            raise ValueError(f"no corrected object-motion queries for {pair}")
        changed = group.group.isin(["rigid", "nonrigid"])
        inlier = group.consensus_inlier.astype(bool)
        semantic_correct = group.semantic_correct.astype(bool)
        changed_queries = int(changed.sum())
        changed_inliers = int((changed & inlier).sum())
        total_inliers = int(inlier.sum())
        changed_semantic = int((changed & inlier & semantic_correct).sum())
        other_inliers = int((~changed & inlier).sum())
        other_semantic = int((~changed & inlier & semantic_correct).sum())
        old = legacy_pairs.loc[pair]
        corrected_pose = corrected_poses.loc[pair]
        rows.append(
            f"\\texttt{{{pair}}} & {int(old.annotated_rigid_objects)}/{int(old.annotated_nonrigid_objects)} & "
            f"{pct(changed_queries, len(group))} / {pct(changed_inliers, total_inliers)} & "
            f"{pct(changed_semantic, changed_inliers)} / {pct(other_semantic, other_inliers)} & "
            f"{corrected_pose.rotation_error_degrees:.1f}$^\\circ$ & {corrected_pose.translation_error_m:.1f} "
            + r"\\"
        )
    return "\n".join([
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Pair & Rigid/nonrigid objs. & Query/inlier share & Sem. acc. changed/other & Rot. err. & Trans. err. (m) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", "",
    ])


def generate_rigid_transform_candidate_table(root: Path) -> str:
    """Candidate-level support under global and official object-pose oracles."""
    summary = json.loads((root / "outputs/reports/20261008_rigid_transform_valid_candidate_availability_corrected_units/summary.json").read_text())
    pooled = summary["pooled"]
    row_end = chr(92) * 2
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"$K$ & Physical object & Scene pose & Object-pose oracle & Full-mesh support " + row_end,
        r" & candidate (\%) & valid (\%) & valid (\%) & object oracle (\%) " + row_end,
        r"\midrule",
    ]
    for k in (1, 5, 10, 20, 50):
        physical = 100 * pooled[f"physical_candidate_k{k}"]
        scene = 100 * pooled[f"scene_valid_candidate_k{k}"]
        obj = 100 * pooled[f"object_valid_candidate_k{k}"]
        surface = 100 * pooled["reference_object_surface_support_object"]
        lines.append(f"{k} & {physical:.1f} & {scene:.1f} & {obj:.1f} & {surface:.1f} " + row_end)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"

def generate_baseline_transfer_table(root: Path) -> str:
    frame = pd.read_csv(root / "outputs/runs/20260923_baseline_label_transfer/baseline_transfer_table.csv")
    frame = frame[frame.method != "FPFH one-way"]
    labels = {
        ("source", "ViT-G consensus"): "Source / ViT-G",
        ("source", "FCGF top-20"): "Source / FCGF",
        ("target", "ViT-G consensus"): "Target / ViT-G",
        ("target", "FCGF top-20"): "Target / FCGF",
    }
    lines = [
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Dataset / matcher & Cov. & Cond. sem. mIoU & Cond. global-label mIoU & Sem. acc. \\",
        r" & (\%) & (\%) & (\%) & (\%) \\", r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{labels[(row.dataset, row.method)]} & {100 * row.coverage:.1f} & "
            f"{100 * row.semantic_mIoU:.1f} & {100 * row.instance_mIoU:.1f} & "
            f"{100 * row.semantic_accuracy:.1f} \\\\" 
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_fpfh_diagnostic_table(root: Path) -> str:
    """Report FPFH on its voxel query population, separate from patch methods."""
    transfer = pd.read_csv(root / "outputs/runs/20260923_baseline_label_transfer/baseline_transfer_table.csv")
    transfer = transfer[transfer.method == "FPFH one-way"].copy()
    control_root = root / "outputs/reports/20261008_source_geometry_controls_corrected_units"
    pair_rows = pd.concat([pd.read_csv(control_root / pair / "fpfh/summary.csv").assign(pair=pair)
                           for pair in ORDER], ignore_index=True)
    pair_rows = pair_rows[pair_rows.method == "fpfh_one_way"].copy()
    pair_rows["split"] = pair_rows.pair.astype(str).str.contains("target").map(
        {True: "target", False: "source"}
    )
    row_end = chr(92) * 2
    lines = [
        r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Split & Pairs & Evaluable voxel queries & Inliers & Coverage & Sem. acc. & Cond. sem. mIoU & Strict pose " + row_end,
        r" & & & & (\%) & (\%) & (\%) & " + row_end, r"\midrule",
    ]
    for row in transfer.itertuples(index=False):
        subset = pair_rows[pair_rows.split == row.dataset]
        pose = int(((subset.rotation_error_deg < 5) & (subset.translation_error_m < 0.25)).sum())
        lines.append(
            f"{row.dataset.capitalize()} & {int(row.pairs)} & {int(row.queries):,} & "
            f"{int(row.retained):,} & {100 * row.coverage:.2f} & "
            f"{100 * row.semantic_accuracy:.1f} & {100 * row.semantic_mIoU:.1f} & "
            f"{pose}/{len(subset)} " + row_end
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_mast3r_dense_budget_table(root: Path) -> str:
    """Compare MASt3R dense top-20 and ViT-G on their shared query budget."""
    base = root / "outputs/reports/20261003_mast3r_dense_topk_comparison"
    frame = pd.read_csv(base / "per_pair.csv")
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    corrected_m = pd.read_csv(root / "outputs/reports/20261008_source_dense_mast3r_corrected_units/per_relation_seed.csv")
    corrected_m = corrected_m[corrected_m.ransac_seed.eq(17)].set_index("pair")
    corrected_g = pd.read_csv(root / "outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv").set_index("run_id")
    for index, row in frame.iterrows():
        frame.loc[index, "mast3r_pose"] = corrected_m.loc[row.pair, "strict_pose_success"]
        frame.loc[index, "dino_pose"] = corrected_g.loc[row.pair, "strict_pose_success"]
    row_end = chr(92) * 2
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
        r"Pair & Queries & \multicolumn{2}{c}{Physical cand. recall@20} & \multicolumn{2}{c}{Correct physical transfer} & \multicolumn{2}{c}{Assigned precision} & \multicolumn{2}{c}{Strict pose} " + row_end,
        r" & & MASt3R & ViT-G & MASt3R & ViT-G & MASt3R & ViT-G & MASt3R & ViT-G " + row_end,
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"\\texttt{{{row.pair}}} & {int(row.mast3r_queries):,} & "
            f"{100 * row.mast3r_candidate_recall:.1f}\\% & {100 * row.dino_candidate_recall:.1f}\\% & "
            f"{100 * row.mast3r_transfer:.1f}\\% & {100 * row.dino_transfer:.1f}\\% & "
            f"{100 * row.mast3r_precision:.1f}\\% & {100 * row.dino_precision:.1f}\\% & "
            f"{'pass' if row.mast3r_pose else 'fail'} & {'pass' if row.dino_pose else 'fail'} " + row_end
        )
    weighted = summary["weighted_by_linked_queries"]
    mast3r_pose = int(frame.mast3r_pose.sum())
    dino_pose = int(frame.dino_pose.sum())
    query_count = int(frame.mast3r_queries.sum())
    lines += [r"\midrule"]
    lines.append(
        f"Weighted & {query_count:,} & {100 * weighted['mast3r_candidate_recall']:.1f}\\% & "
        f"{100 * weighted['dino_candidate_recall']:.1f}\\% & "
        f"{100 * weighted['mast3r_physical_transfer']:.1f}\\% & "
        f"{100 * weighted['dino_physical_transfer']:.1f}\\% & -- & -- & "
        f"{mast3r_pose}/4 & {dino_pose}/4 " + row_end
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_mast3r_full_train_table(root: Path) -> str:
    """Three-way paired comparison on all relations with matched physical audits."""
    base = root / "outputs/reports/20261003_mast3r_dense_train_audit"
    frame = pd.read_csv(base / "paired_dino_comparison.csv")
    corrected_m = pd.read_csv(root / "outputs/reports/20261008_source_dense_mast3r_corrected_units/per_relation_seed.csv")
    corrected_m = corrected_m[corrected_m.ransac_seed.eq(17)].set_index("source_run")
    corrected_g = pd.read_csv(root / "outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv").set_index("run_id")
    corrected_sparse = pd.read_csv(root / "outputs/reports/20261008_source_reciprocal_mast3r_corrected_units/per_relation.csv").set_index("source_run")
    if len(corrected_sparse) != 14:
        raise ValueError("complete corrected reciprocal source poses required")
    for index, row in frame.iterrows():
        frame.loc[index, "mast3r_pose_pass"] = corrected_m.loc[row.source_run, "strict_pose_success"]
        frame.loc[index, "sparse_pose_pass"] = corrected_sparse.loc[row.source_run, "strict_pose_success"]
        frame.loc[index, "dino_pose_pass"] = corrected_g.loc[short_pair_name(row.source_run), "strict_pose_success"]
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    row_end = chr(92) * 2
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
        r"Relation & $Q$ & \multicolumn{3}{c}{Physical candidate recall@20} & \multicolumn{3}{c}{Correct transfer coverage} & \multicolumn{3}{c}{Strict pose} " + row_end,
        r" & & Reciprocal & Dense & ViT-G & Reciprocal & Dense & ViT-G & Reciprocal & Dense & ViT-G " + row_end,
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        sparse_pose = r"\checkmark" if row.sparse_pose_pass else r"--"
        dense_pose = r"\checkmark" if row.mast3r_pose_pass else r"--"
        dino_pose = r"\checkmark" if row.dino_pose_pass else r"--"
        lines.append(
            f"\\texttt{{{short_pair_name(row.source_run)}}} & {int(row.linked_queries):,} & "
            f"{100 * row.sparse_candidate_recall:.1f}\\% & {100 * (row.dino_candidate_hits + row.paired_candidate_delta_hits) / row.linked_queries:.1f}\\% & "
            f"{100 * row.dino_candidate_recall:.1f}\\% & "
            f"{100 * row.sparse_transfer_coverage:.1f}\\% & {100 * (row.dino_correct_transfers + row.paired_transfer_delta_hits) / row.linked_queries:.1f}\\% & "
            f"{100 * row.dino_transfer_coverage:.1f}\\% & {sparse_pose} & {dense_pose} & {dino_pose} " + row_end
        )
    p = summary["metrics"]["paired_dino_scope"]
    lines += [r"\midrule"]
    lines.append(
        f"Weighted & {int(p['linked_queries']):,} & {100*p['sparse_candidate_recall']:.1f}\\% & "
        f"{100*p['mast3r_candidate_recall']:.1f}\\% & {100*p['dino_candidate_recall']:.1f}\\% & "
        f"{100*p['sparse_transfer_coverage']:.1f}\\% & {100*p['mast3r_transfer_coverage']:.1f}\\% & {100*p['dino_transfer_coverage']:.1f}\\% & "
        f"{int(frame.sparse_pose_pass.sum())}/{len(frame)} & {int(frame.mast3r_pose_pass.sum())}/{len(frame)} & {int(frame.dino_pose_pass.sum())}/{len(frame)} " + row_end
    )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def generate_bufferx_source_frozen_table(root: Path) -> str:
    directory=root / 'outputs/reports/20261008_bufferx_validation/source_frozen_threshold'
    env=pd.read_csv(directory / 'per_environment.csv')
    relations=pd.read_csv(directory / 'per_relation.csv')
    arms=[('vitg_global_ransac','ViT-G RANSAC'),('bufferx_pose_vitg_k20','BUFFER-X + DINO K20'),
          ('source_frozen_geometry','Source-fixed distance'),('source_frozen_rank','Source-fixed rank fraction')]
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Arm & Semantic cov. & Semantic prec. & Moved cov. & Raw accepts \\',r'\midrule']
    for arm,label in arms:
        f=env[env.arm.eq(arm)]
        metrics=[100*f[c].mean() for c in ['semantic_coverage','semantic_precision','moved_coverage']]
        count=int(relations[relations.arm.eq(arm)].raw_assigned.sum())
        lines.append(label+' & '+' & '.join(f'{v:.1f}\\%' for v in metrics)+f' & {count:,} '+r'\\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}'])+'\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "paper").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "generated_pair_table.tex").write_text(generate_pair_table(consensus_rows(root)), encoding="utf-8")
    (output / "generated_transfer_table.tex").write_text(generate_transfer_table(root), encoding="utf-8")
    (output / "generated_abstention_miou_table.tex").write_text(
        generate_abstention_miou_table(root), encoding="utf-8"
    )
    (output / "generated_calibration_table.tex").write_text(generate_calibration_table(root), encoding="utf-8")
    (output / "generated_risk_controlled_table.tex").write_text(generate_risk_controlled_table(root), encoding="utf-8")
    (output / "generated_fcgf_table.tex").write_text(generate_fcgf_table(root), encoding="utf-8")
    (output / "generated_scale_table.tex").write_text(generate_scale_table(root), encoding="utf-8")
    (output / "generated_k_sweep_table.tex").write_text(generate_k_sweep_table(root), encoding="utf-8")
    (output / "generated_candidate_set_audit_table.tex").write_text(
        generate_candidate_set_audit_table(root), encoding="utf-8"
    )
    (output / "generated_seed_sweep_table.tex").write_text(
        generate_seed_sweep_table(root), encoding="utf-8"
    )
    (output / "generated_negative_control_table.tex").write_text(generate_negative_control_table(root), encoding="utf-8")
    (output / "generated_framelevel_table.tex").write_text(generate_framelevel_table(root), encoding="utf-8")
    (output / "motion_stratified_table.tex").write_text(generate_motion_stratified_table(root), encoding="utf-8")
    (output / "generated_rigid_transform_candidate_table.tex").write_text(
        generate_rigid_transform_candidate_table(root), encoding="utf-8"
    )
    (output / "generated_baseline_transfer_table.tex").write_text(
        generate_baseline_transfer_table(root), encoding="utf-8"
    )
    (output / "generated_fpfh_diagnostic_table.tex").write_text(
        generate_fpfh_diagnostic_table(root), encoding="utf-8"
    )
    (output / "generated_mast3r_dense_budget_table.tex").write_text(
        generate_mast3r_dense_budget_table(root), encoding="utf-8"
    )
    (output / "generated_mast3r_full_train_table.tex").write_text(
        generate_mast3r_full_train_table(root), encoding="utf-8"
    )
    (output / 'generated_bufferx_source_frozen_table.tex').write_text(generate_bufferx_source_frozen_table(root), encoding='utf-8')
    print(f"wrote supplementary tables to {output}")


if __name__ == "__main__":
    main()
