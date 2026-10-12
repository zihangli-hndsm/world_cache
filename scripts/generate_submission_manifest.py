"""Create a portable, hashed snapshot of the current paper and its inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


FILES = [
    'scripts/verify_corrected_scene_unit_evidence.py',
    'scripts/compare_corrected_validation_poses.py',
    'scripts/aggregate_corrected_geometry_control_poses.py',
    'scripts/rescore_validation_seed_pose_units.py',
    'scripts/rescore_fcgf_validation_seed_pose_units.py',
    'scripts/aggregate_source_dense_mast3r_corrected_poses.py',
    'scripts/aggregate_bufferx_validation_pose.py',
    'scripts/score_bufferx_pose_conditioned_transfer.py',
    'scripts/audit_bufferx_transfer_matched_budget.py',
    'scripts/audit_3rscan_scene_transform_units.py',
    'outputs/reports/20261008_scene_transform_unit_audit/summary.json',
    'outputs/reports/20261008_corrected_validation_pose_comparison/summary.json',
    'outputs/reports/20261008_corrected_validation_pose_comparison/per_relation.csv',
    'outputs/reports/20261008_corrected_validation_pose_comparison/per_environment.csv',
    'outputs/reports/20261008_bufferx_validation/aggregate/summary.json',
    'outputs/reports/20261008_bufferx_validation/pose_conditioned_transfer/summary.json',
    'outputs/reports/20261008_bufferx_validation/matched_assignment_budget/summary.json',
    'outputs/reports/20261008_vitg_source_corrected_pose_units/summary.json',
    'outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv',
    'outputs/reports/20261008_vitg_validation_corrected_pose_units/summary.json',
    'outputs/reports/20261008_vitg_validation_corrected_pose_units/per_relation.csv',
    'outputs/reports/20261008_mast3r_official_validation_corrected_units/summary.json',
    'outputs/reports/20261008_mast3r_official_validation_corrected_units/per_relation.csv',
    'outputs/reports/20261008_mast3r_dense_official_validation_corrected_units/summary.json',
    'outputs/reports/20261008_mast3r_dense_official_validation_corrected_units/per_relation.csv',
    'outputs/reports/20261008_source_reciprocal_mast3r_corrected_units/summary.json',
    'outputs/reports/20261008_source_reciprocal_mast3r_corrected_units/per_relation.csv',
    'outputs/reports/20261008_source_dense_mast3r_full17_corrected_units/summary.json',
    'outputs/reports/20261008_source_dense_mast3r_full17_corrected_units/per_relation.csv',
    'outputs/reports/20261008_vitg_validation_seed_corrected_units/summary.json',
    'outputs/reports/20261008_vitg_validation_seed_corrected_units/per_relation_seed.csv',
    'outputs/reports/20261008_mast3r_dense_validation_seed_corrected_units/summary.json',
    'outputs/reports/20261008_mast3r_dense_validation_seed_corrected_units/per_relation_seed.csv',
    'outputs/reports/20261008_fcgf_validation_seed_corrected_units/summary.json',
    'outputs/reports/20261008_fcgf_validation_seed_corrected_units/per_relation_seed.csv',
    'outputs/reports/20261008_geometry_controls_corrected_units/aggregate/summary.json',
    'outputs/reports/20261008_geometry_controls_corrected_units/aggregate/per_relation.csv',
    'outputs/reports/20261008_validation_transform_valid_candidate_availability_corrected_units_sensitivity/summary.json',
    'scripts/audit_3rscan_test_split_feasibility.py',
    'scripts/benchmark_dino_source_bank_encoder.py',
    'outputs/reports/20261009_3rscan_test_split_feasibility/report.md',
    'outputs/reports/20261009_3rscan_test_split_feasibility/summary.json',
    'outputs/reports/20261009_dino_source_bank_encoder_runtime/report.md',
    'outputs/reports/20261009_dino_source_bank_encoder_runtime/summary.json',
    'scripts/analyze_dino_multiview_budget_curve.py',
    'scripts/analyze_dino_multiview_bank_increment.py',
    'scripts/analyze_remind_pairwise_comparison.py',
    'outputs/reports/20261009_dino_multiview_source_bank/view_budget_curve.json',
    'outputs/reports/20261009_dino_multiview_source_bank_train8/view_budget_curve.json',
    'outputs/reports/20261109_remind_pairwise_validation/report.md',
    'outputs/reports/20261109_remind_pairwise_validation/summary.json',
    'outputs/reports/20261009_dino_multiview_bank_increment/report.md',
    'outputs/reports/20261009_dino_multiview_bank_increment/summary.json',
    'outputs/reports/20261009_dino_multiview_bank_increment_train8/report.md',
    'outputs/reports/20261009_dino_multiview_bank_increment_train8/summary.json',
    'outputs/reports/20261009_core_finding_audit.md',
    'outputs/reports/20261009_venue_opportunity_update.md',
    "README.md",
    "DATASETS.md",
    "WORLD_CACHE_FINDINGS.md",
    "download.py",
    "environment.yml",
    "pyproject.toml",
    "LICENSE",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "paper/main.tex",
    "paper/supplementary.tex",
    "paper/main.pdf",
    "paper/supplementary.pdf",
    "paper/main.log",
    "paper/supplementary.log",
    "paper/Makefile",
    "paper/README.md",
    "paper/PRISM_EXTENSION_NOTE.md",
    "scripts/evaluate_3rscan_spatial_consensus.py",
    "scripts/evaluate_3rscan_framelevel_layers.py",
    "scripts/evaluate_3rscan_instance_correspondence.py",
    "scripts/evaluate_3rscan_pose_free_registration.py",
    "scripts/evaluate_3rscan_fpfh_baseline.py",
    "scripts/evaluate_3rscan_fcgf_baseline.py",
    "scripts/generate_supplementary_tables.py",
    "scripts/verify_current_paper.py",
    "scripts/compare_vitl_vitg_candidate_audit.py",
    "scripts/audit_backbone_scaleup.py",
    "scripts/aggregate_candidate_set_audit.py",
    "scripts/audit_3rscan_moved_instance_candidates.py",
    "scripts/compare_physical_instance_candidate_audit.py",
    "scripts/run_physical_instance_candidate_audit_matched14.py",
    "scripts/evaluate_3rscan_multimotion.py",
    "scripts/analyze_3rscan_multimotion_rescue_mechanisms.py",
    "scripts/analyze_pose_locked_multimotion_extended_source.py",
    "scripts/analyze_pose_locked_motion_identity_permutation.py",
    "scripts/analyze_pose_locked_selector_fit_bootstrap.py",
    "scripts/analyze_pose_locked_selector_aware_identity_permutation.py",
    "scripts/analyze_3rscan_multimotion_precision_lopo.py",
    "scripts/analyze_3rscan_multimotion_confidence_lopo.py",
    "scripts/aggregate_pose_locked_multimotion_seed_robustness.py",
    "scripts/aggregate_pose_locked_extended_source_seed_robustness.py",
    "scripts/analyze_seed_agreement_selective_tradeoff.py",
    "scripts/aggregate_multimotion_seed_vote.py",
    "scripts/aggregate_multimotion_seed_robustness.py",
    "scripts/run_3rscan_validation_pose_locked_motion.py",
    "scripts/generate_submission_manifest.py",
    "scripts/build_submission_bundle.py",
    "third_party/warpconvnet_fcgf.py",
    "outputs/reports/20260929_validation_access/validation_scan_manifest.csv",
    "outputs/reports/20261003_storage_recovery_and_rerun.md",
    "outputs/reports/20261003_vitg_validation/protocol_status.json",
    "outputs/reports/20261003_vitg_validation/validation_run_manifest.csv",
    "outputs/reports/20261003_vitg_validation/aggregate/per_relation.csv",
    "outputs/reports/20261003_vitg_validation/aggregate/per_environment.csv",
    "outputs/reports/20261003_vitg_validation/aggregate/summary.json",
    "outputs/reports/20261003_vitg_validation/aggregate/report.md",
    "outputs/reports/20261003_3rscan_cross_split_overlap/cross_split_pairs.csv",
    "outputs/reports/20261003_3rscan_cross_split_overlap/same_environment_controls.csv",
    "outputs/reports/20261003_3rscan_cross_split_overlap/summary.json",
    "outputs/reports/20261003_3rscan_cross_split_overlap/report.md",
    "outputs/reports/20261003_3rscan_cross_split_overlap/train_reference_download_manifest.json",
    "outputs/reports/20261003_3rscan_cross_split_overlap/validation_sensitivity/summary.json",
    "outputs/reports/20261003_3rscan_cross_split_overlap/validation_sensitivity/report.md",
    "outputs/reports/20261003_3rscan_cross_split_overlap/train_full_targeted/cross_split_pairs.csv",
    "outputs/reports/20261003_3rscan_cross_split_overlap/train_full_targeted/same_environment_controls.csv",
    "outputs/reports/20261003_3rscan_cross_split_overlap/train_full_targeted/summary.json",
    "outputs/reports/20261003_3rscan_cross_split_overlap/train_full_targeted/report.md",
    "outputs/reports/20260929_multimotion_seed_robustness/seed_17/source_lopo_selection.csv",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/protocol_plan.json",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/frozen_train_selection.json",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/protocol_status.json",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/run_manifest.csv",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/summary.json",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/per_relation.csv",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/per_environment.csv",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/report.md",
    "outputs/reports/20261008_anonymous_submission_pdfs.zip",
    "outputs/reports/20261002_vitl_vitg_candidate_layer_comparison/summary.json",
    "outputs/reports/20261002_physical_instance_candidate_layer_comparison/summary.json",
    "outputs/reports/20261002_rigid_transform_valid_candidate_availability/summary.json",
    "outputs/reports/20261003_seed_agreement_selective_tradeoff/summary.json",
    "outputs/reports/20261003_seed_agreement_selective_tradeoff/target_by_object.csv",
    "outputs/reports/20261003_mast3r_baseline_comparison/summary.json",
    "outputs/reports/20261003_mast3r_baseline_comparison/per_pair.csv",
    "outputs/reports/20261003_mast3r_baseline_comparison/report.md",
    "outputs/reports/20261003_mast3r_dense_topk_comparison/summary.json",
    "outputs/reports/20261003_mast3r_dense_topk_comparison/per_pair.csv",
    "outputs/reports/20261003_mast3r_dense_topk_comparison/report.md",
    "outputs/reports/20261003_mast3r_dense_train_audit/summary.json",
    "outputs/reports/20261003_mast3r_dense_train_audit/per_relation.csv",
    "outputs/reports/20261003_mast3r_dense_train_audit/paired_dino_comparison.csv",
    "outputs/reports/20261003_mast3r_dense_train_audit/selection_manifest.csv",
    "outputs/reports/20261003_mast3r_dense_train_audit/report.md",
    "outputs/reports/20261003_mast3r_transfer_conversion/summary.json",
    "outputs/reports/20261003_mast3r_transfer_conversion/per_relation.csv",
    "outputs/reports/20261003_mast3r_transfer_conversion/report.md",
    "outputs/reports/20261003_mast3r_dense_seed_robustness/summary.json",
    "outputs/reports/20261003_mast3r_dense_seed_robustness/summary_by_seed.csv",
    "outputs/reports/20261003_mast3r_dense_seed_robustness/per_relation_seed.csv",
    "outputs/reports/20261003_mast3r_dense_seed_robustness/report.md",
    "outputs/reports/20261003_mast3r_seed_agreement/summary.json",
    "outputs/reports/20261003_mast3r_seed_agreement/per_relation.csv",
    "outputs/reports/20261003_mast3r_seed_agreement/report.md",
    "outputs/reports/20261004_pose_locked_object_cluster_sensitivity/summary.json",
    "outputs/reports/20261004_pose_locked_object_cluster_sensitivity/per_relation.csv",
    "outputs/reports/20261004_pose_locked_object_cluster_sensitivity/per_object.csv",
    "outputs/reports/20261004_pose_locked_object_cluster_sensitivity/report.md",
    "outputs/reports/20261004_pose_locked_identity_permutation/summary.json",
    "outputs/reports/20261004_pose_locked_identity_permutation/per_relation.csv",
    "outputs/reports/20261004_pose_locked_identity_permutation/permutation_draws.csv.gz",
    "outputs/reports/20261004_pose_locked_identity_permutation/report.md",
    "outputs/reports/20261005_pose_locked_selector_fit_bootstrap/summary.json",
    "outputs/reports/20261005_pose_locked_selector_fit_bootstrap/per_relation.csv",
    "outputs/reports/20261005_pose_locked_selector_fit_bootstrap/bootstrap_draws.csv.gz",
    "outputs/reports/20261005_pose_locked_selector_fit_bootstrap/report.md",
    "outputs/reports/20261005_pose_locked_selector_aware_identity_permutation/summary.json",
    "outputs/reports/20261005_pose_locked_selector_aware_identity_permutation/per_relation.csv",
    "outputs/reports/20261005_pose_locked_selector_aware_identity_permutation/permutation_draws.csv.gz",
    "outputs/reports/20261005_pose_locked_selector_aware_identity_permutation/report.md",
    "outputs/reports/20261006_source_precision_motion/fixed_top20_run/summary.csv",
    "outputs/reports/20261006_source_precision_motion/fixed_top20_run/baseline_regression.csv",
    "outputs/reports/20261006_source_precision_motion/precision_lopo/source_lopo.csv",
    "outputs/reports/20261006_source_precision_motion/precision_lopo/target_once.csv",
    "outputs/reports/20261006_source_precision_motion/precision_lopo/query_outcomes.csv",
    "outputs/reports/20261006_source_precision_motion/precision_lopo/summary.json",
    "outputs/reports/20261006_source_precision_motion/precision_lopo/report.md",
    "outputs/reports/20261006_source_precision_motion/precision_lopo/failure_case_4aca.json",
    "outputs/reports/20261003_source_confidence_motion_lopo/source_lopo.csv",
    "outputs/reports/20261003_source_confidence_motion_lopo/query_outcomes.csv",
    "outputs/reports/20261003_source_confidence_motion_lopo/summary.json",
    "outputs/reports/20261003_source_confidence_motion_lopo/report.md",
    "outputs/reports/20261003_pose_locked_multimotion_seed_robustness/summary.json",
    "outputs/reports/20261003_pose_locked_extended_source_seed_robustness/summary.json",
    "outputs/reports/20260924_motion_stratified_audit/motion_stratified_pairs.csv",
    "outputs/reports/20260924_object_motion_oracle/object_motion_oracle_queries.csv",
    "outputs/reports/20261003_submission_artifacts/README.md",
    "outputs/reports/20261003_submission_artifacts/fcgf_summary_all.csv",
    "outputs/runs/20260922_3rscan_consensus_vitg_clean/summary_all.csv",
    "outputs/runs/20260922_3rscan_consensus_vitg_clean_v2/summary_all.csv",
    "outputs/runs/20260922_3rscan_consensus_vitl_clean/summary_all.csv",
    "outputs/runs/20260922_fpfh_17pair_clean/summary_all.csv",
    "outputs/runs/20260922_negative_control_vitg/summary_all.csv",
    "outputs/runs/20260922_framelevel_selection_clean/aggregate.csv",
    "outputs/runs/20260922_framelevel_selection_clean/summary.json",
    "outputs/runs/20260922_framelevel_vitg_clean_target/summary.csv",
    "outputs/runs/20260922_k_sweep_vitg_clean/summary.csv",
    "outputs/runs/20260922_risk_control_selection_audit_vitg_clean/summary.csv",
    "outputs/runs/20260922_risk_controlled_calibration_vitg_clean/summary.csv",
    "outputs/runs/20260922_candidate_set_audit_vitg_clean/per_pair.csv",
    "outputs/runs/20260922_backbone_scaleup_audit/summary.csv",
    "outputs/runs/20260922_seed_sweep_vitg_clean/summary.csv",
    "outputs/runs/20260922_semantic_label_transfer_vitg_clean_calibrated_v2/summary.csv",
    "outputs/runs/20260922_semantic_label_transfer_vitg_clean_calibrated_v2/macro_summary.csv",
    "outputs/runs/20260923_selective_calibration_vitg_clean_v2/leave_one_pair_out.csv",
    "outputs/runs/20260923_baseline_label_transfer/baseline_transfer_table.csv",
]

GLOB_FILES = [
    'outputs/reports/20261008_*corrected*/*.json',
    'outputs/reports/20261008_*corrected*/*.csv',
    'outputs/reports/20261008_*corrected*/**/regression.json',
    'outputs/reports/20261008_scene_transform_unit_audit/official_code/*',
    'outputs/reports/20261008_scene_transform_unit_audit/*.json',
    'outputs/reports/20261008_scene_transform_unit_audit/*.csv',
    'outputs/reports/20261008_bufferx_validation/*manifest.csv',
    'outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json',
    'outputs/reports/20261008_bufferx_source16/run_manifest.csv',
    'outputs/reports/20261008_bufferx_source16/relations/*/predicted_rescan_to_reference.npy',
    'outputs/reports/20261008_bufferx_validation/source_frozen_threshold/*.csv',
    'outputs/reports/20261008_bufferx_validation/source_frozen_threshold/*.json',
    'outputs/reports/20261008_unseen_train_replication/*.json',
    'outputs/reports/20261008_unseen_train_replication/*.csv',
    'outputs/reports/20261008_unseen_train_replication/matched_assignment_budget/*.json',
    'outputs/reports/20261008_unseen_train_replication/matched_assignment_budget/*.csv',
    'outputs/reports/20261008_frozen_control_precision_decomposition/*.json',
    'outputs/reports/20261008_frozen_control_semantic_strata/*.csv',
    'outputs/reports/20261008_bufferx_validation/linked_instance_transfer/*.json',
    'outputs/reports/20261008_bufferx_validation/linked_instance_transfer/*.csv',
    'outputs/reports/20261006_source_precision_motion/fixed_top20_run/descriptor_candidates/*.npz',
    'outputs/reports/20261008_bufferx_validation/relations/*/result.json',
    'outputs/reports/20261008_bufferx_validation/aggregate/*.csv',
    'outputs/reports/20261008_bufferx_validation/pose_conditioned_transfer/*.csv',
    'outputs/reports/20261008_bufferx_validation/matched_assignment_budget/*.csv',
    'outputs/reports/20261008_geometry_controls_corrected_units/aggregate/*.csv',
    'outputs/reports/20261008_geometry_controls_corrected_units/aggregate/*.json',
    "scripts/**/*.py",
    "src/**/*.py",
    "tests/**/*.py",
    "paper/generated_*.tex",
    "paper/motion_stratified_table.tex",
    "paper/figures/*.pdf",
    "outputs/reports/20260923_review_figures/*.pdf",
    "outputs/runs/20260922_3rscan_consensus_*_vitg_diag_clean/*.csv",
    "outputs/runs/20260922_3rscan_consensus_*_vitg_diag_clean/*.json",
    "outputs/runs/20260922_candidate_audit_*_vitg_clean/*.csv",
    "outputs/runs/20260922_candidate_audit_*_vitg_clean/*.json",
    "outputs/runs/20260922_semantic_label_transfer_vitg_clean_calibrated_v2/*.csv",
    "outputs/runs/20260922_risk_control_selection_audit_vitg_clean/*.csv",
    "outputs/runs/20260922_risk_control_selection_audit_vitg_clean/*.json",
    "outputs/runs/20260922_risk_controlled_calibration_vitg_clean/*.csv",
    "outputs/runs/20260922_risk_controlled_calibration_vitg_clean/*.json",
    "outputs/runs/20260922_framelevel_selection_clean/*.csv",
    "outputs/runs/20260922_framelevel_selection_clean/*.json",
    "outputs/runs/20260922_framelevel_vitg_clean_target/*.csv",
    "outputs/runs/20260922_k_sweep_vitg_clean/*.csv",
    "outputs/runs/20260922_k_sweep_vitg_clean/*.json",
    "outputs/runs/20260922_seed_sweep_vitg_clean/*.csv",
    "outputs/runs/20260922_seed_sweep_vitg_clean/*.json",
    "outputs/runs/20260922_backbone_scaleup_audit/*.csv",
    "outputs/runs/20260922_backbone_scaleup_audit/*.json",
    "outputs/runs/20260923_baseline_label_transfer/*.csv",
    "outputs/reports/20261003_mast3r_*_probe/*.csv",
    "outputs/reports/20261003_mast3r_*_probe/*.json",
    "outputs/runs/20261003_mast3r_dense_topk_*/summary.json",
    "outputs/runs/20261003_mast3r_dense_topk_*/query_diagnostics.csv",
    "outputs/runs/20261003_mast3r_dense_topk_*/status.json",
    "outputs/runs/20261003_mast3r_reciprocal_fixed_cuda_*/summary.json",
    "outputs/runs/20261003_mast3r_reciprocal_fixed_cuda_*/query_diagnostics.csv",
    "outputs/runs/20261003_mast3r_reciprocal_fixed_cuda_*/status.json",
    "outputs/runs/20261003_mast3r_dense_seed_sweep/*/seed_*/query_diagnostics.csv",
    "outputs/runs/20261003_mast3r_dense_seed_sweep/*/seed_*/summary.json",
    "outputs/runs/20261003_mast3r_dense_seed_sweep/*/seed_*/status.json",
    "outputs/runs/20261003_mast3r_dense_seed_sweep/*.csv",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/relations/*/summary.csv",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/relations/*/query_diagnostics.csv",
    "outputs/reports/20261006_vitg_pose_locked_motion_validation/relations/*/status.json",
    "outputs/reports/20261006_3rscan_score_threshold_cluster_bootstrap/*.json",
    "outputs/reports/20261006_*/*.json",
    "outputs/reports/20261006_*/*.csv",
    "outputs/reports/20261008_*/**/*.json",
    "outputs/reports/20261008_*/**/*.csv",
    "outputs/reports/20261009_*/**/*.json",
    "outputs/reports/20261009_*/**/*.csv",
    "outputs/reports/20261009_*/**/*.md",
    "outputs/reports/20261009_dino_source_bank_encoder_runtime/*.json",
    "outputs/reports/20261010_*/**/*.json",
    "outputs/reports/20261010_*/**/*.csv",
    "outputs/reports/20261010_*/**/*.md",
    "outputs/reports/20261010_*/**/*.pdf",
    "outputs/reports/20261010_*/**/*.png",
]


def manifest_files(root: Path) -> list[str]:
    paths = list(FILES)
    if (root / "PORTABLE_COPY_PROVENANCE.json").is_file():
        paths.append("PORTABLE_COPY_PROVENANCE.json")
    for pattern in GLOB_FILES:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"manifest glob matched no files: {pattern}")
        paths.extend(str(path.relative_to(root)) for path in matches)
    return list(dict.fromkeys(paths))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/reports/20261009_submission_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    entries = []
    for relative in manifest_files(root):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"manifest path is not portable: {relative}")
        content = path.read_bytes()
        if re.search(rb"/home/[A-Za-z0-9._-]+(?:/|\b)", content):
            raise ValueError(f"machine-local home path in submission input: {relative}")
        entries.append({
            "path": relative,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    status = git_value(root, "status", "--porcelain")
    payload = {
        "status": "complete",
        "format": "portable_sha256_manifest_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": git_value(root, "rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "absolute_paths_used": False,
        "external_inputs_not_bundled": [
            "3RScan RGB-D sequences and annotations",
            "DINOv2, FCGF, MASt3R, and BUFFER-X pretrained weights",
        ],
        "files": entries,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} hashed entries to {output}")


if __name__ == "__main__":
    main()
