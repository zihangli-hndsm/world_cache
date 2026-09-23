"""Generate a portable hash manifest for the current anonymized submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = [
    "paper/main.tex",
    "paper/supplementary.tex",
    "paper/main.pdf",
    "paper/supplementary.pdf",
    "paper/Makefile",
    "paper/README.md",
    "paper/figures/failure_taxonomy.pdf",
    "paper/figures/method_pipeline.pdf",
    "paper/figures/selective_correspondence_overview.pdf",
    "paper/generated_calibration_table.tex",
    "paper/generated_candidate_set_audit_table.tex",
    "paper/generated_fcgf_table.tex",
    "paper/generated_framelevel_table.tex",
    "paper/generated_k_sweep_table.tex",
    "paper/generated_negative_control_table.tex",
    "paper/generated_pair_table.tex",
    "paper/generated_risk_controlled_table.tex",
    "paper/generated_scale_table.tex",
    "paper/generated_seed_sweep_table.tex",
    "paper/generated_transfer_table.tex",
    "paper/generated_abstention_miou_table.tex",
    "paper/generated_baseline_transfer_table.tex",
    "scripts/evaluate_3rscan_spatial_consensus.py",
    "scripts/evaluate_3rscan_fcgf_baseline.py",
    "scripts/aggregate_candidate_set_audit.py",
    "scripts/audit_backbone_scaleup.py",
    "scripts/aggregate_seed_sweep.py",
    "scripts/generate_supplementary_tables.py",
    "scripts/plot_submission_figures.py",
    "scripts/verify_submission_consistency.py",
    "scripts/generate_submission_manifest.py",
    "scripts/build_submission_bundle.py",
    "third_party/warpconvnet_fcgf.py",
    "outputs/reports/20261004_risk_control_selection_audit.md",
    "outputs/reports/20261005_candidate_set_audit.md",
    "outputs/reports/20261006_backbone_scaleup_statistics.md",
    "outputs/reports/20261007_ransac_seed_sensitivity.md",
    "outputs/reports/20260922_submission_dossier.md",
    "outputs/reports/20260923_review_update.md",
    "outputs/reports/20261001_annotation_projection_audit.json",
    "outputs/reports/20261003_k_sensitivity.md",
    "outputs/reports/20261002_3rscan_scope_audit.md",
    "outputs/reports/20261009_1776_external_validation.md",
    "outputs/reports/20261009_1776_geometry_audit.json",
    "outputs/reports/20261009_1776_annotation_projection_audit.csv",
    "outputs/reports/20261009_1776_annotation_projection_audit.json",
    "outputs/reports/20261010_4aca_external_validation.md",
    "outputs/reports/20261010_4aca_geometry_audit.json",
    "outputs/reports/20261010_4aca_annotation_projection_audit.csv",
    "outputs/reports/20261010_4aca_annotation_projection_audit.json",
    "outputs/reports/20261012_baseline_label_transfer.md",
    "outputs/reports/20261013_mast3r_feasibility_audit.md",
    "outputs/reports/20261015_cross_fitted_calibration.md",
    "outputs/runs/20261002_3rscan_consensus_vitg_clean_v2/summary_all.csv",
    "outputs/runs/20261002_3rscan_consensus_vitg_clean/summary_all.csv",
    "outputs/runs/20261002_3rscan_consensus_vitl_clean/summary_all.csv",
    "outputs/runs/20261002_fpfh_17pair_clean/summary_all.csv",
    "outputs/runs/20261002_fcgf_17pair_clean/summary_all.csv",
    "outputs/runs/20261002_negative_control_vitg/summary_all.csv",
    "outputs/runs/20261002_runtime_target_vitg_clean/summary.csv",
    "outputs/runs/20261002_runtime_target_vitg_clean/status.json",
    "outputs/runs/20261003_k_sweep_vitg_clean/summary.csv",
    "outputs/runs/20261004_risk_control_selection_audit_vitg_clean/summary.csv",
    "outputs/runs/20261005_candidate_set_audit_vitg_clean/summary.csv",
    "outputs/runs/20261005_candidate_set_audit_vitg_clean/per_pair.csv",
    "outputs/runs/20261006_backbone_scaleup_audit/summary.csv",
    "outputs/runs/20261007_seed_sweep_vitg_clean/summary.csv",
    "outputs/runs/20261007_seed_sweep_vitg_clean/pair_stability.csv",
    "outputs/reports/20261002_selective_correspondence_vitg_clean_v2/selective_correspondence_overview.pdf",
    "outputs/reports/20261002_submission_figures_clean_v2/failure_taxonomy.pdf",
    "outputs/reports/20261002_submission_figures_clean_v2/method_pipeline.pdf",
]

# These small CSV/JSON/status artifacts are the complete dependency closure of
# the table generator and numerical verifier.  Raw RGB-D NPZ frames and model
# outputs are intentionally not part of the paper bundle.
GLOB_FILES = [
    "src/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
    "outputs/runs/20261002_3rscan_consensus_*_vitg_diag_clean/*.csv",
    "outputs/runs/20261002_3rscan_consensus_*_vitg_diag_clean/*.json",
    "outputs/runs/20261005_candidate_audit_*_vitg_clean/*.csv",
    "outputs/runs/20261005_candidate_audit_*_vitg_clean/*.json",
    "outputs/runs/20261002_negative_control_*_vitg/*.json",
    "outputs/runs/20261002_negative_control_target_vitg_v2/*.json",
    "outputs/runs/20260917_3rscan_pair*_s1/config.json",
    "outputs/runs/20260917_3rscan_target_pair_s1/config.json",
    "outputs/runs/20260921_3rscan_train_*/config.json",
    "outputs/runs/20260922_3rscan_semantic_*/config.json",
    "outputs/runs/20261001_3rscan_pair0[6-8]/config.json",
    "outputs/runs/20261002_regression_target_vitg_after_controls/*",
    "outputs/runs/20261002_framelevel_selection_clean/*.csv",
    "outputs/runs/20261002_framelevel_selection_clean/*.json",
    "outputs/runs/20261002_framelevel_vitg_clean_target/*.csv",
    "outputs/runs/20261002_selective_calibration_vitg_clean_v2/*.csv",
    "outputs/runs/20261002_semantic_label_transfer_vitg_clean_calibrated_v2/*.csv",
    "outputs/runs/20261002_risk_controlled_calibration_vitg_clean/*.csv",
    "outputs/runs/20261002_risk_controlled_calibration_vitg_clean/*.json",
    "outputs/runs/20261004_risk_control_selection_audit_vitg_clean/*.csv",
    "outputs/runs/20261004_risk_control_selection_audit_vitg_clean/*.json",
    "outputs/runs/20261005_candidate_set_audit_vitg_clean/*.csv",
    "outputs/runs/20261005_candidate_set_audit_vitg_clean/*.json",
    "outputs/runs/20261006_backbone_scaleup_audit/*.csv",
    "outputs/runs/20261006_backbone_scaleup_audit/*.json",
    "outputs/runs/20261007_seed_sweep_vitg_clean/*.csv",
    "outputs/runs/20261007_seed_sweep_vitg_clean/*.json",
    "outputs/runs/20261003_k_sweep_vitg_clean/*.csv",
    "outputs/runs/20261003_k_sweep_vitg_clean/*.json",
    "outputs/runs/20261009_3rscan_extra_1776_vitg20_clean/*.csv",
    "outputs/runs/20261009_3rscan_extra_1776_vitg20_clean/*.json",
    "outputs/runs/20261009_3rscan_extra_1776_vitg_k_sweep/*.csv",
    "outputs/runs/20261009_3rscan_extra_1776_vitg_k_sweep/*.json",
    "outputs/runs/20261009_3rscan_extra_1776_pair20_s1/*.json",
    "outputs/runs/20261010_3rscan_extra_4aca_vitg20_clean/*.csv",
    "outputs/runs/20261010_3rscan_extra_4aca_vitg20_clean/*.json",
    "outputs/runs/20261010_3rscan_extra_4aca_vitg_k_sweep/*.csv",
    "outputs/runs/20261010_3rscan_extra_4aca_vitg_k_sweep/*.json",
    "outputs/runs/20261010_3rscan_extra_4aca_pair20_s1/*.json",
    "outputs/runs/20261011_external_calibration_4aca_source16/*.csv",
    "outputs/runs/20261011_risk_calibration_4aca_source16/*.csv",
    "outputs/runs/20261011_risk_calibration_4aca_source16/*.json",
    "outputs/runs/20261014_external_calibration_1776_source16/*.csv",
    "outputs/runs/20261014_risk_calibration_1776_source16/*.csv",
    "outputs/runs/20261014_risk_calibration_1776_source16/*.json",
    "outputs/runs/20261012_fpfh_17pair_diag/*/*.csv",
    "outputs/runs/20261012_fpfh_17pair_diag/*/*.json",
    "outputs/runs/20261012_fcgf_17pair_diag/*/*.csv",
    "outputs/runs/20261012_fcgf_17pair_diag/*/*.json",
    "outputs/runs/20261012_baseline_label_transfer/*.csv",
    "outputs/runs/20261012_baseline_label_transfer/*.json",
    "outputs/runs/20261012_baseline_label_transfer/*.tex",
]


def manifest_files(root: Path) -> list[str]:
    paths = list(FILES)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/20261008_submission_manifest.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    entries = []
    for relative in manifest_files(root):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "status": "complete",
        "format": "portable_sha256_manifest_v1",
        "absolute_paths_used": False,
        "files": entries,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries to {output}")


if __name__ == "__main__":
    main()
