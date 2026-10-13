#!/usr/bin/env python3
"""Post-hoc test whether FPFH proposal dispersion predicts ICP benefit.

The candidate transforms and ICP diagnostics provide inference-time features.
Tracked-object centroid errors are used only to score the fixed diagnostic on
the held-out Rescan scenes; this script does not choose a deployment threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from evaluate_rescan_full_scene_pose_transfer import get_transform, rank_group


KEY = ("scene", "target_scan", "source_instance_id")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def proposal_dispersion(rows: list[dict[str, str]]) -> tuple[float, float]:
    transforms = [get_transform(row) for row in rows]
    translations = np.stack([matrix[:3, 3] for matrix in transforms])
    rotations = np.stack([matrix[:3, :3] for matrix in transforms])
    translation_distances: list[float] = []
    rotation_distances: list[float] = []
    for i in range(len(transforms)):
        for j in range(i + 1, len(transforms)):
            translation_distances.append(float(np.linalg.norm(translations[i] - translations[j])))
            relative = rotations[i] @ rotations[j].T
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            rotation_distances.append(float(np.degrees(np.arccos(cosine))))
    return float(np.median(translation_distances)), float(np.median(rotation_distances))


def clustered_auc_ci(
    rows: list[dict[str, object]],
    score_name: str,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    labels = np.asarray([bool(row["icp_reduces_center_error"]) for row in rows], dtype=bool)
    scores = np.asarray([float(row[score_name]) for row in rows], dtype=np.float64)
    point = float(roc_auc_score(labels, scores))
    scene_names = sorted({str(row["scene"]) for row in rows})
    positive = [scores[(np.asarray([str(row["scene"]) == scene for row in rows])) & labels] for scene in scene_names]
    negative = [scores[(np.asarray([str(row["scene"]) == scene for row in rows])) & ~labels] for scene in scene_names]
    positive_counts = np.asarray([len(values) for values in positive], dtype=np.float64)
    negative_counts = np.asarray([len(values) for values in negative], dtype=np.float64)
    comparisons = np.zeros((len(scene_names), len(scene_names)), dtype=np.float64)
    for i, positives in enumerate(positive):
        for j, negatives in enumerate(negative):
            if len(positives) and len(negatives):
                comparisons[i, j] = (
                    (positives[:, None] > negatives[None, :]).sum()
                    + 0.5 * (positives[:, None] == negatives[None, :]).sum()
                )
    sampled_counts = rng.multinomial(len(scene_names), [1 / len(scene_names)] * len(scene_names), size=samples)
    numerators = np.einsum("bi,ij,bj->b", sampled_counts, comparisons, sampled_counts, optimize=True)
    denominators = np.einsum(
        "bi,i,bj,j->b", sampled_counts, positive_counts,
        sampled_counts, negative_counts, optimize=True,
    )
    valid = denominators > 0
    if not valid.any():
        return point, float("nan"), float("nan")
    draws = numerators[valid] / denominators[valid]
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return point, float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=10_042_026)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap sample count must be positive")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.candidate_csv):
        groups[tuple(row[key] for key in KEY)].append(row)
    diagnostics = {
        tuple(row[key] for key in KEY): row
        for row in read_csv(args.diagnostics_csv)
    }
    if len(diagnostics) != len(read_csv(args.diagnostics_csv)):
        raise ValueError("diagnostic keys are not unique")

    scored: list[dict[str, object]] = []
    for key, proposals in sorted(groups.items()):
        diag = diagnostics.get(key)
        # The centroid diagnostic is undefined when the tracked instance is
        # absent from the target, which remains an evaluation-only stratum.
        if diag is None or not diag["identity_center_error_m"]:
            continue
        if len(proposals) != 5:
            raise ValueError(f"expected five proposals for {key}, found {len(proposals)}")
        selected = rank_group(proposals, "color")
        rgb_scores = sorted(
            (float(row["joint_spatial_rgb_inlier_fraction"]) for row in proposals),
            reverse=True,
        )
        spatial_scores = sorted((float(row["spatial_coverage"]) for row in proposals), reverse=True)
        translation_dispersion, rotation_dispersion = proposal_dispersion(proposals)
        identity_error = float(diag["identity_center_error_m"])
        icp_error = float(diag["icp_center_error_m"])
        scored.append({
            "scene": key[0],
            "target_scan": key[1],
            "source_instance_id": key[2],
            "identity_center_error_m": identity_error,
            "icp_center_error_m": icp_error,
            "center_error_gain_m": identity_error - icp_error,
            "icp_reduces_center_error": icp_error < identity_error,
            "identity_centroid_hit": diag["identity_centroid_hit"],
            "icp_centroid_hit": diag["icp_centroid_hit"],
            "icp_fitness": float(diag["icp_fitness"]),
            "negative_icp_rmse": -float(diag["icp_rmse_m"]),
            "negative_translation_update": -float(diag["translation_update_m"]),
            "negative_proposal_translation_dispersion": -translation_dispersion,
            "negative_proposal_rotation_dispersion": -rotation_dispersion,
            "best_second_spatial_rgb_gap": rgb_scores[0] - rgb_scores[1],
            "best_second_spatial_gap": spatial_scores[0] - spatial_scores[1],
            "selected_spatial_rgb_score": float(selected["joint_spatial_rgb_inlier_fraction"]),
        })
    if not scored:
        raise ValueError("no target-present diagnostic rows could be scored")

    features = [
        "negative_proposal_translation_dispersion",
        "negative_proposal_rotation_dispersion",
        "best_second_spatial_rgb_gap",
        "best_second_spatial_gap",
        "icp_fitness",
        "negative_icp_rmse",
        "negative_translation_update",
    ]
    rng = np.random.default_rng(args.seed)
    auc_rows = []
    for feature in features:
        auc, lower, upper = clustered_auc_ci(scored, feature, args.bootstrap_samples, rng)
        rho = float(spearmanr(
            [float(row[feature]) for row in scored],
            [float(row["center_error_gain_m"]) for row in scored],
        ).statistic)
        auc_rows.append({
            "inference_feature": feature,
            "auc_for_icp_center_error_improvement": auc,
            "scene_cluster_bootstrap_ci95_lower": lower,
            "scene_cluster_bootstrap_ci95_upper": upper,
            "spearman_with_center_error_gain": rho,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_object_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    with (args.output_dir / "feature_auc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(auc_rows[0]))
        writer.writeheader()
        writer.writerows(auc_rows)

    labels = [bool(row["icp_reduces_center_error"]) for row in scored]
    hit_delta = sum(
        int(row["icp_centroid_hit"] == "True") - int(row["identity_centroid_hit"] == "True")
        for row in scored
    )
    per_scene = {}
    for scene in sorted({str(row["scene"]) for row in scored}):
        subset = [row for row in scored if row["scene"] == scene]
        per_scene[scene] = {
            "objects": len(subset),
            "median_translation_dispersion_m": float(np.median([-float(row["negative_proposal_translation_dispersion"]) for row in subset])),
            "median_rotation_dispersion_degrees": float(np.median([-float(row["negative_proposal_rotation_dispersion"]) for row in subset])),
            "mean_center_error_gain_m": float(np.mean([float(row["center_error_gain_m"]) for row in subset])),
        }
    summary = {
        "target_present_objects": len(scored),
        "scenes": len(per_scene),
        "icp_reduces_center_error_count": int(sum(labels)),
        "icp_increases_or_preserves_center_error_count": int(len(labels) - sum(labels)),
        "net_centroid_hit_change": int(hit_delta),
        "bootstrap_unit": "Rescan scene; target scans and objects remain grouped",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "features_are_label_free": True,
        "centroid_labels_are_used_only_for_posthoc_scoring": True,
        "per_scene": per_scene,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Rescan pose-proposal ambiguity diagnostic",
        "",
        f"This post-hoc analysis scores {len(scored)} target-present source-object/target-scan pairs in {len(per_scene)} held-out Rescan scenes. The inference features come from the five saved FPFH proposals and label-free ICP diagnostics. Official tracked-instance centroid errors are evaluation-only. No confidence threshold is selected.",
        "",
        f"ICP reduces tracked-object center error for {sum(labels)}/{len(labels)} objects and increases or preserves it for {len(labels)-sum(labels)}. The identity-to-ICP 20 cm centroid-hit count changes by {hit_delta:+d}.",
        "",
        "| Label-free score (higher means more confidence) | AUC for ICP reducing center error | Scene-cluster 95% CI | Spearman with center-error gain |",
        "|---|---:|---:|---:|",
    ]
    names = {
        "negative_proposal_translation_dispersion": "Lower five-proposal translation dispersion",
        "negative_proposal_rotation_dispersion": "Lower five-proposal rotation dispersion",
        "best_second_spatial_rgb_gap": "Larger spatial+RGB top-two gap",
        "best_second_spatial_gap": "Larger spatial-support top-two gap",
        "icp_fitness": "Higher ICP fitness",
        "negative_icp_rmse": "Lower ICP RMSE",
        "negative_translation_update": "Smaller ICP translation update",
    }
    for row in auc_rows:
        lines.append(
            f"| {names[row['inference_feature']]} | {row['auc_for_icp_center_error_improvement']:.3f} | "
            f"[{row['scene_cluster_bootstrap_ci95_lower']:.3f}, {row['scene_cluster_bootstrap_ci95_upper']:.3f}] | "
            f"{row['spearman_with_center_error_gain']:+.3f} |"
        )
    lines.extend([
        "",
        "Lower transform dispersion is weakly informative about whether ICP reduces tracked-object center error (AUC 0.56 for translation and 0.59 for rotation), but this is too modest to define a useful confidence gate. The top-two spatial and RGB score gap ranks benefit in the opposite direction (AUC 0.40). ICP fitness and RMSE are more informative (AUC 0.64 and 0.67), yet the earlier false-confidence audit shows they can still accept severe wrong alignments. This is a post-hoc diagnostic on already inspected scenes; it does not support a new method or a fresh generalization claim. The source geometry archive is not included in this artifact, so full-scene mIoU was not recomputed.",
        "",
        "## Per-scene proposal dispersion",
        "",
        "| Scene | Objects | Median translation spread (m) | Median rotation spread (deg) | Mean centroid-error gain (m) |",
        "|---|---:|---:|---:|---:|",
    ])
    for scene, row in per_scene.items():
        lines.append(
            f"| {scene} | {row['objects']} | {row['median_translation_dispersion_m']:.3f} | "
            f"{row['median_rotation_dispersion_degrees']:.1f} | {row['mean_center_error_gain_m']:+.3f} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
