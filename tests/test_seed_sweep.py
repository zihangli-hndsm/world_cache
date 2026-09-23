import pandas as pd
import pytest

from scripts.aggregate_seed_sweep import summarize_pair_stability, summarize_seed_metrics


def test_seed_summary_uses_inlier_weighting():
    frame = pd.DataFrame([
        {
            "split": "source", "pair": "a", "ransac_seed": 17,
            "queries": 10, "inliers": 2, "inlier_coverage": 0.2,
            "inlier_semantic_accuracy": 0.5, "inlier_instance_accuracy": 1.0,
            "rotation_error_deg": 1.0, "translation_error_m": 0.1,
        },
        {
            "split": "source", "pair": "b", "ransac_seed": 17,
            "queries": 10, "inliers": 8, "inlier_coverage": 0.8,
            "inlier_semantic_accuracy": 1.0, "inlier_instance_accuracy": 0.75,
            "rotation_error_deg": 10.0, "translation_error_m": 1.0,
        },
    ])

    result = summarize_seed_metrics(frame, "source").iloc[0]

    assert result.weighted_coverage == 0.5
    assert result.weighted_semantic_precision == 0.9
    assert result.strict_pose_successes == 1


def test_pair_stability_reports_ranges_and_pose_count():
    frame = pd.DataFrame([
        {
            "split": "target", "pair": "target", "ransac_seed": 17,
            "inlier_coverage": 0.5, "inlier_semantic_accuracy": 0.8,
            "rotation_error_deg": 2.0, "translation_error_m": 0.1,
        },
        {
            "split": "target", "pair": "target", "ransac_seed": 29,
            "inlier_coverage": 0.6, "inlier_semantic_accuracy": 0.9,
            "rotation_error_deg": 8.0, "translation_error_m": 0.1,
        },
    ])

    result = summarize_pair_stability(frame).iloc[0]

    assert result.semantic_range == pytest.approx(0.1)
    assert result.coverage_range == pytest.approx(0.1)
    assert result.strict_pose_successes == 1
