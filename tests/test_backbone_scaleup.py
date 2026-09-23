import pandas as pd

from scripts.audit_backbone_scaleup import bootstrap_delta


def test_paired_scale_bootstrap_is_zero_for_identical_methods():
    frame = pd.DataFrame({
        "pair": ["a", "b", "c"],
        "inliers": [10, 20, 30],
        "inlier_coverage": [0.5, 0.6, 0.7],
        "inlier_semantic_accuracy": [0.8, 0.9, 1.0],
    })

    result = bootstrap_delta(frame, frame.copy(), samples=200, seed=17)

    assert result["semantic_delta"] == 0.0
    assert result["semantic_delta_ci95_lower"] == 0.0
    assert result["semantic_delta_ci95_upper"] == 0.0
    assert result["coverage_delta"] == 0.0


def test_paired_scale_bootstrap_counts_pairwise_direction():
    vitl = pd.DataFrame({
        "pair": ["a", "b"],
        "inliers": [10, 10],
        "inlier_coverage": [0.5, 0.5],
        "inlier_semantic_accuracy": [0.5, 1.0],
    })
    vitg = vitl.copy()
    vitg["inlier_semantic_accuracy"] = [1.0, 0.5]

    result = bootstrap_delta(vitl, vitg, samples=200, seed=17)

    assert result["semantic_pair_positive"] == 1
    assert result["semantic_pair_negative"] == 1
    assert result["semantic_pair_ties"] == 0
