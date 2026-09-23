import pandas as pd

from scripts.aggregate_candidate_set_audit import summarize


def test_candidate_set_summary_separates_candidate_recall_from_consensus_recovery():
    frame = pd.DataFrame([
        {
            "split": "source", "queries": 4, "candidate_semantic_hits": 3,
            "candidate_instance_hits": 2, "inliers": 2,
            "semantic_correct_inliers": 1, "instance_correct_inliers": 1,
            "semantic_miss_inliers": 0,
        },
        {
            "split": "source", "queries": 2, "candidate_semantic_hits": 1,
            "candidate_instance_hits": 1, "inliers": 1,
            "semantic_correct_inliers": 1, "instance_correct_inliers": 1,
            "semantic_miss_inliers": 0,
        },
    ])

    result = summarize(frame, "source")

    assert result["candidate_semantic_recall"] == 4 / 6
    assert result["inlier_coverage"] == 3 / 6
    assert result["semantic_coverage"] == 2 / 6
    assert result["semantic_precision"] == 2 / 3
    assert result["semantic_recovery_given_candidate"] == 2 / 4


def test_candidate_set_summary_preserves_zero_candidate_hit_recovery():
    frame = pd.DataFrame([
        {
            "split": "target", "queries": 3, "candidate_semantic_hits": 0,
            "candidate_instance_hits": 0, "inliers": 1,
            "semantic_correct_inliers": 0, "instance_correct_inliers": 0,
            "semantic_miss_inliers": 1,
        },
    ])

    result = summarize(frame, "target")

    assert result["candidate_semantic_recall"] == 0.0
    assert result["semantic_recovery_given_candidate"] == 0.0
    assert result["semantic_miss_inliers"] == 1
