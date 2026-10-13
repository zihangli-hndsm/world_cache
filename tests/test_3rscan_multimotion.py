import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evaluate_3rscan_multimotion import (  # noqa: E402
    PairFeatures,
    consensus_variable_k,
    fit_fixed_consensus,
    fit_modes,
    assign_modes,
    run_inference_arms,
)
from analyze_3rscan_multimotion_lopo import evaluate_mode_prefix  # noqa: E402
from probe_3rscan_mode_interventions import paired_intervention  # noqa: E402
from audit_3rscan_moved_instance_candidates import (  # noqa: E402
    candidate_hit_counts,
    validate_candidate_query_alignment,
)
from probe_3rscan_mode_margin_lopo import gated_mode_counts  # noqa: E402
from evaluate_3rscan_spatial_consensus import consensus  # noqa: E402


def synthetic_pair(seed=4):
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(80, 3))
    query = reference[:40] + np.asarray([0.3, -0.2, 0.1])
    candidates = np.empty((len(query), 50), dtype=np.int64)
    for index in range(len(query)):
        others = rng.permutation(np.delete(np.arange(len(reference)), index))[:49]
        # Give RANSAC several equivalent candidate slots for the ground-truth
        # correspondence so this geometry unit test does not test luck.
        candidates[index] = np.concatenate([np.full(20, index), others[:30]])
    scores = np.tile(np.linspace(1.0, 0.0, 50), (len(query), 1))
    return PairFeatures(
        pair="synthetic",
        reference_id="ref",
        rescan_id="rescan",
        pair_run=Path("."),
        reference_points=reference,
        query_points=query,
        query_frame_ids=np.arange(len(query)) % 4,
        candidate_indices=candidates,
        candidate_scores=scores,
        top20_reference_points=reference[candidates[:, :20]],
        top50_reference_points=reference[candidates],
        gap20=scores[:, 0] - scores[:, 19],
    )


def test_baseline_wrapper_is_exactly_the_existing_consensus():
    pair = synthetic_pair()
    direct = consensus(
        pair.query_points,
        pair.reference_points,
        pair.candidate_indices[:, :20],
        threshold=1e-7,
        iterations=400,
        seed=17,
    )
    wrapped = fit_fixed_consensus(
        pair.query_points,
        pair.reference_points,
        pair.candidate_indices[:, :20],
        threshold=1e-7,
        iterations=400,
        seed=17,
    )
    assert np.array_equal(wrapped.selected, direct[1])
    assert np.array_equal(wrapped.inliers, direct[2])
    assert np.allclose(wrapped.matrix, direct[0])


def test_variable_k_consensus_recovers_a_synthetic_rigid_transform():
    pair = synthetic_pair()
    result = consensus_variable_k(
        pair.query_points,
        pair.reference_points,
        pair.candidate_indices,
        np.full(len(pair.query_points), 20, dtype=np.int64),
        threshold=1e-7,
        iterations=400,
        seed=17,
    )
    repeated = consensus_variable_k(
        pair.query_points,
        pair.reference_points,
        pair.candidate_indices,
        np.full(len(pair.query_points), 20, dtype=np.int64),
        threshold=1e-7,
        iterations=400,
        seed=17,
    )
    assert result.inliers.all()
    assert np.allclose(result.matrix[:3, 3], [-0.3, 0.2, -0.1], atol=1e-7)
    assert np.array_equal(result.selected, repeated.selected)
    assert np.array_equal(result.inliers, repeated.inliers)
    assert np.allclose(result.matrix, repeated.matrix)


def test_residual_consensus_recovers_two_synthetic_rigid_motions():
    rng = np.random.default_rng(21)
    query = rng.normal(size=(60, 3))
    reference = query.copy()
    reference[30:] += np.asarray([1.0, 0.0, 0.0])
    candidates = np.empty((60, 50), dtype=np.int64)
    for index in range(60):
        correct_index = index
        distractors = rng.permutation(np.delete(np.arange(60), correct_index))[:30]
        candidates[index] = np.concatenate([np.full(20, correct_index), distractors])
    scores = np.tile(np.linspace(1.0, 0.0, 50), (60, 1))
    pair = PairFeatures(
        pair="two-motion",
        reference_id="ref",
        rescan_id="rescan",
        pair_run=Path("."),
        reference_points=reference,
        query_points=query,
        query_frame_ids=np.arange(60) % 3,
        candidate_indices=candidates,
        candidate_scores=scores,
        top20_reference_points=reference[candidates[:, :20]],
        top50_reference_points=reference[candidates],
        gap20=scores[:, 0] - scores[:, 19],
    )
    baseline = fit_fixed_consensus(query, reference, candidates[:, :20], 0.01, 300, 17)
    modes = fit_modes(
        pair,
        np.full(60, 20, dtype=np.int64),
        threshold=0.01,
        iterations=300,
        secondary_iterations=300,
        seed=17,
        max_modes=3,
        min_support=8,
        fixed_baseline=baseline,
    )
    assignment, _, _, inliers, _, _ = assign_modes(pair, np.full(60, 20), modes, 0.01)
    assert len(modes) == 2
    assert inliers.all()
    assert set(assignment.tolist()) == {0, 1}


def test_all_arms_keep_the_same_query_population_and_inference_has_no_labels():
    pair = synthetic_pair()
    arms = run_inference_arms(
        pair,
        adaptive_threshold=0.05,
        threshold=1e-7,
        iterations=400,
        secondary_iterations=100,
        seed=17,
        max_modes=2,
        min_support=6,
    )
    assert set(arms) == {"global_ransac", "multi_motion", "adaptive_k", "combined"}
    assert all(len(arm["selected"]) == len(pair.query_points) for arm in arms.values())
    assert all(len(arm["assigned"]) == len(pair.query_points) for arm in arms.values())
    assert "annotation_root" not in inspect.signature(run_inference_arms).parameters
    assert "metadata" not in inspect.signature(run_inference_arms).parameters
    assert not hasattr(pair, "query_object_ids")


def test_mode_prefix_reassigns_queries_instead_of_filtering_full_model_assignments():
    import pandas as pd

    frame = pd.DataFrame({
        "query_evaluable": [True, True],
        "official_rigid_reference_instance_id": [2, 3],
        "hypothesis_0_residual_m": [0.02, 0.05],
        "hypothesis_0_selected_reference_instance_id": [1, 3],
        "hypothesis_1_residual_m": [0.03, 0.02],
        "hypothesis_1_selected_reference_instance_id": [2, 4],
    })
    one_mode = evaluate_mode_prefix(frame, 1)
    two_modes = evaluate_mode_prefix(frame, 2)
    assert one_mode["correct_transfers"] == 1
    assert two_modes["correct_transfers"] == 0


def test_paired_intervention_aligns_queries_and_counts_rescues_and_losses():
    import pandas as pd

    method = pd.DataFrame({
        "query_index": [0, 1],
        "query_evaluable": [True, True],
        "official_rigid_reference_instance_id": [2, 3],
        "hypothesis_0_residual_m": [0.05, 0.02],
        "hypothesis_0_selected_reference_instance_id": [1, 3],
        "hypothesis_1_residual_m": [0.02, 0.01],
        "hypothesis_1_selected_reference_instance_id": [2, 4],
    })
    baseline = method.drop(columns=[
        "hypothesis_1_residual_m", "hypothesis_1_selected_reference_instance_id"
    ]).copy()
    paired = paired_intervention(method, baseline, mode_count=2)
    assert paired["query_set_match"] is True
    assert paired["rescued_vs_baseline"] == 1
    assert paired["lost_vs_baseline"] == 1
    assert paired["net_gain"] == 0


def test_exact_object_candidate_recall_uses_official_local_instance_mapping():
    candidate_indices = np.asarray([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    reference_object_ids = np.asarray([7, 9, 11])
    expected = np.asarray([9, 7, -1])
    evaluable = np.asarray([True, True, True])
    result = candidate_hit_counts(
        candidate_indices,
        reference_object_ids,
        expected,
        evaluable,
        k_values=(1, 2, 3),
    )
    assert result["moved_rigid_queries"] == 2
    assert result["top1_hits"] == 0
    assert result["top2_hits"] == 2
    assert result["top3_hits"] == 2


def test_candidate_audit_requires_exact_query_and_frame_order():
    candidates = np.zeros((3, 5), dtype=np.int64)
    frame_ids = np.asarray([4, 4, 5])
    validate_candidate_query_alignment(np.arange(3), frame_ids, candidates, frame_ids.copy())
    with pytest.raises(ValueError, match="incomplete or misordered"):
        validate_candidate_query_alignment(np.asarray([0, 2, 1]), frame_ids, candidates, frame_ids)
    with pytest.raises(ValueError, match="frame order mismatch"):
        validate_candidate_query_alignment(np.arange(3), frame_ids, candidates, np.asarray([4, 5, 4]))


def test_residual_switch_margin_can_reject_a_worse_motion_hypothesis():
    import pandas as pd

    frame = pd.DataFrame({
        "query_evaluable": [True, True],
        "official_rigid_reference_instance_id": [2, 3],
        "hypothesis_0_residual_m": [0.05, 0.03],
        "hypothesis_0_selected_reference_instance_id": [1, 3],
        "hypothesis_1_residual_m": [0.02, 0.01],
        "hypothesis_1_selected_reference_instance_id": [2, 4],
    })
    conservative = gated_mode_counts(frame, mode_count=2, switch_margin_m=0.03)
    permissive = gated_mode_counts(frame, mode_count=2, switch_margin_m=0.0)
    assert conservative["correct_transfers"] == 2
    assert permissive["correct_transfers"] == 1
