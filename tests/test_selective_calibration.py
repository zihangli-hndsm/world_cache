import numpy as np
import pandas as pd

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evaluate_selective_calibration import evaluate, fit_threshold, summarize_cross_fitted  # noqa: E402


def test_fit_threshold_uses_tied_score_groups_and_evaluates_selection():
    frame = pd.DataFrame({
        "semantic_correct": [True, True, False, False],
        "instance_correct": [True, True, False, False],
    })
    threshold = fit_threshold(frame, np.asarray([0.9, 0.8, 0.8, 0.1]), 0.66, 2)

    assert threshold == (0.8, 3, 2 / 3)
    metrics = evaluate(frame, np.asarray([0.9, 0.8, 0.8, 0.1]), threshold[0])
    assert metrics["queries"] == 3
    assert metrics["semantic_precision"] == 2 / 3


def test_cross_fitted_summary_pools_only_held_out_pairs():
    result = pd.DataFrame({
        "precision_target": [0.9, 0.9],
        "held_pair": ["a", "b"],
        "score": ["low_spatial_residual", "low_spatial_residual"],
        "status": ["calibrated", "calibrated"],
        "held_queries": [2, 1],
        "held_coverage": [0.5, 0.25],
        "held_semantic_precision": [1.0, 0.0],
        "held_instance_precision": [1.0, 0.0],
    })
    source = {"a": pd.DataFrame(index=range(4)), "b": pd.DataFrame(index=range(4))}

    summary = summarize_cross_fitted(result, source).iloc[0]

    assert summary.status == "complete"
    assert summary.selected_queries == 3
    assert summary.coverage == 3 / 8
    assert summary.weighted_semantic_precision == 2 / 3
    assert summary.pairs_below_precision_target == 1
