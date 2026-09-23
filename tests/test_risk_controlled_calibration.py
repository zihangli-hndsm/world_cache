from __future__ import annotations

from scripts.evaluate_risk_controlled_calibration import fit_threshold, lower_confidence_bound

import pandas as pd


def test_exact_lower_bound_is_conservative() -> None:
    assert lower_confidence_bound(9, 10, 0.95) < 0.9
    assert lower_confidence_bound(100, 100, 0.95) > 0.95


def test_fit_threshold_chooses_largest_valid_prefix() -> None:
    frame = pd.DataFrame({
        "spatial_residual_m": [0.01, 0.02, 0.03, 0.04],
        "semantic_correct": [True, True, True, False],
    })
    fit = fit_threshold(frame, precision_target=0.3, confidence_level=0.95, min_queries=2)
    assert fit["queries"] == 3
    assert fit["threshold"] == 0.03
