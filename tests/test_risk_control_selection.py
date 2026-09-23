from __future__ import annotations

import pandas as pd

from scripts.audit_risk_control_selection import familywise_scan


def test_familywise_scan_rejects_in_sample_prefix_selection_when_correction_is_strict() -> None:
    frame = pd.DataFrame({
        "spatial_residual_m": [0.01, 0.02, 0.03, 0.04, 0.05],
        "semantic_correct": [True, True, True, False, False],
    })
    result = familywise_scan(frame, precision_target=0.90, confidence_level=0.95, min_queries=2)
    assert result["candidate_prefixes"] == 4
    assert result["familywise_feasible"] is False


def test_familywise_scan_can_pass_with_a_large_clean_prefix() -> None:
    frame = pd.DataFrame({
        "spatial_residual_m": [0.01] * 200,
        "semantic_correct": [True] * 200,
    })
    result = familywise_scan(frame, precision_target=0.90, confidence_level=0.95, min_queries=100)
    assert result["familywise_feasible"] is True
