import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evaluate_selective_correspondence import risk_curve  # noqa: E402


def test_risk_curve_orders_high_scores_and_preserves_coverage():
    frame = pd.DataFrame({
        "semantic_correct": [True, False, True, False],
        "instance_correct": [True, False, True, False],
        "consensus_inlier": [True, True, False, False],
    })

    result = risk_curve(frame, "source", "score", np.asarray([0.1, 0.9, 0.8, 0.2]))

    at_half = result[result["coverage"] == 0.5].iloc[0]
    assert at_half["queries"] == 2
    assert at_half["semantic_precision"] == 0.5
    assert at_half["instance_precision"] == 0.5
    assert at_half["consensus_inlier_fraction"] == 0.5
