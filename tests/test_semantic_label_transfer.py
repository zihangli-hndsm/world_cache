import numpy as np
import pandas as pd

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evaluate_semantic_label_transfer import mean_iou, metrics  # noqa: E402


def test_mean_iou_and_transfer_metrics_are_class_balanced():
    assert mean_iou(np.asarray([1, 1, 2]), np.asarray([1, 2, 2])) == (0.5 + 0.5) / 2
    frame = pd.DataFrame({
        "query_semantic_id": [1, 1, 2],
        "selected_reference_semantic_id": [1, 2, 2],
        "query_instance_id": [10, 10, 20],
        "selected_reference_instance_id": [10, 20, 20],
    })
    result = metrics(frame, np.asarray([True, True, False]))
    assert result["queries"] == 2
    assert result["coverage"] == 2 / 3
    assert result["semantic_accuracy"] == 0.5
