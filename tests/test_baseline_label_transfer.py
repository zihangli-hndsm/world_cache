import numpy as np
import pandas as pd

from scripts.summarize_baseline_label_transfer import mean_iou, metrics


def test_mean_iou_handles_union_of_labels():
    assert np.isclose(mean_iou(np.array([1, 1, 2]), np.array([1, 2, 2])), 0.5)


def test_metrics_reports_retained_coverage_and_accuracy():
    frame = pd.DataFrame(
        {
            "query_semantic_id": [1, 1, 2],
            "selected_reference_semantic_id": [1, 2, 2],
            "query_instance_id": [10, 10, 20],
            "selected_reference_instance_id": [10, 11, 20],
        }
    )
    result = metrics(frame, np.array([True, False, True]))
    assert result["queries"] == 3
    assert result["retained"] == 2
    assert np.isclose(result["coverage"], 2 / 3)
    assert np.isclose(result["semantic_accuracy"], 1.0)
