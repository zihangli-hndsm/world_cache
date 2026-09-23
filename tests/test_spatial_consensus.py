import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evaluate_3rscan_spatial_consensus import consensus  # noqa: E402


def test_consensus_recovers_rigid_transform_and_unique_assignment():
    target = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [0.2, -0.1, 0.3]
    source = target - matrix[:3, 3]
    candidate_indices = np.tile(np.arange(len(target)), (len(source), 1))

    estimated, selected, inliers = consensus(
        source,
        target,
        candidate_indices,
        threshold=1e-8,
        iterations=200,
        seed=7,
        assignment="unique",
    )

    assert inliers.all()
    assert np.allclose(estimated, matrix, atol=1e-8)
    assert len(set(selected[inliers].tolist())) == int(inliers.sum())
