import numpy as np

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from bootstrap_pair_statistics import bootstrap_ci  # noqa: E402


def test_bootstrap_ci_is_deterministic_and_contains_point_estimate():
    values = np.asarray([0.2, 0.4, 0.6, 0.8])
    first = bootstrap_ci(values, np.mean, seed=3, samples=1000)
    second = bootstrap_ci(values, np.mean, seed=3, samples=1000)
    assert first == second
    assert first[1] <= first[0] <= first[2]
