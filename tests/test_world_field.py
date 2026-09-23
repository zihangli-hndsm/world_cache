import pytest
import torch

from worldcache.reuse import encode_world_receptive_field


def test_world_receptive_field_preserves_shape_and_normalizes() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    descriptors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pooled = encode_world_receptive_field(points, descriptors, radius=0.10)
    assert pooled.shape == descriptors.shape
    assert torch.allclose(torch.linalg.vector_norm(pooled, dim=1), torch.ones(2), atol=1e-6)
    assert pooled[0, 0] > pooled[0, 1]


def test_world_receptive_field_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        encode_world_receptive_field(torch.zeros(2, 2), torch.zeros(2, 3), radius=0.1)
    with pytest.raises(ValueError):
        encode_world_receptive_field(torch.zeros(2, 3), torch.zeros(3, 3), radius=0.1)
    with pytest.raises(ValueError):
        encode_world_receptive_field(torch.zeros(2, 3), torch.zeros(2, 3), radius=0.0)
