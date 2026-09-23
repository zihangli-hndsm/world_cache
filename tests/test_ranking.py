import pytest
import torch

from worldcache.reuse.ranking import fuse_descriptor_geometry_ranks, fuse_descriptor_geometry_scores


def test_geometry_descriptor_fusion_penalizes_far_candidates() -> None:
    scores = torch.tensor([[0.80, 0.79]])
    distances = torch.tensor([[0.40, 0.05]])
    fused = fuse_descriptor_geometry_scores(scores, distances, alpha=1.0)
    assert int(torch.argmax(scores)) == 0
    assert int(torch.argmax(fused)) == 1


def test_geometry_descriptor_fusion_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        fuse_descriptor_geometry_scores(torch.zeros(1, 2), torch.zeros(1, 3))
    with pytest.raises(ValueError):
        fuse_descriptor_geometry_scores(torch.zeros(1, 2), torch.zeros(1, 2), alpha=-1.0)


def test_rank_fusion_is_scale_invariant() -> None:
    scores = torch.tensor([[0.80, 0.79, 0.20]])
    distances = torch.tensor([[0.40, 0.05, 0.10]])
    scaled = torch.tensor([[80.0, 79.0, 20.0]])
    first = torch.argmax(fuse_descriptor_geometry_ranks(scores, distances, beta=1.0))
    second = torch.argmax(fuse_descriptor_geometry_ranks(scaled, distances * 100.0, beta=1.0))
    assert int(first) == int(second)
