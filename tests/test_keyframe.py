import pytest
import torch

from worldcache.reuse.keyframe import KeyframeBank


def test_keyframe_bank_interpolates_tensor() -> None:
    bank = KeyframeBank([0.0, 2.0], [torch.zeros(1, 2), torch.full((1, 2), 2.0)])
    assert torch.equal(bank.query(0.0), torch.zeros(1, 2))
    assert torch.equal(bank.query(1.0), torch.ones(1, 2))
    assert torch.equal(bank.query(2.0), torch.full((1, 2), 2.0))


def test_keyframe_bank_rejects_outside_and_invalid_coordinates() -> None:
    with pytest.raises(ValueError):
        KeyframeBank([0.0], [torch.zeros(1)])
    with pytest.raises(ValueError):
        KeyframeBank([0.0, 0.0], [torch.zeros(1), torch.ones(1)])
    bank = KeyframeBank([0.0, 1.0], [torch.zeros(1), torch.ones(1)])
    with pytest.raises(ValueError):
        bank.query(-0.1)
