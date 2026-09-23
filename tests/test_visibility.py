import torch

from worldcache.geometry.visibility import depth_visibility_mask


def test_depth_visibility() -> None:
    depth = torch.full((5, 6), 2.0)
    pixels = torch.tensor([[1.0, 1.0], [2.0, 2.0], [-1.0, 0.0], [5.0, 4.0]])
    projected_depth = torch.tensor([2.01, 1.5, 2.0, -1.0])
    visible = depth_visibility_mask(pixels, projected_depth, depth, absolute_tolerance_m=0.02)
    assert visible.tolist() == [True, False, False, False]

