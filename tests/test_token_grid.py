import torch

from worldcache.geometry.correspondence import patch_sample_pixels, pixel_to_patch_index


def test_patch_index_mapping() -> None:
    pixels = torch.tensor([[0.0, 0.0], [13.9, 13.9], [14.0, 0.0], [223.9, 223.9], [224.0, 0.0]])
    indices, valid = pixel_to_patch_index(pixels, image_height=224, image_width=224, patch_size=14)
    assert indices.tolist() == [0, 0, 1, 255, -1]
    assert valid.tolist() == [True, True, True, True, False]


def test_patch_sample_pixels_stay_in_patch() -> None:
    pixels = patch_sample_pixels(2, 3, patch_size=14, samples_per_axis=4)
    assert pixels.shape == (16, 2)
    assert torch.all((pixels[:, 0] >= 42) & (pixels[:, 0] < 56))
    assert torch.all((pixels[:, 1] >= 28) & (pixels[:, 1] < 42))
