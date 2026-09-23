import torch

from worldcache.geometry.projection import backproject_pixels, project_world_points


def test_projection_round_trip() -> None:
    intrinsics = torch.tensor([[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, -2.0, 0.5])
    pixels = torch.tensor([[0.5, 1.5], [80.0, 60.0], [159.5, 119.5]])
    depth = torch.tensor([1.0, 2.0, 3.0])
    world = backproject_pixels(pixels, depth, intrinsics, pose)
    projected, recovered_depth = project_world_points(world, intrinsics, pose)
    torch.testing.assert_close(projected, pixels, atol=1e-5, rtol=0)
    torch.testing.assert_close(recovered_depth, depth, atol=1e-5, rtol=0)


def test_identity_pose_alignment() -> None:
    intrinsics = torch.eye(3)
    pixels = torch.tensor([[2.0, 3.0], [7.0, 11.0]])
    depth = torch.tensor([1.0, 4.0])
    world = backproject_pixels(pixels, depth, intrinsics, torch.eye(4))
    projected, camera_depth = project_world_points(world, intrinsics, torch.eye(4))
    torch.testing.assert_close(projected, pixels)
    torch.testing.assert_close(camera_depth, depth)

