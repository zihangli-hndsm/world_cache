"""Torch-only pinhole camera operations using camera-to-world poses.

Pixel coordinates follow image convention: x/right is u, y/down is v. Depth is
positive metric z in the camera coordinate frame. A pose named ``T_wc`` maps
camera-frame homogeneous points to world-frame homogeneous points.
"""

from __future__ import annotations

import torch


def _as_homogeneous(points: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError(f"expected [..., 3] points, got {tuple(points.shape)}")
    return torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)


def invert_rigid_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert one or more 4x4 rigid transforms without a general matrix inverse."""
    if transform.shape[-2:] != (4, 4):
        raise ValueError("transform must end in [4, 4]")
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3:4]
    result = torch.eye(4, dtype=transform.dtype, device=transform.device).expand_as(transform).clone()
    result[..., :3, :3] = rotation.transpose(-1, -2)
    result[..., :3, 3:4] = -rotation.transpose(-1, -2) @ translation
    return result


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 transform to [..., 3] points."""
    homogeneous = _as_homogeneous(points)
    return (homogeneous @ transform.transpose(-1, -2))[..., :3]


def backproject_pixels(
    pixels_uv: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor, camera_to_world: torch.Tensor
) -> torch.Tensor:
    """Lift image pixels with metric z depth into world coordinates.

    Args:
        pixels_uv: ``[N, 2]`` pixel centers in (u, v) order.
        depth: ``[N]`` positive camera-z values.
        intrinsics: ``[3, 3]`` pinhole calibration matrix.
        camera_to_world: ``[4, 4]`` transform.
    """
    if pixels_uv.ndim != 2 or pixels_uv.shape[-1] != 2:
        raise ValueError("pixels_uv must have shape [N, 2]")
    if depth.shape != pixels_uv.shape[:1]:
        raise ValueError("depth must have shape [N]")
    rays = torch.cat((pixels_uv, torch.ones_like(pixels_uv[:, :1])), dim=-1)
    camera_points = (rays @ torch.linalg.inv(intrinsics).transpose(-1, -2)) * depth[:, None]
    return transform_points(camera_points, camera_to_world)


def project_world_points(
    world_points: torch.Tensor, intrinsics: torch.Tensor, camera_to_world: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points to pixels and return ``(pixels_uv, camera_z)``."""
    world_to_camera = invert_rigid_transform(camera_to_world)
    camera_points = transform_points(world_points, world_to_camera)
    camera_z = camera_points[:, 2]
    safe_z = camera_z.clamp_min(torch.finfo(camera_z.dtype).eps)
    normalized = camera_points / safe_z[:, None]
    pixels_h = normalized @ intrinsics.transpose(-1, -2)
    return pixels_h[:, :2], camera_z

