"""Camera geometry, visibility, and patch-token correspondence."""

from .projection import (
    backproject_pixels,
    invert_rigid_transform,
    project_world_points,
    transform_points,
)
from .visibility import depth_visibility_mask

__all__ = [
    "backproject_pixels",
    "depth_visibility_mask",
    "invert_rigid_transform",
    "project_world_points",
    "transform_points",
]

