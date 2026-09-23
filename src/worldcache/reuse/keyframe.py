"""Pose/route-indexed interpolation cache for static-scene embeddings."""

from __future__ import annotations

from bisect import bisect_left

import torch


class KeyframeBank:
    """Interpolate cached tensors between ordered scalar route coordinates.

    The caller maps a known camera pose to a scalar route coordinate.  Queries
    outside the stored interval are rejected so a caller can trigger a real
    backbone refresh instead of silently extrapolating.
    """

    def __init__(self, coordinates: list[float], outputs: list[torch.Tensor]) -> None:
        if len(coordinates) != len(outputs) or len(coordinates) < 2:
            raise ValueError("KeyframeBank needs at least two coordinate/output pairs")
        if any(right <= left for left, right in zip(coordinates, coordinates[1:])):
            raise ValueError("coordinates must be strictly increasing")
        shape = outputs[0].shape
        if any(output.shape != shape for output in outputs):
            raise ValueError("all cached outputs must have the same shape")
        self.coordinates = tuple(float(value) for value in coordinates)
        self.outputs = tuple(output.detach() for output in outputs)

    @torch.inference_mode()
    def query(self, coordinate: float) -> torch.Tensor:
        if coordinate < self.coordinates[0] or coordinate > self.coordinates[-1]:
            raise ValueError("query coordinate is outside the cached keyframe interval")
        right = bisect_left(self.coordinates, coordinate)
        if right == 0:
            return self.outputs[0].clone()
        if right == len(self.coordinates):
            return self.outputs[-1].clone()
        if self.coordinates[right] == coordinate:
            return self.outputs[right].clone()
        left = right - 1
        alpha = (coordinate - self.coordinates[left]) / (self.coordinates[right] - self.coordinates[left])
        return torch.lerp(self.outputs[left], self.outputs[right], alpha)
