"""Small, dependency-light reader for the official 3RScan RGB-D format.

The reader intentionally does not unpack ``sequence.zip``.  A scan directory
contains the archive plus mesh/annotation files; RGB-D frames are read on
demand from the archive.  Poses in the archive are camera-to-scan transforms.
The metadata transform maps rescan coordinates into the reference scan frame.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _matrix(values: Iterable[float], name: str) -> np.ndarray:
    values = list(values)
    if len(values) != 16:
        raise ValueError(f"{name} must contain 16 values, got {len(values)}")
    return np.asarray(values, dtype=np.float32).reshape(4, 4)


def _metadata_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size != 16:
        raise ValueError(f"{name} must contain 16 values, got shape {array.shape}")
    # 3RScan.json stores the transform as a flattened column-major matrix;
    # pose.txt uses the usual row-major text layout with translation in the
    # last column.  Convert metadata to the same internal convention.  The
    # official FAQ specifies scene-transform translations in millimetres,
    # while camera poses and meshes use metres.
    matrix = array.reshape(4, 4).T
    matrix[:3, 3] *= 0.001
    return matrix


def _read_info(text: str) -> tuple[np.ndarray, float]:
    """Read the standard ``_info.txt`` intrinsics and depth scale."""
    fields: dict[str, list[float]] = {}
    for line in text.splitlines():
        separator = ":" if ":" in line else "=" if "=" in line else None
        if separator is None:
            continue
        key, raw = line.split(separator, 1)
        values = [float(item) for item in _FLOAT_RE.findall(raw)]
        if values:
            fields[key.strip()] = values
    color_values = fields.get("m_calibrationColorIntrinsic")
    depth_values = fields.get("m_calibrationDepthIntrinsic")
    if color_values is None and depth_values is None:
        raise ValueError("_info.txt has no color/depth intrinsic matrix")
    color = _matrix(color_values or depth_values, "color intrinsics")[:3, :3]
    depth = _matrix(depth_values or color_values, "depth intrinsics")[:3, :3]
    depth_extrinsic = _matrix(fields.get("m_calibrationDepthExtrinsic", np.eye(4).reshape(-1)), "depth extrinsic")
    depth_shift = fields.get("m_depthShift", [1000.0])[0]
    return color, depth, depth_extrinsic, float(depth_shift), depth_values is not None


def _zip_member(names: Iterable[str], prefix: str, kind: str, frame_id: int) -> str:
    names = list(names)
    if kind == "color":
        candidates = [
            f"{prefix}frames/color/{frame_id}.jpg",
            f"{prefix}frames/color/{frame_id:06d}.jpg",
            f"{prefix}frame-{frame_id:06d}.color.jpg",
            f"{prefix}frame-{frame_id}.color.jpg",
        ]
        suffixes = (".jpg", ".jpeg")
    elif kind == "depth":
        candidates = [
            f"{prefix}frames/depth/{frame_id}.png",
            f"{prefix}frames/depth/{frame_id:06d}.png",
            f"{prefix}frame-{frame_id:06d}.depth.pgm",
            f"{prefix}frame-{frame_id}.depth.pgm",
        ]
        suffixes = (".png", ".pgm")
    else:
        candidates = [
            f"{prefix}frames/pose/{frame_id}.txt",
            f"{prefix}frames/pose/{frame_id:06d}.txt",
            f"{prefix}frame-{frame_id:06d}.pose.txt",
            f"{prefix}frame-{frame_id}.pose.txt",
        ]
        suffixes = (".txt",)
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    matches = [
        name for name in names
        if name.lower().startswith(prefix.lower()) and name.lower().endswith(suffixes)
        and (Path(name).stem.isdigit() and int(Path(name).stem) == frame_id
             or f"frame-{frame_id:06d}.{kind}." in name.lower())
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"no unique {kind} frame for id {frame_id}")


def _pose_member(names: Iterable[str], frame_id: int) -> str:
    names = list(names)
    exact = f"frames/pose/{frame_id}.txt"
    for name in names:
        if name.lower() == exact.lower():
            return name
    matches = [
        name for name in names
        if name.lower().startswith("frames/pose/") and name.lower().endswith(".txt")
        and Path(name).stem.isdigit() and int(Path(name).stem) == frame_id
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"no unique pose for frame id {frame_id}")


@dataclass(frozen=True)
class ThreeRScanSequence:
    """On-demand access to one 3RScan sequence."""

    scan_id: str
    root: Path

    @property
    def archive_path(self) -> Path:
        path = self.root / self.scan_id / "sequence.zip"
        if path.exists():
            return path
        monolithic = self.root / "3RScan.v2.zip"
        if monolithic.exists():
            return monolithic
        raise FileNotFoundError(f"missing 3RScan sequence archive for {self.scan_id}: {path} or {monolithic}")

    def _open(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(self.archive_path)

    def _prefix(self, archive: zipfile.ZipFile) -> str:
        names = archive.namelist()
        sequence_prefix = f"{self.scan_id}/sequence/"
        if any(name.startswith(sequence_prefix) for name in names):
            return sequence_prefix
        if any(name.startswith("frames/") for name in names) or any(name.startswith("frame-") for name in names):
            return ""
        raise FileNotFoundError(f"scan {self.scan_id} is not present in {self.archive_path}")

    def _info(self, archive: zipfile.ZipFile, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        names = archive.namelist()
        candidates = [name for name in names if name == f"{prefix}_info.txt"]
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one _info.txt in {self.archive_path}")
        return _read_info(archive.read(candidates[0]).decode("utf-8"))

    def frame_ids(self) -> list[int]:
        with self._open() as archive:
            prefix = self._prefix(archive)
            ids = {
                int(re.search(r"(?:frame-)?(\d+)(?:\.color)?$", Path(name).stem).group(1))
                for name in archive.namelist()
                if name.lower().startswith(f"{prefix}frames/color/")
                and Path(name).stem.isdigit()
                or name.lower().startswith(prefix.lower())
                and ".color." in name.lower()
                and Path(name).suffix.lower() in {".jpg", ".jpeg"}
            }
        return sorted(ids)

    def read_frame(self, frame_id: int, global_transform: np.ndarray | None = None) -> dict[str, np.ndarray]:
        with self._open() as archive:
            names = archive.namelist()
            prefix = self._prefix(archive)
            color = np.asarray(Image.open(io.BytesIO(archive.read(_zip_member(names, prefix, "color", frame_id)))).convert("RGB"))
            depth = np.asarray(Image.open(io.BytesIO(archive.read(_zip_member(names, prefix, "depth", frame_id)))), dtype=np.float32)
            if depth.ndim != 2:
                raise ValueError(f"depth frame {frame_id} is not single-channel")
            pose_values = [float(item) for item in _FLOAT_RE.findall(archive.read(_zip_member(names, prefix, "pose", frame_id)).decode("utf-8"))]
            pose = _matrix(pose_values, "camera pose")
            color_intrinsic, depth_intrinsic, depth_extrinsic, depth_shift, has_depth_intrinsic = self._info(archive, prefix)
        if depth_shift <= 0:
            raise ValueError(f"invalid depth shift {depth_shift}")
        depth /= depth_shift
        if has_depth_intrinsic and depth.shape != color.shape[:2]:
            yy, xx = np.meshgrid(np.arange(depth.shape[0], dtype=np.float32), np.arange(depth.shape[1], dtype=np.float32), indexing="ij")
            pixels = np.stack((xx.reshape(-1), yy.reshape(-1), np.ones(xx.size, dtype=np.float32)), axis=1)
            valid = np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
            points = (pixels[valid] @ np.linalg.inv(depth_intrinsic).T) * depth.reshape(-1)[valid, None]
            points_h = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
            points_color = (points_h @ depth_extrinsic.T)[:, :3]
            projected = points_color @ color_intrinsic.T
            projected_uv = projected[:, :2] / projected[:, 2:3]
            aligned = np.zeros(color.shape[:2], dtype=np.float32)
            u = np.rint(projected_uv[:, 0]).astype(np.int64)
            v = np.rint(projected_uv[:, 1]).astype(np.int64)
            visible = points_color[:, 2] > 0
            visible &= u >= 0
            visible &= v >= 0
            visible &= u < color.shape[1]
            visible &= v < color.shape[0]
            for uu, vv, zz in zip(u[visible], v[visible], points_color[visible, 2]):
                if aligned[vv, uu] == 0 or zz < aligned[vv, uu]:
                    aligned[vv, uu] = zz
            depth = aligned
        if global_transform is not None:
            pose = np.asarray(global_transform, dtype=np.float32) @ pose
        return {
            "rgb": color,
            "depth": depth.astype(np.float32),
            "intrinsics": color_intrinsic.astype(np.float32),
            "camera_to_world": pose.astype(np.float32),
            "frame_id": np.asarray(frame_id, dtype=np.int64),
        }


@dataclass(frozen=True)
class ThreeRScanPair:
    reference: ThreeRScanSequence
    rescan: ThreeRScanSequence
    rescan_to_reference: np.ndarray

    @classmethod
    def from_metadata(cls, root: Path, metadata_path: Path, reference_id: str, rescan_id: str) -> "ThreeRScanPair":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for scene in metadata:
            if scene.get("reference") != reference_id:
                continue
            for scan in scene.get("scans", []):
                if scan.get("reference") == rescan_id:
                    transform = _metadata_matrix(scan["transform"], "rescan transform")
                    return cls(
                        ThreeRScanSequence(reference_id, root),
                        ThreeRScanSequence(rescan_id, root),
                        transform,
                    )
        raise KeyError(f"3RScan pair not found: reference={reference_id}, rescan={rescan_id}")
