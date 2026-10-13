#!/usr/bin/env python3
"""Check whether ObChange point clouds share the recorded RGB-D map frame.

Requires the optional `rosbags` package, Open3D, and NumPy. The audit projects
the saved annotated-scan surface points into sampled raw RGB/depth frames using
only the bag's camera calibration and TF tree, then compares projected z with
registered sensor depth. Labels are not used.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


RGB_TOPIC = "/hsrb/head_rgbd_sensor/rgb/image_raw"
DEPTH_TOPIC = "/hsrb/head_rgbd_sensor/depth_registered/image_raw"
CAMERA_TOPIC = "/hsrb/head_rgbd_sensor/rgb/camera_info"


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def transform_matrix(transform: Any) -> np.ndarray:
    q = transform.rotation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    t = transform.translation
    matrix[:3, 3] = [float(t.x), float(t.y), float(t.z)]
    return matrix


def selected_indices(count: int, samples: int) -> list[int]:
    if count <= 0:
        return []
    return sorted(set(np.linspace(0, count - 1, min(samples, count)).round().astype(int).tolist()))


def collect_bag_inputs(
    bag: Path, samples: int
) -> tuple[dict[str, Any], list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]], dict[tuple[str, str], list[tuple[int, np.ndarray]]], dict[tuple[str, str], np.ndarray]]:
    with Reader(bag) as reader:
        store = get_typestore(Stores.ROS1_NOETIC)
        for connection in reader.connections:
            if connection.msgtype not in store.FIELDDEFS:
                store.register(
                    get_types_from_msg(
                        connection.msgdef,
                        connection.msgtype.replace("/msg/", "/"),
                    )
                )

        rgb_connections = [c for c in reader.connections if c.topic == RGB_TOPIC]
        depth_connections = [c for c in reader.connections if c.topic == DEPTH_TOPIC]
        rgb_count = sum(c.msgcount for c in rgb_connections)
        depth_count = sum(c.msgcount for c in depth_connections)
        rgb_indices = set(selected_indices(rgb_count, samples))
        depth_targets = {
            i: int(round(i * (depth_count - 1) / max(1, rgb_count - 1)))
            for i in rgb_indices
        }
        depth_indices = {
            candidate
            for target in depth_targets.values()
            for candidate in range(max(0, target - 3), min(depth_count, target + 4))
        }

        camera: dict[str, Any] | None = None
        rgb_frames: list[tuple[int, np.ndarray]] = []
        depth_frames: list[tuple[int, np.ndarray]] = []
        rgb_i = depth_i = 0
        dynamic: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
        static: dict[tuple[str, str], np.ndarray] = {}

        for connection, bag_time, raw in reader.messages():
            topic = connection.topic
            if topic in ("/tf", "/tf_static"):
                message = store.deserialize_ros1(raw, connection.msgtype)
                for item in message.transforms:
                    parent, child = item.header.frame_id, item.child_frame_id
                    matrix = transform_matrix(item.transform)
                    key = (parent, child)
                    if topic == "/tf_static":
                        static[key] = matrix
                    else:
                        timestamp = stamp_ns(item.header.stamp) or int(bag_time)
                        dynamic[key].append((timestamp, matrix))
            elif topic == CAMERA_TOPIC and camera is None:
                message = store.deserialize_ros1(raw, connection.msgtype)
                camera = {
                    "width": int(message.width),
                    "height": int(message.height),
                    "K": np.asarray(message.K, dtype=np.float64).reshape(3, 3),
                    "D": np.asarray(message.D, dtype=np.float64),
                    "distortion_model": str(message.distortion_model),
                    "frame_id": str(message.header.frame_id),
                }
            elif topic == RGB_TOPIC:
                index = rgb_i
                rgb_i += 1
                if index in rgb_indices:
                    message = store.deserialize_ros1(raw, connection.msgtype)
                    if message.encoding != "rgb8":
                        raise ValueError(f"unexpected RGB encoding {message.encoding!r}")
                    image = np.frombuffer(message.data, dtype=np.uint8).copy()
                    image = image.reshape(message.height, message.width, 3)
                    rgb_frames.append((stamp_ns(message.header.stamp), image))
            elif topic == DEPTH_TOPIC:
                index = depth_i
                depth_i += 1
                if index in depth_indices:
                    message = store.deserialize_ros1(raw, connection.msgtype)
                    if message.encoding != "16UC1":
                        raise ValueError(f"unexpected depth encoding {message.encoding!r}")
                    image = np.frombuffer(message.data, dtype="<u2").copy()
                    image = image.reshape(message.height, message.width)
                    depth_frames.append((stamp_ns(message.header.stamp), image))

    if camera is None or not rgb_frames or not depth_frames:
        raise ValueError("bag is missing the expected camera calibration or RGB-D frames")
    for values in dynamic.values():
        values.sort(key=lambda pair: pair[0])
    return camera, rgb_frames, depth_frames, dynamic, static


def map_to_camera(
    timestamp: int,
    dynamic: dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    static: dict[tuple[str, str], np.ndarray],
) -> np.ndarray | None:
    edges = dict(static)
    max_age_ns = 500_000_000
    for key, values in dynamic.items():
        times = [value[0] for value in values]
        index = int(np.searchsorted(times, timestamp, side="right")) - 1
        if index >= 0 and timestamp - times[index] < max_age_ns:
            edges[key] = values[index][1]

    adjacency: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for (parent, child), matrix in edges.items():
        adjacency[parent].append((child, matrix))
        adjacency[child].append((parent, np.linalg.inv(matrix)))

    target = "head_rgbd_sensor_rgb_frame"
    queue: deque[tuple[str, np.ndarray]] = deque([("map", np.eye(4))])
    visited = {"map"}
    while queue:
        frame, map_to_frame = queue.popleft()
        if frame == target:
            return map_to_frame
        for next_frame, frame_to_next in adjacency[frame]:
            if next_frame not in visited:
                visited.add(next_frame)
                queue.append((next_frame, map_to_frame @ frame_to_next))
    return None


def render_cloud_depth(
    points: np.ndarray, map_to_cam: np.ndarray, camera: dict[str, Any]
) -> np.ndarray:
    width, height = camera["width"], camera["height"]
    intrinsics, distortion = camera["K"], camera["D"]
    camera_to_map = np.linalg.inv(map_to_cam)
    xyz = points @ camera_to_map[:3, :3].T + camera_to_map[:3, 3]
    z = xyz[:, 2]
    keep = np.isfinite(xyz).all(axis=1) & (z > 0.2) & (z < 8.0)
    z, xyz = z[keep], xyz[keep]
    x, y = xyz[:, 0] / z, xyz[:, 1] / z
    if camera["distortion_model"] == "plumb_bob" and distortion.size >= 5:
        k1, k2, p1, p2, k3 = distortion[:5]
        radius2 = x * x + y * y
        radial = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        x, y = (
            x * radial + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
            y * radial + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y,
        )
    u = np.rint(intrinsics[0, 0] * x + intrinsics[0, 2]).astype(np.int64)
    v = np.rint(intrinsics[1, 1] * y + intrinsics[1, 2]).astype(np.int64)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    z, u, v = z[inside], u[inside], v[inside]
    depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(depth, v * width + u, z.astype(np.float32))
    return depth.reshape(height, width)


def evaluate_frame(
    points: np.ndarray,
    timestamp: int,
    measured_depth: np.ndarray,
    camera: dict[str, Any],
    dynamic: dict[tuple[str, str], list[tuple[int, np.ndarray]]],
    static: dict[tuple[str, str], np.ndarray],
) -> dict[str, Any] | None:
    pose = map_to_camera(timestamp, dynamic, static)
    if pose is None:
        return None
    predicted = render_cloud_depth(points, pose, camera)
    measured = measured_depth.astype(np.float32) / 1000.0
    overlap = np.isfinite(predicted) & (measured >= 0.2) & (measured <= 8.0)
    if not overlap.any():
        return {"timestamp_ns": timestamp, "overlap_pixels": 0}
    errors = np.abs(predicted[overlap] - measured[overlap])
    return {
        "timestamp_ns": int(timestamp),
        "overlap_pixels": int(overlap.sum()),
        "valid_depth_pixels": int(((measured >= 0.2) & (measured <= 8.0)).sum()),
        "overlap_fraction": float(overlap.sum() / ((measured >= 0.2) & (measured <= 8.0)).sum()),
        "median_abs_error_m": float(np.median(errors)),
        "p95_abs_error_m": float(np.quantile(errors, 0.95)),
        "within_5cm": float(np.mean(errors < 0.05)),
        "within_10cm": float(np.mean(errors < 0.10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--pcd-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    camera, rgb, depth, dynamic, static = collect_bag_inputs(args.bag, args.samples)
    clouds = []
    for path in sorted(args.pcd_directory.glob("*/merged_plane_clouds_ds002.pcd")):
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points)
        if len(points):
            clouds.append(points)
    if not clouds:
        raise FileNotFoundError(f"no surface PCDs below {args.pcd_directory}")
    points = np.concatenate(clouds, axis=0)

    frames = []
    for rgb_timestamp, _rgb_image in rgb:
        depth_timestamp, depth_image = min(depth, key=lambda pair: abs(pair[0] - rgb_timestamp))
        result = evaluate_frame(points, rgb_timestamp, depth_image, camera, dynamic, static)
        if result is not None:
            result["rgb_depth_sync_ms"] = abs(depth_timestamp - rgb_timestamp) / 1e6
            frames.append(result)
    successful = [item for item in frames if item.get("overlap_pixels", 0) > 0]
    summary = {
        "bag": str(args.bag),
        "pcd_directory": str(args.pcd_directory),
        "camera": {
            "width": camera["width"],
            "height": camera["height"],
            "frame_id": camera["frame_id"],
            "distortion_model": camera["distortion_model"],
        },
        "point_count": int(len(points)),
        "rgb_samples": len(rgb),
        "depth_samples": len(depth),
        "evaluated_frames": len(successful),
        "median_overlap_fraction": float(np.median([x["overlap_fraction"] for x in successful])) if successful else None,
        "median_abs_error_m": float(np.median([x["median_abs_error_m"] for x in successful])) if successful else None,
        "median_within_5cm": float(np.median([x["within_5cm"] for x in successful])) if successful else None,
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
