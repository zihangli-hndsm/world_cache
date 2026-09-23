"""Generate restart-safe RGB-D pose pairs from a fixed ReplicaCAD scene.

The generated camera-to-world matrices use the conventional vision camera frame:
right=x, down=y, forward=z. Habitat's native sensor frame is OpenGL-like
(right=x, up=y, forward=-z), so a fixed ``diag(1, -1, -1)`` basis conversion
is applied before records are written.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import habitat_sim
import numpy as np
import quaternion


CV_FROM_HABITAT_CAMERA = np.diag([1.0, -1.0, -1.0])


def _sensor_spec(uuid: str, sensor_type: habitat_sim.SensorType, resolution: int) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [resolution, resolution]
    spec.position = [0.0, 1.5, 0.0]
    spec.hfov = 90.0
    return spec


def make_simulator(root: Path, scene_name: str, resolution: int) -> habitat_sim.Simulator:
    config = habitat_sim.SimulatorConfiguration()
    config.scene_dataset_config_file = str(root / "replicaCAD.scene_dataset_config.json")
    config.scene_id = str(root / "configs/scenes" / f"{scene_name}.scene_instance.json")
    config.enable_physics = False
    config.gpu_device_id = 0
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [
        _sensor_spec("color", habitat_sim.SensorType.COLOR, resolution),
        _sensor_spec("depth", habitat_sim.SensorType.DEPTH, resolution),
    ]
    return habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))


def intrinsics_for_sensor(resolution: int, hfov_degrees: float = 90.0) -> np.ndarray:
    focal = (resolution / 2.0) / math.tan(math.radians(hfov_degrees) / 2.0)
    center = (resolution - 1.0) / 2.0
    return np.array([[focal, 0.0, center], [0.0, focal, center], [0.0, 0.0, 1.0]], dtype=np.float32)


def set_agent_pose(agent: habitat_sim.agent.Agent, position_xyz: np.ndarray, yaw_degrees: float) -> None:
    state = agent.get_state()
    state.position = np.asarray(position_xyz, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector([0.0, math.radians(yaw_degrees), 0.0])
    agent.set_state(state, reset_sensors=True)


def observation(sim: habitat_sim.Simulator, resolution: int) -> dict[str, np.ndarray]:
    observed = sim.get_sensor_observations()
    sensor_state = sim.get_agent(0).get_state().sensor_states["depth"]
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation).astype(np.float32) @ CV_FROM_HABITAT_CAMERA
    transform[:3, 3] = np.asarray(sensor_state.position, dtype=np.float32)
    return {
        "rgb": np.asarray(observed["color"])[..., :3].copy(),
        "depth": np.asarray(observed["depth"], dtype=np.float32).copy(),
        "intrinsics": intrinsics_for_sensor(resolution),
        "camera_to_world": transform,
    }


def write_observation(path: Path, item: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **item)
    temporary.replace(path)


def read_pose(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["camera_to_world"], dtype=np.float32)


def relative_pose(reference_to_world: np.ndarray, current_to_world: np.ndarray) -> np.ndarray:
    return np.linalg.inv(current_to_world) @ reference_to_world


def pose_record(pair_id: int, pair_type: str, scene: str, ref_path: Path, cur_path: Path, output_dir: Path) -> dict[str, Any]:
    ref_pose, cur_pose = read_pose(ref_path), read_pose(cur_path)
    return {
        "pair_id": pair_id,
        "pair_type": pair_type,
        "scene_id": scene,
        "reference": str(ref_path.relative_to(output_dir)),
        "current": str(cur_path.relative_to(output_dir)),
        "relative_pose": relative_pose(ref_pose, cur_pose).round(8).tolist(),
    }


def small_motion_schedule(pair_count: int, identity_count: int) -> list[tuple[str, np.ndarray, float]]:
    schedule: list[tuple[str, np.ndarray, float]] = []
    for _ in range(identity_count):
        schedule.append(("identity", np.zeros(3, dtype=np.float32), 0.0))
    # Stable, nearby offsets around a validated reference camera origin. They
    # are intentionally small for the Day-1 geometry gate, not a navigation policy.
    translations = [
        (-0.20, 0.0), (-0.10, 0.0), (0.10, 0.0), (0.20, 0.0),
        (0.0, -0.20), (0.0, -0.10), (0.0, 0.10), (0.0, 0.20),
        (-0.15, -0.15), (-0.15, 0.15), (0.15, -0.15), (0.15, 0.15),
    ]
    yaws = [-10.0, -6.0, -3.0, 3.0, 6.0, 10.0]
    for index in range(pair_count - identity_count):
        x, z = translations[index % len(translations)]
        yaw = yaws[(index // len(translations)) % len(yaws)]
        schedule.append(("small_motion", np.array([x, 0.0, z], dtype=np.float32), yaw))
    return schedule


def micro_schedule(pair_count: int, identity_count: int) -> list[tuple[str, np.ndarray, float]]:
    """Very short-horizon viewpoint changes for temporal keyframe caching."""
    schedule: list[tuple[str, np.ndarray, float]] = []
    for _ in range(identity_count):
        schedule.append(("identity", np.zeros(3, dtype=np.float32), 0.0))
    translations = [
        (-0.005, 0.0), (-0.0025, 0.0), (0.0025, 0.0), (0.005, 0.0),
        (0.0, -0.005), (0.0, -0.0025), (0.0, 0.0025), (0.0, 0.005),
        (-0.010, 0.0), (0.010, 0.0), (0.0, -0.010), (0.0, 0.010),
        (-0.020, 0.0), (0.020, 0.0), (0.0, -0.020), (0.0, 0.020),
    ]
    yaws = [-0.5, 0.5, -1.0, 1.0, -2.0, 2.0]
    for index in range(pair_count - identity_count):
        x, z = translations[index % len(translations)]
        yaw = yaws[(index // len(translations)) % len(yaws)]
        schedule.append(("micro_motion", np.array([x, 0.0, z], dtype=np.float32), yaw))
    return schedule


def trajectory_schedule(pair_count: int, identity_count: int) -> list[tuple[str, np.ndarray, float]]:
    """Monotone short video trajectory with small per-frame pose changes."""
    schedule: list[tuple[str, np.ndarray, float]] = []
    for _ in range(identity_count):
        schedule.append(("identity", np.zeros(3, dtype=np.float32), 0.0))
    for index in range(pair_count - identity_count):
        step = index + 1
        schedule.append(
            (
                "trajectory",
                np.array([0.002 * step, 0.0, 0.0], dtype=np.float32),
                0.1 * step,
            )
        )
    return schedule


def diagonal_trajectory_schedule(pair_count: int, identity_count: int) -> list[tuple[str, np.ndarray, float]]:
    """Second smooth trajectory with diagonal translation and rotation."""
    schedule: list[tuple[str, np.ndarray, float]] = []
    for _ in range(identity_count):
        schedule.append(("identity", np.zeros(3, dtype=np.float32), 0.0))
    for index in range(pair_count - identity_count):
        step = index + 1
        schedule.append(
            (
                "diagonal_trajectory",
                np.array([0.0015 * step, 0.0, 0.0010 * step], dtype=np.float32),
                0.08 * step,
            )
        )
    return schedule


def coverage_schedule(pair_count: int, identity_count: int) -> list[tuple[str, np.ndarray, float]]:
    """Deterministic nontrivial pose sweep for the S02 stability gate."""
    schedule: list[tuple[str, np.ndarray, float]] = [
        ("identity", np.zeros(3, dtype=np.float32), 0.0) for _ in range(identity_count)
    ]
    # The four groups correspond to planned rotation bins: 0–10, 10–30,
    # 30–60, and >60 degrees. Translation is varied independently across
    # 0–0.25, 0.25–0.5, 0.5–1.0, and >1.0m bins.
    yaws = [-8.0, 8.0, -20.0, 20.0, -45.0, 45.0, -75.0, 75.0]
    translations = [
        (-0.15, 0.00), (0.15, 0.00),
        (0.35, 0.00), (-0.35, 0.00),
        (0.70, 0.00), (-0.70, 0.00),
        (0.00, 1.20), (0.00, -1.20),
    ]
    for index in range(pair_count - identity_count):
        x, z = translations[index % len(translations)]
        yaw = yaws[(index // len(translations)) % len(yaws)]
        schedule.append(("viewpoint_sweep", np.array([x, 0.0, z], dtype=np.float32), yaw))
    return schedule


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene", default="apt_1")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--identity-pairs", type=int, default=10)
    parser.add_argument("--profile", choices=("small", "micro", "trajectory", "diagonal", "coverage"), default="small")
    parser.add_argument(
        "--reference-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.0],
        help="Validated agent-space start position; use a navigable point for staging scenes.",
    )
    args = parser.parse_args()
    if args.pairs < args.identity_pairs or args.identity_pairs < 1:
        raise ValueError("pairs must be >= identity-pairs >= 1")

    output_dir = args.output_dir
    pair_root = output_dir / "pairs"
    pair_root.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "replicacad",
        "scene": args.scene,
        "root": str(args.root),
        "resolution": args.resolution,
        "pairs": args.pairs,
        "identity_pairs": args.identity_pairs,
        "profile": args.profile,
        "reference_position": args.reference_position,
        "commit_hash": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if args.profile == "small":
        schedule = small_motion_schedule(args.pairs, args.identity_pairs)
    elif args.profile == "micro":
        schedule = micro_schedule(args.pairs, args.identity_pairs)
    elif args.profile == "trajectory":
        schedule = trajectory_schedule(args.pairs, args.identity_pairs)
    elif args.profile == "diagonal":
        schedule = diagonal_trajectory_schedule(args.pairs, args.identity_pairs)
    else:
        schedule = coverage_schedule(args.pairs, args.identity_pairs)
    sim = make_simulator(args.root, args.scene, args.resolution)
    records: list[dict[str, Any]] = []
    try:
        agent = sim.get_agent(0)
        reference_position = np.asarray(args.reference_position, dtype=np.float32)
        for pair_id, (pair_type, delta, yaw) in enumerate(schedule):
            pair_dir = pair_root / f"pair_{pair_id:04d}"
            reference_path, current_path = pair_dir / "reference.npz", pair_dir / "current.npz"
            if not reference_path.exists():
                pair_dir.mkdir(parents=True, exist_ok=True)
                set_agent_pose(agent, reference_position, 0.0)
                write_observation(reference_path, observation(sim, args.resolution))
            if not current_path.exists():
                set_agent_pose(agent, reference_position + delta, yaw)
                write_observation(current_path, observation(sim, args.resolution))
            records.append(pose_record(pair_id, pair_type, args.scene, reference_path, current_path, output_dir))
    finally:
        sim.close()

    with (output_dir / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "status": "complete",
        "pair_count": len(records),
        "identity_pairs": args.identity_pairs,
        "small_motion_pairs": len(records) - args.identity_pairs,
        "output": str(output_dir),
    }
    (output_dir / "status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
