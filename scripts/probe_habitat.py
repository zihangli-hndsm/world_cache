"""Minimal RGB-D/pose probe for validating local Habitat + ReplicaCAD wiring."""

from __future__ import annotations

import argparse
from pathlib import Path

import habitat_sim
import numpy as np
import quaternion


def sensor_spec(uuid: str, sensor_type: habitat_sim.SensorType, resolution: int) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [resolution, resolution]
    spec.position = [0.0, 1.5, 0.0]
    spec.hfov = 90.0
    return spec


def make_simulator(root: Path, scene: str, resolution: int) -> habitat_sim.Simulator:
    config = habitat_sim.SimulatorConfiguration()
    config.scene_dataset_config_file = str(root / "replicaCAD.scene_dataset_config.json")
    config.scene_id = str(root / "configs/scenes" / f"{scene}.scene_instance.json")
    config.enable_physics = False
    config.gpu_device_id = 0
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [
        sensor_spec("color", habitat_sim.SensorType.COLOR, resolution),
        sensor_spec("depth", habitat_sim.SensorType.DEPTH, resolution),
    ]
    return habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scene", default="apt_1")
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()
    sim = make_simulator(args.root, args.scene, args.resolution)
    try:
        print("pathfinder_loaded", sim.pathfinder.is_loaded)
        # ReplicaCAD's public navmesh set intentionally omits held-out variants;
        # never call PathFinder sampling until a navmesh is confirmed loaded.
        point = sim.pathfinder.get_random_navigable_point() if sim.pathfinder.is_loaded else np.zeros(3, dtype=np.float32)
        print("probe_point", list(point))
        state = sim.get_agent(0).get_state()
        state.position = point
        sim.get_agent(0).set_state(state, reset_sensors=True)
        observations = sim.get_sensor_observations()
        sensor_state = sim.get_agent(0).get_state().sensor_states["depth"]
        print("color", observations["color"].shape, observations["color"].dtype)
        print("depth", observations["depth"].shape, observations["depth"].dtype)
        print("valid_depth_fraction", float(np.isfinite(observations["depth"]).mean() * (observations["depth"] > 0).mean()))
        print("sensor_position", sensor_state.position.tolist())
        print("sensor_rotation_matrix")
        print(quaternion.as_rotation_matrix(sensor_state.rotation))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
