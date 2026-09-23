"""Evaluate the published FCGF 3DMatch checkpoint on a 3RScan pair.

This is intentionally a geometry-only control.  It uses the same 14-pixel
query/reference sampling and the same pose-free top-K + spatial-consensus
protocol as the visual evaluator, but replaces DINO descriptors with FCGF
descriptors extracted from the sampled 3-D point clouds.

The maintained FCGF repository currently supplies a WarpConvNet adapter while
some released WarpConvNet wheels do not package the adapter module.  The
``--fcgf-source`` argument points at that official adapter source; the script
loads it without modifying the installed package.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_observation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def patch_world_points(observation: dict[str, np.ndarray], patch_size: int = 14) -> np.ndarray:
    height, width = observation["depth"].shape
    ys = np.arange(patch_size / 2, height, patch_size, dtype=np.float32)
    xs = np.arange(patch_size / 2, width, patch_size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pixels = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    depth = observation["depth"][pixels[:, 1].astype(np.int64), pixels[:, 0].astype(np.int64)]
    valid = np.isfinite(depth) & (depth > 0)
    pixels, depth = pixels[valid], depth[valid]
    intrinsics = observation["intrinsics"]
    camera = np.stack(
        (
            (pixels[:, 0] - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (pixels[:, 1] - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=1,
    )
    homogeneous = np.concatenate((camera, np.ones((len(camera), 1), dtype=np.float32)), axis=1)
    return (homogeneous @ observation["camera_to_world"].T)[:, :3]


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    return (homogeneous @ matrix.T)[:, :3]


def read_annotated_mesh(root: Path, scan_id: str) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    payload = (root / scan_id / "labels.instances.annotated.v2.ply").read_bytes()
    header_end = payload.index(b"end_header\n") + len(b"end_header\n")
    header = payload[:header_end].decode("ascii")
    vertex_count = int(re.search(r"element vertex (\d+)", header).group(1))
    values = np.fromstring(payload[header_end:].decode("ascii"), sep=" ", dtype=np.float64)
    vertices = values[: vertex_count * 11].reshape(vertex_count, 11)
    return (
        cKDTree(vertices[:, :3]),
        vertices[:, 7].astype(np.int64),
        vertices[:, 8].astype(np.int64),
    )


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = target_mean - rotation @ source_mean
    return matrix


def consensus(
    query: np.ndarray,
    reference: np.ndarray,
    candidates: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    best_score = -1
    best_matrix = np.eye(4, dtype=np.float64)
    best_inliers = np.zeros(len(query), dtype=bool)
    best_selected = candidates[:, 0].copy()
    for _ in range(iterations):
        sample = rng.choice(len(query), size=3, replace=False)
        matrix = rigid_fit(query[sample], reference[candidates[sample, rng.integers(0, candidates.shape[1], 3)]])
        distances = np.linalg.norm(
            transform_points(query, matrix)[:, None, :] - reference[candidates], axis=2
        )
        selected = np.argmin(distances, axis=1)
        residuals = distances[np.arange(len(query)), selected]
        inliers = residuals <= threshold
        score = int(inliers.sum())
        if score > best_score:
            best_score, best_matrix = score, matrix
            best_selected, best_inliers = candidates[np.arange(len(query)), selected], inliers
    if best_inliers.sum() >= 3:
        best_matrix = rigid_fit(query[best_inliers], reference[best_selected[best_inliers]])
        distances = np.linalg.norm(
            transform_points(query, best_matrix)[:, None, :] - reference[candidates], axis=2
        )
        selected = np.argmin(distances, axis=1)
        best_selected = candidates[np.arange(len(query)), selected]
        best_inliers = distances[np.arange(len(query)), selected] <= threshold
    return best_matrix, best_selected, best_inliers


def load_model(args: argparse.Namespace, device: str):
    # The official conversion helper imports warpconvnet.models.fcgf.  Inject
    # the adapter source before importing it because some wheels omit it.
    load_module("warpconvnet.models.fcgf", args.fcgf_source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    config = {
        "conv1_kernel_size": getattr(config, "conv1_kernel_size", 7),
        "model_n_out": getattr(config, "model_n_out", 32),
        "normalize_feature": getattr(config, "normalize_feature", True),
    }
    from warpconvnet.models.fcgf import ResUNetBN2C

    model = ResUNetBN2C(
        in_channels=1,
        out_channels=config["model_n_out"],
        normalize_feature=config["normalize_feature"],
        conv1_kernel_size=config["conv1_kernel_size"],
    )
    state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if ".bn." in key:
            prefix, parameter = key.split(".bn.")
            if prefix.startswith("block"):
                block, norm = prefix.split(".")
                target = f"{block}.conv{norm[-1]}.1.{parameter}"
            elif prefix.endswith("_tr"):
                target = f"{prefix.replace('norm', 'conv')}.norm_act.0.{parameter}"
            else:
                target = f"{prefix.replace('norm', 'conv')}.1.{parameter}"
        elif key.endswith(".kernel"):
            prefix = key[: -len(".kernel")]
            target = f"{prefix}.weight" if prefix in ("conv1_tr", "final") else (
                f"{prefix}.conv_tr.weight" if prefix.endswith("_tr") else f"{prefix}.0.weight"
            )
        elif key == "final.bias":
            target = key
        else:
            continue
        if key.endswith(".kernel"):
            if value.ndim == 2:
                value = value.reshape(1, value.shape[0], value.shape[1])
            else:
                kernel = round(value.shape[0] ** (1.0 / 3.0))
                kernel_shape = value.shape
                value = value.reshape(kernel, kernel, kernel, value.shape[1], value.shape[2])
                value = value.permute(2, 1, 0, 3, 4).contiguous().reshape(kernel_shape)
        elif key == "final.bias":
            value = value.reshape(-1)
        state_dict[target] = value.clone()
    model.load_state_dict(state_dict, strict=True)
    config["kernel_perm"] = "reverse"
    model = model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def extract_features(model, xyz: np.ndarray, voxel_size: float, device: str):
    from warpconvnet.geometry.types.voxels import Voxels

    coords = np.floor(xyz / voxel_size).astype(np.int32)
    _, selected = np.unique(coords, axis=0, return_index=True)
    selected = np.sort(selected)
    coordinates = torch.from_numpy(coords[selected])
    features = torch.ones((len(selected), 1), dtype=torch.float32)
    voxels = Voxels([coordinates], [features]).to(device)
    output = model(voxels)
    return xyz[selected], output.feature_tensor.detach().cpu().numpy()


def collect_cloud(pair_run: Path, entries: list[dict[str, object]]) -> np.ndarray:
    points = [patch_world_points(load_observation(pair_run / str(entry["path"]))) for entry in entries]
    return np.concatenate(points, axis=0).astype(np.float64)


def evaluate(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[int, list[dict[str, object]]]]:
    config = json.loads((args.pair_run / "config.json").read_text(encoding="utf-8"))
    if config.get("coordinate_convention") != "native_scan_frames":
        raise ValueError("pair manifest must store native scan coordinates; run migrate_3rscan_rescan_coordinates.py")
    manifest = json.loads((args.pair_run / "manifest.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = load_model(args, device)
    reference_tree, reference_ids_mesh, reference_semantics_mesh = read_annotated_mesh(
        args.annotation_root, config["reference"]
    )
    rescan_tree, rescan_ids_mesh, rescan_semantics_mesh = read_annotated_mesh(
        args.annotation_root, config["rescan"]
    )

    reference_cloud = collect_cloud(args.pair_run, manifest["reference"])
    rescan_cloud = collect_cloud(args.pair_run, manifest["rescan"])
    rescan_to_reference = np.asarray(config["rescan_to_reference"], dtype=np.float64)

    reference_feature_points, reference_features = extract_features(
        model, reference_cloud, args.voxel_size, device
    )
    rescan_feature_points, rescan_features = extract_features(model, rescan_cloud, args.voxel_size, device)
    reference_feature_tree, rescan_feature_tree = cKDTree(reference_feature_points), cKDTree(rescan_feature_points)

    # Evaluate at the same sampled points used to create the clouds, mapping
    # each point to the nearest FCGF voxel representative.
    reference_eval_points = reference_cloud
    query_eval_points = rescan_cloud
    reference_feature_indices = reference_feature_tree.query(reference_eval_points, k=1)[1]
    query_feature_indices = rescan_feature_tree.query(query_eval_points, k=1)[1]
    reference_labels = reference_ids_mesh[reference_tree.query(reference_eval_points, k=1)[1]]
    reference_semantics = reference_semantics_mesh[reference_tree.query(reference_eval_points, k=1)[1]]
    query_labels = rescan_ids_mesh[rescan_tree.query(query_eval_points, k=1)[1]]
    query_semantics = rescan_semantics_mesh[rescan_tree.query(query_eval_points, k=1)[1]]
    # Annotation labels are carried for scoring only. The descriptor matcher
    # receives every depth-valid sampled point, including points whose mesh
    # projection has no non-zero annotation.
    reference_points = reference_eval_points
    reference_features_eval = reference_features[reference_feature_indices]
    reference_ids = reference_labels
    reference_semantics_eval = reference_semantics
    query_points = query_eval_points
    query_features_eval = rescan_features[query_feature_indices]
    query_ids = query_labels
    query_semantics_eval = query_semantics
    query_evaluable = (query_ids != 0) & (query_semantics_eval != 0)

    scores = torch.from_numpy(query_features_eval) @ torch.from_numpy(reference_features_eval.T)
    rows: list[dict[str, object]] = []
    diagnostics: dict[int, list[dict[str, object]]] = {}
    for top_k in args.top_k:
        _, top_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)
        matrix, selected, inliers = consensus(
            query_points,
            reference_points,
            top_indices.cpu().numpy(),
            args.threshold,
            args.iterations,
            args.seed,
        )
        transformed = transform_points(query_points, matrix)
        residuals = np.linalg.norm(transformed - reference_points[selected], axis=1)
        evaluated_inliers = inliers & query_evaluable
        evaluated_query_count = int(query_evaluable.sum())
        rotation_error = Rotation.from_matrix(matrix[:3, :3].T @ rescan_to_reference[:3, :3]).magnitude()
        translation_error = np.linalg.norm(matrix[:3, 3] - rescan_to_reference[:3, 3])
        rows.append(
            {
                "pair": args.pair_run.name,
                "top_k": top_k,
                "reference_queries": len(reference_points),
                "query_queries": evaluated_query_count,
                "matching_query_points": len(query_points),
                "matching_inliers": int(inliers.sum()),
                "inliers": int(evaluated_inliers.sum()),
                "inlier_coverage": float(evaluated_inliers.sum() / evaluated_query_count) if evaluated_query_count else 0.0,
                "inlier_instance_accuracy": float(np.mean(reference_ids[selected[evaluated_inliers]] == query_ids[evaluated_inliers])) if evaluated_inliers.any() else float("nan"),
                "inlier_semantic_accuracy": float(np.mean(reference_semantics_eval[selected[evaluated_inliers]] == query_semantics_eval[evaluated_inliers])) if evaluated_inliers.any() else float("nan"),
                "median_inlier_residual_m": float(np.median(residuals[evaluated_inliers])) if evaluated_inliers.any() else float("nan"),
                "rotation_error_deg": float(np.degrees(rotation_error)),
                "translation_error_m": float(translation_error),
                "pose_success_lt5deg_lt025m": bool(np.degrees(rotation_error) < 5.0 and translation_error < 0.25),
                "voxel_size_m": args.voxel_size,
                "checkpoint": str(args.checkpoint),
                "model_config": json.dumps(model_config, sort_keys=True),
            }
        )
        diagnostics[top_k] = [
            {
                "query_semantic_id": int(query_semantics_eval[index]),
                "selected_reference_semantic_id": int(reference_semantics_eval[selected[index]]),
                "query_instance_id": int(query_ids[index]),
                "selected_reference_instance_id": int(reference_ids[selected[index]]),
                "query_evaluable": bool(query_evaluable[index]),
                "consensus_inlier": bool(inliers[index]),
                "spatial_residual_m": float(residuals[index]),
            }
            for index in range(len(query_points))
        ]
    return rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument(
        "--fcgf-source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party/warpconvnet_fcgf.py",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-query-diagnostics", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = evaluate(args)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "known_alignment_used_for_selection": False,
                "protocol": "same 14-pixel sample, top-K descriptor candidates, 0.10m spatial-consensus RANSAC",
                "top_k": args.top_k,
                "voxel_size_m": args.voxel_size,
                "query_diagnostics_saved": args.save_query_diagnostics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.save_query_diagnostics:
        for top_k, diagnostic_rows in diagnostics.items():
            with (args.output_dir / f"query_diagnostics_k{top_k}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]))
                writer.writeheader()
                writer.writerows(diagnostic_rows)
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
