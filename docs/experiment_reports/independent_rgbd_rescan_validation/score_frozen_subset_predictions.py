#!/usr/bin/env python3
"""Score saved label-free predictions against MultiScan labels and meshes.

This script must run only after run_frozen_subset_matching.py has completed.
It reads annotation IDs and annotated meshes solely for post-prediction scoring.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import time
import zipfile
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from audit_multiscan_rgbd_projection import (  # noqa: E402
    REVISION,
    read_mesh,
    scaled_intrinsic_parameters,
    unique_member,
)

PATCH = 14
INPUT_SIZE = 224
THRESHOLD_M = 0.05
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261011


def load_posthoc_labels(archive_path: Path, scan_id: str, point_frames: np.ndarray,
                        patch_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    started = time.perf_counter()
    with zipfile.ZipFile(archive_path) as archive:
        meta = json.loads(archive.read(unique_member(archive, scan_id + ".json")))
        color = next(s for s in meta["streams"] if s["type"] == "color_camera")
        depth = next(s for s in meta["streams"] if s["type"] == "lidar_sensor")
        color_h, color_w = map(int, color["resolution"])
        depth_h, depth_w = map(int, depth["resolution"])
        poses = [json.loads(line) for line in archive.read(unique_member(archive, scan_id + ".jsonl")).splitlines() if line]
        annotations = json.loads(archive.read(unique_member(archive, scan_id + ".annotations.json")))
        annotation_by_id = {int(row["objectId"]): str(row["label"]) for row in annotations.get("objects", [])}
        vertices, triangles, face_object_ids = read_mesh(archive, scan_id)
        alignment = json.loads(archive.read(unique_member(archive, scan_id + ".align.json")))

    if annotations.get("scanId") != scan_id:
        raise ValueError(f"annotation scanId mismatch for {scan_id}")
    T_align = np.asarray(alignment["coordinate_transform"], dtype=np.float64).reshape(4, 4, order="F")
    vertices = vertices @ T_align[:3, :3].T + T_align[:3, 3]
    mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices.astype(np.float32)),
        o3d.core.Tensor(triangles.astype(np.uint32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    labels = np.empty(len(point_frames), dtype=object)
    labels[:] = None
    categories = np.empty(len(point_frames), dtype=object)
    categories[:] = None
    patch_centers = np.arange(PATCH // 2, INPUT_SIZE, PATCH, dtype=np.float64)
    arkit_to_o3d = np.diag([1.0, -1.0, -1.0, 1.0])
    sx = INPUT_SIZE / color_w
    sy = INPUT_SIZE / color_h
    for frame_id in np.unique(point_frames):
        positions = np.flatnonzero(point_frames == frame_id)
        patch = patch_indices[positions]
        xs = patch_centers[patch % 16]
        ys = patch_centers[patch // 16]
        pose = poses[int(frame_id)]
        fx, fy, cx, cy = scaled_intrinsic_parameters(pose["intrinsics"], sx, sy)
        dirs_cam = np.stack([(xs-cx)/fx, (ys-cy)/fy, np.ones(len(xs))], axis=1)
        T_arkit = np.asarray(pose["transform"], dtype=np.float64).reshape(4, 4, order="F")
        T_cw = T_arkit @ arkit_to_o3d
        dirs_world = dirs_cam @ T_cw[:3, :3].T
        origins = np.broadcast_to(T_cw[:3, 3], dirs_world.shape)
        rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
        primitive = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))["primitive_ids"].numpy()
        for local, face_id in enumerate(primitive):
            face_id = int(face_id)
            if face_id >= len(face_object_ids):
                continue
            object_id = int(face_object_ids[face_id])
            label = annotation_by_id.get(object_id)
            if not label or label == "remove" or "." not in label:
                continue
            labels[positions[local]] = label
            categories[positions[local]] = label.rsplit(".", 1)[0]
    return labels, categories, {"label_projection_seconds": time.perf_counter() - started,
                               "annotated_point_fraction": float(np.mean(labels != None))}


def topk_hits(query_labels, query_categories, reference_labels, reference_categories, candidates):
    n = len(query_labels)
    sem_hit = np.zeros(n, dtype=bool)
    phys_hit = np.zeros(n, dtype=bool)
    for i in range(n):
        qcat, qid = query_categories[i], query_labels[i]
        refs = candidates[i]
        if qcat is not None:
            sem_hit[i] = any(reference_categories[j] == qcat for j in refs)
        if qid is not None:
            phys_hit[i] = any(reference_labels[j] == qid for j in refs)
    return sem_hit, phys_hit


def accepted_metrics(query_labels, reference_labels, assigned, inliers):
    correct = np.asarray([
        bool(query_labels[i] is not None and reference_labels[int(assigned[i])] == query_labels[i])
        for i in range(len(query_labels))
    ], dtype=bool)
    accepted = inliers.astype(bool)
    accepted_count = int(accepted.sum())
    correct_count = int((accepted & correct).sum())
    return {
        "accepted_query_count": accepted_count,
        "accepted_query_coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "accepted_physical_instance_precision": correct_count / accepted_count if accepted_count else None,
        "correct_accepted_query_coverage": correct_count / len(accepted) if len(accepted) else 0.0,
    }


def main() -> None:
    scoring_started = time.perf_counter()
    run_manifest_path = REPORT_DIR / "predictions" / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise SystemExit("No completed prediction run manifest; scoring is forbidden before predictions are saved.")
    run_manifest = json.loads(run_manifest_path.read_text())
    if run_manifest.get("predictions_saved_before_label_scoring") is not True:
        raise SystemExit("Prediction manifest does not certify label-free prediction persistence.")
    subset = json.loads((REPORT_DIR / "quota_subset_manifest.json").read_text())
    state = json.loads((REPORT_DIR / "download_state.json").read_text())
    run_rows = list(csv.DictReader((REPORT_DIR / "predictions" / "run_summary.csv").open(newline="")))
    if len(run_rows) != len(subset["relations"]) or any(r["status"] != "complete" for r in run_rows):
        raise SystemExit("Predictions do not cover every relation in the frozen subset; scoring all is blocked.")

    results = []
    total_label_seconds = 0.0
    for relation in subset["relations"]:
        env = relation["environment_id"]
        pred_dir = REPORT_DIR / "predictions" / env
        metadata = json.loads((pred_dir / "prediction_metadata.json").read_text())
        if metadata.get("status") != "predictions_saved_before_scoring" or metadata.get("environment_id") != env:
            raise ValueError(f"unverified prediction file for {env}")
        pred = np.load(pred_dir / "predictions.npz")
        ref_id, qry_id = relation["reference_scan_id"], relation["query_scan_id"]
        ref_file = next(a["filename"] for a in subset["archives"] if Path(a["filename"]).stem == ref_id)
        qry_file = next(a["filename"] for a in subset["archives"] if Path(a["filename"]).stem == qry_id)
        ref_path = ROOT / state["downloaded_archives"][ref_file]["cache_path"]
        qry_path = ROOT / state["downloaded_archives"][qry_file]["cache_path"]
        ref_labels, ref_categories, ref_timing = load_posthoc_labels(
            ref_path, ref_id, pred["reference_frame_ids"], pred["reference_patch_indices"]
        )
        qry_labels, qry_categories, qry_timing = load_posthoc_labels(
            qry_path, qry_id, pred["query_frame_ids"], pred["query_patch_indices"]
        )
        total_label_seconds += ref_timing["label_projection_seconds"] + qry_timing["label_projection_seconds"]
        row = {
            "environment_id": env,
            "query_count": len(qry_labels),
            "reference_count": len(ref_labels),
            "query_identity_coverage": float(np.mean(qry_labels != None)),
            "reference_identity_coverage": float(np.mean(ref_labels != None)),
            "query_missing_identity_count": int(np.sum(qry_labels == None)),
            "reference_label_projection_seconds": ref_timing["label_projection_seconds"],
            "query_label_projection_seconds": qry_timing["label_projection_seconds"],
            "dino_semantic_recall_at20": float(topk_hits(qry_labels, qry_categories, ref_labels, ref_categories, pred["dino_top20"])[0].mean()),
            "dino_physical_instance_recall_at20": float(topk_hits(qry_labels, qry_categories, ref_labels, ref_categories, pred["dino_top20"])[1].mean()),
            "rgb_semantic_recall_at20": float(topk_hits(qry_labels, qry_categories, ref_labels, ref_categories, pred["rgb_top20"])[0].mean()),
            "rgb_physical_instance_recall_at20": float(topk_hits(qry_labels, qry_categories, ref_labels, ref_categories, pred["rgb_top20"])[1].mean()),
        }
        row["dino_semantic_minus_physical_pp"] = 100 * (row["dino_semantic_recall_at20"] - row["dino_physical_instance_recall_at20"])
        row["rgb_semantic_minus_physical_pp"] = 100 * (row["rgb_semantic_recall_at20"] - row["rgb_physical_instance_recall_at20"])
        row.update({f"dino_{k}": v for k, v in accepted_metrics(qry_labels, ref_labels,
                    pred["dino_assigned_reference"], pred["dino_inliers"]).items()})
        row.update({f"rgb_{k}": v for k, v in accepted_metrics(qry_labels, ref_labels,
                    pred["rgb_assigned_reference"], pred["rgb_inliers"]).items()})
        row["dino_prediction_file"] = str((pred_dir / "predictions.npz").relative_to(ROOT))
        results.append(row)
        print(f"scored {env}: DINO physical R@20={row['dino_physical_instance_recall_at20']:.3f}; RGB={row['rgb_physical_instance_recall_at20']:.3f}", flush=True)

    out_dir = REPORT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_environment_metrics.csv", results)
    statistics_started = time.perf_counter()
    metrics = [k for k, v in results[0].items() if isinstance(v, float) and not k.endswith(("_pp", "_seconds"))]
    means = {metric: float(np.mean([r[metric] for r in results])) for metric in metrics}
    paired = {
        "dino_semantic_minus_physical_pp": [r["dino_semantic_minus_physical_pp"] for r in results],
        "rgb_semantic_minus_physical_pp": [r["rgb_semantic_minus_physical_pp"] for r in results],
        "dino_minus_rgb_semantic_recall_pp": [100 * (r["dino_semantic_recall_at20"]-r["rgb_semantic_recall_at20"]) for r in results],
        "dino_minus_rgb_physical_recall_pp": [100 * (r["dino_physical_instance_recall_at20"]-r["rgb_physical_instance_recall_at20"]) for r in results],
        "dino_minus_rgb_instance_precision_pp": [
            100 * (r["dino_accepted_physical_instance_precision"]-r["rgb_accepted_physical_instance_precision"])
            if r["dino_accepted_physical_instance_precision"] is not None and r["rgb_accepted_physical_instance_precision"] is not None
            else float("nan") for r in results
        ],
        "dino_minus_rgb_correct_coverage_pp": [100 * (r["dino_correct_accepted_query_coverage"]-r["rgb_correct_accepted_query_coverage"]) for r in results],
    }
    bootstrap = cluster_bootstrap(results, paired)
    statistics_seconds = time.perf_counter() - statistics_started
    (out_dir / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2, allow_nan=False) + "\n")
    summary = {
        "dataset_revision": REVISION,
        "subset_manifest": "../quota_subset_manifest.json",
        "environment_count": len(results),
        "relation_count": len(results),
        "query_count": int(sum(r["query_count"] for r in results)),
        "mean_query_identity_coverage": float(np.mean([r["query_identity_coverage"] for r in results])),
        "environment_macro_metrics": means,
        "bootstrap": "10,000 environment-cluster resamples; seed 20261011",
        "scoring_runtime_seconds": total_label_seconds,
        "statistics_bootstrap_runtime_seconds": statistics_seconds,
        "scoring_command_wall_seconds": time.perf_counter() - scoring_started,
        "label_access_timing": "only after all 17 relation prediction files were persisted",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


def cluster_bootstrap(results: list[dict], paired: dict[str, list[float]]) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(results)
    output = {"unit": "environment_id", "clusters": n, "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "metrics": {}}
    keys = [k for k, v in results[0].items() if isinstance(v, float) and not k.endswith(("_pp", "_seconds"))]
    arrays = {k: np.asarray([r[k] for r in results], dtype=np.float64) for k in keys}
    for key, values in paired.items():
        arrays[key] = np.asarray(values, dtype=np.float64)
    for key, values in arrays.items():
        finite = np.isfinite(values)
        if not finite.any():
            output["metrics"][key] = {"estimate": None, "ci95_percentile": [None, None]}
            continue
        samples = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
        for i in range(BOOTSTRAP_DRAWS):
            ids = rng.integers(0, n, size=n)
            selected = values[ids]
            selected = selected[np.isfinite(selected)]
            samples[i] = selected.mean() if len(selected) else np.nan
        valid_samples = samples[np.isfinite(samples)]
        interval = [float(np.quantile(valid_samples, .025)), float(np.quantile(valid_samples, .975))] if len(valid_samples) else [None, None]
        output["metrics"][key] = {
            "estimate": float(values[finite].mean()),
            "ci95_percentile": interval,
        }
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
