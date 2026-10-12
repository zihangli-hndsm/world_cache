"""Measure scene-level versus 3RScan object-transform oracle support.

This post-hoc audit reuses the exact depth-valid query samples behind saved
top-K diagnostics. PLY objectId identifies the local instance; PLY globalId is
a semantic label and is never used as an instance key. Official instance labels
and transforms are evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from audit_motion_stratified_failures import object_ids, read_pair_table
from evaluate_3rscan_instance_correspondence import patch_world_points


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    marker = b"end_header\n"
    offset = payload.index(marker) + len(marker)
    header = payload[:offset].decode("ascii")
    match = re.search(r"element vertex (\d+)", header)
    if match is None:
        raise ValueError(f"PLY has no vertex count: {path}")
    count = int(match.group(1))
    values = np.fromstring(payload[offset:].decode("ascii"), sep=" ", dtype=np.float64)
    if values.size < count * 11:
        raise ValueError(f"truncated PLY data: {path}")
    vertices = values[: count * 11].reshape(count, 11)
    return (
        vertices[:, :3].astype(np.float32),
        vertices[:, 6].astype(np.int64),
        vertices[:, 7].astype(np.int64),
    )


def scene_matrix(flat: Any) -> np.ndarray:
    matrix = np.asarray(flat, dtype=np.float64).reshape(4, 4).T
    # Released metadata and annotated meshes use metres; see the full
    # static-surface audit in 20261008_scene_transform_unit_audit.
    return matrix


def object_matrix(flat: Any) -> np.ndarray:
    # The object transform is encoded with translation in metres and maps
    # reference-instance coordinates into rescan-instance coordinates.
    return np.asarray(flat, dtype=np.float64).reshape(4, 4).T


def apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def load_query_points(pair_run: Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads((pair_run / "manifest.json").read_text(encoding="utf-8"))
    all_points: list[np.ndarray] = []
    all_frames: list[np.ndarray] = []
    for entry in manifest["rescan"]:
        path = pair_run / entry["path"]
        with np.load(path) as archive:
            observation = {
                key: torch.from_numpy(archive[key].copy())
                for key in archive.files
            }
        points, _ = patch_world_points(observation, patch_size=14)
        # Match the float32 precision used by the original annotation reader;
        # it resolves exact-distance vertex ties identically.
        values = points.numpy().astype(np.float32)
        all_points.append(values)
        all_frames.append(np.full(len(values), int(entry["frame_id"]), dtype=np.int64))
    return np.concatenate(all_points), np.concatenate(all_frames)


def find_pair_run(
    runs_root: Path, reference_id: str, rescan_id: str
) -> Path:
    matches: list[Path] = []
    for config_path in runs_root.rglob("config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            config.get("dataset") == "3rscan"
            and config.get("reference") == reference_id
            and config.get("rescan") == rescan_id
            and config.get("coordinate_convention") == "native_scan_frames"
            and (config_path.parent / "manifest.json").is_file()
        ):
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"expected one native-frame cache for {reference_id}/{rescan_id}, "
            f"found {matches}"
        )
    return matches[0]


def load_diagnostics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[
        (frame.geometry_weight == 0.0)
        & (frame.top_k == 20)
    ].copy()
    if "ransac_seed" in frame.columns:
        frame = frame[frame.ransac_seed == 17]
    if frame.query_index.duplicated().any():
        raise ValueError(f"duplicate query indices in {path}")
    frame = frame.sort_values("query_index").reset_index(drop=True)
    if not np.array_equal(frame.query_index.to_numpy(), np.arange(len(frame))):
        raise ValueError(f"non-contiguous query indices in {path}")
    return frame


def group_summary(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    n = len(frame)
    if not n:
        return {"queries": 0}
    inlier = frame.consensus_inlier.astype(bool).to_numpy()
    correct_semantic = frame.semantic_correct.astype(bool).to_numpy()
    candidate = frame.candidate_semantic_hit.astype(bool).to_numpy()
    scene_distance = frame.scene_oracle_distance_m.to_numpy(dtype=np.float64)
    valid_scene = np.isfinite(scene_distance)
    object_distance = frame.object_oracle_distance_m.to_numpy(dtype=np.float64)
    valid_object = np.isfinite(object_distance)
    correct_inliers = inlier & correct_semantic
    return {
        "queries": n,
        "semantic_candidate_recall": float(candidate.mean()),
        "correct_semantic_inlier_coverage": float(correct_inliers.mean()),
        "correct_recovery_given_candidate": (
            float(correct_inliers.sum() / candidate.sum()) if candidate.any() else float("nan")
        ),
        "current_inlier_coverage": float(inlier.mean()),
        "current_semantic_accuracy_among_inliers": (
            float(correct_inliers.sum() / inlier.sum()) if inlier.any() else float("nan")
        ),
        "scene_oracle_query_count": int(valid_scene.sum()),
        "scene_oracle_support_at_threshold": (
            float(np.mean(scene_distance[valid_scene] <= threshold))
            if valid_scene.any()
            else float("nan")
        ),
        "scene_oracle_median_residual_m": (
            float(np.median(scene_distance[valid_scene])) if valid_scene.any() else float("nan")
        ),
        "object_oracle_query_count": int(valid_object.sum()),
        "object_oracle_support_at_threshold": (
            float(np.mean(object_distance[valid_object] <= threshold))
            if valid_object.any()
            else float("nan")
        ),
        "object_oracle_median_residual_m": (
            float(np.median(object_distance[valid_object])) if valid_object.any() else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--diagnostics-template", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-threshold", type=float, default=0.10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    pair_records: list[dict[str, object]] = []
    query_records: list[dict[str, object]] = []
    object_checks: list[dict[str, object]] = []

    for pair, reference_id, rescan_id in read_pair_table(args.pair_table):
        relation = next(
            (
                item
                for item in metadata
                if item.get("reference") == reference_id
                and any(scan.get("reference") == rescan_id for scan in item.get("scans", []))
            ),
            None,
        )
        if relation is None:
            raise ValueError(f"official relation not found for {pair}")
        scan = next(item for item in relation["scans"] if item.get("reference") == rescan_id)
        pair_run = find_pair_run(args.runs_root, reference_id, rescan_id)
        config = json.loads((pair_run / "config.json").read_text(encoding="utf-8"))
        scene_transform = np.asarray(config["rescan_to_reference"], dtype=np.float64)
        metadata_scene_transform = scene_matrix(scan["transform"])
        if not np.allclose(scene_transform, metadata_scene_transform, atol=1e-5):
            raise ValueError(f"scene transform mismatch for {pair}")

        ref_points, ref_object_ids, ref_global_ids = read_ply(
            args.annotation_root / reference_id / "labels.instances.annotated.v2.ply"
        )
        rescan_points, rescan_object_ids, rescan_global_ids = read_ply(
            args.annotation_root / rescan_id / "labels.instances.annotated.v2.ply"
        )
        rescan_tree = cKDTree(rescan_points)
        ref_trees: dict[int, cKDTree] = {}
        for object_id in np.unique(ref_object_ids):
            if object_id == 0:
                continue
            points = ref_points[ref_object_ids == object_id]
            if len(points):
                ref_trees[int(object_id)] = cKDTree(points)

        changed: dict[int, dict[str, object]] = {}
        rigid = scan.get("rigid", [])
        for item in rigid:
            reference_object = int(item.get("instance_reference", item.get("id", -1)))
            rescan_object = int(item.get("instance_rescan", item.get("id", -1)))
            if reference_object not in ref_trees or not np.any(rescan_object_ids == rescan_object):
                continue
            if rescan_object in changed:
                raise ValueError(f"multiple motion annotations for rescan objectId {rescan_object} in {pair}")
            changed[rescan_object] = {
                "kind": "rigid",
                "reference_object_id": reference_object,
                "transform": object_matrix(item["transform"]),
            }

        for item in scan.get("nonrigid", []):
            for rescan_object in object_ids([item], side="rescan"):
                if rescan_object not in changed:
                    changed[rescan_object] = {
                        "kind": "nonrigid",
                        "reference_object_id": rescan_object,
                    }

        diagnostics_path = args.runs_root / args.diagnostics_template.format(pair=pair)
        if diagnostics_path.is_dir():
            diagnostics_path = diagnostics_path / "query_diagnostics.csv"
        if not diagnostics_path.is_file():
            raise FileNotFoundError(diagnostics_path)
        diagnostics = load_diagnostics(diagnostics_path)
        query_points, query_frames = load_query_points(pair_run)
        if len(query_points) != len(diagnostics):
            raise ValueError(
                f"query cache/diagnostic length mismatch for {pair}: "
                f"{len(query_points)} vs {len(diagnostics)}"
            )
        if not np.array_equal(query_frames, diagnostics.frame_id.to_numpy(dtype=np.int64)):
            raise ValueError(f"frame order mismatch between cache and diagnostics for {pair}")
        nearest_rescan_vertices = rescan_tree.query(query_points, k=1)[1]
        reconstructed_semantic_ids = rescan_global_ids[nearest_rescan_vertices]
        if not np.array_equal(
            reconstructed_semantic_ids,
            diagnostics.query_instance_id.to_numpy(dtype=np.int64),
        ):
            raise ValueError(f"global semantic labels do not match the saved audit for {pair}")
        reconstructed_query_object_ids = rescan_object_ids[nearest_rescan_vertices]

        for row_index, row in diagnostics.iterrows():
            if not bool(row.query_evaluable):
                continue
            query_object_id = int(reconstructed_query_object_ids[row_index])
            change = changed.get(
                query_object_id,
                {"kind": "other", "reference_object_id": query_object_id},
            )
            reference_object_id = int(change["reference_object_id"])
            # Only the official rigid relation supplies a validated cross-scan
            # object mapping for this oracle. Do not infer physical identity
            # for all remaining local object IDs.
            reference_tree = ref_trees.get(reference_object_id) if change["kind"] == "rigid" else None
            scene_distance = float("nan")
            object_distance = float("nan")
            point = query_points[row_index : row_index + 1]
            if reference_tree is not None:
                scene_point = apply_transform(point, scene_transform)
                scene_distance = float(reference_tree.query(scene_point, k=1)[0][0])
                if change["kind"] == "rigid":
                    object_point = apply_transform(point, np.linalg.inv(change["transform"]))
                    object_distance = float(reference_tree.query(object_point, k=1)[0][0])
            if change["kind"] == "rigid" and reference_tree is None:
                group = "rigid_no_reference_instance"
            else:
                group = str(change["kind"])
            query_records.append({
                "pair": pair,
                "split": "target" if pair == "target" else "source",
                "group": group,
                "query_index": int(row.query_index),
                "frame_id": int(row.frame_id),
                "query_object_id": query_object_id,
                "reference_object_id": reference_object_id,
                "candidate_semantic_hit": bool(row.candidate_semantic_hit),
                "consensus_inlier": bool(row.consensus_inlier),
                "semantic_correct": bool(row.semantic_correct),
                "scene_oracle_distance_m": scene_distance,
                "object_oracle_distance_m": object_distance,
            })

        pair_records.append({
            "pair": pair,
            "pair_run": str(pair_run),
            "diagnostics": str(diagnostics_path),
            "rigid_object_count": len(rigid),
            "nonrigid_object_count": len(scan.get("nonrigid", [])),
            "evaluated_queries": int(diagnostics.query_evaluable.astype(bool).sum()),
        })
        for object_item in rigid:
            ref_oid = int(object_item.get("instance_reference", -1))
            res_oid = int(object_item.get("instance_rescan", -1))
            ref_mask = ref_object_ids == ref_oid
            res_mask = rescan_object_ids == res_oid
            if not np.any(ref_mask) or not np.any(res_mask):
                continue
            ref_tree = cKDTree(ref_points[ref_mask])
            moved = apply_transform(rescan_points[res_mask], np.linalg.inv(object_matrix(object_item["transform"])))
            static = apply_transform(rescan_points[res_mask], scene_transform)
            object_checks.append({
                "pair": pair,
                "reference_object_id": ref_oid,
                "rescan_object_id": res_oid,
                "surface_points": int(len(moved)),
                "scene_transform_median_residual_m": float(np.median(ref_tree.query(static, k=1)[0])),
                "object_transform_median_residual_m": float(np.median(ref_tree.query(moved, k=1)[0])),
            })

    queries = pd.DataFrame(query_records)
    queries.to_csv(args.output_dir / "object_motion_oracle_queries.csv", index=False)
    pd.DataFrame(pair_records).to_csv(args.output_dir / "object_motion_oracle_pairs.csv", index=False)
    object_frame = pd.DataFrame(object_checks)
    object_frame.to_csv(args.output_dir / "object_transform_surface_checks.csv", index=False)

    summaries: dict[str, dict[str, dict[str, float | int]]] = {}
    motion_group_aggregates: dict[str, dict[str, float | int]] = {}
    for split in ("source", "target"):
        summaries[split] = {}
        split_frame = queries[queries.split == split]
        for group in ("rigid", "nonrigid", "other", "rigid_no_reference_instance"):
            summaries[split][group] = group_summary(
                split_frame[split_frame.group == group], args.support_threshold
            )
        changed = split_frame[split_frame.group.isin(("rigid", "nonrigid"))]
        other = split_frame[split_frame.group == "other"]
        changed_inliers = changed.consensus_inlier.astype(bool)
        other_inliers = other.consensus_inlier.astype(bool)
        all_inliers = split_frame.consensus_inlier.astype(bool)
        motion_group_aggregates[split] = {
            "changed_queries": int(len(changed)),
            "all_queries": int(len(split_frame)),
            "changed_query_share": float(len(changed) / len(split_frame)) if len(split_frame) else float("nan"),
            "changed_consensus_inliers": int(changed_inliers.sum()),
            "all_consensus_inliers": int(all_inliers.sum()),
            "changed_inlier_share": float(changed_inliers.sum() / all_inliers.sum()) if all_inliers.any() else float("nan"),
            "changed_retained_semantic_accuracy": (
                float((changed_inliers & changed.semantic_correct.astype(bool)).sum() / changed_inliers.sum())
                if changed_inliers.any() else float("nan")
            ),
            "other_retained_semantic_accuracy": (
                float((other_inliers & other.semantic_correct.astype(bool)).sum() / other_inliers.sum())
                if other_inliers.any() else float("nan")
            ),
        }
    summary_json = {
        "status": "complete",
        "support_threshold_m": args.support_threshold,
        "scene_oracle": "3RScan rescan_to_reference",
        "object_oracle": "inverse of official per-instance reference-to-rescan rigid transform",
        "object_transform_translation_units": "meters, empirically checked against annotated surfaces",
        "matching_changed": False,
        "pair_count": len(pair_records),
        "summaries": summaries,
        "combined_motion_groups": motion_group_aggregates,
        "rigid_surface_checks": {
            "objects": int(len(object_frame)),
            "points": int(object_frame.surface_points.sum()) if len(object_frame) else 0,
            "scene_transform_median_of_object_medians_m": (
                float(object_frame.scene_transform_median_residual_m.median())
                if len(object_frame) else float("nan")
            ),
            "object_transform_median_of_object_medians_m": (
                float(object_frame.object_transform_median_residual_m.median())
                if len(object_frame) else float("nan")
            ),
            "objects_improved_by_object_transform": (
                int((object_frame.object_transform_median_residual_m
                     < object_frame.scene_transform_median_residual_m).sum())
                if len(object_frame) else 0
            ),
        },
        "provenance": {
            "metadata": str(args.metadata),
            "pair_table": str(args.pair_table),
            "diagnostics_template": args.diagnostics_template,
            "pair_data_root": str(args.runs_root),
        },
        "caveat": (
            "geometric oracle is computed only for explicitly mapped rigid instances; "
            "it uses hidden annotations and reference mesh points and is an upper-bound "
            "diagnostic, not a deployable method result or causal estimate"
        ),
    }
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value

    (args.output_dir / "object_motion_oracle_summary.json").write_text(
        json.dumps(json_safe(summary_json), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = [
        "# Object-transform oracle audit",
        "",
        "This evaluation-only oracle reuses the saved ViT-G top-20 query population and corresponding cached RGB-D patch centers. For queries assigned to an officially mapped rigid object, it measures nearest distance to that object's reference-mesh vertices after either the scene-level rescan-to-reference transform or the inverse per-instance transform. PLY objectId is used only to locate scan-local mesh vertices; PLY globalId is a semantic label. Matching and RANSAC are unchanged.",
        "",
        f"Support is the fraction of explicitly mapped rigid-object query points within {args.support_threshold:.2f} m of the corresponding reference mesh. The oracle uses hidden object mappings and transforms, so it measures an upper bound on geometric support, not an achievable method result.",
        "",
        "## Pooled query results",
        "",
        "| Split | Query group | Queries | Semantic candidate hit | Scene-transform support | Object-transform support | Correct semantic inlier coverage | Correct recovery given semantic candidate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    def format_metric(item: dict[str, float | int], key: str) -> str:
        value = item.get(key, float("nan"))
        return f"{100 * float(value):.1f}%" if value is not None and np.isfinite(value) else "--"

    for split, groups in summaries.items():
        for group, item in groups.items():
            if int(item.get("queries", 0)) == 0:
                continue
            report.append(
                f"| {split} | {group} | {item['queries']} | "
                f"{format_metric(item, 'semantic_candidate_recall')} | "
                f"{format_metric(item, 'scene_oracle_support_at_threshold')} | "
                f"{format_metric(item, 'object_oracle_support_at_threshold')} | "
                f"{format_metric(item, 'correct_semantic_inlier_coverage')} | "
                f"{format_metric(item, 'correct_recovery_given_candidate')} |"
            )
    report.extend([
        "",
        "## Combined annotated-change stratification",
        "",
        "This descriptive comparison combines rigid and nonrigid rescan-side objectId groups. Unannotated objects are not assumed to be static, and this split does not attribute cause.",
        "",
        "| Split | Changed queries / all | Query share | Changed inliers / all inliers | Inlier share | Semantic accuracy, changed / other |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for split, item in motion_group_aggregates.items():
        report.append(
            f"| {split} | {item['changed_queries']} / {item['all_queries']} | "
            f"{format_metric(item, 'changed_query_share')} | "
            f"{item['changed_consensus_inliers']} / {item['all_consensus_inliers']} | "
            f"{format_metric(item, 'changed_inlier_share')} | "
            f"{format_metric(item, 'changed_retained_semantic_accuracy')} / "
            f"{format_metric(item, 'other_retained_semantic_accuracy')} |"
        )
    report.extend([
        "",
        "## Object-transform surface check",
        "",
        f"For {len(object_frame)} explicitly mapped rigid instances, the median across per-object median surface residuals is {object_frame.scene_transform_median_residual_m.median():.3f} m under the scene transform and {object_frame.object_transform_median_residual_m.median():.3f} m under the object-transform oracle; the object transform improves {int((object_frame.object_transform_median_residual_m < object_frame.scene_transform_median_residual_m).sum())}/{len(object_frame)} instances.",
        "",
        "These results locate a geometric model ceiling. They do not show that a visual matcher can discover or select the object transforms without annotations. Partial scans and mesh reconstruction differences also make same-instance surface distance an imperfect proxy for true point correspondence.",
        "",
    ])
    (args.output_dir / "object_motion_oracle_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(json_safe(summary_json), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
