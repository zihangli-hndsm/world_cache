#!/usr/bin/env python3
"""Prepare segmentation-assisted input, keeping identity targets separate.

Masks are projected from scan-local annotated meshes using measured depth and
released camera poses. This is privileged segmentation input, not a deployable
segmenter and not independently certified physical identity.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_3rscan_instance_correspondence import read_annotated_mesh
from generate_3rscan_pairs import resize_observation
from worldcache.datasets.threerscan import ThreeRScanSequence


def prepare(scan_dir, output, frames, mask_source, frame_selection):
    sequence = ThreeRScanSequence(root=scan_dir.parent, scan_id=scan_dir.name)
    tree, object_ids, _, _, _ = read_annotated_mesh(scan_dir.parent, scan_dir.name)
    if mask_source == 'mesh_render':
        from render_native_annotation_masks import read_faces, render
        faces = read_faces(scan_dir / 'labels.instances.annotated.v2.ply')
    records = []
    available = sequence.frame_ids()
    chosen = available[:frames] if frame_selection == 'first' else [available[i] for i in np.linspace(0, len(available)-1, min(frames, len(available)), dtype=int)]
    for frame_id in chosen:
        observation = sequence.read_frame(frame_id)
        depth = observation['depth']
        y, x = np.indices(depth.shape)
        pixels = np.column_stack((x.ravel(), y.ravel(), np.ones(depth.size)))
        valid = np.isfinite(depth.ravel()) & (depth.ravel() > 0)
        camera = (pixels[valid] @ np.linalg.inv(observation['intrinsics']).T) * depth.ravel()[valid, None]
        pose = observation['camera_to_world']
        points = camera @ pose[:3, :3].T + pose[:3, 3]
        distance, nearest = tree.query(points, k=1)
        # Bin all valid measured points directly to the final grid. Resizing a
        # sparsely registered full-resolution depth image first loses samples.
        source_x, source_y = pixels[valid, 0], pixels[valid, 1]
        u = np.floor((source_x + .5) * 224 / depth.shape[1]).astype(int)
        v = np.floor((source_y + .5) * 224 / depth.shape[0]).astype(int)
        target = v * 224 + u
        # Keep the frontmost sample at each target pixel, including unlabeled
        # foreground samples so that annotated background cannot shine through.
        order = np.lexsort((np.arange(len(camera)), camera[:, 2], target))
        _, first = np.unique(target[order], return_index=True)
        selected = order[first]
        labels = np.zeros(224 * 224, dtype=np.int64)
        supported = selected[distance[selected] <= .05]
        labels[target[supported]] = object_ids[nearest[supported]]
        measured_labels = labels.copy()
        if mask_source == 'mesh_render':
            rendered_labels, rendered_depth = render(tree.data, object_ids, faces, observation)
            labels = rendered_labels.ravel()
            np.savez_compressed(output / f'{scan_dir.name}_frame_{frame_id:06d}_projection_audit.npz',
                                measured_labels=measured_labels.reshape(224, 224), rendered_labels=rendered_labels,
                                rendered_depth=rendered_depth)
        ids, counts = np.unique(labels, return_counts=True)
        ids = ids[(ids > 0) & (counts >= 4)]
        masks = np.stack([labels.reshape(224, 224) == i for i in ids]) if len(ids) else np.zeros((0, 224, 224), bool)
        name = f'{scan_dir.name}_frame_{frame_id:06d}'
        input_path = output / (name + '_input.npz')
        target_path = output / (name + '_identity_ids.npy')
        # Matching receives only image and anonymous masks, never identity IDs.
        np.savez_compressed(input_path, rgb=resize_observation(observation, 224)['rgb'], masks=masks)
        np.save(target_path, ids)
        records.append(dict(frame_id=frame_id, masks=len(ids), input=str(input_path),
                            identity_targets=str(target_path), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--rescan', required=True)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--mask-source', choices=['measured_depth', 'mesh_render'], default='measured_depth')
    parser.add_argument('--frame-selection', choices=['first', 'uniform'], default='first')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError('frames must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = {role: prepare(args.data_root / scan, args.output_dir, args.frames, args.mask_source, args.frame_selection)
               for role, scan in [('reference', args.reference), ('rescan', args.rescan)]}
    report = dict(resolution=224, mesh_distance_m=.05, min_mask_pixels=4,
                  mask_source=('scan-local annotation triangle z-buffer using released camera calibration and poses; measured-depth projection retained for audit'
                               if args.mask_source == 'mesh_render' else
                               'all valid measured points binned to224 with frontmost-depth selection and scan-local annotation lookup'),
                  selected_mask_source=args.mask_source,
                  frame_selection=args.frame_selection,
                  mesh_render_limit='Mixed-label triangles unknown; triangles crossing near plane omitted',
                  scope='Segmentation-assisted diagnostic; separate denominator from fixed-patch experiment',
                  records=records)
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({r: [(f['frame_id'], f['masks']) for f in rows] for r, rows in records.items()}))


if __name__ == '__main__':
    main()
