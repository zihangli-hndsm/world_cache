#!/usr/bin/env python3
"""Reproduce frozen cached consensus exactly and correct only GT pose units."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from evaluate_3rscan_multimotion import fit_fixed_consensus


def rescore(task: tuple[dict, str, str, list, str]) -> dict:
    row, source_name, output_name, metadata, layout = task
    source, output = Path(source_name), Path(output_name)
    run = row['run_id']
    with np.load(source / 'descriptor_candidates' / (run + '.npz')) as z:
        q, ref, candidates = z['query_points'].copy(), z['reference_points'].copy(), z['candidate_indices'][:, :20].copy()
        frame_ids = z['query_frame_ids'].copy()
    diagnostics = pd.read_csv(source / 'relations' / run / 'query_diagnostics.csv' if layout == 'validation' else source / 'query_diagnostics.csv')
    if layout == 'source':
        diagnostics = diagnostics[diagnostics.pair.eq(run)]
    base = diagnostics[diagnostics.arm.eq('global_ransac')].sort_values('query_index').reset_index(drop=True)
    if not np.array_equal(base.query_index, np.arange(len(q))) or not np.array_equal(base.frame_id, frame_ids):
        raise ValueError('fixed query population mismatch')
    hypothesis = fit_fixed_consensus(q, ref, candidates, threshold=.1, iterations=1500, seed=17)
    if not np.array_equal(hypothesis.selected, base.selected_reference_index) or not np.array_equal(hypothesis.inliers, base.consensus_inlier):
        raise ValueError(f'{run}: frozen consensus did not reproduce selections/inliers')
    np.testing.assert_allclose(hypothesis.residuals, base.spatial_residual_m, atol=1e-7, rtol=1e-6)
    scan = next(s for r in metadata if r['reference'] == row['reference_scan'] for s in r['scans'] if s['reference'] == row['rescan_scan'])
    # Use the same float32 metadata parser precision as the original config.
    gt = np.asarray(scan['transform'], dtype=np.float32).reshape(4, 4).T.copy()
    old_gt = gt.copy()
    old_gt[:3, 3] *= .001
    pose = hypothesis.matrix
    rotation = float(np.degrees(Rotation.from_matrix(pose[:3, :3].T @ gt[:3, :3]).magnitude()))
    old_translation = float(np.linalg.norm(pose[:3, 3] - old_gt[:3, 3]))
    summary = pd.read_csv(source / 'relations' / run / 'summary.csv' if layout == 'validation' else source / 'summary.csv')
    if layout == 'source':
        summary = summary[summary.pair.eq(run)]
    saved = summary[summary.arm.eq('global_ransac')].iloc[0]
    np.testing.assert_allclose(rotation, saved.scene_rotation_error_deg, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(old_translation, saved.scene_translation_error_m, atol=1e-6, rtol=1e-6)
    translation = float(np.linalg.norm(pose[:3, 3] - gt[:3, 3]))
    relation = output / 'relations' / run
    relation.mkdir(parents=True, exist_ok=True)
    np.save(relation / 'predicted_rescan_to_reference.npy', pose)
    return {'run_id': run, 'environment_id': row['environment_id'], 'reference_scan': row['reference_scan'],
            'rescan_scan': row['rescan_scan'], 'rotation_error_degrees': rotation, 'translation_error_m': translation,
            'old_translation_error_m': old_translation, 'old_strict_pose_success': bool(saved.strict_scene_pose_success),
            'strict_pose_success': rotation < 5 and translation < .25,
            'selected_and_inlier_arrays_reproduced': True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/reports/20261006_vitg_pose_locked_motion_validation'))
    parser.add_argument('--data-root', type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/reports/20261008_vitg_validation_corrected_pose_units'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--layout', choices=['validation', 'source'], default='validation')
    args = parser.parse_args()
    if args.layout == 'validation':
        manifest = pd.read_csv(args.source / 'run_manifest.csv')
        if len(manifest) != 110 or manifest.environment_id.nunique() != 47:
            raise ValueError('full fixed population required')
    else:
        records = []
        for path in sorted((args.source / 'descriptor_candidates').glob('*.npz')):
            with np.load(path) as z:
                records.append({'run_id': path.stem, 'reference_scan': str(z['reference_id'].item()),
                                'rescan_scan': str(z['rescan_id'].item()), 'environment_id': str(z['reference_id'].item())})
        manifest = pd.DataFrame(records)
        if len(manifest) != 17 or 'target' not in set(manifest.run_id):
            raise ValueError('full 16-source / 1-target population required')
    metadata = json.loads((args.data_root / '3RScan.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(row, str(args.source), str(args.output_dir), metadata, args.layout) for row in manifest.to_dict('records')]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(rescore, tasks):
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output_dir / 'per_relation.csv', index=False)
            print(f"{len(rows)}/{len(manifest)} {row['run_id']} old={row['old_translation_error_m']:.3f} corrected={row['translation_error_m']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    env = frame.groupby('environment_id')[['strict_pose_success', 'old_strict_pose_success']].mean()
    env.to_csv(args.output_dir / 'per_environment.csv')
    summary = {'relations': len(frame), 'environments': len(env), 'old_pose_successes': int(frame.old_strict_pose_success.sum()),
               'corrected_pose_successes': int(frame.strict_pose_success.sum()), 'corrected_environment_macro_pose_success': float(env.strict_pose_success.mean()),
               'all_frozen_assignments_reproduced': bool(frame.selected_and_inlier_arrays_reproduced.all()),
               'changed': 'GT scene translation units only; predictions and transfer assignments unchanged',
               'scene_transform_translation_units': 'metres, raw released metadata values'}
    if args.layout == 'source':
        summary['source_corrected_pose_successes'] = int(frame.loc[frame.run_id.ne('target'), 'strict_pose_success'].sum())
        summary['target_corrected_pose_success'] = bool(frame.loc[frame.run_id.eq('target'), 'strict_pose_success'].iloc[0])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
