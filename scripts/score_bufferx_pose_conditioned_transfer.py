#!/usr/bin/env python3
"""Score pose-conditioned transfers on fixed queries, not native BUFFER-X matches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from evaluate_3rscan_instance_correspondence import read_annotated_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/reports/20261008_bufferx_validation'))
    parser.add_argument('--vitg-dir', type=Path, default=Path('outputs/reports/20261006_vitg_pose_locked_motion_validation'))
    parser.add_argument('--data-root', type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    args = parser.parse_args()
    manifest = pd.read_csv(args.run_dir / 'run_manifest.csv')
    scheduled = json.loads((args.run_dir / 'scheduled_population.json').read_text())
    if len(manifest) != 110 or manifest.run_id.duplicated().any() or set(manifest.run_id) != {r['run_id'] for r in scheduled}:
        raise ValueError('all 110 scheduled relations required')
    if not manifest.status.isin(['complete', 'failed']).all():
        raise ValueError('all relations must terminate')
    output = args.run_dir / 'pose_conditioned_transfer'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    previous_ref, ref_tree, object_ids, semantic_ids = None, None, None, None
    arms = ['vitg_global_ransac', 'bufferx_pose_nearest_patch', 'bufferx_pose_vitg_k20']
    for relation in manifest.itertuples(index=False):
        with np.load(args.vitg_dir / 'descriptor_candidates' / (relation.run_id + '.npz')) as z:
            ref, query, frames, candidates = z['reference_points'], z['query_points'], z['query_frame_ids'], z['candidate_indices'][:, :20]
            if str(z['reference_id'].item()) != relation.reference_scan or str(z['rescan_id'].item()) != relation.rescan_scan:
                raise ValueError('fixed cache relation mismatch')
        all_diag = pd.read_csv(args.vitg_dir / 'relations' / relation.run_id / 'query_diagnostics.csv')
        diag = all_diag[all_diag.arm.eq('global_ransac')].sort_values('query_index').reset_index(drop=True)
        if not np.array_equal(diag.query_index, np.arange(len(query))) or not np.array_equal(diag.frame_id, frames):
            raise ValueError('fixed query alignment mismatch')
        if previous_ref != relation.reference_scan:
            ref_tree, object_ids, _, semantic_ids, _ = read_annotated_mesh(args.data_root, relation.reference_scan)
            previous_ref = relation.reference_scan
        nearest_annotation = ref_tree.query(ref.astype(np.float32), k=1)[1]
        ref_objects, ref_semantics = object_ids[nearest_annotation], semantic_ids[nearest_annotation]
        evaluable = diag.query_evaluable.to_numpy(bool)
        moved = diag.moved_rigid.to_numpy(bool)
        if not np.array_equal(moved, evaluable & (diag.official_rigid_reference_instance_id.to_numpy() > 0)):
            raise ValueError('moved-rigid denominator mismatch')
        assignments = {'vitg_global_ransac': (diag.selected_reference_index.to_numpy(int), diag.consensus_inlier.to_numpy(bool), diag.spatial_residual_m.to_numpy())}
        if relation.status == 'complete':
            pose = np.load(args.run_dir / 'relations' / relation.run_id / 'predicted_rescan_to_reference.npy')
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError('invalid prediction')
            predicted = query @ pose[:3, :3].T + pose[:3, 3]
            # Both selections use no labels; labels are applied below only for scoring.
            distances, selected = cKDTree(ref).query(predicted, k=1)
            assignments['bufferx_pose_nearest_patch'] = selected, distances <= .1, distances
            residuals = np.linalg.norm(predicted[:, None, :] - ref[candidates], axis=2)
            slots = residuals.argmin(axis=1)
            selected = candidates[np.arange(len(query)), slots]
            distances = residuals[np.arange(len(query)), slots]
            assignments['bufferx_pose_vitg_k20'] = selected, distances <= .1, distances
        else:
            for arm in arms[1:]:
                assignments[arm] = np.zeros(len(query), dtype=int), np.zeros(len(query), dtype=bool), np.full(len(query), np.inf)
        query_tables = []
        for arm in arms:
            selected, assigned, distances = assignments[arm]
            sem_correct = ref_semantics[selected] == diag.query_semantic_id.to_numpy()
            physical_correct = ref_objects[selected] == diag.official_rigid_reference_instance_id.to_numpy()
            if arm == 'vitg_global_ransac':
                np.testing.assert_array_equal(sem_correct, diag.semantic_correct.to_numpy(bool))
                np.testing.assert_array_equal(physical_correct & moved, diag.official_rigid_instance_correct.to_numpy(bool) & moved)
            record = {'run_id': relation.run_id, 'environment_id': relation.environment_id, 'arm': arm,
                      'semantic_queries': int(evaluable.sum()), 'semantic_assigned': int((assigned & evaluable).sum()),
                      'semantic_correct': int((assigned & evaluable & sem_correct).sum()),
                      'moved_rigid_queries': int(moved.sum()), 'moved_rigid_assigned': int((assigned & moved).sum()),
                      'moved_rigid_correct': int((assigned & moved & physical_correct).sum())}
            records.append(record)
            query_tables.append(pd.DataFrame({'query_index': np.arange(len(query)), 'frame_id': frames, 'arm': arm,
                                             'selected_reference_index': selected, 'assigned': assigned, 'residual_m': distances,
                                             'semantic_evaluable': evaluable, 'moved_rigid': moved,
                                             'semantic_correct': sem_correct & assigned & evaluable,
                                             'moved_rigid_correct': physical_correct & assigned & moved}))
        pd.concat(query_tables).to_csv(output / (relation.run_id + '.csv.gz'), index=False)
        print(relation.run_id, relation.status, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'per_relation.csv', index=False)
    env = frame.groupby(['environment_id', 'arm']).sum(numeric_only=True).reset_index()
    env['semantic_coverage'] = env.semantic_correct / env.semantic_queries
    env['moved_rigid_coverage'] = env.moved_rigid_correct / env.moved_rigid_queries.replace(0, np.nan)
    env.to_csv(output / 'per_environment.csv', index=False)
    rng = np.random.default_rng(10082026)
    comparisons = []
    for metric in ['semantic_coverage', 'moved_rigid_coverage']:
        wide = env.pivot(index='environment_id', columns='arm', values=metric).dropna()
        for arm in arms[1:]:
            delta = (wide[arm] - wide[arms[0]]).to_numpy()
            draws = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(axis=1)
            comparisons.append({'arm': arm, 'metric': metric, 'environments': len(delta),
                                'macro': float(wide[arm].mean()), 'vitg_macro': float(wide[arms[0]].mean()),
                                'delta': float(delta.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()})
    summary = {'relations': 110, 'environments': 47, 'threshold_m': .1, 'bootstrap_samples': 20000,
               'bootstrap_seed': 10082026, 'comparisons': comparisons,
               'scope': 'pose-conditioned fixed-patch transfers; not native BUFFER-X descriptor correspondence scores',
               'caveats': 'denser RGB-D pose input; same fixed query/reference patch support; failures retained; explored split; no independent generalization'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
