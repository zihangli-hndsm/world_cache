#!/usr/bin/env python3
"""Compare BUFFER-X and frozen ViT-G pose success on the entire scheduled split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/reports/20261008_bufferx_validation'))
    parser.add_argument('--baseline-dir', type=Path, default=Path('outputs/reports/20261006_vitg_pose_locked_motion_validation'))
    parser.add_argument('--corrected-vitg-dir', type=Path, default=Path('outputs/reports/20261008_vitg_validation_corrected_pose_units'))
    parser.add_argument('--data-root', type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    args = parser.parse_args()
    manifest = pd.read_csv(args.run_dir / 'run_manifest.csv', keep_default_na=False)
    baseline = pd.read_csv(args.baseline_dir / 'per_relation.csv')
    corrected = pd.read_csv(args.corrected_vitg_dir / 'per_relation.csv')
    if len(corrected) != 110 or not corrected.selected_and_inlier_arrays_reproduced.all():
        raise ValueError('complete reproduced ViT-G poses with corrected units required')
    metadata = json.loads((args.data_root / '3RScan.json').read_text())
    scheduled = pd.DataFrame(json.loads((args.run_dir / 'scheduled_population.json').read_text()))
    if len(manifest) != 110 or manifest.run_id.duplicated().any() or set(manifest.run_id) != set(scheduled.run_id):
        raise ValueError('all 110 scheduled relations must terminate before aggregate comparison')
    if not manifest.status.isin(['complete', 'failed']).all():
        raise ValueError('nonterminal baseline run')
    rows = []
    for row in manifest.itertuples(index=False):
        base = baseline[baseline.run_id == row.run_id]
        if len(base) != 1:
            raise ValueError('baseline relation must be unique')
        base = base.iloc[0]
        for key in ('environment_id', 'reference_scan', 'rescan_scan'):
            if getattr(row, key) != base[key]:
                raise ValueError('paired relation identity mismatch')
        corrected_row = corrected[corrected.run_id.eq(row.run_id)]
        if len(corrected_row) != 1:
            raise ValueError('corrected ViT-G pose must be unique')
        record = {'run_id': row.run_id, 'environment_id': row.environment_id, 'status': row.status,
                  'bufferx_success': False, 'vitg_success': bool(corrected_row.iloc[0].strict_pose_success),
                  'rotation_error_degrees': None, 'translation_error_m': None}
        if row.status == 'complete':
            result = json.loads((args.run_dir / 'relations' / row.run_id / 'result.json').read_text())
            if result['early_exit'] or result['pose_solver'] != 'ransac' or result['gt_used_for_inference']:
                raise ValueError('protocol drift')
            # Rescore all persisted predictions consistently, including those
            # produced before the parser correction during this run.
            pose = np.load(args.run_dir / 'relations' / row.run_id / 'predicted_rescan_to_reference.npy')
            scan = next(s for r in metadata if r['reference'] == row.reference_scan for s in r['scans'] if s['reference'] == row.rescan_scan)
            gt = np.asarray(scan['transform'], dtype=np.float32).reshape(4, 4).T
            rotation = float(np.degrees(Rotation.from_matrix(pose[:3, :3].T @ gt[:3, :3]).magnitude()))
            translation = float(np.linalg.norm(pose[:3, 3] - gt[:3, 3]))
            record.update(bufferx_success=rotation < 5 and translation < .25,
                          rotation_error_degrees=rotation, translation_error_m=translation)
        rows.append(record)
    frame = pd.DataFrame(rows)
    env = frame.groupby('environment_id')[['bufferx_success', 'vitg_success']].mean()
    if len(env) != 47:
        raise ValueError('47 environment clusters required')
    env['paired_delta'] = env.bufferx_success - env.vitg_success
    rng = np.random.default_rng(10082026)
    values = env.paired_delta.to_numpy()
    boot = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    summary = {'relations': len(frame), 'environments': len(env), 'failed_bufferx_relations': int(frame.status.eq('failed').sum()),
               'bufferx_pose_successes': int(frame.bufferx_success.sum()), 'vitg_pose_successes': int(frame.vitg_success.sum()),
               'bufferx_macro_success': float(env.bufferx_success.mean()), 'vitg_macro_success': float(env.vitg_success.mean()),
               'bufferx_minus_vitg_macro_delta': float(values.mean()),
               'paired_environment_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
               'bootstrap_samples': 20000, 'bootstrap_seed': 10082026,
               'scene_transform_units': 'raw released metadata in metres; all predictions rescored',
               'success_gate': 'rotation < 5 degrees AND translation < 0.25 m; failures count as unsuccessful',
               'scope': 'pose only; dense RGB-D geometry adaptation; explored validation split, no independent generalization or transfer claim'}
    output = args.run_dir / 'aggregate'
    output.mkdir(exist_ok=True)
    frame.to_csv(output / 'per_relation.csv', index=False)
    env.to_csv(output / 'per_environment.csv')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
