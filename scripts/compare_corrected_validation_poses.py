#!/usr/bin/env python3
"""Paired comparison of fully reproduced validation poses in released metre units."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path('outputs/reports')
OUT = ROOT / '20261008_corrected_validation_pose_comparison'
SOURCES = {
    'vitg': ROOT / '20261008_vitg_validation_corrected_pose_units/per_relation.csv',
    'mast3r_reciprocal': ROOT / '20261008_mast3r_official_validation_corrected_units/per_relation.csv',
    'mast3r_dense': ROOT / '20261008_mast3r_dense_official_validation_corrected_units/per_relation.csv',
    'bufferx': ROOT / '20261008_bufferx_validation/aggregate/per_relation.csv',
}

def main():
    frames = {}
    for method, path in SOURCES.items():
        frame = pd.read_csv(path).set_index('run_id')
        if len(frame) != 110 or not frame.index.is_unique or frame.environment_id.nunique() != 47:
            raise ValueError(f'incomplete or duplicated population: {method}')
        frames[method] = frame
    controls_path = ROOT / '20261008_geometry_controls_corrected_units/aggregate/per_relation.csv'
    control_summary = json.loads((controls_path.parent / 'summary.json').read_text())
    if control_summary['relations'] != 110 or not control_summary['all_transfer_rows_reproduced']:
        raise ValueError('complete reproduced geometry controls required')
    controls = pd.read_csv(controls_path)
    for method in ['fcgf', 'fpfh_one_way', 'fpfh_mutual']:
        frame = controls[controls.method.eq(method)].set_index('run_id')
        if len(frame) != 110 or not frame.index.is_unique:
            raise ValueError(f'incomplete geometry control {method}')
        frames[method] = frame
    base = frames['vitg']
    combined = base[['environment_id']].copy()
    for method, frame in frames.items():
        if set(frame.index) != set(base.index):
            raise ValueError(f'relation mismatch: {method}')
        frame = frame.reindex(base.index)
        np.testing.assert_array_equal(frame.environment_id, base.environment_id)
        column = 'bufferx_success' if method == 'bufferx' else 'strict_pose_success'
        if not frame[column].isin([True, False]).all():
            raise ValueError('invalid pose success values')
        combined[method] = frame[column].astype(bool)
    env = combined.groupby('environment_id')[list(frames)].mean()
    rng = np.random.default_rng(10082026)
    draws = rng.integers(0, len(env), (20000, len(env)))
    rows = []
    for method in frames:
        delta = (env[method] - env.vitg).to_numpy()
        low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
        rows.append({'method': method, 'strict_pose_successes': int(combined[method].sum()),
            'environment_macro_success': float(env[method].mean()),
            'minus_vitg_macro_delta': float(delta.mean()), 'ci95_low': float(low), 'ci95_high': float(high)})
    OUT.mkdir(exist_ok=True)
    combined.to_csv(OUT / 'per_relation.csv')
    env.to_csv(OUT / 'per_environment.csv')
    pd.DataFrame(rows).to_csv(OUT / 'method_comparison.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps({'relations': 110, 'environments': 47,
        'methods': rows, 'bootstrap_samples': 20000, 'bootstrap_seed': 10082026,
        'scene_units': 'raw released metadata in metres',
        'scope': 'complete validation pose comparison; does not establish transfer quality'}, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
