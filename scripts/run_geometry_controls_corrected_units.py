#!/usr/bin/env python3
"""Rerun geometry controls with corrected GT; verify saved transfer rows unchanged."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def compare(old: Path, new: Path) -> None:
    left, right = pd.read_csv(old), pd.read_csv(new)
    columns = list(left.columns)
    if not set(columns).issubset(right.columns) or len(left) != len(right):
        raise ValueError(f'query schema/population mismatch: {old}')
    for column in columns:
        if column == 'spatial_residual_m':
            np.testing.assert_allclose(left[column], right[column], atol=1e-6, rtol=1e-5)
        else:
            np.testing.assert_array_equal(left[column], right[column])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, default=Path('outputs/reports/20261003_vitg_validation'))
    parser.add_argument('--manifest', type=Path, default=Path('outputs/reports/20261006_vitg_pose_locked_motion_validation/run_manifest.csv'))
    parser.add_argument('--data-root', type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/reports/20261008_geometry_controls_corrected_units'))
    parser.add_argument('--max-relations', type=int)
    args = parser.parse_args()
    scripts = Path(__file__).resolve().parent
    rows = list(csv.DictReader(args.manifest.open()))
    if len(rows) != 110 or len({r['environment_id'] for r in rows}) != 47:
        raise ValueError('full fixed validation population required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'scheduled_population.json').write_text(json.dumps(rows, indent=2)+'\n')
    records = []
    for row in rows[:args.max_relations] if args.max_relations else rows:
        output = args.output_dir / 'relations' / row['run_id']
        output.mkdir(parents=True, exist_ok=True)
        record = {key: row[key] for key in ('run_id', 'environment_id', 'reference_scan', 'rescan_scan')}
        try:
            if not (output / 'regression.json').exists():
                with tempfile.TemporaryDirectory(prefix='wc_units_controls_') as scratch:
                    pair = Path(scratch) / 'pair'
                    common = ['--pair-run', str(pair), '--annotation-root', str(args.data_root), '--save-query-diagnostics']
                    commands = [
                        [sys.executable, str(scripts/'generate_3rscan_pairs.py'), '--root', str(args.data_root), '--metadata', str(args.data_root/'3RScan.json'),
                         '--reference', row['reference_scan'], '--rescan', row['rescan_scan'], '--output-dir', str(pair),
                         '--resolution', '224', '--reference-frames', '20', '--rescan-frames', '20'],
                        [sys.executable, str(scripts/'evaluate_3rscan_fpfh_baseline.py'), *common, '--output-dir', str(output/'fpfh'),
                         '--voxel-size', '.05', '--normal-radius', '.10', '--feature-radius', '.25', '--threshold', '.10', '--iterations', '5000'],
                        [str(Path.home() / 'data/conda/envs/worldcache_fcgf/bin/python'), str(scripts/'evaluate_3rscan_fcgf_baseline.py'), *common,
                         '--output-dir', str(output/'fcgf'), '--checkpoint', str(Path.home() / 'data/worldcache_models/fcgf/fcgf-3dmatch.pth'),
                         '--voxel-size', '.025', '--top-k', '20', '--threshold', '.10', '--iterations', '3000', '--seed', '17'],
                    ]
                    with (output/'runner.log').open('w') as log:
                        for command in commands:
                            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                old = args.old_root/'relations'/row['run_id']
                compare(old/'fcgf/query_diagnostics_k20.csv', output/'fcgf/query_diagnostics_k20.csv')
                for arm in ('fpfh_one_way', 'fpfh_mutual'):
                    compare(old/f'fpfh/query_diagnostics_{arm}.csv', output/f'fpfh/query_diagnostics_{arm}.csv')
                (output/'regression.json').write_text(json.dumps({'status':'pass','all_transfer_rows_reproduced':True,
                    'scope':'query labels, assignments, residuals reproduced; GT scene translation now in metres'},indent=2)+'\n')
            record.update(status='complete', error='')
        except Exception as error:
            record.update(status='failed', error=str(error))
        records.append(record)
        with (args.output_dir/'run_manifest.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(record));writer.writeheader();writer.writerows(records)
        print(f"{len(records)}/{len(rows)} {row['run_id']} {record['status']} {record['error']}",flush=True)


if __name__ == '__main__':
    main()
