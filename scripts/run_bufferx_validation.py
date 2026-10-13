#!/usr/bin/env python3
"""Schedule all fixed validation relations, retaining failures and bounded scratch."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose-root', type=Path, default=Path('outputs/reports/20261006_vitg_pose_locked_motion_validation'))
    parser.add_argument('--data-root', type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/reports/20261008_bufferx_validation'))
    parser.add_argument('--max-relations', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    rows = list(csv.DictReader((args.pose_root / 'run_manifest.csv').open()))
    if len(rows) != 110 or len({row['environment_id'] for row in rows}) != 47:
        raise ValueError('fixed 110-relation / 47-environment validation population required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'scheduled_population.json').write_text(json.dumps(rows, indent=2) + '\n')
    completed = []
    for row in rows[:args.max_relations] if args.max_relations else rows:
        output = args.output_dir / 'relations' / row['run_id']
        output.mkdir(parents=True, exist_ok=True)
        record = {k: row[k] for k in ('run_id', 'environment_id', 'reference_scan', 'rescan_scan')}
        try:
            if not (output / 'result.json').exists():
                with tempfile.TemporaryDirectory(prefix='wc_bufferx_val_') as scratch:
                    pair = Path(scratch) / 'pair'
                    geometry = output / 'geometry'
                    commands = [
                        [sys.executable, str(root / 'scripts/generate_3rscan_pairs.py'), '--root', str(args.data_root),
                         '--metadata', str(args.data_root / '3RScan.json'), '--reference', row['reference_scan'],
                         '--rescan', row['rescan_scan'], '--output-dir', str(pair), '--resolution', '224',
                         '--reference-frames', '20', '--rescan-frames', '20'],
                        [sys.executable, str(root / 'scripts/prepare_bufferx_rgbd_geometry.py'),
                         '--pair-run', str(pair), '--output-dir', str(geometry)],
                        ['bash', str(root / 'scripts/run_bufferx_runtime.sh'), str(root / 'scripts/evaluate_bufferx_native_pair.py'),
                         '--geometry-dir', str(geometry), '--output-dir', str(output)],
                    ]
                    with (output / 'runner.log').open('w') as log:
                        subprocess.run(commands[0], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                        manifest = json.loads((pair / 'manifest.json').read_text())
                        observed = {entry['frame_id'] for entry in manifest['rescan']}
                        with np.load(args.pose_root / 'descriptor_candidates' / (row['run_id'] + '.npz')) as cache:
                            if not set(map(int, cache['query_frame_ids'])).issubset(observed):
                                raise ValueError('regenerated observations do not cover fixed query frames')
                        subprocess.run(commands[1], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                        subprocess.run(commands[2], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            result = json.loads((output / 'result.json').read_text())
            if result['reference'] != row['reference_scan'] or result['rescan'] != row['rescan_scan']:
                raise ValueError('baseline relation identity mismatch')
            record.update(status='complete', error='')
        except Exception as error:
            record.update(status='failed', error=str(error))
        completed.append(record)
        path = args.output_dir / 'run_manifest.csv'
        temp = path.with_suffix('.tmp')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(completed)
        temp.replace(path)
        print(f"{len(completed)}/{len(rows)} {row['run_id']} {record['status']} {record['error']}", flush=True)


if __name__ == '__main__':
    main()
