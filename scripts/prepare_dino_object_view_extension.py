#!/usr/bin/env python3
"""Prepare frozen mesh-render inputs for the second eight train relations.

This is an exploratory extension of an already analyzed cohort; it does not
create an independent or preregistered split. Masks and frame selection follow
the existing 3RScan uniform20 DINO/REMIND protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from prepare_native_segment_pair import prepare
from worldcache.datasets.threerscan import ThreeRScanSequence


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path('data/3rscan')
RELATIONS = ROOT / 'outputs/reports/20261009_unseen_train_replication_extension/relations.csv'
OUTPUT = ROOT / 'outputs/reports/20261009_dino_object_view_budget_curve_extension'


def prepare_scan(scan: str, *, workers_root: Path, scan_inputs: Path) -> tuple[str, list[dict]]:
    destination = scan_inputs / scan
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        records = payload['records']
        for record in records:
            input_path = Path(record['input'])
            if not input_path.is_absolute():
                input_path = workers_root / input_path
            if hashlib.sha256(input_path.read_bytes()).hexdigest() != record['input_sha256']:
                raise ValueError(f'cached input checksum mismatch: {input_path}')
        return scan, records

    sequence = ThreeRScanSequence(root=DATA_ROOT, scan_id=scan)
    available = sequence.frame_ids()
    if not available:
        raise ValueError(f'no frames found for {scan}')
    expected = [available[i] for i in np.linspace(0, len(available) - 1, min(20, len(available)), dtype=int)]
    records = prepare(DATA_ROOT / scan, destination, 20, 'mesh_render', 'uniform')
    if [int(record['frame_id']) for record in records] != expected:
        raise ValueError(f'uniform frame schedule mismatch for {scan}')
    manifest_path.write_text(json.dumps({'records': records}, indent=2) + '\n')
    return scan, records


def write_same_or_new(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f'refusing to change frozen artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('--workers must be between 1 and 4')

    rows = list(csv.DictReader(RELATIONS.open()))
    if len(rows) != 8:
        raise ValueError(f'expected eight extension relations, got {len(rows)}')
    scans = sorted({row[key] for row in rows for key in ('reference_scan', 'rescan_scan')})
    scan_inputs = OUTPUT / 'scan_inputs'
    scan_inputs.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(prepare_scan, scan, workers_root=ROOT, scan_inputs=scan_inputs): scan
            for scan in scans
        }
        for future in as_completed(futures):
            scan, _ = future.result()
            done.add(scan)
            print(f'prepared {len(done)}/{len(scans)} scans: {scan}', flush=True)

    for index, row in enumerate(rows, start=8):
        run_id = f"{index:03d}_{row['reference_scan'][:8]}_{row['rescan_scan'][:8]}"
        role_scans = {'reference': row['reference_scan'], 'rescan': row['rescan_scan']}
        records = {
            role: json.loads((scan_inputs / scan / 'manifest.json').read_text())['records']
            for role, scan in role_scans.items()
        }
        destination = OUTPUT / 'uniform20_inputs' / run_id
        write_same_or_new(destination / 'manifest.json', {
            'selected_mask_source': 'mesh_render',
            'frame_selection': 'uniform',
            'resolution': 224,
            'records': records,
            'scope': 'post-hoc second hash-selected train extension; same frozen mesh-render mask protocol',
        })
        pair_map = {
            'records': [
                {'query_frame': int(query['frame_id']), 'reference_frame': int(reference['frame_id'])}
                for query, reference in zip(records['rescan'], records['reference'])
            ]
        }
        write_same_or_new(OUTPUT / 'dino_segment_predictions' / run_id / 'inference_summary.json', pair_map)
        print(f'frozen {run_id}: ref={len(records["reference"])} query={len(records["rescan"])}', flush=True)

    relation_copy = OUTPUT / 'relations.csv'
    expected_bytes = RELATIONS.read_bytes()
    if relation_copy.exists() and relation_copy.read_bytes() != expected_bytes:
        raise FileExistsError(f'relation manifest differs: {relation_copy}')
    relation_copy.write_bytes(expected_bytes)
    digest = hashlib.sha256(expected_bytes).hexdigest()
    print(f'complete: 8 relations / {len(scans)} scans; relations SHA-256 {digest}')


if __name__ == '__main__':
    main()
