"""Freeze full validation before matching; prepare each shared scan once."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path

# Bound ordinary preparation workers; no neural model runs in these workers.
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np

from worldcache.datasets.threerscan import ThreeRScanSequence


OUTPUT = Path('outputs/reports/20261008_segmast3r_validation')
ROOT = Path('data/3rscan')


def prepare_scan(scan):
    from prepare_native_segment_pair import prepare
    destination = OUTPUT / 'scan_inputs' / scan
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'manifest.json'
    if manifest.exists():
        return scan, json.loads(manifest.read_text())
    records = prepare(ROOT / scan, destination, 20, 'mesh_render', 'uniform')
    manifest.write_text(json.dumps(records, indent=2) + '\n')
    return scan, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('use one to four bounded CPU workers')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = Path('outputs/reports/20260929_validation_access/validation_pair_manifest.csv')
    relations = list(csv.DictReader(manifest_path.open()))
    if len(relations) != 110 or len({r['environment_id'] for r in relations}) != 47:
        raise ValueError('expected full official validation110/47')
    scans = sorted({r[k] for r in relations for k in ['reference_scan', 'rescan_scan']})
    frames = {}
    for scan in scans:
        available = ThreeRScanSequence(root=ROOT, scan_id=scan).frame_ids()
        if not available:
            raise ValueError(f'empty native input:{scan}; retain failure, do not silently omit')
        frames[scan] = [available[i] for i in np.linspace(0, len(available)-1, min(20, len(available)), dtype=int)]
    prototype = Path('outputs/reports/20261008_segmast3r_feasibility')
    protocol = json.loads((prototype / 'SEGMENT_PROTOCOL.json').read_text())
    protocol.update(next_scope='All110official validation relations/47released-reference groups; up to20uniform frames per scan, chronological-rank pairing; the native single-frame rescan retained',
                    inference_preprocessing='official-dataset: PIL bilinear512square and RGB[-1,1] on the common224source RGB',
                    scope='Rules fixed from exploratory eight-group integration; new validation segment matching outcomes, not independent sites or another acquisition dataset',
                    source_prototype_results_sha256=hashlib.sha256((prototype / 'segment_comparison_dataset/summary.json').read_bytes()).hexdigest())
    protocol_path = OUTPUT / 'SEGMENT_PROTOCOL.json'
    serialized = json.dumps(protocol, indent=2) + '\n'
    if protocol_path.exists() and protocol_path.read_text() != serialized:
        raise ValueError('frozen protocol differs; do not overwrite')
    protocol_path.write_text(serialized)
    population = [dict(r, run_id=f"{i:03d}_{r['reference_scan'][:8]}_{r['rescan_scan'][:8]}",
                       reference_frames=frames[r['reference_scan']], rescan_frames=frames[r['rescan_scan']])
                  for i, r in enumerate(relations)]
    frozen = dict(source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(), population=population,
                  frame_pairs=sum(min(len(r['reference_frames']), len(r['rescan_frames'])) for r in population),
                  scope=protocol['scope'])
    path = OUTPUT / 'frozen_segment_population.json'
    serialized = json.dumps(frozen, indent=2) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise ValueError('frozen validation population changed')
    path.write_text(serialized)
    print(f"frozen110relations/47groups/{frozen['frame_pairs']}frame pairs/{len(scans)}shared scans", flush=True)
    completed, failed = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(prepare_scan, scan): scan for scan in scans}
        for future in as_completed(futures):
            scan = futures[future]
            try:
                _, records = future.result()
                if [r['frame_id'] for r in records] != frames[scan]:
                    raise ValueError('cached frame selection mismatch')
                for record in records:
                    if hashlib.sha256(Path(record['input']).read_bytes()).hexdigest() != record['input_sha256']:
                        raise ValueError('input cache changed')
                completed[scan] = records
            except Exception as error:
                failed[scan] = str(error)
            (OUTPUT / 'scan_preparation_progress.json').write_text(json.dumps(dict(completed_scans=sorted(completed), failed=failed), indent=2) + '\n')
            print(f"scans{len(completed)}/{len(scans)} failures{len(failed)} latest{scan}", flush=True)
    attempts = []
    for r in population:
        row = dict(run_id=r['run_id'], status='failed', error='')
        if r['reference_scan'] in completed and r['rescan_scan'] in completed:
            destination = OUTPUT / 'uniform20_inputs' / r['run_id']
            destination.mkdir(parents=True, exist_ok=True)
            m = dict(selected_mask_source='mesh_render', frame_selection='uniform', resolution=224,
                     records=dict(reference=completed[r['reference_scan']], rescan=completed[r['rescan_scan']]))
            p = destination / 'manifest.json'
            p.write_text(json.dumps(m, indent=2) + '\n')
            row.update(status='complete', manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        else:
            row['error'] = 'input preparation failure retained; see scan_preparation_progress.json'
        attempts.append(row)
    (OUTPUT / 'uniform20_preparation_manifest.json').write_text(json.dumps(attempts, indent=2) + '\n')
    print(f"completed{sum(r['status']=='complete' for r in attempts)}/110relation preparations", flush=True)


if __name__ == '__main__':
    main()
