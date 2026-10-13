"""Prepare all frozen segmentation-assisted attempts, retaining failures."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    output = Path('outputs/reports/20261008_segmast3r_feasibility')
    frozen = json.loads((output / 'frozen_segment_population.json').read_text())
    if hashlib.sha256((output / 'SEGMENT_PROTOCOL.json').read_bytes()).hexdigest() != frozen['protocol_sha256']:
        raise ValueError('frozen segment protocol changed')
    rows = []
    for r in frozen['population']:
        directory = output / 'uniform20_inputs' / r['run_id']
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(run_id=r['run_id'], status='failed', error='')
        try:
            if not (directory / 'manifest.json').exists():
                command = [sys.executable, 'scripts/prepare_native_segment_pair.py',
                           '--data-root', 'data/3rscan',
                           '--reference', r['reference_scan'], '--rescan', r['rescan_scan'],
                           '--mask-source', 'mesh_render', '--frame-selection', 'uniform',
                           '--frames', '20', '--output-dir', str(directory)]
                env = dict(os.environ, PYTHONPATH='src')
                with (directory / 'preparation.log').open('w') as log:
                    subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest['selected_mask_source'] != 'mesh_render' or manifest['frame_selection'] != 'uniform':
                raise ValueError('preparation rule mismatch')
            for role in ['reference', 'rescan']:
                if [a['frame_id'] for a in manifest['records'][role]] != r[role + '_frames']:
                    raise ValueError('frame selection differs from frozen population')
                for a in manifest['records'][role]:
                    if hashlib.sha256(Path(a['input']).read_bytes()).hexdigest() != a['input_sha256']:
                        raise ValueError('prepared input hash mismatch')
            row.update(status='complete', manifest_sha256=hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest())
        except Exception as error:
            row['error'] = str(error)
        rows.append(row)
        (output / 'uniform20_preparation_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(f"{len(rows)}/{len(frozen['population'])} {row}", flush=True)


if __name__ == '__main__':
    main()
