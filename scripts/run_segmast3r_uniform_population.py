"""Run every frozen uniform-frame attempt; persist failed attempts."""
import hashlib
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-preprocessing', choices=['official-demo', 'official-dataset'], default='official-demo')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/reports/20261008_segmast3r_feasibility'))
    args = parser.parse_args()
    output = args.output_root
    frozen = json.loads((output / 'frozen_segment_population.json').read_text())
    prepared = {r['run_id']: r for r in json.loads((output / 'uniform20_preparation_manifest.json').read_text())}
    rows = []
    for r in frozen['population']:
        variant = 'segment_predictions' if args.input_preprocessing == 'official-demo' else 'segment_predictions_dataset'
        destination = output / variant / r['run_id']
        destination.mkdir(parents=True, exist_ok=True)
        manifest = output / 'uniform20_inputs' / r['run_id'] / 'manifest.json'
        record = dict(run_id=r['run_id'], status='failed', error='')
        try:
            if prepared[r['run_id']]['status'] != 'complete':
                raise ValueError('input preparation failed')
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != prepared[r['run_id']]['manifest_sha256']:
                raise ValueError('prepared manifest changed')
            if not (destination / 'inference_summary.json').exists():
                command = [sys.executable, 'scripts/infer_segmast3r_segment_pair.py',
                           '--weights', 'data/models/segmast3r/segmast3r_spp_inference.pt',
                           '--input-manifest', str(manifest), '--output-dir', str(destination),
                           '--input-preprocessing', args.input_preprocessing]
                with (destination / 'runner.log').open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            summary = json.loads((destination / 'inference_summary.json').read_text())
            expected = list(zip(r['reference_frames'], r['rescan_frames']))
            actual = [(a['reference_frame'], a['query_frame']) for a in summary['records']]
            if actual != expected or summary['identity_targets_read'] or summary['semantic_labels_read']:
                raise ValueError('prediction population or label isolation mismatch')
            record.update(status='complete', frame_pairs=len(actual))
        except Exception as error:
            record['error'] = str(error)
        rows.append(record)
        manifest_name = 'segment_inference_manifest.json' if args.input_preprocessing == 'official-demo' else 'segment_inference_manifest_dataset.json'
        (output / manifest_name).write_text(json.dumps(rows, indent=2) + '\n')
        print(f"{len(rows)}/{len(frozen['population'])} {record}", flush=True)


if __name__ == '__main__':
    main()
