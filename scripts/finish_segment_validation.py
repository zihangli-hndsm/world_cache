"""Score the frozen population once both inference manifests are complete."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--timeout-hours', type=float, default=6)
    args = parser.parse_args()
    root = args.output_root
    expected = {r['run_id'] for r in json.loads((root / 'frozen_segment_population.json').read_text())['population']}
    deadline = time.monotonic() + args.timeout_hours * 3600
    previous = None
    while time.monotonic() < deadline:
        manifests = []
        try:
            for name in ['dino_inference_manifest.json', 'segment_inference_manifest_dataset.json']:
                manifests.append(json.loads((root / name).read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(5)
            continue
        progress = tuple(sum(r['status'] == 'complete' for r in rows) for rows in manifests)
        if progress != previous:
            print('complete DINO/SegMASt3R:', progress, flush=True)
            previous = progress
        if all(len(rows) == len(expected) and {r['run_id'] for r in rows} == expected for rows in manifests):
            if any(r['status'] != 'complete' for rows in manifests for r in rows):
                raise RuntimeError('Full manifest contains failed attempts: explicit failure scoring is required.')
            subprocess.run([sys.executable, 'scripts/score_segment_matcher_population.py',
                            '--output-root', str(root), '--input-preprocessing', 'official-dataset'], check=True)
            print('full frozen scoring complete', flush=True)
            return
        time.sleep(5)
    raise TimeoutError('Inference not complete; preserve current jobs and inspect before restarting.')


if __name__ == '__main__':
    main()
