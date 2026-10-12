"""Integrity audit for saved DINO segment scores and nearest-neighbor controls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--allow-incomplete', action='store_true', help='Audit completed runs only and label the artifact partial.')
    args = ap.parse_args()
    root = args.output_root
    frozen = json.loads((root / 'frozen_segment_population.json').read_text())['population']
    manifest = json.loads((root / 'dino_inference_manifest.json').read_text())
    expected_ids = {x['run_id'] for x in frozen}
    manifest_ids = {x['run_id'] for x in manifest}
    if (not manifest_ids <= expected_ids or len(manifest_ids) != len(manifest)
            or (not args.allow_incomplete and manifest_ids != expected_ids)
            or any(x['status'] != 'complete' for x in manifest)):
        raise ValueError('DINO inference manifest is incomplete or has failures')
    frame_pairs = total_query_masks = total_nn = total_mutual = 0
    relations_to_audit = [x for x in frozen if x['run_id'] in manifest_ids]
    for relation in relations_to_audit:
        prepared = json.loads((root / 'uniform20_inputs' / relation['run_id'] / 'manifest.json').read_text())
        summary = json.loads((root / 'dino_segment_predictions' / relation['run_id'] / 'inference_summary.json').read_text())
        if summary.get('identity_targets_read') or summary.get('semantic_labels_read'):
            raise ValueError('labels read during DINO matching')
        expected = list(zip(relation['reference_frames'], relation['rescan_frames']))
        records = summary['records']
        if [(x['reference_frame'], x['query_frame']) for x in records] != expected:
            raise ValueError(f"frame order mismatch: {relation['run_id']}")
        for ref, query, record in zip(prepared['records']['reference'], prepared['records']['rescan'], records):
            for src, digest in [(ref['input'], ref['input_sha256']), (query['input'], query['input_sha256'])]:
                if hashlib.sha256(Path(src).read_bytes()).hexdigest() != digest:
                    raise ValueError('frozen DINO input hash mismatch')
            with np.load(record['predictions']) as prediction:
                scores = prediction['scores'].copy()
                selected = prediction['query_to_reference'].copy()
                mutual = prediction['mutual_assigned'].copy()
            if scores.shape != (query['masks'], ref['masks']) or selected.shape != (query['masks'],) or mutual.shape != selected.shape:
                raise ValueError(f"candidate support mismatch: {relation['run_id']} {record['reference_frame']}/{record['query_frame']}")
            if not np.isfinite(scores).all() or np.any(selected < -1) or np.any(selected >= ref['masks']):
                raise ValueError('nonfinite score or out-of-range nearest neighbor')
            if min(scores.shape, default=0) > 0:
                expected_selected = scores.argmax(1)
                reverse = scores.argmax(0)
                expected_mutual = reverse[expected_selected] == np.arange(len(expected_selected))
                np.testing.assert_array_equal(selected, expected_selected)
                np.testing.assert_array_equal(mutual, expected_mutual)
            elif np.any(selected != -1) or np.any(mutual):
                raise ValueError('empty candidate support must have no assignments')
            frame_pairs += 1
            total_query_masks += len(selected)
            total_nn += int((selected >= 0).sum())
            total_mutual += int(mutual.sum())
    complete = len(relations_to_audit) == len(frozen)
    result = dict(status='pass' if complete else 'partial_pass', relations=len(relations_to_audit), frame_pairs=frame_pairs,
                  query_masks=total_query_masks, nearest_neighbor_assignments=total_nn,
                  mutual_assignments=total_mutual,
                  scope='Saved anonymous DINO scores/indices and exact frozen inputs; this is prediction integrity, not identity accuracy.')
    target = root / ('dino_segment_integrity_audit.json' if complete else 'dino_segment_integrity_audit_partial.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
