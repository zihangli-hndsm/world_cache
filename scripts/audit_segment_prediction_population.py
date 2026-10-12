"""Integrity audit for anonymous saved segment assignments and their inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--preprocessing', choices=['official-demo', 'official-dataset'], default='official-dataset')
    args = ap.parse_args()
    root = args.output_root
    frozen = json.loads((root / 'frozen_segment_population.json').read_text())['population']
    variant = 'segment_predictions' if args.preprocessing == 'official-demo' else 'segment_predictions_dataset'
    manifest_name = 'segment_inference_manifest.json' if args.preprocessing == 'official-demo' else 'segment_inference_manifest_dataset.json'
    manifest = json.loads((root / manifest_name).read_text())
    expected_ids = {x['run_id'] for x in frozen}
    if len(manifest) != len(frozen) or {x['run_id'] for x in manifest} != expected_ids or any(x['status'] != 'complete' for x in manifest):
        raise ValueError('inference manifest is incomplete or has failed attempts')
    pairs = 0
    assigned = 0
    total_masks = 0
    for relation in frozen:
        prepared = json.loads((root / 'uniform20_inputs' / relation['run_id'] / 'manifest.json').read_text())
        summary = json.loads((root / variant / relation['run_id'] / 'inference_summary.json').read_text())
        if summary.get('identity_targets_read') or summary.get('semantic_labels_read'):
            raise ValueError('labels read during matching')
        frozen_pairs = list(zip(relation['reference_frames'], relation['rescan_frames']))
        records = summary['records']
        if [(x['reference_frame'], x['query_frame']) for x in records] != frozen_pairs:
            raise ValueError(f"frame order mismatch: {relation['run_id']}")
        for ref, query, record in zip(prepared['records']['reference'], prepared['records']['rescan'], records):
            status = record.get('status', 'complete')
            if status not in {'complete', 'empty_mask_input'}:
                raise ValueError('incomplete frame-pair inference')
            if status == 'empty_mask_input' and ref['masks'] != 0 and query['masks'] != 0:
                raise ValueError('empty-mask status requires an empty side')
            if record['input_hashes'] != [ref['input_sha256'], query['input_sha256']]:
                raise ValueError('recorded input hash differs from frozen inputs')
            for source, expected in zip([ref['input'], query['input']], record['input_hashes']):
                if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
                    raise ValueError('anonymous input hash mismatch')
            with np.load(record['predictions']) as pred:
                match = pred['reference_to_query'].copy()
                scores = pred['scores_with_dustbin']
            if match.shape != (ref['masks'],) or scores.shape != (ref['masks'] + 1, query['masks'] + 1):
                raise ValueError(f"prediction support mismatch {relation['run_id']} frames {record['reference_frame']}/{record['query_frame']}: match={match.shape}, scores={scores.shape}, ref_masks={ref['masks']}, query_masks={query['masks']}")
            if not np.isfinite(scores).all() or np.any(match < -1) or np.any(match >= query['masks']):
                raise ValueError('nonfinite score or out-of-range assignment')
            valid = match >= 0
            if len(np.unique(match[valid])) != int(valid.sum()):
                raise ValueError('reference-to-query assignment is not injective')
            if status == 'empty_mask_input' and int(valid.sum()) != 0:
                raise ValueError('empty-mask input cannot produce assignments')
            pairs += 1
            assigned += int(valid.sum())
            total_masks += len(match)
    result = dict(status='pass', relations=len(frozen), frame_pairs=pairs,
                  reference_masks=total_masks, assigned_segments=assigned,
                  preprocessing=args.preprocessing,
                  scope='Saved anonymous predictions and exact frozen inputs; this is prediction integrity, not identity accuracy.')
    target = root / f"{variant}_integrity_audit.json"
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
