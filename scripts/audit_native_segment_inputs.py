"""Audit saved anonymous masks against their separate annotation targets."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for manifest in sorted((args.output_root / 'scan_inputs').glob('*/manifest.json')):
        records = json.loads(manifest.read_text())
        for record in records:
            source = Path(record['input'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != record['input_sha256']:
                raise ValueError(f'input hash mismatch: {source}')
            ids = np.load(record['identity_targets'])
            audit_path = source.with_name(source.name.replace('_input.npz', '_projection_audit.npz'))
            with np.load(source) as data, np.load(audit_path) as audit:
                masks = data['masks']
                labels = audit['rendered_labels']
                measured = audit['measured_labels']
                eligible, counts = np.unique(labels, return_counts=True)
                expected = eligible[(eligible > 0) & (counts >= 4)]
                if not np.array_equal(ids, expected) or masks.shape != (len(ids), 224, 224):
                    raise ValueError(f'annotation support mismatch: {source}')
                if record['masks'] != len(ids) or data['rgb'].shape != (224, 224, 3):
                    raise ValueError(f'input metadata mismatch: {source}')
                for index, identity in enumerate(ids):
                    if not np.array_equal(masks[index], labels == identity):
                        raise ValueError(f'mask/identity mismatch: {source}, index {index}')
                common = (labels > 0) & (measured > 0)
                rows.append(dict(scan=manifest.parent.name, frame=record['frame_id'], masks=len(ids),
                                 common_pixels=int(common.sum()), agreeing_pixels=int(((labels == measured) & common).sum())))
    result = dict(status='pass', scans=len({r['scan'] for r in rows}), images=len(rows),
                  masks=sum(r['masks'] for r in rows), records=rows,
                  scope='Exact saved mask/UID/render correspondence. Shared annotations and poses; not independent physical identity certification.')
    destination = args.output_root / 'annotation_input_audit.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
