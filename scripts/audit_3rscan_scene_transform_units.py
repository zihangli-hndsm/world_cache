#!/usr/bin/env python3
"""Empirically audit metadata scale/direction on static annotated mesh surfaces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from audit_object_motion_oracle import read_ply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument('--manifest', type=Path, default=Path('outputs/reports/20261006_vitg_pose_locked_motion_validation/run_manifest.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/reports/20261008_scene_transform_unit_audit'))
    parser.add_argument('--max-relations', type=int)
    args = parser.parse_args()
    metadata = json.loads((args.data_root / '3RScan.json').read_text())
    relations = list(csv.DictReader(args.manifest.open()))
    if args.max_relations:
        relations = relations[:args.max_relations]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cached_ref, ref_xyz, ref_oid = None, None, None
    for relation in relations:
        reference, rescan = relation['reference_scan'], relation['rescan_scan']
        record = next(s for r in metadata if r['reference'] == reference for s in r['scans'] if s['reference'] == rescan)
        if reference != cached_ref:
            ref_xyz, ref_oid, _ = read_ply(args.data_root / reference / 'labels.instances.annotated.v2.ply')
            cached_ref = reference
        xyz, oid, _ = read_ply(args.data_root / rescan / 'labels.instances.annotated.v2.ply')
        changed = [item for item in record.get('rigid', []) if isinstance(item, dict)]
        ref_changed = [int(item['instance_reference']) for item in changed]
        cur_changed = [int(item['instance_rescan']) for item in changed]
        # Exclude all annotated rigid motions; use only unchanged local mesh surfaces.
        static_ref = ref_xyz[~np.isin(ref_oid, ref_changed)]
        static_cur = xyz[~np.isin(oid, cur_changed)]
        if len(static_ref) == 0 or len(static_cur) == 0:
            raise ValueError('no static surfaces')
        sampled = static_cur[np.linspace(0, len(static_cur)-1, min(20000, len(static_cur)), dtype=int)]
        tree = cKDTree(static_ref)
        raw = np.asarray(record['transform'], dtype=float).reshape(4, 4).T
        row = {'run_id': relation['run_id'], 'reference_scan': reference, 'rescan_scan': rescan,
               'metadata_translation_norm': float(np.linalg.norm(raw[:3, 3])), 'sampled_static_vertices': len(sampled)}
        for name, scale, inverse in [('raw_forward', 1., False), ('scaled_001_forward', .001, False),
                                      ('raw_inverse', 1., True), ('scaled_001_inverse', .001, True)]:
            matrix = raw.copy()
            matrix[:3, 3] *= scale
            if inverse:
                matrix = np.linalg.inv(matrix)
            points = sampled @ matrix[:3, :3].T + matrix[:3, 3]
            distances = tree.query(points, k=1)[0]
            row[name + '_median_m'] = float(np.median(distances))
            row[name + '_fraction_within_5cm'] = float(np.mean(distances <= .05))
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir / 'per_relation.csv', index=False)
        print(f"{len(rows)}/{len(relations)} {relation['run_id']} raw={row['raw_forward_median_m']:.4f} scaled={row['scaled_001_forward_median_m']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    result = {'relations': len(frame), 'raw_forward_better_median_relations': int((frame.raw_forward_median_m < frame.scaled_001_forward_median_m).sum()),
              'raw_forward_better_5cm_support_relations': int((frame.raw_forward_fraction_within_5cm > frame.scaled_001_forward_fraction_within_5cm).sum()),
              'median_relation_raw_forward_median_m': float(frame.raw_forward_median_m.median()),
              'median_relation_scaled_001_forward_median_m': float(frame.scaled_001_forward_median_m.median()),
              'scope': 'evaluation-only geometric consistency audit; no fitted registration or descriptor use',
              'interpretation': 'compare observed static-surface alignment against the documentation; do not select scale separately for each relation'}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
