#!/usr/bin/env python3
"""Reproduce cached MASt3R consensus and rescore using raw metre metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from evaluate_mast3r_consensus_seeds import evaluate


def rescore(task: tuple[str, str, list]) -> dict:
    relation_name, output_name, metadata = task
    relation, output = Path(relation_name), Path(output_name)
    identity = json.loads((relation/'relation.json').read_text())
    scan = next(s for r in metadata if r['reference']==identity['reference_scan'] for s in r['scans'] if s['reference']==identity['rescan_scan'])
    gt = np.asarray(scan['transform'], dtype=np.float32).reshape(4,4).T.astype(float)
    saved = json.loads((relation/'summary.json').read_text())
    new = evaluate(relation/'candidate_cache.npz', output, int(saved['ransac_seed']), .1, int(saved['ransac_iterations']), gt)
    left, right = pd.read_csv(relation/'query_diagnostics.csv'), pd.read_csv(output/'query_diagnostics.csv')
    for column in right:
        if column not in left:
            raise ValueError(f'missing original diagnostic {column}')
        if column=='spatial_residual_m':
            np.testing.assert_allclose(left[column], right[column], atol=1e-7, rtol=1e-6)
        else:
            np.testing.assert_array_equal(left[column], right[column])
    matrix = np.load(output/'predicted_rescan_to_reference.npy')
    with np.load(relation/'candidate_cache.npz') as z:
        old_gt = z['ground_truth_transform']
    old_rot = float(np.degrees(Rotation.from_matrix(matrix[:3,:3].T @ old_gt[:3,:3]).magnitude()))
    old_translation = float(np.linalg.norm(matrix[:3,3]-old_gt[:3,3]))
    np.testing.assert_allclose(old_rot,saved['scene_rotation_error_deg'],atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(old_translation,saved['scene_translation_error_m'],atol=1e-6,rtol=1e-6)
    regression={'status':'pass','all_transfer_rows_reproduced':True,'old_pose_errors_reproduced':True,
                'scene_transform_units':'raw released metadata in metres'}
    (output/'regression.json').write_text(json.dumps(regression,indent=2)+'\n')
    return {'run_id':identity['run_id'],'environment_id':identity['environment_id'],
            'reference_scan':identity['reference_scan'],'rescan_scan':identity['rescan_scan'],
            'old_pose_success':bool(saved['strict_scene_pose_success']),
            'strict_pose_success':new['strict_scene_pose_success'],
            'rotation_error_degrees':new['scene_rotation_error_deg'],'translation_error_m':new['scene_translation_error_m']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=(Path.home() / 'data/worldcache_datasets/3rscan'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    paths=sorted((args.source/'relations').glob('*/candidate_cache.npz'))
    if len(paths)!=110:
        raise ValueError('full 110-relation validation population required')
    metadata=json.loads((args.data_root/'3RScan.json').read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    tasks=[(str(p.parent),str(args.output_dir/'relations'/p.parent.name),metadata) for p in paths]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(rescore,tasks):
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output_dir/'per_relation.csv',index=False)
            print(f"{len(rows)}/110 {row['run_id']} corrected pose={row['strict_pose_success']}",flush=True)
    frame=pd.DataFrame(rows)
    if frame.environment_id.nunique()!=47:
        raise ValueError('47 validation environments required')
    env=frame.groupby('environment_id')[['old_pose_success','strict_pose_success']].mean()
    env.to_csv(args.output_dir/'per_environment.csv')
    summary={'relations':110,'environments':47,'old_pose_successes':int(frame.old_pose_success.sum()),
             'corrected_pose_successes':int(frame.strict_pose_success.sum()),
             'corrected_environment_macro_pose_success':float(env.strict_pose_success.mean()),
             'all_transfer_rows_reproduced':True,'scene_transform_units':'raw released metadata in metres'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
