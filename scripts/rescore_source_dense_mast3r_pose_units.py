#!/usr/bin/env python3
"""Reproduce source dense MASt3R caches before correcting scene pose units."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from evaluate_mast3r_consensus_seeds import evaluate
from generate_supplementary_tables import short_pair_name

OUT=Path('outputs/reports/20261008_source_dense_mast3r_corrected_units')

def rescore(task):
    name,seed=task
    source=Path('outputs/runs/20261003_mast3r_dense_seed_sweep')/name/f'seed_{seed}'
    saved=json.loads((source/'summary.json').read_text())
    cache=Path('outputs/runs/20261003_mast3r_dense_seed_cache')/name/'candidate_cache.npz'
    pair=short_pair_name(name)
    config=json.loads((Path('outputs/reports/20261008_corrected_source_pair_views')/pair/'config.json').read_text())
    gt=np.asarray(config['rescan_to_reference'])
    output=OUT/name/f'seed_{seed}'
    new=evaluate(cache,output,seed,.1,1500,gt)
    old_diag=pd.read_csv(source/'query_diagnostics.csv');new_diag=pd.read_csv(output/'query_diagnostics.csv')
    for column in new_diag:
        if column=='spatial_residual_m':
            np.testing.assert_allclose(old_diag[column],new_diag[column],atol=1e-7,rtol=1e-6)
        else:np.testing.assert_array_equal(old_diag[column],new_diag[column])
    matrix=np.load(output/'predicted_rescan_to_reference.npy')
    with np.load(cache) as z: old_gt=z['ground_truth_transform']
    old_rotation=np.degrees(Rotation.from_matrix(matrix[:3,:3].T@old_gt[:3,:3]).magnitude())
    old_translation=np.linalg.norm(matrix[:3,3]-old_gt[:3,3])
    np.testing.assert_allclose([old_rotation,old_translation],[saved['scene_rotation_error_deg'],saved['scene_translation_error_m']],atol=1e-6,rtol=1e-6)
    (output/'regression.json').write_text(json.dumps({'status':'pass','all_transfer_rows_reproduced':True,'old_pose_errors_reproduced':True})+'\n')
    return dict(pair=pair,source_run=name,ransac_seed=seed,strict_pose_success=new['strict_scene_pose_success'],rotation_error_deg=new['scene_rotation_error_deg'],translation_error_m=new['scene_translation_error_m'])

def main():
    caches=sorted(Path('outputs/runs/20261003_mast3r_dense_seed_cache').glob('*/candidate_cache.npz'))
    if len(caches)!=14:raise ValueError('14 matched source caches required')
    tasks=[(p.parent.name,seed) for p in caches for seed in [17,29,41]]
    OUT.mkdir(exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(rescore,tasks):
            rows.append(row);pd.DataFrame(rows).to_csv(OUT/'per_relation_seed.csv',index=False)
            print(len(rows),'/',len(tasks),row['pair'],row['ransac_seed'],flush=True)
    frame=pd.DataFrame(rows)
    counts=frame.groupby('ransac_seed').strict_pose_success.sum().to_dict()
    (OUT/'summary.json').write_text(json.dumps({'relations':14,'all_transfer_rows_reproduced':True,'corrected_pose_counts':counts},indent=2)+'\n')

if __name__=='__main__':main()
