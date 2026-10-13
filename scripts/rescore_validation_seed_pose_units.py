#!/usr/bin/env python3
"""Reproduce all saved ViT-G or dense MASt3R validation consensus seeds."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from evaluate_3rscan_multimotion import fit_fixed_consensus
from evaluate_mast3r_consensus_seeds import evaluate

ROOT=Path('outputs/reports')

def rescore(task):
    identity,method,metadata=task
    run=identity['run_id'];out=ROOT/f'20261008_{method}_validation_seed_corrected_units'/run
    scan=next(s for r in metadata if r['reference']==identity['reference_scan'] for s in r['scans'] if s['reference']==identity['rescan_scan'])
    gt=np.asarray(scan['transform'],dtype=np.float32).reshape(4,4).T.copy();old_gt=gt.copy();old_gt[:3,3]*=.001
    rows=[]
    if method=='vitg':
        source=ROOT/'20261004_vitg_validation_seed_sensitivity/relations'/run/'vitg'
        summaries=pd.read_csv(source/'summary.csv').set_index('ransac_seed')
        diagnostics=pd.read_csv(source/'query_diagnostics.csv')
        with np.load(ROOT/'20261006_vitg_pose_locked_motion_validation/descriptor_candidates'/f'{run}.npz') as z:
            q,ref,candidates=z['query_points'],z['reference_points'],z['candidate_indices'][:,:20]
    for seed in [17,29,41]:
        output=out/f'seed_{seed}';output.mkdir(parents=True,exist_ok=True)
        if method=='vitg':
            saved=summaries.loc[seed]
            diag=diagnostics[diagnostics.ransac_seed.eq(seed)].sort_values('query_index')
            hypothesis=fit_fixed_consensus(q,ref,candidates,threshold=.1,iterations=1500,seed=seed)
            np.testing.assert_array_equal(diag.consensus_inlier,hypothesis.inliers)
            np.testing.assert_allclose(diag.spatial_residual_m,hypothesis.residuals,atol=1e-7,rtol=1e-6)
            matrix=hypothesis.matrix
            old_rotation=saved.rotation_error_deg;old_translation=saved.translation_error_m
            np.save(output/'predicted_rescan_to_reference.npy',matrix)
        else:
            cache=ROOT/'20261004_mast3r_dense_official_validation/relations'/run/'candidate_cache.npz'
            source=Path('outputs/runs/20261004_mast3r_dense_validation_seed_sweep')/run/f'seed_{seed}'
            saved=json.loads((source/'summary.json').read_text())
            evaluate(cache,output,seed,.1,1500,gt)
            left=pd.read_csv(source/'query_diagnostics.csv');right=pd.read_csv(output/'query_diagnostics.csv')
            for column in right:
                if column=='spatial_residual_m':np.testing.assert_allclose(left[column],right[column],atol=1e-7,rtol=1e-6)
                else:np.testing.assert_array_equal(left[column],right[column])
            matrix=np.load(output/'predicted_rescan_to_reference.npy')
            old_rotation=saved['scene_rotation_error_deg'];old_translation=saved['scene_translation_error_m']
        rotation=float(np.degrees(Rotation.from_matrix(matrix[:3,:3].T@gt[:3,:3]).magnitude()))
        np.testing.assert_allclose([rotation,np.linalg.norm(matrix[:3,3]-old_gt[:3,3])],[old_rotation,old_translation],atol=1e-6,rtol=1e-6)
        translation=float(np.linalg.norm(matrix[:3,3]-gt[:3,3]))
        (output/'regression.json').write_text(json.dumps({'status':'pass','saved_inliers_and_residuals_reproduced':True,'old_pose_errors_reproduced':True})+'\n')
        rows.append(dict(run_id=run,environment_id=identity['environment_id'],ransac_seed=seed,rotation_error_deg=rotation,translation_error_m=translation,strict_pose_success=rotation<5 and translation<.25))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['vitg','mast3r_dense'],required=True);a=p.parse_args()
    identities=pd.read_csv(ROOT/'20261008_vitg_validation_corrected_pose_units/per_relation.csv').to_dict('records')
    if len(identities)!=110 or len({r['environment_id'] for r in identities})!=47:raise ValueError('110/47 population required')
    metadata=json.loads((Path.home() / 'data/worldcache_datasets/3rscan/3RScan.json').read_text())
    out=ROOT/f'20261008_{a.method}_validation_seed_corrected_units';out.mkdir(exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,records in enumerate(pool.map(rescore,[(r,a.method,metadata) for r in identities]),1):
            rows+=records;pd.DataFrame(rows).to_csv(out/'per_relation_seed.csv',index=False);print(i,'/110',records[0]['run_id'],flush=True)
    frame=pd.DataFrame(rows);counts=frame.groupby('ransac_seed').strict_pose_success.sum().to_dict()
    (out/'summary.json').write_text(json.dumps({'relations':110,'environments':47,'saved_consensus_reproduced':True,'corrected_pose_counts':counts},indent=2)+'\n')

if __name__=='__main__':main()
