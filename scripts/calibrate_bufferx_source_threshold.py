"""Freeze a geometric residual threshold using 16 source relations, without labels."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def main():
    root=Path('outputs/reports/20261008_bufferx_source16')
    manifest=pd.read_csv(root/'run_manifest.csv')
    population=json.loads((root/'scheduled_population.json').read_text())
    assert len(manifest)==16 and set(manifest.run_id)=={p['run_id'] for p in population}
    assert manifest.status.isin(['complete','failed']).all(), 'all source relations must terminate before calibration'
    distances=[];budget=0;records=[]
    for r in manifest.itertuples(index=False):
        path=Path('outputs/reports/20261006_source_precision_motion/fixed_top20_run/descriptor_candidates')/f'{r.run_id}.npz'
        pose_path=root/'relations'/r.run_id/'predicted_rescan_to_reference.npy'
        with np.load(path) as z:
            assert str(z['reference_id'].item())==r.reference_scan
            assert str(z['rescan_id'].item())==r.rescan_scan
            ref,q,c=z['reference_points'],z['query_points'],z['candidate_indices'][:,:20]
        if r.status=='failed':
            distances.append(np.full(len(q),np.inf))
            records.append(dict(run_id=r.run_id,status='failed',raw_queries=len(q),dino_k20_accepted_at_10cm=0,
                candidate_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pose_sha256=None))
            continue
        pose=np.load(pose_path);pred=q@pose[:3,:3].T+pose[:3,3]
        nn=cKDTree(ref).query(pred,k=1)[0]
        restricted=np.linalg.norm(pred[:,None,:]-ref[c],axis=2).min(axis=1)
        accepted=int((restricted<=.1).sum());budget+=accepted;distances.append(nn)
        records.append(dict(run_id=r.run_id,status='complete',raw_queries=len(q),dino_k20_accepted_at_10cm=accepted,
            candidate_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pose_sha256=hashlib.sha256(pose_path.read_bytes()).hexdigest()))
    nn=np.concatenate(distances)
    assert 0<budget<len(nn)
    ordered=np.sort(nn)
    threshold=float((ordered[budget-1]+ordered[budget])/2)
    payload=dict(status='frozen',source_relations=16,failed_source_relations=int(manifest.status.eq('failed').sum()),target_excluded=True,raw_queries=len(nn),dino_raw_budget=budget,
        geometric_accepted=int((nn<=threshold).sum()),threshold_m=threshold,records=records,
        selection='Pooled source geometry residual cutoff matching DINO K20 raw acceptance at 10cm. No annotation labels loaded. Failed source poses retain their query denominator and accept no matches.',
        validation_policy='Apply this single threshold unchanged to all fixed validation queries; do not use validation acceptance counts or labels.',
        caveat='Post-hoc design after validation exploration; source-calibrated control, not independent confirmation.')
    output=root/'frozen_source_threshold.json'
    if output.exists():
        assert json.loads(output.read_text())==payload,'refusing to change frozen calibration'
    else:output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='records'},indent=2))

if __name__=='__main__':main()
