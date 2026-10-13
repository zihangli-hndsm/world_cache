"""Apply the source-frozen geometric cutoff to every validation relation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    root=Path('outputs/reports/20261008_bufferx_validation')
    calibration_path=Path('outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json')
    calibration=json.loads(calibration_path.read_text());threshold=calibration['threshold_m']
    assert calibration['status']=='frozen' and calibration['source_relations']==16 and 0<threshold<=.1
    manifest=pd.read_csv(root/'run_manifest.csv');assert len(manifest)==110 and manifest.status.eq('complete').all()
    source_ids={r['run_id'] for r in calibration['records']};assert not source_ids.intersection(manifest.run_id)
    raw_fraction=calibration['dino_raw_budget']/calibration['raw_queries']
    rows=[]
    for r in manifest.itertuples(index=False):
        q=pd.read_csv(root/'pose_conditioned_transfer'/f'{r.run_id}.csv.gz')
        for arm in ['vitg_global_ransac','bufferx_pose_nearest_patch','bufferx_pose_vitg_k20','source_frozen_rank']:
            f=q[q.arm.eq('bufferx_pose_nearest_patch' if arm=='source_frozen_rank' else arm)].sort_values('query_index').reset_index(drop=True)
            assigned=f.assigned.to_numpy(bool)
            if arm=='bufferx_pose_nearest_patch':
                assigned=assigned&(f.residual_m.to_numpy()<=threshold)
                name='source_frozen_geometry'
            elif arm=='source_frozen_rank':
                eligible=np.flatnonzero(assigned)
                order=np.lexsort((f.query_index.to_numpy()[eligible],f.residual_m.to_numpy()[eligible]))
                budget=int(np.floor(raw_fraction*len(f)))
                assigned=np.zeros(len(f),bool);assigned[eligible[order[:budget]]]=True
                name=arm
            else:name=arm
            rows.append(dict(run_id=r.run_id,environment_id=r.environment_id,arm=name,raw_assigned=int(assigned.sum()),
                semantic_queries=int(f.semantic_evaluable.sum()),semantic_assigned=int((assigned&f.semantic_evaluable).sum()),
                semantic_correct=int((assigned&f.semantic_correct).sum()),moved_queries=int(f.moved_rigid.sum()),
                moved_assigned=int((assigned&f.moved_rigid).sum()),moved_correct=int((assigned&f.moved_rigid_correct).sum())))
    out=root/'source_frozen_threshold';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_relation.csv',index=False)
    env=frame.groupby(['environment_id','arm']).sum(numeric_only=True)
    env['semantic_coverage']=env.semantic_correct/env.semantic_queries
    env['semantic_precision']=env.semantic_correct/env.semantic_assigned.replace(0,np.nan)
    env['moved_coverage']=env.moved_correct/env.moved_queries.replace(0,np.nan)
    env.to_csv(out/'per_environment.csv')
    effects=[];rng=np.random.default_rng(10082026)
    for metric in ['semantic_coverage','semantic_precision','moved_coverage']:
        wide=env[metric].unstack('arm')
        for geometry_arm in ['source_frozen_geometry','source_frozen_rank']:
            for comparator in ['vitg_global_ransac','bufferx_pose_vitg_k20']:
                pair=wide[[geometry_arm,comparator]].dropna()
                delta=(pair[geometry_arm]-pair[comparator]).to_numpy()
                boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
                effects.append(dict(metric=metric,geometry_arm=geometry_arm,comparator=comparator,environments=len(delta),geometry_macro=float(pair[geometry_arm].mean()),comparator_macro=float(pair[comparator].mean()),delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),bonferroni_familywise_ci95=np.quantile(boot,[.05/24,1-.05/24]).tolist()))
    payload=dict(relations=110,environments=47,threshold_m=threshold,source_raw_acceptance_fraction=raw_fraction,
        calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),effects=effects,
        pooled=frame.groupby('arm').sum(numeric_only=True).to_dict('index'),
        scope='Source-frozen geometric threshold and source-frozen raw acceptance fraction (rank by residual, capped at 10cm); no validation DINO counts used for acceptance. Fixed BUFFER-X poses; same validation patches.',
        limitations='Explored validation population; differing acceptance budgets reported explicitly. Source calibration is not prospective independent validation. ci95 is pointwise; additional Bonferroni percentile intervals cover the family of 12 exploratory effects approximately (bootstrap approximation).')
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
