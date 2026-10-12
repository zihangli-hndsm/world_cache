"""Apply the existing label-free matched-budget diagnostic to all eight new outcomes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from evaluate_3rscan_multimotion import load_feature_cache,annotation_labels,relation_record,rigid_object_mapping


def main():
    root=Path('outputs/reports/20261008_unseen_train_replication')
    frozen=json.loads((root/'frozen_population.json').read_text())
    manifest=pd.read_csv(root/'run_manifest.csv')
    assert len(manifest)==8 and manifest.status.eq('complete').all()
    assert set(manifest.run_id)=={r['run_id'] for r in frozen['population']}
    data=Path.home()/'data/worldcache_datasets/3rscan';rows=[]
    for r in manifest.itertuples(index=False):
        f=load_feature_cache(root/'descriptor_candidates'/f'{r.run_id}.npz',r.run_id,r.reference_scan,r.rescan_scan,Path('.'),'giant',(2,20,40),224)
        directory=root/'relations'/r.run_id
        pose=np.load(directory/'bufferx/predicted_rescan_to_reference.npy')
        pred=f.query_points@pose[:3,:3].T+pose[:3,3]
        distances,selected=cKDTree(f.reference_points).query(pred,k=1)
        with np.load(directory/'frozen_policy_assignments.npz') as z:
            ds,da=z['bufferx_pose_vitg_k20_selected'],z['bufferx_pose_vitg_k20_assigned']
        budget=int(da.sum());eligible=np.flatnonzero(distances<=.1)
        assert not (da & (distances>.1)).any()
        order=np.lexsort((eligible,distances[eligible]))
        ga=np.zeros(len(pred),bool);ga[eligible[order[:budget]]]=True
        assert ga.sum()==budget
        np.savez_compressed(directory/'matched_budget_assignments.npz',geometry_selected=selected,geometry_assigned=ga,dino_selected=ds,dino_assigned=da)
        labels=annotation_labels(data,f)
        valid=(labels['query_object_id']!=0)&(labels['query_semantic_id']!=0)
        mapping=rigid_object_mapping(relation_record(json.loads((data/'3RScan.json').read_text()),r.reference_scan,r.rescan_scan)['scan'])
        moved=valid&np.isin(labels['query_object_id'],list(mapping));expected=np.array([mapping.get(int(x),-1) for x in labels['query_object_id']])
        for arm,s,a in [('geometry_low_residual_matched_budget',selected,ga),('dino_k20',ds,da)]:
            sem=labels['reference_semantic_id'][s]==labels['query_semantic_id'];physical=labels['reference_object_id'][s]==expected
            rows.append(dict(run_id=r.run_id,environment_id=r.environment_id,arm=arm,raw_budget=budget,raw_assigned=int(a.sum()),
                semantic_queries=int(valid.sum()),semantic_assigned=int((a&valid).sum()),semantic_correct=int((a&valid&sem).sum()),
                moved_queries=int(moved.sum()),moved_assigned=int((a&moved).sum()),moved_correct=int((a&moved&physical).sum())))
    out=root/'matched_assignment_budget';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_relation.csv',index=False)
    env=frame.groupby(['environment_id','arm']).sum(numeric_only=True)
    env['semantic_coverage']=env.semantic_correct/env.semantic_queries
    env['semantic_precision']=env.semantic_correct/env.semantic_assigned.replace(0,np.nan)
    env['moved_coverage']=env.moved_correct/env.moved_queries.replace(0,np.nan)
    env.to_csv(out/'per_environment.csv')
    rng=np.random.default_rng(10082026);effects=[]
    for metric in ['semantic_coverage','semantic_precision','moved_coverage']:
        pair=env[metric].unstack('arm').dropna()
        delta=(pair.dino_k20-pair.geometry_low_residual_matched_budget).to_numpy()
        boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
        effects.append(dict(metric=metric,environments=len(delta),delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),bonferroni_ci95=np.quantile(boot,[.05/6,1-.05/6]).tolist()))
    pooled=frame.groupby('arm').sum(numeric_only=True).to_dict('index')
    original=pd.read_csv(root/'policy_per_relation.csv');dino=original[original.arm.eq('bufferx_pose_vitg_k20')].set_index('run_id').sort_index()
    fresh=frame[frame.arm.eq('dino_k20')].set_index('run_id').sort_index()
    for col in ['raw_assigned','semantic_queries','semantic_assigned','semantic_correct','moved_queries','moved_assigned','moved_correct']:
        np.testing.assert_array_equal(dino[col],fresh[col])
    summary=dict(relations=8,environments=8,pooled=pooled,effects=effects,
        population_sha256=hashlib.sha256((root/'frozen_population.json').read_bytes()).hexdigest(),
        scope='Existing matched-budget rule applied after inspecting new outcomes. Labels only score. Not a preregistered endpoint or deployable budget selector; DINO supplies per-relation raw acceptance budgets. No equivalence margin defined.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
