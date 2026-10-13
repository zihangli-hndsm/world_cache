"""Compare modern geometry transfers on the exact saved linked-instance target."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate_3rscan_instance_correspondence import read_annotated_mesh


def main():
    reports=Path('outputs/reports');root=reports/'20261008_bufferx_validation'
    manifest=pd.read_csv(root/'run_manifest.csv');assert len(manifest)==110 and manifest.status.eq('complete').all()
    pose_root=reports/'20261006_vitg_pose_locked_motion_validation'
    physical_root=reports/'20261004_vitg_validation_candidate_audit'
    data=Path.home()/'data/worldcache_datasets/3rscan';rows=[]
    previous=None
    for r in manifest.itertuples(index=False):
        with np.load(pose_root/'descriptor_candidates'/f'{r.run_id}.npz') as z:
            ref=z['reference_points'];query_frames=z['query_frame_ids']
        if previous!=r.reference_scan:
            tree,objects,_,_,_=read_annotated_mesh(data,r.reference_scan);previous=r.reference_scan
        reference_objects=objects[tree.query(ref.astype(np.float32),k=1)[1]]
        identity=pd.read_csv(physical_root/'relations'/r.run_id/'vitg/query_diagnostics.csv')
        identity=identity[identity.top_k.eq(20)&identity.geometry_weight.eq(0)].sort_values('query_index').reset_index(drop=True)
        assert np.array_equal(identity.query_index,np.arange(len(query_frames)))
        np.testing.assert_array_equal(identity.frame_id,query_frames)
        linked=identity.physical_instance_evaluable.to_numpy(bool)
        expected=identity.mapped_reference_object_id.to_numpy(int)
        transfer=pd.read_csv(root/'pose_conditioned_transfer'/f'{r.run_id}.csv.gz')
        geo=transfer[transfer.arm.eq('bufferx_pose_nearest_patch')].sort_values('query_index').reset_index(drop=True)
        dino=transfer[transfer.arm.eq('bufferx_pose_vitg_k20')].sort_values('query_index').reset_index(drop=True)
        eligible=np.flatnonzero(geo.assigned.to_numpy(bool));budget=int(dino.assigned.sum())
        rank=np.lexsort((geo.query_index.to_numpy()[eligible],geo.residual_m.to_numpy()[eligible]))
        matched=np.zeros(len(geo),bool);matched[eligible[rank[:budget]]]=True
        assert matched.sum()==budget
        arms=[]
        for arm in ['vitg_global_ransac','bufferx_pose_nearest_patch','bufferx_pose_vitg_k20']:
            f=transfer[transfer.arm.eq(arm)].sort_values('query_index').reset_index(drop=True)
            np.testing.assert_array_equal(f.query_index,identity.query_index)
            selected=f.selected_reference_index.to_numpy(int);assigned=f.assigned.to_numpy(bool)
            correct=reference_objects[selected]==expected
            if arm=='vitg_global_ransac':
                np.testing.assert_array_equal(correct[linked],identity.physical_instance_correct.to_numpy(bool)[linked])
            arms.append((arm,assigned,correct))
        arms.append(('geometry_low_residual_matched_budget',matched,reference_objects[geo.selected_reference_index.to_numpy(int)]==expected))
        for arm,assigned,correct in arms:
            rows.append(dict(run_id=r.run_id,environment_id=r.environment_id,arm=arm,linked_queries=int(linked.sum()),
                raw_assigned=int(assigned.sum()),linked_assigned=int((assigned&linked).sum()),linked_correct=int((assigned&linked&correct).sum())))
    out=root/'linked_instance_transfer';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_relation.csv',index=False)
    assert frame.groupby('arm').linked_queries.sum().eq(98221).all()
    env=frame.groupby(['environment_id','arm']).sum(numeric_only=True)
    env['coverage']=env.linked_correct/env.linked_queries
    env['precision']=env.linked_correct/env.linked_assigned.replace(0,np.nan)
    env.to_csv(out/'per_environment.csv')
    rng=np.random.default_rng(10082026);effects=[]
    for metric in ['coverage','precision']:
        wide=env[metric].unstack('arm')
        for a,b in [('bufferx_pose_nearest_patch','vitg_global_ransac'),('bufferx_pose_vitg_k20','vitg_global_ransac'),('bufferx_pose_vitg_k20','geometry_low_residual_matched_budget')]:
            pair=wide[[a,b]].dropna();delta=(pair[a]-pair[b]).to_numpy()
            boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
            effects.append(dict(metric=metric,arm=a,comparator=b,environments=len(delta),arm_macro=float(pair[a].mean()),comparator_macro=float(pair[b].mean()),delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),bonferroni_ci95=np.quantile(boot,[.05/12,1-.05/12]).tolist()))
    summary=dict(relations=110,environments=47,linked_queries=98221,pooled=frame.groupby('arm').sum(numeric_only=True).to_dict('index'),effects=effects,
        scope='Post-hoc fixed-query linked-instance transfer, using exactly the saved unambiguous physical candidate-audit mapping; explicit moved links plus unchanged shared-ID semantic-consistent fallback, metadata ambiguity excluded. No ground truth enters assignments.',
        limitations='These are audited annotation links, not independently verified world identity. Acceptance budgets differ except the matched pair, whose raw budget comes from DINO. Same explored validation split; pointwise and6-effect conservative intervals.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
