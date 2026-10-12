"""Score all eight frozen attempts, retaining failed geometry as zero acceptance."""
from pathlib import Path
import argparse
import hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from evaluate_3rscan_multimotion import load_feature_cache,fit_fixed_consensus,annotation_labels,relation_record,rigid_object_mapping


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-dir',type=Path,default=Path('outputs/reports/20261008_unseen_train_replication'))
    args=parser.parse_args()
    project=Path(__file__).resolve().parent.parent
    out=args.experiment_dir if args.experiment_dir.is_absolute() else project/args.experiment_dir
    frozen=json.loads((out/'frozen_population.json').read_text())
    population=frozen['population']
    manifest=pd.read_csv(out/'run_manifest.csv')
    assert len(manifest)==len(population) and set(manifest.run_id)=={r['run_id'] for r in population}
    assert manifest.status.isin(['complete','failed']).all()
    calibration_path=project/'outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json'
    assert hashlib.sha256(calibration_path.read_bytes()).hexdigest()==frozen['calibration_sha256']
    c=json.loads(calibration_path.read_text());fraction=c['dino_raw_budget']/c['raw_queries'];tau=c['threshold_m']
    data=Path.home()/'data/worldcache_datasets/3rscan'
    rows=[]
    for r in population:
        directory=out/'relations'/r['run_id']
        features=load_feature_cache(out/'descriptor_candidates'/f"{r['run_id']}.npz",r['run_id'],r['reference_scan'],r['rescan_scan'],Path('.'),'giant',(2,20,40),224)
        assert features is not None,'missing query support prevents full-population scoring; do not omit the relation'
        q,ref=features.query_points,features.reference_points
        base=fit_fixed_consensus(q,ref,features.candidate_indices[:,:20],threshold=.1,iterations=1500,seed=17)
        np.testing.assert_allclose(base.matrix,np.load(directory/'vitg_predicted_pose.npy'),atol=1e-12)
        assignments={'vitg_global_ransac':(base.selected,base.inliers)}
        pose_path=directory/'bufferx/predicted_rescan_to_reference.npy'
        if pose_path.exists():
            pose=np.load(pose_path);pred=q@pose[:3,:3].T+pose[:3,3]
            distance,selected=cKDTree(ref).query(pred,k=1)
            candidates=features.candidate_indices[:,:20]
            residual=np.linalg.norm(pred[:,None,:]-ref[candidates],axis=2)
            slots=residual.argmin(axis=1)
            assignments['bufferx_pose_vitg_k20']=(candidates[np.arange(len(q)),slots],residual[np.arange(len(q)),slots]<=.1)
            assignments['source_frozen_geometry']=(selected,distance<=tau)
            eligible=np.flatnonzero(distance<=.1)
            order=np.lexsort((eligible,distance[eligible]))
            accept=np.zeros(len(q),bool);accept[eligible[order[:int(np.floor(fraction*len(q)))]]]=True
            assignments['source_frozen_rank']=(selected,accept)
        else:
            for arm in ['bufferx_pose_vitg_k20','source_frozen_geometry','source_frozen_rank']:
                assignments[arm]=(np.zeros(len(q),int),np.zeros(len(q),bool))
        # Persist label-free assignments before loading annotations.
        np.savez_compressed(directory/'frozen_policy_assignments.npz',**{f'{arm}_{key}':value for arm,pair in assignments.items() for key,value in zip(['selected','assigned'],pair)})
        labels=annotation_labels(data,features)
        metadata=json.loads((data/'3RScan.json').read_text())
        mapping=rigid_object_mapping(relation_record(metadata,r['reference_scan'],r['rescan_scan'])['scan'])
        evaluable=(labels['query_object_id']!=0)&(labels['query_semantic_id']!=0)
        moved=evaluable&np.isin(labels['query_object_id'],list(mapping))
        expected=np.array([mapping.get(int(x),-1) for x in labels['query_object_id']])
        for arm,(selected,assigned) in assignments.items():
            semantic=labels['reference_semantic_id'][selected]==labels['query_semantic_id']
            physical=labels['reference_object_id'][selected]==expected
            rows.append(dict(run_id=r['run_id'],environment_id=r['environment_id'],arm=arm,buffer_pose_available=pose_path.exists(),
                raw_assigned=int(assigned.sum()),semantic_queries=int(evaluable.sum()),semantic_assigned=int((assigned&evaluable).sum()),
                semantic_correct=int((assigned&evaluable&semantic).sum()),moved_queries=int(moved.sum()),moved_assigned=int((assigned&moved).sum()),moved_correct=int((assigned&moved&physical).sum())))
    frame=pd.DataFrame(rows);frame.to_csv(out/'policy_per_relation.csv',index=False)
    env=frame.groupby(['environment_id','arm']).sum(numeric_only=True)
    env['semantic_coverage']=env.semantic_correct/env.semantic_queries.replace(0,np.nan)
    env['semantic_precision']=env.semantic_correct/env.semantic_assigned.replace(0,np.nan)
    env['moved_coverage']=env.moved_correct/env.moved_queries.replace(0,np.nan)
    env.to_csv(out/'policy_per_environment.csv')
    rng=np.random.default_rng(10082026);effects=[]
    for metric in ['semantic_coverage','semantic_precision','moved_coverage']:
        wide=env[metric].unstack('arm')
        for arm in ['source_frozen_geometry','source_frozen_rank']:
            pair=wide[[arm,'bufferx_pose_vitg_k20']].dropna()
            delta=(pair.bufferx_pose_vitg_k20-pair[arm]).to_numpy()
            if len(delta):
                boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
                effects.append(dict(metric=metric,geometry_arm=arm,environments=len(delta),dino_minus_geometry=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),bonferroni_ci95=np.quantile(boot,[.05/12,1-.05/12]).tolist()))
    summary=dict(relations=len(population),environments=int(frame.environment_id.nunique()),buffer_pose_failures=int((~frame[frame.arm.eq('source_frozen_geometry')].buffer_pose_available).sum()),
        calibration_sha256=frozen['calibration_sha256'],population_sha256=hashlib.sha256((out/'frozen_population.json').read_bytes()).hexdigest(),
        threshold_m=tau,raw_fraction=fraction,effects=effects,pooled=frame.groupby('arm').sum(numeric_only=True).to_dict('index'),
        scope=f"{len(population)} frozen relations from hash-selected unused train sequences; same acquisition dataset, physical-site independence unproven. This cohort was selected after the first cohort outcomes were observed, so it is an exploratory post-hoc extension, not an independent or preregistered confirmation; report all attempts.")
    (out/'policy_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
