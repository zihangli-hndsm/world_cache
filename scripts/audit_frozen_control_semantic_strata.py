"""Post-hoc semantic composition audit of unchanged frozen policy assignments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate_3rscan_multimotion import load_feature_cache,annotation_labels


def main():
    reports=Path('outputs/reports');data=Path.home()/'data/worldcache_datasets/3rscan'
    calibration=json.loads((reports/'20261008_bufferx_source16/frozen_source_threshold.json').read_text())
    fraction=calibration['dino_raw_budget']/calibration['raw_queries'];tau=calibration['threshold_m']
    rows=[]
    for cohort,root in [('validation',reports/'20261008_bufferx_validation'),('new_native',reports/'20261008_unseen_train_replication')]:
        manifest=pd.read_csv(root/'run_manifest.csv')
        for r in manifest.itertuples(index=False):
            if cohort=='validation':
                table=pd.read_csv(root/'pose_conditioned_transfer'/f'{r.run_id}.csv.gz')
                geo=table[table.arm.eq('bufferx_pose_nearest_patch')].sort_values('query_index').reset_index(drop=True)
                dino=table[table.arm.eq('bufferx_pose_vitg_k20')].sort_values('query_index').reset_index(drop=True)
                diag=pd.read_csv(reports/'20261006_vitg_pose_locked_motion_validation/relations'/r.run_id/'query_diagnostics.csv')
                diag=diag[diag.arm.eq('global_ransac')].sort_values('query_index').reset_index(drop=True)
                np.testing.assert_array_equal(geo.query_index,diag.query_index)
                np.testing.assert_array_equal(geo.semantic_evaluable,diag.query_evaluable)
                evaluable=geo.semantic_evaluable.to_numpy(bool);classes=diag.query_semantic_id.to_numpy(int)
                eligible=np.flatnonzero(geo.assigned.to_numpy(bool))
                order=np.lexsort((geo.query_index.to_numpy()[eligible],geo.residual_m.to_numpy()[eligible]))
                rank=np.zeros(len(geo),bool);rank[eligible[order[:int(np.floor(fraction*len(geo)))]]]=True
                arms={'bufferx_pose_vitg_k20':(dino.assigned.to_numpy(bool),dino.semantic_correct.to_numpy(bool)),
                    'source_frozen_geometry':(geo.assigned.to_numpy(bool)&(geo.residual_m.to_numpy()<=tau),geo.semantic_correct.to_numpy(bool)),
                    'source_frozen_rank':(rank,geo.semantic_correct.to_numpy(bool))}
            else:
                features=load_feature_cache(root/'descriptor_candidates'/f'{r.run_id}.npz',r.run_id,r.reference_scan,r.rescan_scan,Path('.'),'giant',(2,20,40),224)
                labels=annotation_labels(data,features)
                classes=labels['query_semantic_id'];evaluable=(labels['query_object_id']!=0)&(classes!=0)
                arms={}
                with np.load(root/'relations'/r.run_id/'frozen_policy_assignments.npz') as z:
                    for arm in ['bufferx_pose_vitg_k20','source_frozen_geometry','source_frozen_rank']:
                        selected=z[arm+'_selected'];assigned=z[arm+'_assigned']
                        arms[arm]=(assigned,labels['reference_semantic_id'][selected]==classes)
            for arm,(assigned,correct) in arms.items():
                for stratum,mask in [('nyu40_1_2_22',np.isin(classes,[1,2,22])),('other_nyu40',~np.isin(classes,[1,2,22]))]:
                    valid=mask&evaluable
                    rows.append(dict(cohort=cohort,run_id=r.run_id,environment_id=r.environment_id,arm=arm,stratum=stratum,
                        queries=int(valid.sum()),assigned=int((valid&assigned).sum()),correct=int((valid&assigned&correct).sum())))
    out=reports/'20261008_frozen_control_semantic_strata';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_relation.csv',index=False)
    env=frame.groupby(['cohort','environment_id','arm','stratum']).sum(numeric_only=True)
    env['precision']=env.correct/env.assigned.replace(0,np.nan)
    env['coverage']=env.correct/env.queries.replace(0,np.nan)
    env.to_csv(out/'per_environment.csv')
    pooled=frame.groupby(['cohort','arm','stratum']).sum(numeric_only=True)
    pooled['precision']=pooled.correct/pooled.assigned.replace(0,np.nan);pooled.to_csv(out/'pooled.csv')
    effects=[];rng=np.random.default_rng(10082026)
    for cohort in ['validation','new_native']:
        for stratum in ['nyu40_1_2_22','other_nyu40']:
            wide=env.loc[cohort].xs(stratum,level='stratum').precision.unstack('arm')
            for arm in ['source_frozen_geometry','source_frozen_rank']:
                pair=wide[['bufferx_pose_vitg_k20',arm]].dropna()
                delta=(pair.bufferx_pose_vitg_k20-pair[arm]).to_numpy()
                if len(delta):
                    draws=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
                    effects.append(dict(cohort=cohort,stratum=stratum,geometry_arm=arm,environments=len(delta),delta=float(delta.mean()),ci95=np.quantile(draws,[.025,.975]).tolist()))
    summary=dict(effects=effects,scope='Post-hoc fixed-assignment stratification by query NYU40 IDs1,2,22 versus other mapped categories. No policy refitting.',
        limitations='Labels define diagnostic strata, not deployable selection. Different accepted queries and within-stratum class mixtures remain. Precision contrasts condition on nonzero accepts in both arms. Eight pointwise exploratory intervals; no causal explanation or independent new replication.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print(pooled.to_string())

if __name__=='__main__':main()
