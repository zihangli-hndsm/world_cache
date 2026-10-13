"""Describe influence and cohort shift without fitting or choosing new selectors."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    reports=Path('outputs/reports')
    cohorts={'validation':reports/'20261008_bufferx_validation/source_frozen_threshold/per_environment.csv',
             'new_native':reports/'20261008_unseen_train_replication/policy_per_environment.csv'}
    rows=[];vectors={};rng=np.random.default_rng(10082026)
    for cohort,path in cohorts.items():
        frame=pd.read_csv(path)
        for metric in ['semantic_coverage','semantic_precision','moved_coverage']:
            wide=frame.pivot(index='environment_id',columns='arm',values=metric)
            for arm in ['source_frozen_geometry','source_frozen_rank']:
                pair=wide[['bufferx_pose_vitg_k20',arm]].dropna()
                d=(pair.bufferx_pose_vitg_k20-pair[arm]).to_numpy()
                loo=(d.sum()-d)/(len(d)-1)
                vectors[(cohort,metric,arm)]=d
                rows.append(dict(cohort=cohort,metric=metric,geometry_arm=arm,environments=len(d),delta=float(d.mean()),
                    median_delta=float(np.median(d)),dino_better=int((d>0).sum()),tied=int((d==0).sum()),
                    dino_worse=int((d<0).sum()),loo_min=float(loo.min()),loo_max=float(loo.max())))
    effects=[]
    for metric in ['semantic_coverage','semantic_precision','moved_coverage']:
        for arm in ['source_frozen_geometry','source_frozen_rank']:
            a=vectors[('validation',metric,arm)];b=vectors[('new_native',metric,arm)]
            draws=b[rng.integers(0,len(b),(20000,len(b)))].mean(axis=1)-a[rng.integers(0,len(a),(20000,len(a)))].mean(axis=1)
            effects.append(dict(metric=metric,geometry_arm=arm,new_minus_validation=float(b.mean()-a.mean()),
                ci95=np.quantile(draws,[.025,.975]).tolist(),bonferroni_ci95=np.quantile(draws,[.05/12,1-.05/12]).tolist()))
    out=reports/'20261008_frozen_control_cohort_stability';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'within_cohort.csv',index=False)
    (out/'summary.json').write_text(json.dumps(dict(cohort_effects=effects,
        scope='Exploratory contrast of environment-macro DINO-minus-geometry effects; unpaired environment bootstrap between cohorts, paired within each cohort.',
        limitations='Eight new train environments versus47 validation environments. Different cohort populations; prior mesh screening, physical-site independence unproven. LOO influence checks are descriptive, not causal attribution. No new policy fitted.'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(effects,indent=2))

if __name__=='__main__':main()
