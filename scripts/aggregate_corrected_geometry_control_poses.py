#!/usr/bin/env python3
"""Aggregate complete geometry-control runs after exact transfer reproduction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('outputs/reports/20261008_geometry_controls_corrected_units')

def main():
    manifest=pd.read_csv(ROOT/'run_manifest.csv',keep_default_na=False)
    if len(manifest)!=110 or manifest.run_id.duplicated().any() or manifest.environment_id.nunique()!=47 or not manifest.status.eq('complete').all():
        raise ValueError('110 complete relations in 47 environments required')
    rows=[]
    for item in manifest.itertuples(index=False):
        directory=ROOT/'relations'/item.run_id
        regression=json.loads((directory/'regression.json').read_text())
        if regression['status']!='pass' or not regression['all_transfer_rows_reproduced']:raise ValueError('unverified control')
        controls={'fcgf':pd.read_csv(directory/'fcgf/summary.csv').iloc[0]}
        controls.update({r.method:r for r in pd.read_csv(directory/'fpfh/summary.csv').itertuples(index=False)})
        for method,record in controls.items():
            rotation=float(record.rotation_error_deg);translation=float(record.translation_error_m)
            rows.append(dict(run_id=item.run_id,environment_id=item.environment_id,method=method,rotation_error_deg=rotation,translation_error_m=translation,strict_pose_success=rotation<5 and translation<.25))
    frame=pd.DataFrame(rows)
    env=frame.groupby(['environment_id','method']).strict_pose_success.mean().unstack()
    base=pd.read_csv('outputs/reports/20261008_vitg_validation_corrected_pose_units/per_relation.csv').set_index('run_id')
    np.testing.assert_array_equal(manifest.environment_id,base.reindex(manifest.run_id).environment_id)
    baseline=base.groupby('environment_id').strict_pose_success.mean().reindex(env.index)
    draws=np.random.default_rng(10082026).integers(0,len(env),(20000,len(env)))
    methods=[]
    for method in env:
        delta=(env[method]-baseline).to_numpy();lo,hi=np.quantile(delta[draws].mean(axis=1),[.025,.975])
        methods.append(dict(method=method,strict_pose_successes=int(frame.loc[frame.method.eq(method),'strict_pose_success'].sum()),environment_macro_success=float(env[method].mean()),minus_vitg_macro_delta=float(delta.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    output=ROOT/'aggregate';output.mkdir(exist_ok=True)
    frame.to_csv(output/'per_relation.csv',index=False);env.to_csv(output/'per_environment.csv')
    (output/'summary.json').write_text(json.dumps(dict(relations=110,environments=47,all_transfer_rows_reproduced=True,scene_units='raw released metadata in metres',bootstrap_samples=20000,bootstrap_seed=10082026,methods=methods),indent=2)+'\n')
    print(json.dumps(methods,indent=2))

if __name__=='__main__':main()
