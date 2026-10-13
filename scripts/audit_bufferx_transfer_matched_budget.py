#!/usr/bin/env python3
"""Compare DINO filtering to lowest-residual geometry at equal raw assignment budgets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root=Path('outputs/reports/20261008_bufferx_validation')
    manifest=pd.read_csv(root/'run_manifest.csv')
    if len(manifest)!=110 or manifest.environment_id.nunique()!=47 or not manifest.status.eq('complete').all():
        raise ValueError('all 110 completed relations required for this diagnostic')
    source=root/'pose_conditioned_transfer'
    out=root/'matched_assignment_budget';out.mkdir(exist_ok=True)
    rows=[]
    for relation in manifest.itertuples(index=False):
        q=pd.read_csv(source/f'{relation.run_id}.csv.gz')
        geo=q[q.arm.eq('bufferx_pose_nearest_patch')].sort_values('query_index').reset_index(drop=True)
        dino=q[q.arm.eq('bufferx_pose_vitg_k20')].sort_values('query_index').reset_index(drop=True)
        for column in ('query_index','frame_id','semantic_evaluable','moved_rigid'):
            np.testing.assert_array_equal(geo[column],dino[column])
        if (dino.assigned&~geo.assigned).any():raise ValueError('DINO assignment outside nearest-geometry support')
        budget=int(dino.assigned.sum())
        eligible=np.flatnonzero(geo.assigned.to_numpy(bool))
        # The budget uses every raw depth-valid query, not annotation-valid labels.
        order=np.lexsort((geo.query_index.to_numpy()[eligible],geo.residual_m.to_numpy()[eligible]))
        chosen=eligible[order[:budget]]
        selected=np.zeros(len(geo),dtype=bool);selected[chosen]=True
        if selected.sum()!=budget:raise ValueError('raw assignment budgets differ')
        for arm,frame,assigned in [('geometry_low_residual_matched_budget',geo,selected),('dino_k20',dino,dino.assigned.to_numpy(bool))]:
            rows.append({'run_id':relation.run_id,'environment_id':relation.environment_id,'arm':arm,
                'raw_budget':budget,'raw_assigned':int(assigned.sum()),'semantic_queries':int(frame.semantic_evaluable.sum()),
                'semantic_assigned':int((assigned&frame.semantic_evaluable).sum()),
                'semantic_correct':int((assigned&frame.semantic_correct).sum()),
                'moved_rigid_queries':int(frame.moved_rigid.sum()),'moved_rigid_assigned':int((assigned&frame.moved_rigid).sum()),
                'moved_rigid_correct':int((assigned&frame.moved_rigid_correct).sum())})
    f=pd.DataFrame(rows);f.to_csv(out/'per_relation.csv',index=False)
    env=f.groupby(['environment_id','arm']).sum(numeric_only=True)
    env['semantic_coverage']=env.semantic_correct/env.semantic_queries
    env['semantic_precision']=env.semantic_correct/env.semantic_assigned.replace(0,np.nan)
    env['moved_coverage']=env.moved_rigid_correct/env.moved_rigid_queries.replace(0,np.nan)
    env.to_csv(out/'per_environment.csv')
    rng=np.random.default_rng(10082026);effects=[]
    for metric in ('semantic_coverage','semantic_precision','moved_coverage'):
        wide=env[metric].unstack('arm').dropna()
        delta=(wide.dino_k20-wide.geometry_low_residual_matched_budget).to_numpy()
        boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
        effects.append({'metric':metric,'environments':len(wide),'geometry_macro':float(wide.geometry_low_residual_matched_budget.mean()),
            'dino_macro':float(wide.dino_k20.mean()),'dino_minus_geometry':float(delta.mean()),
            'paired_ci95':np.quantile(boot,[.025,.975]).tolist()})
    pooled=f.groupby('arm').sum(numeric_only=True)
    pooled['semantic_precision']=pooled.semantic_correct/pooled.semantic_assigned
    summary={'relations':110,'environments':47,'all_raw_budgets_equal':True,'pooled':pooled.to_dict('index'),
        'effects':effects,'bootstrap_samples':20000,'bootstrap_seed':10082026,
        'scope':'post-hoc matched-budget diagnostic; every arm uses identical BUFFER-X poses and fixed patch support',
        'caveat':'geometry budget is supplied by the DINO acceptance count; this is a controlled comparison, not a standalone deployable geometry selector; no labels used in ordering or budget'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
