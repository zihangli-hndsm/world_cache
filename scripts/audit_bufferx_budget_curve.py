"""Post-hoc paired budget curves with fixed poses, no label-based selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root=Path('outputs/reports/20261008_bufferx_validation')
    manifest=pd.read_csv(root/'run_manifest.csv')
    assert len(manifest)==110 and manifest.environment_id.nunique()==47
    assert manifest.status.eq('complete').all()
    fractions=[.1,.25,.5,.75,1.]
    rows=[];overlaps=[]
    for r in manifest.itertuples(index=False):
        table=pd.read_csv(root/'pose_conditioned_transfer'/f'{r.run_id}.csv.gz')
        arms={a:table[table.arm.eq(a)].sort_values('query_index').reset_index(drop=True) for a in ['bufferx_pose_nearest_patch','bufferx_pose_vitg_k20']}
        geo,dino=arms.values()
        for col in ['query_index','frame_id','semantic_evaluable','moved_rigid']:
            np.testing.assert_array_equal(geo[col],dino[col])
        assert not (dino.assigned & ~geo.assigned).any()
        maximum=int(dino.assigned.sum())
        orders={}
        for name,f in arms.items():
            eligible=np.flatnonzero(f.assigned.to_numpy(bool))
            orders[name]=eligible[np.lexsort((f.query_index.to_numpy()[eligible],f.residual_m.to_numpy()[eligible]))]
        for fraction in fractions:
            budget=int(np.floor(maximum*fraction));masks={}
            for name,f in arms.items():
                mask=np.zeros(len(f),bool);mask[orders[name][:budget]]=True
                assert mask.sum()==budget
                masks[name]=mask
                rows.append(dict(run_id=r.run_id,environment_id=r.environment_id,arm=name,fraction=fraction,raw_budget=budget,
                    semantic_queries=int(f.semantic_evaluable.sum()),semantic_assigned=int((mask&f.semantic_evaluable).sum()),
                    semantic_correct=int((mask&f.semantic_correct).sum()),moved_queries=int(f.moved_rigid.sum()),
                    moved_correct=int((mask&f.moved_rigid_correct).sum())))
            gm,dm=masks.values()
            overlaps.append(dict(run_id=r.run_id,fraction=fraction,budget=budget,shared=int((gm&dm).sum()),
                shared_reference=int((gm&dm&(geo.selected_reference_index==dino.selected_reference_index)).sum())))
    out=root/'budget_curve';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'per_relation.csv',index=False)
    pd.DataFrame(overlaps).to_csv(out/'assignment_overlap.csv',index=False)
    env=frame.groupby(['fraction','environment_id','arm']).sum(numeric_only=True)
    env['semantic_coverage']=env.semantic_correct/env.semantic_queries
    env['semantic_precision']=env.semantic_correct/env.semantic_assigned.replace(0,np.nan)
    env['moved_coverage']=env.moved_correct/env.moved_queries.replace(0,np.nan)
    env.to_csv(out/'per_environment.csv')
    rng=np.random.default_rng(10082026);effects=[]
    for fraction in fractions:
        for metric in ['semantic_coverage','semantic_precision','moved_coverage']:
            wide=env.loc[fraction,metric].unstack('arm').dropna()
            delta=(wide.bufferx_pose_vitg_k20-wide.bufferx_pose_nearest_patch).to_numpy()
            boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
            effects.append(dict(fraction=fraction,metric=metric,environments=len(delta),geometry_macro=float(wide.bufferx_pose_nearest_patch.mean()),dino_macro=float(wide.bufferx_pose_vitg_k20.mean()),delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    summary=dict(relations=110,environments=47,fractions=fractions,effects=effects,
        scope='Exploratory budget curve; budgets are fractions of DINO acceptance per relation. Both arms sort by residual with query-index tie break. Labels only score.',
        limitations='Same explored validation population; pointwise bootstrap intervals across 15 comparisons, no simultaneous coverage. This is not an independent generalization test.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    pooled=pd.DataFrame(overlaps).groupby('fraction')[['budget','shared','shared_reference']].sum()
    pooled['query_overlap']=pooled.shared/pooled.budget
    pooled['same_reference_fraction']=pooled.shared_reference/pooled.budget
    pooled.to_csv(out/'pooled_assignment_overlap.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10,3.2))
    results=pd.DataFrame(effects)
    for ax,metric,title in zip(axes,['semantic_coverage','semantic_precision','moved_coverage'],
                              ['Semantic coverage','Semantic precision','Moved-instance coverage']):
        d=results[results.metric.eq(metric)]
        ax.plot(d.fraction,100*d.geometry_macro,'o-',label='Geometry')
        ax.plot(d.fraction,100*d.dino_macro,'s-',label='DINO K20 filter')
        ax.set(title=title,xlabel='Fraction of DINO acceptance budget',ylabel='Environment macro (%)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'budget_curves.pdf');fig.savefig(out/'budget_curves.png',dpi=160);plt.close(fig)
    for e in effects:print(e)

if __name__=='__main__':main()
