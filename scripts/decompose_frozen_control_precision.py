"""Exact descriptive precision decomposition into within-environment and allocation terms."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    reports=Path('outputs/reports');rng=np.random.default_rng(10082026);rows=[]
    for cohort,path in [('validation',reports/'20261008_bufferx_validation/source_frozen_threshold/per_environment.csv'),
                        ('new_native',reports/'20261008_unseen_train_replication/policy_per_environment.csv')]:
        f=pd.read_csv(path)
        for arm in ['source_frozen_geometry','source_frozen_rank']:
            a=f[f.arm.eq('bufferx_pose_vitg_k20')].set_index('environment_id').sort_index()
            b=f[f.arm.eq(arm)].set_index('environment_id').reindex(a.index)
            na,nb=a.semantic_assigned.to_numpy(),b.semantic_assigned.to_numpy()
            ca,cb=a.semantic_correct.to_numpy(),b.semantic_correct.to_numpy()
            assert (na>0).all() and (nb>0).all(),'both arms must have positive support in every environment for this identity'
            pa,pb=ca/na,cb/nb;wa,wb=na/na.sum(),nb/nb.sum()
            within=float(((wa+wb)/2*(pa-pb)).sum())
            allocation=float(((pa+pb)/2*(wa-wb)).sum())
            pooled=float(ca.sum()/na.sum()-cb.sum()/nb.sum())
            np.testing.assert_allclose(pooled,within+allocation,atol=1e-14)
            indices=rng.integers(0,len(a),(20000,len(a)))
            nda,ndb=na[indices],nb[indices];pda,pdb=pa[indices],pb[indices]
            wda=nda/nda.sum(axis=1)[:,None];wdb=ndb/ndb.sum(axis=1)[:,None]
            within_draw=((wda+wdb)/2*(pda-pdb)).sum(axis=1)
            allocation_draw=((pda+pdb)/2*(wda-wdb)).sum(axis=1)
            pooled_draw=ca[indices].sum(axis=1)/nda.sum(axis=1)-cb[indices].sum(axis=1)/ndb.sum(axis=1)
            np.testing.assert_allclose(pooled_draw,within_draw+allocation_draw,atol=1e-14)
            rows.append(dict(cohort=cohort,geometry_arm=arm,environments=len(a),macro_delta=float((pa-pb).mean()),
                pooled_delta=pooled,within_environment_component=within,allocation_component=allocation,
                pooled_ci95=np.quantile(pooled_draw,[.025,.975]).tolist(),within_ci95=np.quantile(within_draw,[.025,.975]).tolist(),allocation_ci95=np.quantile(allocation_draw,[.025,.975]).tolist()))
    out=reports/'20261008_frozen_control_precision_decomposition';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,
        identity='P_D-P_G = sum_e ((w_D+w_G)/2)(p_D-p_G) + sum_e ((p_D+p_G)/2)(w_D-w_G); w are accepted semantic-query shares, p are environment precisions.',
        scope='Exact algebra for observed accepted sets; conservation checked in all20000paired bootstrap draws per contrast. Four comparisons; component intervals are pointwise.',
        limitations='Not causal attribution; selection changes query and category mixes within environments too. Precision summaries answer different weighting questions. No new method or tuning; small8-environment confirmation cohort.'),indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,list)} for r in rows]).to_csv(out/'components.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
    for ax,cohort in zip(axes,['validation','new_native']):
        values=[r for r in rows if r['cohort']==cohort]
        x=np.arange(2)
        ax.bar(x-.18,[100*r['within_environment_component'] for r in values],.36,label='Within environment')
        ax.bar(x+.18,[100*r['allocation_component'] for r in values],.36,label='Acceptance allocation')
        ax.scatter(x,[100*r['pooled_delta'] for r in values],marker='D',color='black',label='Pooled difference',zorder=4)
        ax.scatter(x,[100*r['macro_delta'] for r in values],marker='x',color='red',label='Macro difference',zorder=4)
        ax.axhline(0,color='black',linewidth=.6);ax.set_xticks(x,['Fixed distance','Fixed rank fraction'])
        ax.set_title('Validation (47 environments)' if cohort=='validation' else 'New native cohort (8 environments)');ax.set_ylabel('DINO minus geometry (percentage points)')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'precision_decomposition.pdf');fig.savefig(out/'precision_decomposition.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
