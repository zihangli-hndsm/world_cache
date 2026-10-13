"""Plot full-validation semantic and linked-instance utility on fixed patch support."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('outputs/reports/20261008_bufferx_validation')
    semantics=pd.read_csv(root/'pose_conditioned_transfer/per_environment.csv')
    semantics['coverage']=semantics.semantic_correct/semantics.semantic_queries
    semantics['precision']=semantics.semantic_correct/semantics.semantic_assigned.replace(0,np.nan)
    matched=pd.read_csv(root/'matched_assignment_budget/per_environment.csv')
    matched=matched[matched.arm.eq('geometry_low_residual_matched_budget')].copy()
    matched['coverage']=matched.semantic_coverage;matched['precision']=matched.semantic_precision
    semantics=pd.concat([semantics,matched],ignore_index=True)
    identity=pd.read_csv(root/'linked_instance_transfer/per_environment.csv')
    definitions=[('vitg_global_ransac','ViT-G consensus','o','#687b91'),('bufferx_pose_nearest_patch','BUFFER-X geometry','s','#d89032'),
        ('bufferx_pose_vitg_k20','BUFFER-X + DINO K20','^','#328f71'),('geometry_low_residual_matched_budget','BUFFER-X equal-budget geometry','D','#66549c')]
    fig,axes=plt.subplots(1,2,figsize=(10,3.3));rows=[];rng=np.random.default_rng(10082026)
    indices=rng.integers(0,47,(20000,47))
    for ax,frame,title in zip(axes,[semantics,identity],['Semantic transfer','Linked-instance transfer']):
        for arm,label,marker,color in definitions:
            part=frame[frame.arm.eq(arm)].sort_values('environment_id')
            assert len(part)==47 and part.coverage.notna().all() and part.precision.notna().all()
            values=part[['coverage','precision']].to_numpy()
            point=values.mean(axis=0);bounds=np.quantile(values[indices].mean(axis=1),[.025,.975],axis=0)
            ax.errorbar(100*point[0],100*point[1],xerr=100*np.array([[point[0]-bounds[0,0]],[bounds[1,0]-point[0]]]),
                yerr=100*np.array([[point[1]-bounds[0,1]],[bounds[1,1]-point[1]]]),fmt=marker,color=color,label=label,capsize=2,markersize=5,elinewidth=.8)
            rows.append(dict(outcome=title,arm=arm,coverage=float(point[0]),precision=float(point[1]),coverage_ci95=bounds[:,0].tolist(),precision_ci95=bounds[:,1].tolist()))
        ax.set(title=title,xlabel='Correct-query coverage (%)',ylabel='Accepted-set precision (%)')
        ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8,bbox_to_anchor=(.5,-.04))
    fig.tight_layout(rect=[0,.10,1,1]);out=Path('outputs/reports/20261008_fixed_query_geometry_figure');out.mkdir(exist_ok=True)
    fig.savefig(out/'fixed_query_geometry_controls.pdf',bbox_inches='tight');fig.savefig(out/'fixed_query_geometry_controls.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(points=rows,scope='All110relations/47environments; same fixed query/reference patches. Coordinate-wise pointwise bootstrap intervals for arm means, not paired difference or joint regions. Raw budgets differ except DINO/equal-budget geometry. Labels only score.'),indent=2)+'\n')

if __name__=='__main__':main()
