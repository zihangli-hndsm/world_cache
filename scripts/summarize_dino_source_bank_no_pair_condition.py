#!/usr/bin/env python3
"""Summarize the fixed, source-only DINO bank selection audit."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/20261009_dino_source_bank_no_pair_condition'
d=json.loads((OUT/'summary.json').read_text())
raw=pd.read_csv(OUT/'per_relation.csv')
env=raw.groupby(['bank_size','strategy','repeat','environment_id'],as_index=False).agg(
    query_masks=('query_masks','sum'), accepted=('accepted','sum'), correct=('correct','sum'),
    remind_budget=('remind_budget','sum'), source_descriptors=('source_descriptors','sum'),
    full_source_descriptors=('full_source_descriptors','sum'))
env['precision']=env.correct/env.accepted
env['coverage']=env.correct/env.query_masks
envs=sorted(env.environment_id.unique())
rng=np.random.default_rng(20261010)
ix=rng.integers(0,len(envs),size=(100000,len(envs)))
# Average randomized bank selections within each group; retain paired group-level rows.
macro=env.groupby(['bank_size','strategy','environment_id'],as_index=False).agg(
    precision=('precision','mean'),coverage=('coverage','mean'))
remind=json.loads((ROOT/'outputs/reports/20261009_dino_multiview_source_bank/summary.json').read_text())
remind_by={r['environment_id']:r['remind_one_to_one'] for r in remind['reference_groups']}
full={}
for strat in ['random','temporal_uniform','feature_diverse']:
    full[strat]=macro[(macro.bank_size==20)&(macro.strategy==strat)].set_index('environment_id').reindex(envs)
contrasts={}
for k in [1,2,4,8,12]:
    contrasts[str(k)]={}
    for strat in ['random','temporal_uniform','feature_diverse']:
        a=macro[(macro.bank_size==k)&(macro.strategy==strat)].set_index('environment_id').reindex(envs)
        contrasts[str(k)][strat]={}
        for metric in ['precision','coverage']:
            base=full[strat][metric].to_numpy(float)
            v=a[metric].to_numpy(float)
            delta=v-base
            boot=delta[ix].mean(axis=1)
            contrasts[str(k)][strat][metric]={
                'delta_vs_full_estimate':float(delta.mean()),
                'delta_vs_full_ci95_pointwise':[float(x) for x in np.quantile(boot,[.025,.975])],
                'delta_vs_full_ci95_bonferroni_two_outcomes':[float(x) for x in np.quantile(boot,[.0125,.9875])],
                'groups_positive':int((delta>0).sum()),'groups_tie':int(np.isclose(delta,0).sum())}
# Primary descriptive comparisons for temporal 12-view bank: paired group bootstrap.
k=12; strat='temporal_uniform'
a=macro[(macro.bank_size==k)&(macro.strategy==strat)].set_index('environment_id').reindex(envs)
contrasts['12']['temporal_uniform']['vs_remind']={}
for metric in ['precision','coverage']:
    base=np.asarray([remind_by[g][metric] for g in envs],float)
    v=a[metric].to_numpy(float); delta=v-base; boot=delta[ix].mean(axis=1)
    contrasts['12']['temporal_uniform']['vs_remind'][metric]={
      'estimate':float(delta.mean()),'ci95_pointwise':[float(x) for x in np.quantile(boot,[.025,.975])],
      'groups_positive':int((delta>0).sum())}
d['summary']['difference_vs_full_bank']=contrasts
(OUT/'summary.json').write_text(json.dumps(d,indent=2)+'\n')
rows=[]
for k in [1,2,4,8,12,20]:
    sub=env[env.bank_size==k]
    # Common accepted count is identical across all strategies/repeats per relation.
    budget=float(sub.groupby('environment_id').accepted.first().sum())
    remind_total=float(sub.groupby('environment_id').remind_budget.first().sum())
    for strat in ['random','temporal_uniform','feature_diverse']:
        storage=d['summary']['feature_storage'][f'{k}:{strat}']
        x=d['summary']['macro'][str(k)].get(strat)
        if x is None: continue
        rows.append((k,strat,storage['mean_descriptor_fraction'],x['precision']['estimate'],x['coverage']['estimate'],budget/remind_total))
lines=[
'# Fixed source-only reference bank: view-pairing sensitivity', '',
'This re-score uses the cached DINOv2 mask embeddings from 110 official 3RScan validation relations (47 reference groups). For each relation, a single selected reference-frame subset is shared by every query frame; the query-to-reference frame pairing used in the previous view-count curve is not supplied. Selection is label-blind: random valid-frame subsets, temporal-uniform subsets, or greedy farthest-first subsets over the reference-only mean mask embeddings. Ground-truth identity IDs are loaded only after candidate assignments are generated. The instance masks remain privileged inputs.', '',
'For each relation and bank size, all strategies and random repeats use the largest common number of one-to-one assignments supported by every subset, capped at the REMIND accepted-link count. From 2 selected views upward, the full per-relation REMIND count is supported. At 1 view, the common budget is 98.3% of REMIND because some subsets lack enough distinct source masks. At each budget, assigned links are ranked by DINO cosine. This is a post-hoc fixed-budget diagnostic, not a deployment policy.', '',
'## Bank size, descriptor storage, and outcomes', '',
'| Views | Selector | Stored mask descriptors (% of 20-view bank) | Link precision | Correct coverage | Accepted budget (% of REMIND) |',
'|---:|---|---:|---:|---:|---:|']
for k,st,frac,prec,cov,bfrac in rows:
    lines.append(f'| {k} | {st.replace("_"," ")} | {frac:.1%} | {prec:.1%} | {cov:.2%} | {bfrac:.1%} |')
lines += ['', 'At 12 temporally uniform views, DINO macro precision is {:.1%} and correct coverage {:.2%}, versus {:.1%} and {:.2%} for the full 20-view bank. The paired 47-group cluster-bootstrap difference (12 minus 20) is {:.2f} pp in precision (pointwise 95% CI [{:.2f},{:.2f}]; two-outcome Bonferroni interval [{:.2f},{:.2f}]) and {:.2f} pp in coverage (pointwise [{:.2f},{:.2f}]; Bonferroni [{:.2f},{:.2f}]). The 12-view bank stores {:.1%} as many mask descriptors, a {:.1%} reduction. This indicates a promising storage/performance tradeoff on the explored masks, but does not establish equivalence.'.format(
    d['summary']['macro']['12']['temporal_uniform']['precision']['estimate'],
    d['summary']['macro']['12']['temporal_uniform']['coverage']['estimate'],
    d['summary']['macro']['20']['temporal_uniform']['precision']['estimate'],
    d['summary']['macro']['20']['temporal_uniform']['coverage']['estimate'],
    d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['precision']['delta_vs_full_estimate']*100,
    *[x*100 for x in d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['precision']['delta_vs_full_ci95_pointwise']],
    *[x*100 for x in d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['precision']['delta_vs_full_ci95_bonferroni_two_outcomes']],
    d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['coverage']['delta_vs_full_estimate']*100,
    *[x*100 for x in d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['coverage']['delta_vs_full_ci95_pointwise']],
    *[x*100 for x in d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['coverage']['delta_vs_full_ci95_bonferroni_two_outcomes']],
    d['summary']['feature_storage']['12:temporal_uniform']['mean_descriptor_fraction'],
    d['summary']['feature_storage']['12:temporal_uniform']['mean_storage_reduction']),
'','The 12-view bank is also higher than REMIND at the matched accepted count: macro precision {:.1%} vs {:.1%}, correct coverage {:.2%} vs {:.2%}. Paired 47-group differences are +{:.2f} pp precision (pointwise 95% CI [{:.2f},{:.2f}]) and +{:.2f} pp coverage ([{:.2f},{:.2f}]). The all-20-view bank reproduces the existing exact-budget result because the full relation-wide reference bank has no special paired-frame inclusion.'.format(
 d['summary']['macro']['12']['temporal_uniform']['precision']['estimate'],
 np.mean([remind_by[g]['precision'] for g in envs]),
 d['summary']['macro']['12']['temporal_uniform']['coverage']['estimate'],
 np.mean([remind_by[g]['coverage'] for g in envs]),
 d['summary']['difference_vs_remind']['12:temporal_uniform']['precision']['estimate']*100,
 *[x*100 for x in d['summary']['difference_vs_remind']['12:temporal_uniform']['precision']['ci95']],
 d['summary']['difference_vs_remind']['12:temporal_uniform']['coverage']['estimate']*100,
 *[x*100 for x in d['summary']['difference_vs_remind']['12:temporal_uniform']['coverage']['ci95']]),
'', 'Feature-diverse selection does not clearly improve over temporal-uniform selection at the same bank size; the evaluation does not show a specific view-selection algorithm is necessary. The selector/size grid is exploratory and all 95% intervals are pointwise except the stated two-outcome correction for the 12-versus-20 comparison. The data are the explored 3RScan validation cohort, use annotation-derived masks, and do not provide independent-site evidence. Descriptor counts estimate cache storage; the separate timing below covers only post-encoder matching, not end-to-end memory or a downstream robot task.', '',
'Reproduce the feature-only candidate generation with `python scripts/analyze_dino_source_bank_no_pair_condition.py --repeats 20 --draws 100000`; reproduce the report with `python scripts/summarize_dino_source_bank_no_pair_condition.py`. Per-relation predictions, selected frames, exact budget counts and descriptor counts are in `per_relation.csv`; machine summary is `summary.json`.', '']
# correction: correct accidental awkward guard expression above while preserving value
runtime = json.loads((ROOT / "outputs/reports/20261009_dino_source_bank_runtime/summary.json").read_text())
r12, r20 = runtime["results"]
needle = next(i for i, line in enumerate(lines) if line.startswith("Feature-diverse selection"))
lines.insert(needle, f"A separate single-thread CPU microbenchmark on these cached features takes {r12['median_seconds']:.3f} seconds for 12 views and {r20['median_seconds']:.3f} seconds for 20 views across 110 relations ({(1-runtime['ratios']['median_runtime_12_over_20'])*100:.1f}% less matching time; 50 alternating epochs). It includes cosine products, Hungarian assignment, and score sorting, but excludes feature extraction and disk reads. Full timing limits are in `outputs/reports/20261009_dino_source_bank_runtime/report.md`.")
(OUT/'report.md').write_text('\n'.join(lines))
print(OUT/'report.md')
print('key 12-view vs full precision', d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['precision'])
print('key 12-view vs full coverage', d['summary']['difference_vs_full_bank']['12']['temporal_uniform']['coverage'])
