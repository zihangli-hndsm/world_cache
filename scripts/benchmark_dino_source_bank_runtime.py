#!/usr/bin/env python3
"""Measure CPU matching cost for fixed 12-view vs full 20-view DINO banks."""
from __future__ import annotations
import csv, json, platform, random, statistics, time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits, threadpool_info
ROOT=Path(__file__).resolve().parents[1]
FEATURES=ROOT/'outputs/reports/20261009_dino_multiview_source_bank'
SELECTIONS=ROOT/'outputs/reports/20261009_dino_source_bank_no_pair_condition/per_relation.csv'
OUT=ROOT/'outputs/reports/20261009_dino_source_bank_runtime'
REPEATS=50
SEED=20261011


def load_items():
    sel=[]
    with SELECTIONS.open() as f:
        for row in csv.DictReader(f):
            if row['strategy']=='temporal_uniform' and row['repeat']=='0' and row['bank_size'] in {'12','20'}:
                sel.append(row)
    if len(sel)!=220: raise RuntimeError(f'expected 220 selection rows, got {len(sel)}')
    by_run={}
    for row in sel: by_run.setdefault(row['run_id'],{})[int(row['bank_size'])]=row
    items=[]
    for i,(run_id,rs) in enumerate(sorted(by_run.items())):
        if set(rs)!={12,20}: raise RuntimeError(f'incomplete selection rows: {run_id}')
        with np.load(FEATURES/f'{run_id}_features.npz') as z:
            sf=z['source_embeddings'].astype(np.float32)
            sf/=np.maximum(np.linalg.norm(sf,axis=1,keepdims=True),1e-8)
            sframe=z['source_frame_ids'].astype(np.int32)
            qf=z['query_embeddings'].astype(np.float32)
            qf/=np.maximum(np.linalg.norm(qf,axis=1,keepdims=True),1e-8)
            qframe=z['query_frame_ids'].astype(np.int32)
        selected={}
        for k in (12,20):
            frames=np.array([int(x) for x in rs[k]['selected_reference_frames'].split(',')],dtype=np.int32)
            idx=np.flatnonzero(np.isin(sframe,frames))
            selected[k]=idx
        qgroups=[np.flatnonzero(qframe==f) for f in np.unique(qframe)]
        if any(len(q)==0 for q in qgroups): raise RuntimeError(f'empty query group {run_id}')
        items.append({'run_id':run_id,'features':sf,'queries':qf,'source_idx':selected,'query_groups':qgroups,
                      'budget':int(rs[12]['accepted']),'full_count':len(sf),'target_count':int(rs[12]['source_descriptors'])})
    return items


def score_one(item,k):
    source=item['features'][item['source_idx'][k]]
    candidates=[]
    sims_total=0
    for qidx in item['query_groups']:
        q=item['queries'][qidx]
        sims=q @ source.T
        sims_total+=int(sims.size)
        rr,cc=linear_sum_assignment(-sims)
        for qi,si in zip(rr.tolist(),cc.tolist()):
            candidates.append((float(sims[qi,si]),int(qidx[qi])))
    candidates.sort(key=lambda z:(-z[0],z[1]))
    top=candidates[:item['budget']]
    return len(candidates),len(top),sims_total


def run_epoch(items,order):
    out={}
    for k in order:
        candidates=accepted=similarity_entries=0
        for item in items:
            c,a,s=score_one(item,k)
            candidates+=c;accepted+=a;similarity_entries+=s
        out[k]={'candidates':candidates,'accepted':accepted,'similarity_entries':similarity_entries}
    return out


def main():
    items=load_items()
    with threadpool_limits(limits=1,user_api='blas'):
        runtime_threadpools = threadpool_info()
        # Warm-up both paths before recording paired alternating epochs.
        run_epoch(items,[12,20])
        times={12:[],20:[]}; totals={}
        rng=random.Random(SEED)
        for epoch in range(REPEATS):
            order=[12,20]
            if epoch%2: order.reverse()
            for k in order:
                t=time.perf_counter(); results=run_epoch(items,[k]); elapsed=time.perf_counter()-t
                times[k].append(elapsed); totals[k]=results[k]
    rows=[]
    for k in (12,20):
        rows.append({'bank_size':k,'epochs':REPEATS,'median_seconds':statistics.median(times[k]),
                     'mean_seconds':statistics.mean(times[k]),'p10_seconds':float(np.quantile(times[k],.10)),
                     'p90_seconds':float(np.quantile(times[k],.90)),**totals[k]})
    by={r['bank_size']:r for r in rows}
    summary={
      'status':'complete','protocol':{
        'input':'cached normalized DINOv2 ViT-G mask embeddings from the 110 official 3RScan validation relations',
        'operation':'per-query-frame CPU cosine matrix multiplication, SciPy Hungarian assignment, and per-relation score sorting; no encoder inference, file loading, identity scoring, or threshold fitting is timed',
        'selection':'12-view temporal-uniform fixed source subset shared by all query frames vs all 20 source views',
        'threading':'BLAS restricted to one thread; 1 warm-up epoch then 50 paired alternating epochs',
        'machine':platform.platform(),'cpu':platform.processor(),'threadpools':runtime_threadpools,
        'limits':['This measures only post-encoder CPU matching on cached features. It is not end-to-end latency, GPU latency, online feature extraction, or robotics task performance.',
                  'The source bank uses annotation-derived masks, and the 3RScan validation cohort has already been explored.']},
      'relation_count':len(items),'results':rows,
      'ratios':{
        'median_runtime_12_over_20':by[12]['median_seconds']/by[20]['median_seconds'],
        'similarity_entries_12_over_20':by[12]['similarity_entries']/by[20]['similarity_entries'],
        'stored_descriptor_count_12_over_20':sum(x['target_count'] for x in items)/sum(x['full_count'] for x in items),
      },'epoch_seconds':{str(k):times[k] for k in times}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    md=['# DINO source-bank matching runtime and storage audit','',
        'We timed the CPU matching portion on cached, normalized embeddings for all 110 explored validation relations. Each timed epoch runs per-query-frame cosine matrix multiplication, Hungarian assignment, and per-relation score ranking for either a fixed 12-view source bank or the full 20-view bank. DINO feature extraction, disk reads, annotation loading, and label scoring are excluded. BLAS uses one thread; each variant is warmed and 50 epochs alternate order.','',
        '| Bank | Median seconds / 110 relations | p10–p90 seconds | Similarity entries | Accepted links |','|---:|---:|---:|---:|---:|']
    for r in rows: md.append(f"| {r['bank_size']} | {r['median_seconds']:.4f} | {r['p10_seconds']:.4f}–{r['p90_seconds']:.4f} | {r['similarity_entries']:,} | {r['accepted']:,} |")
    md += ['',f"The 12-view bank retains {summary['ratios']['stored_descriptor_count_12_over_20']:.1%} of the full bank's source descriptors and {summary['ratios']['similarity_entries_12_over_20']:.1%} of its measured similarity-matrix entries. Median matching time is {summary['ratios']['median_runtime_12_over_20']:.1%} of the 20-view cost. This is a post-encoder CPU microbenchmark on cached features; it does not measure encoder savings or end-to-end latency. The descriptor cache is float16 with 1,536 dimensions, so each bank mask descriptor occupies 3,072 bytes before container overhead. The 12-view source selection reduces descriptor bytes by the same fraction as the measured count reduction.",'',
        'The evaluation uses the already explored 3RScan validation masks and does not establish independent-site transfer, runtime on a deployed robot, or a new association method. Reproduce with `python scripts/benchmark_dino_source_bank_runtime.py`; raw per-epoch timings and machine details are in `summary.json`.','']
    (OUT/'report.md').write_text('\n'.join(md))
    print(json.dumps({'status':'complete','results':rows,'ratios':summary['ratios']},indent=2))

if __name__=='__main__': main()
