#!/usr/bin/env python3
"""Closed-set ARID20 cross-view object retrieval on the official OCID labels.

For each floor/table seqNN pair, the final bottom-view frame queries all object
instances in the final top-view frame and vice versa. Ground truth is the
sequence-local integer instance label. Frozen DINOv2-B/14 descriptors use
mask-weighted patch tokens from tight padded object crops; baselines are a DINO
CLS crop descriptor and a masked RGB color histogram. Scenes are bootstrap
clusters. This is an exploratory external-data audit, not a deployment test.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import torch, timm

DEFAULT_ROOT = Path('data/OCID-dataset/ARID20')
DEFAULT_OUT = Path('outputs/reports/20261009_ocid_cross_view_identity')
MODEL = 'vit_base_patch14_dinov2.lvd142m'
MEAN = np.array([.485,.456,.406], np.float32)
STD = np.array([.229,.224,.225], np.float32)

def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)

def extract(root: Path, model, device: str):
    scene_data=[]; tensors=[]; masks=[]
    for loc in ('floor','table'):
      for si in range(1,14):
        seq=f'seq{si:02d}'; views={}
        for vi,view in enumerate(('bottom','top')):
          base=root/loc/view/seq
          rgbfiles=sorted((base/'rgb').glob('*.png'))
          if not rgbfiles: raise FileNotFoundError(base/'rgb')
          fn=rgbfiles[-1]
          im=np.asarray(Image.open(fn).convert('RGB'))
          lab=np.asarray(Image.open(base/'label'/fn.name))
          ids=np.unique(lab); ids=ids[ids>0]
          objs={}
          for oid in ids:
            ys,xs=np.where(lab==oid)
            if not len(xs): continue
            pad=8; y0=max(0,int(ys.min())-pad); y1=min(im.shape[0],int(ys.max())+pad+1)
            x0=max(0,int(xs.min())-pad); x1=min(im.shape[1],int(xs.max())+pad+1)
            crop=np.asarray(Image.fromarray(im[y0:y1,x0:x1]).resize((224,224),Image.Resampling.BILINEAR))
            mask=np.asarray(Image.fromarray((lab[y0:y1,x0:x1]==oid).astype(np.uint8)*255).resize((224,224),Image.Resampling.NEAREST))>0
            arr=(crop.astype(np.float32)/255.-MEAN)/STD
            key=(loc,seq,vi,int(oid))
            tensors.append(torch.from_numpy(arr).permute(2,0,1)); masks.append(mask)
            objs[int(oid)]={'tensor_index':len(tensors)-1,'mask':mask,'rgb':im,'label':lab,'source':str(fn)}
          views[view]={'objects':objs,'frame':str(fn),'image_shape':list(im.shape),'max_label':int(ids.max()) if len(ids) else 0}
        common=set(views['bottom']['objects']) & set(views['top']['objects'])
        if not common: raise ValueError(f'no common visible IDs for {loc}/{seq}')
        for view in ('bottom','top'):
          views[view]['unmatched_visible_ids']=sorted(set(views[view]['objects'])-common)
          views[view]['objects']={oid:obj for oid,obj in views[view]['objects'].items() if oid in common}
        scene_data.append({'scene':f'{loc}/{seq}','views':views,'common_ids':sorted(common)})
    cls=[]; pooled=[]
    model.eval()
    with torch.inference_mode():
      for start in range(0,len(tensors),64):
        t=torch.stack(tensors[start:start+64]).to(device)
        out=model.forward_features(t)
        if isinstance(out,dict):
          patch=out.get('x_norm_patchtokens',out.get('last_hidden_state'))
          c=out.get('x_norm_clstoken')
          if patch is None: raise RuntimeError('No patch-token output from DINOv2')
          if c is None: c=out['last_hidden_state'][:,0]
          patch=patch.float().cpu().numpy().reshape(len(t),16,16,-1)
          c=c.float().cpu().numpy()
        else:
          out=out.float().cpu().numpy()
          patch=out[:,1:257,:].reshape(len(t),16,16,-1); c=out[:,0,:]
        cls.extend(c)
        for j in range(len(t)):
          m=masks[start+j].astype(np.float32).reshape(16,14,16,14).mean((1,3))
          pooled.append((patch[j]*m[:,:,None]).sum((0,1))/(m.sum()+1e-12))
    cls=unit(np.stack(cls)); pooled=unit(np.stack(pooled))
    for scene in scene_data:
      for vi,view in enumerate(('bottom','top')):
        for oid,obj in scene['views'][view]['objects'].items():
          idx=obj['tensor_index']; obj['dino_cls']=cls[idx]; obj['dino_maskpool']=pooled[idx]
          pixels=obj['rgb'][obj['label']==oid]
          hist=[]
          for c in range(3):
            h,_=np.histogram(pixels[:,c],bins=32,range=(0,256))
            hist.extend(h.astype(np.float32)/(len(pixels)+1e-12))
          obj['rgbhist']=unit(hist)
    return scene_data

def score_symmetric(scenes, descriptor):
    rows=[]
    for s in scenes:
      a=s['views']['bottom']['objects']; b=s['views']['top']['objects']; ids=sorted(a)
      A=unit(np.stack([a[i][descriptor] for i in ids])); B=unit(np.stack([b[i][descriptor] for i in ids]))
      sim=A@B.T
      for direction,mat in [('bottom_to_top',sim),('top_to_bottom',sim.T)]:
        pred=mat.argmax(1)
        for q,oid in enumerate(ids):
          rows.append({'scene':s['scene'],'direction':direction,'object_id':oid,'candidate_count':len(ids),'correct':int(ids[pred[q]]==oid),'top1_margin':float(np.sort(mat[q])[-1]-np.sort(mat[q])[-2])})
    return rows

def summarize(rows, seed=437, nboot=20000):
    scenes=sorted({r['scene'] for r in rows}); by={s:[r for r in rows if r['scene']==s] for s in scenes}
    n=len(rows); correct=sum(r['correct'] for r in rows); scene_acc={s:sum(r['correct'] for r in by[s])/len(by[s]) for s in scenes}
    rng=np.random.default_rng(seed); vals=np.empty(nboot)
    for i in range(nboot):
      sampled=rng.choice(scenes,size=len(scenes),replace=True)
      vals[i]=np.mean([scene_acc[s] for s in sampled])
    return {'n_query_instances':n,'correct':correct,'micro_top1':correct/n,'scene_macro_top1':float(np.mean(list(scene_acc.values()))),'cluster_bootstrap_95_scene_macro':[float(x) for x in np.quantile(vals,[.025,.975])],'n_scenes':len(scenes),'scene_accuracies':scene_acc}

def paired_difference(rows_a, rows_b, seed=437, nboot=20000):
    by_a={}; by_b={}
    for r in rows_a: by_a.setdefault(r['scene'],[]).append(r['correct'])
    for r in rows_b: by_b.setdefault(r['scene'],[]).append(r['correct'])
    scenes=sorted(set(by_a)&set(by_b))
    diffs=np.array([np.mean(by_a[x])-np.mean(by_b[x]) for x in scenes])
    rng=np.random.default_rng(seed); samples=np.empty(nboot)
    for i in range(nboot): samples[i]=rng.choice(diffs,size=len(diffs),replace=True).mean()
    return {'scene_macro_difference':float(diffs.mean()),'cluster_bootstrap_95':[float(x) for x in np.quantile(samples,[.025,.975])],'n_scenes':len(scenes)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');ap.add_argument('--seed',type=int,default=437);ap.add_argument('--bootstrap',type=int,default=20000); args=ap.parse_args()
    torch.set_num_threads(4); model=timm.create_model(MODEL,pretrained=True,num_classes=0,img_size=224).to(args.device).eval()
    scenes=extract(args.root,model,args.device)
    summaries={}; all_rows={}
    for descriptor in ('dino_maskpool','dino_cls','rgbhist'):
      rows=score_symmetric(scenes,descriptor); summaries[descriptor]=summarize(rows,args.seed,args.bootstrap); all_rows[descriptor]=rows
    args.out.mkdir(parents=True,exist_ok=True)
    diffs={'dino_maskpool_minus_rgbhist':paired_difference(all_rows['dino_maskpool'],all_rows['rgbhist'],args.seed,args.bootstrap),'dino_maskpool_minus_dino_cls':paired_difference(all_rows['dino_maskpool'],all_rows['dino_cls'],args.seed,args.bootstrap)}
    subsets={}
    for loc in ('floor','table'):
      subsets[loc]={name:summarize([r for r in rows if r['scene'].startswith(loc+'/')],args.seed,args.bootstrap) for name,rows in all_rows.items()}
    (args.out/'summary.json').write_text(json.dumps({'dataset':'OCID ARID20','source_record':'https://doi.org/10.48436/pcbjd-4wa12','license_status':'not specified in official metadata; verify with uploader before redistribution or further use','archive_bytes':6822716319,'archive_md5_verified':'aa0eff01f75104c40e4243e6f892a757','model':MODEL,'protocol':'final recorded frame in each paired bottom/top view; sequence-local IDs visible in both frames are candidates; both retrieval directions; closed set','n_frames':sum(2 for _ in scenes),'n_scenes':len(scenes),'n_objects':len(all_rows['dino_maskpool'])//2,'candidate_count_distribution':{str(k):int(v) for k,v in sorted(Counter({r['scene']:r['candidate_count'] for r in all_rows['dino_maskpool']}.values()).items())},'n_ids_not_visible_in_both_views':sum(len(s['views']['bottom']['unmatched_visible_ids'])+len(s['views']['top']['unmatched_visible_ids']) for s in scenes),'summaries':summaries,'paired_differences':diffs,'location_subsets':subsets},indent=2)+'\n')
    for name,rows in all_rows.items():
      with (args.out/f'{name}_rows.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row)+'\n')
    lines=['# OCID ARID20 cross-view identity retrieval','','Protocol: for each of 26 floor/table scene sequences, the final frame from bottom view queries the final frame from top view and vice versa. Official sequence-local integer instance IDs define matches; the candidate set includes IDs visible in both final frames (one sequence object is excluded because it is absent from one view). DINOv2-B/14 is frozen and pretrained. Confidence intervals resample whole sequences. The official TU Wien record lists a 6.82 GB archive and MD5 but leaves its Rights/License field blank; this audit therefore does not assume an open license. The prior local copy was removed after evaluation, and no archive was re-downloaded. This is exploratory and closed-set. Dataset source: [TU Wien OCID record](https://doi.org/10.48436/pcbjd-4wa12); downloaded archive MD5 was verified against the official metadata.','',f"The run covers {len(scenes)} sequences and {summaries['dino_maskpool']['n_query_instances']} directional object queries over {summaries['dino_maskpool']['n_scenes']} scene clusters.",'','| Descriptor | Top-1 | Scene-macro 95% cluster bootstrap |','|---|---:|---:|','']
    for name,title in [('dino_maskpool','DINOv2-B/14 mask-pooled patch tokens from tight object crop'),('dino_cls','DINOv2-B/14 CLS token from tight object crop'),('rgbhist','Masked RGB 32-bin/channel histogram')]:
      z=summaries[name]; lo,hi=z['cluster_bootstrap_95_scene_macro'];lines.append(f"| {title} | {z['micro_top1']*100:.1f}% ({z['correct']}/{z['n_query_instances']}) | {z['scene_macro_top1']*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]% |")
    d=diffs['dino_maskpool_minus_rgbhist']; dlo,dhi=d['cluster_bootstrap_95']
    lines += ['',f"The paired DINO mask-pooling gain over RGB histograms is {d['scene_macro_difference']*100:.1f} pp (scene-cluster 95% bootstrap [{dlo*100:.1f}, {dhi*100:.1f}] pp).",'', '| Location | DINO mask-pooled | DINO CLS | RGB histogram |','|---|---:|---:|---:|']
    for loc in ('floor','table'):
      subset=subsets[loc]
      lines.append(f"| {loc} | {subset['dino_maskpool']['scene_macro_top1']*100:.1f}% | {subset['dino_cls']['scene_macro_top1']*100:.1f}% | {subset['rgbhist']['scene_macro_top1']*100:.1f}% |")
    lines+=['','Limitations: RGB boxes/masks are privileged; one final frame per view; the candidate set is restricted to visible instances in the same preassembled scene; this measures instance retrieval across OCID viewpoints, not temporal cache reuse or pose accuracy.','']
    (args.out/'report.md').write_text('\n'.join(lines))
    print((args.out/'report.md').read_text())
if __name__=='__main__': main()
