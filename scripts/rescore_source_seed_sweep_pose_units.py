#!/usr/bin/env python3
"""Reproduce all saved source seed-sweep summaries and correct only pose GT units."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from evaluate_3rscan_instance_correspondence import read_annotated_mesh
from evaluate_3rscan_multimotion import fit_fixed_consensus


def evaluate_pair(pair: str) -> list[dict]:
    root=Path('outputs/reports/20261006_source_precision_motion/fixed_top20_run')
    with np.load(root/'descriptor_candidates'/f'{pair}.npz') as z:
        q,ref,candidates=z['query_points'],z['reference_points'],z['candidate_indices']
        reference,rescan=str(z['reference_id'].item()),str(z['rescan_id'].item())
    old=pd.read_csv('outputs/runs/20260922_seed_sweep_vitg_clean/per_seed.csv')
    old=old[old.pair.eq(pair)]
    diag=pd.read_csv(root/'query_diagnostics.csv')
    diag=diag[diag.pair.eq(pair)&diag.arm.eq('global_ransac')].sort_values('query_index')
    evaluable=diag.query_evaluable.to_numpy(bool)
    tree,_,_,sem,_=read_annotated_mesh((Path.home() / 'data/worldcache_datasets/3rscan'),reference)
    ref_sem=sem[tree.query(ref.astype(np.float32),k=1)[1]]
    meta=json.load(open(str(Path.home() / 'data/worldcache_datasets/3rscan/3RScan.json')))
    scan=next(s for r in meta if r['reference']==reference for s in r['scans'] if s['reference']==rescan)
    gt=np.asarray(scan['transform'],dtype=np.float32).reshape(4,4).T.copy()
    old_gt=gt.copy();old_gt[:3,3]*=.001
    records=[]
    for saved in old.itertuples(index=False):
        hypothesis=fit_fixed_consensus(q,ref,candidates[:,:saved.top_k],threshold=.1,iterations=1500,seed=int(saved.ransac_seed))
        mask=evaluable&hypothesis.inliers
        semantic=ref_sem[hypothesis.selected]==diag.query_semantic_id.to_numpy()
        rotation=float(np.degrees(Rotation.from_matrix(hypothesis.matrix[:3,:3].T@gt[:3,:3]).magnitude()))
        previous_translation=float(np.linalg.norm(hypothesis.matrix[:3,3]-old_gt[:3,3]))
        actual=[int(mask.sum()),float(semantic[mask].mean()),float(np.median(hypothesis.residuals[mask])),rotation,previous_translation]
        expected=[saved.inliers,saved.inlier_semantic_accuracy,saved.median_inlier_residual_m,saved.rotation_error_deg,saved.translation_error_m]
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6,equal_nan=True)
        translation=float(np.linalg.norm(hypothesis.matrix[:3,3]-gt[:3,3]))
        out=Path('outputs/reports/20261008_source_seed_sweep_corrected_units')/pair
        out.mkdir(parents=True,exist_ok=True)
        np.save(out/f'pose_k{saved.top_k}_seed{saved.ransac_seed}.npy',hypothesis.matrix)
        record=saved._asdict();record.update(translation_error_m=translation,strict_pose_success=rotation<5 and translation<.25,
            old_translation_error_m=saved.translation_error_m,summary_reproduced=True)
        records.append(record)
    return records


def main() -> None:
    saved=pd.read_csv('outputs/runs/20260922_seed_sweep_vitg_clean/per_seed.csv')
    pairs=sorted(saved.pair.unique())
    if len(pairs)!=17:raise ValueError('full source/target sweep required')
    out=Path('outputs/reports/20261008_source_seed_sweep_corrected_units');out.mkdir(exist_ok=True)
    records=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for pair,rows in zip(pairs,pool.map(evaluate_pair,pairs)):
            records.extend(rows);pd.DataFrame(records).to_csv(out/'per_pair.csv',index=False)
            print(pair,'all seed summaries reproduced',flush=True)
    frame=pd.DataFrame(records)
    counts=frame.groupby(['split','ransac_seed']).strict_pose_success.sum().reset_index()
    counts.to_csv(out/'corrected_pose_counts.csv',index=False)
    (out/'summary.json').write_text(json.dumps({'relations':17,'all_saved_summaries_reproduced':True,
        'pose_counts':counts.to_dict('records'),'scene_units':'raw released metadata in metres'},indent=2)+'\n')


if __name__=='__main__':main()
