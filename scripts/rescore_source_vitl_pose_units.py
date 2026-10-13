#!/usr/bin/env python3
"""Reproduce the original fourteen ViT-L results on immutable native-frame views."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from generate_supplementary_tables import short_pair_name


def main() -> None:
    old=pd.read_csv('outputs/runs/20260922_3rscan_consensus_vitl_clean/summary_all.csv')
    if len(old)!=14:raise ValueError('original 14-relation ViT-L scope required')
    out=Path('outputs/reports/20261008_vitl_source_corrected_units');out.mkdir(exist_ok=True)
    script=Path(__file__).with_name('evaluate_3rscan_spatial_consensus.py')
    rows=[]
    for saved in old.itertuples(index=False):
        pair=short_pair_name(saved.pair)
        view=Path('outputs/reports/20261008_corrected_source_pair_views')/pair
        output=out/'relations'/pair;output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(script),'--pair-run',str(view),'--annotation-root',str(Path.home() / 'data/worldcache_datasets/3rscan'),
                 '--output-dir',str(output),'--model-size','large','--layers','2','12','24','--top-k','20',
                 '--threshold','.1','--iterations','1500','--save-query-diagnostics']
        with (output/'runner.log').open('w') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        fresh=pd.read_csv(output/'summary.csv').iloc[0]
        for key in ('queries','matching_query_points','matching_reference_points','matching_inliers','inliers',
                    'inlier_coverage','inlier_semantic_accuracy','median_inlier_residual_m','rotation_error_deg'):
            np.testing.assert_allclose(float(fresh[key]),float(getattr(saved,key)),atol=1e-6,rtol=1e-6)
        pose=np.load(output/'predicted_pose_geometry0_k20_seed17.npy')
        gt=np.asarray(json.loads((view/'config.json').read_text())['rescan_to_reference'])
        old_gt=gt.copy();old_gt[:3,3]*=.001
        translation=float(np.linalg.norm(pose[:3,3]-old_gt[:3,3]))
        np.testing.assert_allclose(translation,saved.translation_error_m,atol=1e-6,rtol=1e-6)
        record=fresh.to_dict();record.update(pair=pair,strict_pose_success=fresh.rotation_error_deg<5 and fresh.translation_error_m<.25,
                                             old_translation_error_m=saved.translation_error_m,summary_reproduced=True)
        rows.append(record);pd.DataFrame(rows).to_csv(out/'per_relation.csv',index=False)
        print(f'{len(rows)}/14 {pair} reproduced',flush=True)
    frame=pd.DataFrame(rows)
    (out/'summary.json').write_text(json.dumps({'relations':14,'all_saved_summaries_reproduced':True,
        'strict_pose_successes':int(frame.strict_pose_success.sum()),'scene_units':'raw released metadata in metres'},indent=2)+'\n')


if __name__=='__main__':main()
