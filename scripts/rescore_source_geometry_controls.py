#!/usr/bin/env python3
"""Reproduce source controls on immutable frame views with corrected scoring units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from generate_supplementary_tables import short_pair_name
from prepare_corrected_scene_unit_pair_view import prepare


def check_summary(saved: pd.Series, fresh: pd.Series, pose: np.ndarray, gt: np.ndarray,
                  columns: list[str]) -> None:
    for column in columns:
        np.testing.assert_allclose(float(saved[column]),float(fresh[column]),atol=1e-6,rtol=1e-6,equal_nan=True)
    old_gt=gt.copy();old_gt[:3,3]*=.001
    rotation=float(np.degrees(Rotation.from_matrix(pose[:3,:3].T@old_gt[:3,:3]).magnitude()))
    translation=float(np.linalg.norm(pose[:3,3]-old_gt[:3,3]))
    np.testing.assert_allclose([rotation,translation],[saved.rotation_error_deg,saved.translation_error_m],atol=1e-5,rtol=1e-5)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/reports/20261008_source_geometry_controls_corrected_units'))
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    data=(Path.home() / 'data/worldcache_datasets/3rscan')
    metadata=json.loads((data/'3RScan.json').read_text())
    pairs=pd.read_csv('outputs/reports/20260924_object_motion_oracle/object_motion_oracle_pairs.csv')
    if len(pairs)!=17 or pairs.pair.duplicated().any():raise ValueError('17 source/target pairs required')
    fcgf_old=pd.read_csv('outputs/runs/20260922_fcgf_17pair_clean/summary_all.csv');fcgf_old['short']=fcgf_old.pair.map(short_pair_name)
    fpfh_old=pd.read_csv('outputs/runs/20260922_fpfh_17pair_clean/summary_all.csv');fpfh_old['short']=fpfh_old.pair.map(short_pair_name)
    scripts=Path(__file__).resolve().parent
    records=[]
    for pair in pairs.itertuples(index=False):
        view=Path('outputs/reports/20261008_corrected_source_pair_views')/pair.pair
        out=a.output_dir/pair.pair;out.mkdir(exist_ok=True)
        prepare(Path(pair.pair_run),view,metadata)
        gt=np.asarray(json.loads((view/'config.json').read_text())['rescan_to_reference'])
        commands=[
            [sys.executable,str(scripts/'evaluate_3rscan_fpfh_baseline.py'),'--pair-run',str(view),'--annotation-root',str(data),
             '--output-dir',str(out/'fpfh'),'--voxel-size','.08','--normal-radius','.16','--feature-radius','.40',
             '--threshold','.1','--iterations','5000','--save-query-diagnostics'],
            [str(Path.home() / 'data/conda/envs/worldcache_fcgf/bin/python'),str(scripts/'evaluate_3rscan_fcgf_baseline.py'),
             '--pair-run',str(view),'--annotation-root',str(data),'--output-dir',str(out/'fcgf'),
             '--checkpoint',str(Path.home() / 'data/worldcache_models/fcgf/fcgf-3dmatch.pth'),'--voxel-size','.025',
             '--top-k','20','--threshold','.1','--iterations','3000','--seed','17','--save-query-diagnostics']]
        with (out/'runner.log').open('w') as log:
            for command in commands:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        new=pd.read_csv(out/'fcgf/summary.csv').iloc[0]
        old=fcgf_old[fcgf_old.short.eq(pair.pair)&fcgf_old.top_k.eq(20)]
        if len(old)!=1:raise ValueError('unique source FCGF K20 row required')
        pose=np.load(out/'fcgf/predicted_pose_k20_seed17.npy')
        check_summary(old.iloc[0],new,pose,gt,['reference_queries','query_queries','matching_inliers','inliers',
            'inlier_coverage','inlier_semantic_accuracy','median_inlier_residual_m'])
        new_fpfh=pd.read_csv(out/'fpfh/summary.csv')
        for arm in ('fpfh_one_way','fpfh_mutual'):
            old=fpfh_old[fpfh_old.short.eq(pair.pair)&fpfh_old.method.eq(arm)]
            if len(old)!=1:raise ValueError('unique source FPFH row required')
            check_summary(old.iloc[0],new_fpfh[new_fpfh.method.eq(arm)].iloc[0],np.load(out/f'fpfh/predicted_pose_{arm}.npy'),gt,
                ['cloud_source_points','cloud_target_points','matches','matching_inliers','inliers','inlier_coverage',
                 'inlier_semantic_accuracy','median_inlier_residual_m'])
        (out/'regression.json').write_text(json.dumps({'status':'pass','saved_counts_precision_residual_and_old_pose_reproduced':True,
            'scope':'source lacks legacy query rows; summary reproduction verified; immutable native frame hashes retained'},indent=2)+'\n')
        records.append({'pair':pair.pair,'source_pair_run':pair.pair_run,'fcgf_corrected_pose_success':bool(new.pose_success_lt5deg_lt025m),
            'fcgf_translation_error_m':float(new.translation_error_m),'fpfh_one_way_pose_success':bool((new_fpfh[new_fpfh.method.eq('fpfh_one_way')].rotation_error_deg<5).iloc[0] and (new_fpfh[new_fpfh.method.eq('fpfh_one_way')].translation_error_m<.25).iloc[0]),'status':'complete'})
        pd.DataFrame(records).to_csv(a.output_dir/'run_manifest.csv',index=False)
        print(f'{len(records)}/17 {pair.pair} reproduced',flush=True)


if __name__=='__main__':main()
