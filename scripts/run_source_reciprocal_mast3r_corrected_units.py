#!/usr/bin/env python3
"""Reproduce the 14 matched source reciprocal runs with corrected scoring GT."""
import sys
from pathlib import Path
import json,subprocess,os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from aggregate_mast3r_dense_train_audit import _sparse_directory
from generate_supplementary_tables import short_pair_name
from run_geometry_controls_corrected_units import compare

ROOT=Path.cwd();OUT=ROOT/'outputs/reports/20261008_source_reciprocal_mast3r_corrected_units'
DATA=(Path.home() / 'data/worldcache_datasets/3rscan')

def main():
    matched=pd.read_csv(ROOT/'outputs/reports/20261003_mast3r_dense_train_audit/paired_dino_comparison.csv')
    if len(matched)!=14:raise ValueError('14 matched source relations required')
    OUT.mkdir(exist_ok=True);records=[]
    for row in matched.itertuples(index=False):
        pair=short_pair_name(row.source_run);directory=OUT/row.source_run;directory.mkdir(exist_ok=True)
        old=_sparse_directory(ROOT,row.source_run)
        if old is None:raise ValueError('missing legacy reciprocal result')
        view=ROOT/'outputs/reports/20261008_corrected_source_pair_views'/pair
        if not (directory/'regression.json').exists():
            command=[sys.executable,str(ROOT/'scripts/evaluate_3rscan_mast3r_baseline.py'),
                '--pair-run',str(view),'--annotation-root',str(DATA),'--metadata',str(DATA/'3RScan.json'),'--output-dir',str(directory),
                '--match-mode','reciprocal','--candidate-cache',str(directory/'candidate_cache.npz')]
            with (directory/'runner.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900, env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(ROOT/"src") + os.pathsep + os.environ.get("PYTHONPATH", "")})
            compare(old/'query_diagnostics.csv',directory/'query_diagnostics.csv')
            saved=json.loads((old/'summary.json').read_text())
            matrix=np.load(directory/'predicted_rescan_to_reference.npy')
            old_gt=np.asarray(json.loads((ROOT/str(saved['pair_run'])/'config.json').read_text())['rescan_to_reference'])
            rotation=np.degrees(Rotation.from_matrix(matrix[:3,:3].T@old_gt[:3,:3]).magnitude())
            translation=np.linalg.norm(matrix[:3,3]-old_gt[:3,3])
            np.testing.assert_allclose([rotation,translation],[saved['scene_rotation_error_deg'],saved['scene_translation_error_m']],atol=1e-6,rtol=1e-6)
            (directory/'regression.json').write_text(json.dumps({'status':'pass','all_transfer_rows_reproduced':True,'old_pose_errors_reproduced':True})+'\n')
        new=json.loads((directory/'summary.json').read_text())
        records.append(dict(source_run=row.source_run,pair=pair,strict_pose_success=new['strict_scene_pose_success'],rotation_error_deg=new['scene_rotation_error_deg'],translation_error_m=new['scene_translation_error_m']))
        pd.DataFrame(records).to_csv(OUT/'per_relation.csv',index=False)
        print(len(records),'/14',pair,'reproduced',flush=True)
    (OUT/'summary.json').write_text(json.dumps({'relations':14,'all_transfer_rows_reproduced':True,'strict_pose_successes':sum(r['strict_pose_success'] for r in records)},indent=2)+'\n')

if __name__=='__main__':main()
