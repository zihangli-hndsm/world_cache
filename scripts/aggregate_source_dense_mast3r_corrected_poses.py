#!/usr/bin/env python3
"""Complete 17-source dense MASt3R strict-pose audit after unit correction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from run_geometry_controls_corrected_units import compare

ROOT=Path.cwd();REPORTS=ROOT/'outputs/reports'

def main():
    corrected=REPORTS/'20261008_source_dense_mast3r_corrected_units'
    replay=pd.read_csv(corrected/'per_relation_seed.csv');replay=replay[replay.ransac_seed.eq(17)].set_index('source_run')
    if len(replay)!=14:raise ValueError('14 corrected matched source relations required')
    legacy=pd.read_csv(REPORTS/'20261003_mast3r_dense_train_audit/per_relation.csv')
    if len(legacy)!=17 or legacy.source_run.duplicated().any():raise ValueError('17 source relations required')
    extra_name='20260922_3rscan_extra_1776_pair20_s1'
    extra=REPORTS/'20261008_source_dense_mast3r_extra1776_corrected_units'
    old=ROOT/'outputs/runs/20261003_mast3r_dense_topk_extra_1776'
    compare(old/'query_diagnostics.csv',extra/'query_diagnostics.csv')
    matrix=np.load(extra/'predicted_rescan_to_reference.npy')
    saved=json.loads((old/'summary.json').read_text())
    old_gt=np.asarray(json.loads((ROOT/'outputs/runs'/extra_name/'config.json').read_text())['rescan_to_reference'])
    rotation=np.degrees(Rotation.from_matrix(matrix[:3,:3].T@old_gt[:3,:3]).magnitude())
    np.testing.assert_allclose([rotation,np.linalg.norm(matrix[:3,3]-old_gt[:3,3])],[saved['scene_rotation_error_deg'],saved['scene_translation_error_m']],atol=1e-6,rtol=1e-6)
    (extra/'regression.json').write_text(json.dumps({'status':'pass','all_transfer_rows_reproduced':True,'old_pose_errors_reproduced':True})+'\n')
    extra_summary=json.loads((extra/'summary.json').read_text())
    records=[]
    for row in legacy.itertuples(index=False):
        if row.source_run in replay.index:
            new=replay.loc[row.source_run]
            regression=json.loads((corrected/row.source_run/'seed_17/regression.json').read_text())
            if regression['status']!='pass':raise ValueError('failed reproduction')
            success=bool(new.strict_pose_success);rotation=new.rotation_error_deg;translation=new.translation_error_m;basis='exact cached consensus reproduction'
        elif row.source_run==extra_name:
            success=extra_summary['strict_scene_pose_success'];rotation=extra_summary['scene_rotation_error_deg'];translation=extra_summary['scene_translation_error_m'];basis='exact inference and consensus reproduction'
        else:
            if row.mast3r_pose_rotation_deg<5:raise ValueError('unreplayed relation may pass fixed gate')
            success=False;rotation=row.mast3r_pose_rotation_deg;translation=None;basis='failure proven by unchanged rotation; legacy translation omitted'
        records.append(dict(source_run=row.source_run,strict_pose_success=success,rotation_error_deg=rotation,translation_error_m=translation,score_basis=basis))
    out=REPORTS/'20261008_source_dense_mast3r_full17_corrected_units';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'per_relation.csv',index=False)
    summary={'relations':17,'strict_pose_successes':int(frame.strict_pose_success.sum()),'scene_units':'raw released metadata in metres','scope':'complete source strict gate audit; transfer statistics unchanged','all_required_replays_reproduced':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)

if __name__=='__main__':main()
