#!/usr/bin/env python3
"""Score FCGF seed poses; replay only relations not disproven by rotation."""
import sys
from pathlib import Path
import json,subprocess,tempfile,os
import numpy as np
import pandas as pd
from run_geometry_controls_corrected_units import compare

ROOT=Path.cwd();REPORTS=ROOT/'outputs/reports';OUT=REPORTS/'20261008_fcgf_validation_seed_corrected_units'
DATA=(Path.home() / 'data/worldcache_datasets/3rscan')

def main():
    identities=pd.read_csv(REPORTS/'20261008_vitg_validation_corrected_pose_units/per_relation.csv').set_index('run_id')
    frames=[]
    for path in (REPORTS/'20261004_fcgf_validation_seed_sensitivity/relations').glob('*/fcgf/summary.csv'):
        frames.append(pd.read_csv(path).assign(run_id=path.parents[1].name))
    saved=pd.concat(frames,ignore_index=True)
    if len(saved)!=330 or set(saved.run_id)!=set(identities.index):raise ValueError('complete seed population required')
    OUT.mkdir(exist_ok=True)
    needed=sorted(saved[saved.ransac_seed.ne(17)&saved.rotation_error_deg.lt(5)].run_id.unique())
    for run in needed:
        identity=identities.loc[run];directory=OUT/run;directory.mkdir(exist_ok=True)
        if not (directory/'regression.json').exists():
            with tempfile.TemporaryDirectory(prefix='wc_fcgf_seed_units_') as scratch:
                pair=Path(scratch)/'pair'
                commands=[[sys.executable,str(ROOT/'scripts/generate_3rscan_pairs.py'),
                    '--root',str(DATA),'--metadata',str(DATA/'3RScan.json'),'--reference',identity.reference_scan,'--rescan',identity.rescan_scan,
                    '--output-dir',str(pair),'--resolution','224','--reference-frames','20','--rescan-frames','20'],
                    [str(Path.home() / 'data/conda/envs/worldcache_fcgf/bin/python'),str(ROOT/'scripts/evaluate_3rscan_fcgf_baseline.py'),
                    '--pair-run',str(pair),'--annotation-root',str(DATA),'--output-dir',str(directory),'--checkpoint',str(Path.home() / 'data/worldcache_models/fcgf/fcgf-3dmatch.pth'),
                    '--voxel-size','.025','--top-k','20','--threshold','.10','--iterations','3000','--seed-sweep','17','29','41','--save-query-diagnostics']]
                with (directory/'runner.log').open('w') as log:
                    for command in commands:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600,env={**os.environ,'PYTHONPATH':str(ROOT/'src')})
            for seed in [17,29,41]:compare(REPORTS/'20261004_fcgf_validation_seed_sensitivity/relations'/run/'fcgf'/f'query_diagnostics_k20_seed{seed}.csv',directory/f'query_diagnostics_k20_seed{seed}.csv')
            new=pd.read_csv(directory/'summary.csv').set_index('ransac_seed')
            original=saved[saved.run_id.eq(run)].set_index('ransac_seed')
            np.testing.assert_allclose(new.rotation_error_deg,original.reindex(new.index).rotation_error_deg,atol=1e-6,rtol=1e-6)
            (directory/'regression.json').write_text(json.dumps({'status':'pass','all_transfer_rows_reproduced':True,'rotation_errors_reproduced':True})+'\n')
        print(run,'all seeds reproduced',flush=True)
    rows=[]
    for record in saved.itertuples(index=False):
        run=record.run_id;seed=int(record.ransac_seed)
        if seed==17:
            current=pd.read_csv(REPORTS/'20261008_geometry_controls_corrected_units/relations'/run/'fcgf/summary.csv').iloc[0]
            success=bool(current.pose_success_lt5deg_lt025m);basis='complete seed17 replay'
        elif record.rotation_error_deg>=5:success=False;basis='rotation error alone exceeds fixed gate; translation not needed'
        else:
            current=pd.read_csv(OUT/run/'summary.csv').set_index('ransac_seed').loc[seed]
            success=bool(current.pose_success_lt5deg_lt025m);basis='targeted replay; exact saved transfer diagnostics reproduced'
        rows.append(dict(run_id=run,environment_id=identities.loc[run].environment_id,ransac_seed=seed,strict_pose_success=success,score_basis=basis))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'per_relation_seed.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps({'relations':110,'environments':47,'corrected_pose_counts':frame.groupby('ransac_seed').strict_pose_success.sum().to_dict(),'targeted_replay_relations':needed,'scope':'strict pose success only; unneeded legacy translations not carried forward'},indent=2)+'\n')

if __name__=='__main__':main()
