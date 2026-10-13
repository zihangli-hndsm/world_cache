"""Run native BUFFER-X on the fixed 16 source relations for threshold calibration."""
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
import numpy as np


def main():
    root=Path(__file__).resolve().parent.parent
    population=pd.read_csv(root/'outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv')
    population=population[population.run_id.ne('target')].copy()
    assert len(population)==16 and population.run_id.nunique()==16
    out=root/'outputs/reports/20261008_bufferx_source16';out.mkdir(exist_ok=True)
    (out/'scheduled_population.json').write_text(population[['run_id','environment_id','reference_scan','rescan_scan']].to_json(orient='records',indent=2)+'\n')
    rows=[]
    for r in population.itertuples(index=False):
        directory=out/'relations'/r.run_id;directory.mkdir(parents=True,exist_ok=True)
        view=root/'outputs/reports/20261008_corrected_source_pair_views'/r.run_id
        row=dict(run_id=r.run_id,environment_id=r.environment_id,reference_scan=r.reference_scan,rescan_scan=r.rescan_scan)
        try:
            config=json.loads((view/'config.json').read_text())
            assert config['reference']==r.reference_scan and config['rescan']==r.rescan_scan
            frame_manifest=json.loads((view/'manifest.json').read_text())
            cache=root/'outputs/reports/20261006_source_precision_motion/fixed_top20_run/descriptor_candidates'/f'{r.run_id}.npz'
            with np.load(cache) as z:
                assert str(z['reference_id'].item())==r.reference_scan and str(z['rescan_id'].item())==r.rescan_scan
                assert set(map(int,z['query_frame_ids'])).issubset({int(f['frame_id']) for f in frame_manifest['rescan']})
            if not (directory/'result.json').exists():
                commands=[[sys.executable,str(root/'scripts/prepare_bufferx_rgbd_geometry.py'),'--pair-run',str(view),'--output-dir',str(directory/'geometry')],
                    ['bash',str(root/'scripts/run_bufferx_runtime.sh'),str(root/'scripts/evaluate_bufferx_native_pair.py'),'--geometry-dir',str(directory/'geometry'),'--output-dir',str(directory)]]
                with (directory/'runner.log').open('w') as log:
                    for command in commands:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600,cwd=root)
            result=json.loads((directory/'result.json').read_text())
            assert result['reference']==r.reference_scan and result['rescan']==r.rescan_scan
            row.update(status='complete',error='')
        except Exception as error:row.update(status='failed',error=str(error))
        rows.append(row);pd.DataFrame(rows).to_csv(out/'run_manifest.csv',index=False)
        print(f'{len(rows)}/16 {r.run_id} {row["status"]} {row["error"]}',flush=True)

if __name__=='__main__':main()
