"""Run the frozen eight-relation confirmation population; retain every failure."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
from evaluate_3rscan_multimotion import extract_pair_features,fit_fixed_consensus,save_feature_cache,load_feature_cache,score_outputs
from run_3rscan_validation_pose_locked_motion import pack_base
from worldcache.backbone.dinov2 import load_dinov2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-dir',type=Path,default=Path('outputs/reports/20261008_unseen_train_replication'))
    args=parser.parse_args()
    project=Path(__file__).resolve().parent.parent
    out=args.experiment_dir if args.experiment_dir.is_absolute() else project/args.experiment_dir
    frozen=json.loads((out/'frozen_population.json').read_text())
    calibration=project/'outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json'
    assert hashlib.sha256(calibration.read_bytes()).hexdigest()==frozen['calibration_sha256']
    population=frozen['population']
    data=Path.home()/'data/worldcache_datasets/3rscan'
    metadata=json.loads((data/'3RScan.json').read_text())
    device=torch.device('cuda');backbone=load_dinov2('giant',224,device)
    records=[]
    for r in frozen['population']:
        directory=out/'relations'/r['run_id'];directory.mkdir(parents=True,exist_ok=True)
        cache=out/'descriptor_candidates'/f"{r['run_id']}.npz"
        record={**r}
        try:
            if not (directory/'complete.json').exists():
                with tempfile.TemporaryDirectory(prefix='wc_unseen_replication_') as temporary:
                    pair=Path(temporary)/'pair'
                    command=[sys.executable,str(project/'scripts/generate_3rscan_pairs.py'),'--root',str(data),'--metadata',str(data/'3RScan.json'),
                        '--reference',r['reference_scan'],'--rescan',r['rescan_scan'],'--output-dir',str(pair),'--resolution','224',
                        '--reference-frames','20','--rescan-frames','20']
                    with (directory/'preparation.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                    features=load_feature_cache(cache,r['run_id'],r['reference_scan'],r['rescan_scan'],pair,'giant',(2,20,40),224)
                    if features is None:
                        features=extract_pair_features(r['run_id'],r['reference_scan'],r['rescan_scan'],pair,backbone,(2,20,40),device,descriptor_chunk=100000,preserve_top_k=20)
                        save_feature_cache(cache,features,'giant',(2,20,40),224)
                    base=fit_fixed_consensus(features.query_points,features.reference_points,features.candidate_indices[:,:20],threshold=.1,iterations=1500,seed=17)
                    np.save(directory/'vitg_predicted_pose.npy',base.matrix)
                    # Native geometry preparation and BUFFER-X prediction precede label scoring.
                    buffer=directory/'bufferx';buffer.mkdir(exist_ok=True)
                    if not (buffer/'result.json').exists():
                        commands=[[sys.executable,str(project/'scripts/prepare_bufferx_rgbd_geometry.py'),'--pair-run',str(pair),'--output-dir',str(buffer/'geometry')],
                            ['bash',str(project/'scripts/run_bufferx_runtime.sh'),str(project/'scripts/evaluate_bufferx_native_pair.py'),'--geometry-dir',str(buffer/'geometry'),'--output-dir',str(buffer)]]
                        with (buffer/'runner.log').open('w') as log:
                            for cmd in commands:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600,cwd=project)
                    config=json.loads((pair/'config.json').read_text())
                    rows,diagnostics=score_outputs(features,{'global_ransac':pack_base(base,len(features.query_points))},data,metadata,config)
                    pd.DataFrame(rows).to_csv(directory/'summary.csv',index=False)
                    diag=pd.concat(diagnostics,ignore_index=True)
                    diag['moved_rigid']=diag.query_evaluable.astype(bool)&(diag.official_rigid_reference_instance_id>0)
                    diag.to_csv(directory/'query_diagnostics.csv',index=False)
                    (directory/'complete.json').write_text(json.dumps(dict(status='complete',run_id=r['run_id'],calibration_sha256=frozen['calibration_sha256'],native_rgbd=True,threshold_changed=False))+'\n')
            record.update(status='complete',error='')
        except Exception as error:record.update(status='failed',error=str(error))
        records.append(record);pd.DataFrame(records).to_csv(out/'run_manifest.csv',index=False)
        print(f"{len(records)}/{len(population)} {r['run_id']} {record['status']} {record['error']}",flush=True)

if __name__=='__main__':main()
