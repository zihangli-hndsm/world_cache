"""Freeze eight unused train environments without looking at matching outcomes."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def main():
    root=Path.home()/'data/worldcache_datasets/3rscan'
    out=Path('outputs/reports/20261008_unseen_train_replication');out.mkdir(exist_ok=True)
    destination=out/'frozen_population.json'
    if destination.exists():
        print('Existing frozen population retained');return
    metadata=json.loads((root/'3RScan.json').read_text())
    source=pd.read_csv('outputs/reports/20261008_vitg_source_corrected_pose_units/per_relation.csv')
    validation=pd.read_csv('outputs/reports/20261008_bufferx_validation/run_manifest.csv')
    excluded=set(source.reference_scan)|set(validation.reference_scan)
    local={p.parent.name for p in root.glob('*/sequence.zip')}
    seed='10082026_unseen_native_replication_v1'
    order=lambda identifier:hashlib.sha256((seed+identifier).encode()).hexdigest()
    eligible=[g for g in metadata if g['type']=='train' and g['reference'] not in excluded
              and g['reference'] not in local and g['scans']]
    eligible.sort(key=lambda g:order(g['reference']))
    rows=[]
    for i,g in enumerate(eligible[:8]):
        candidates=sorted([s['reference'] for s in g['scans'] if s['reference'] not in local],key=order)
        if not candidates:raise ValueError('selected group has no unused scan; do not replace it silently')
        rows.append(dict(run_id=f'{i:03d}_{g["reference"][:8]}_{candidates[0][:8]}',environment_id=g['reference'],reference_scan=g['reference'],rescan_scan=candidates[0]))
    assert len(rows)==8
    calibration=Path('outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json')
    payload=dict(status='frozen',population=rows,seed=seed,eligible_environments=len(eligible),
        selection='Hash order among train groups with no previously downloaded RGB-D reference; exclude original source and validation references. One hash-selected rescan per group.',
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        scope='New matching outcomes and disjoint released scan IDs, same dataset acquisition. Prior all-train mesh screens exist; physical-site independence not established.',
        policy='Apply unchanged source-fixed distance and rank-fraction policies, DINO K20 and ViT-G RANSAC. Retain all eight attempts and failures. Do not choose policies or replace environments by outcome.')
    destination.write_text(json.dumps(payload,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(out/'relations.csv',index=False)
    pd.DataFrame({'scan_id':sorted({r[k] for r in rows for k in ['reference_scan','rescan_scan']})}).to_csv(out/'scan_manifest.csv',index=False)
    print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
