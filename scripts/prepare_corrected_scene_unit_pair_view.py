#!/usr/bin/env python3
"""Link immutable native RGB-D frames into a view with corrected scoring metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(pair: Path, output: Path, metadata: list) -> None:
    config = json.loads((pair/'config.json').read_text())
    if config.get('coordinate_convention') != 'native_scan_frames':
        raise ValueError('native scan frames required')
    manifest = json.loads((pair/'manifest.json').read_text())
    scan = next(s for r in metadata if r['reference']==config['reference'] for s in r['scans'] if s['reference']==config['rescan'])
    corrected = np.asarray(scan['transform'],dtype=np.float32).reshape(4,4).T
    config['rescan_to_reference'] = corrected.tolist()
    config['scene_transform_translation_units'] = 'metres; raw released metadata'
    config['unit_correction_source_pair'] = str(pair.resolve())
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for arm in ('reference','rescan'):
        for entry in manifest[arm]:
            relative=Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('relative frame paths required')
            source=(pair/relative).resolve()
            target=output/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_symlink():
                if target.resolve()!=source:
                    raise ValueError('existing frame link differs')
            elif target.exists():
                raise ValueError('view contains an unexpected frame file')
            else:
                target.symlink_to(source)
            records.append({'path':str(relative),'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'view_provenance.json').write_text(json.dumps({'source_pair':str(pair),'frames':records,
        'change':'GT scene translation units only; immutable native RGB-D arrays linked unchanged'},indent=2)+'\n')


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--metadata',type=Path,default=(Path.home() / 'data/worldcache_datasets/3rscan/3RScan.json'))
    a=p.parse_args();prepare(a.pair_run,a.output_dir,json.loads(a.metadata.read_text()))


if __name__=='__main__':
    main()
