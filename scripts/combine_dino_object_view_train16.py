#!/usr/bin/env python3
"""Assemble the first and second eight-relation view curves for reanalysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COHORTS = (
    (
        ROOT / 'outputs/reports/20261009_unseen_oasis_equal_budget',
        ROOT / 'outputs/reports/20261009_dino_multiview_source_bank_train8',
        ROOT / 'outputs/reports/20261109_remind_style_3rscan_mapping_final',
    ),
    (
        ROOT / 'outputs/reports/20261009_dino_object_view_budget_curve_extension',
        ROOT / 'outputs/reports/20261009_dino_multiview_source_bank_extension',
        ROOT / 'outputs/reports/20261009_remind_style_view_extension',
    ),
)


def link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise FileExistsError(f'link target differs: {destination}')
        return
    if destination.exists():
        raise FileExistsError(f'refusing to replace existing artifact: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def write_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2) + '\n'
    if path.exists() and path.read_text() != serialized:
        raise FileExistsError(f'refusing to change frozen artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reports/20261009_dino_object_view_budget_curve_train16')
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    input_root, feature_root, remind_root = output / 'input', output / 'features', output / 'remind'
    run_results, repeat_rows = [], []

    for input_source, feature_source, remind_source in COHORTS:
        manifest = json.loads((feature_source / 'run_manifest.json').read_text())
        curve = json.loads((feature_source / 'view_budget_curve.json').read_text())
        if manifest.get('status') != 'complete':
            raise ValueError(f'incomplete source bank: {feature_source}')
        run_ids = [item['run_id'] for item in manifest['results']]
        for run_id in run_ids:
            for category in ('uniform20_inputs', 'dino_segment_predictions'):
                link(input_source / category / run_id, input_root / category / run_id)
            link(feature_source / f'{run_id}_features.npz', feature_root / f'{run_id}_features.npz')
            link(remind_source / f'{run_id}.json', remind_root / f'{run_id}.json')
            run_results.append({'run_id': run_id})
        repeat_rows.extend(curve['relation_repeat_rows'])

    write_json(feature_root / 'run_manifest.json', {'status': 'complete', 'results': run_results})
    write_json(feature_root / 'view_budget_curve.json', {'relation_repeat_rows': repeat_rows})
    print(f'assembled {len(run_results)} relations from two cohorts into {output}')


if __name__ == '__main__':
    main()
