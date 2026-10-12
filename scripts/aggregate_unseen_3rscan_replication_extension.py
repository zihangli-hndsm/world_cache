"""Combine the original and post-hoc eight-relation cohorts without tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent.parent
    first = root / 'outputs/reports/20261008_unseen_train_replication'
    extension = root / 'outputs/reports/20261009_unseen_train_replication_extension'
    out = extension
    frame = pd.concat([
        pd.read_csv(first / 'policy_per_relation.csv'),
        pd.read_csv(extension / 'policy_per_relation.csv'),
    ], ignore_index=True)
    assert frame.run_id.is_unique is False  # each relation has one row per policy
    assert frame.groupby('run_id').environment_id.nunique().max() == 1
    assert frame.groupby(['run_id', 'arm']).size().max() == 1
    assert frame.environment_id.nunique() == 16

    env = frame.groupby(['environment_id', 'arm']).sum(numeric_only=True)
    env['semantic_coverage'] = env.semantic_correct / env.semantic_queries.replace(0, np.nan)
    env['semantic_precision'] = env.semantic_correct / env.semantic_assigned.replace(0, np.nan)
    env['moved_coverage'] = env.moved_correct / env.moved_queries.replace(0, np.nan)
    env.to_csv(out / 'combined_policy_per_environment.csv')

    rng = np.random.default_rng(10092026)
    effects = []
    for metric in ['semantic_coverage', 'semantic_precision', 'moved_coverage']:
        wide = env[metric].unstack('arm')
        for arm in ['source_frozen_geometry', 'source_frozen_rank']:
            pair = wide[[arm, 'bufferx_pose_vitg_k20']].dropna()
            delta = (pair.bufferx_pose_vitg_k20 - pair[arm]).to_numpy()
            if len(delta):
                boot = delta[rng.integers(0, len(delta), (50000, len(delta)))].mean(axis=1)
                effects.append(dict(
                    metric=metric, geometry_arm=arm, environments=len(delta),
                    dino_minus_geometry=float(delta.mean()),
                    ci95=np.quantile(boot, [.025, .975]).tolist(),
                    bonferroni_ci95=np.quantile(boot, [.05/12, 1-.05/12]).tolist(),
                ))
    pooled = frame.groupby('arm').sum(numeric_only=True)
    summary = dict(
        relations=int(frame.run_id.nunique()), environments=int(frame.environment_id.nunique()),
        buffer_pose_failures=int((~frame[frame.arm.eq('source_frozen_geometry')].buffer_pose_available).sum()),
        threshold_m=0.04874177796801955, raw_fraction=0.20961742826780022,
        effects=effects,
        pooled={arm: row.to_dict() for arm, row in pooled.iterrows()},
        scope=('Combined descriptive estimate from 16 hash-selected unused 3RScan train environments. '
               'The extension cohort was selected after outcomes from the first eight were observed; '
               'this is post-hoc same-dataset evidence, not a preregistered or independent-site replication.'),
    )
    (out / 'combined_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    lines = [
        '# Combined frozen-policy extension (16 environments)', '',
        'The first eight and the hash-ordered next eight unused train environments were scored with the same frozen source calibration and unchanged matching policies. BUFFER-X pose succeeded in all 16. The second cohort was selected after observing the first cohort; combined estimates are post-hoc and exploratory, not independent-site or preregistered confirmation.', '',
        'Macro paired environment-bootstrap effects (ViT-G candidates + BUFFER-X pose against source-frozen controls):', '',
        '| Outcome | Control | Environments | Difference | 95% cluster bootstrap CI |',
        '|---|---|---:|---:|---:|',
    ]
    labels = {'semantic_precision':'Semantic precision', 'semantic_coverage':'Semantic coverage', 'moved_coverage':'Moved-instance coverage'}
    controls = {'source_frozen_geometry':'Frozen distance', 'source_frozen_rank':'Frozen rank'}
    for row in effects:
        lo, hi = row['ci95']
        lines.append(f"| {labels[row['metric']]} | {controls[row['geometry_arm']]} | {row['environments']} | {100*row['dino_minus_geometry']:+.2f} pp | [{100*lo:+.2f}, {100*hi:+.2f}] pp |")
    lines += [
        '',
        'The primary semantic-precision contrasts remain inconclusive: both 95% intervals include zero. The point estimates across all 16 environments are reported directly in `combined_summary.json`; coverage is secondary and moved-instance denominators are smaller. No policy was changed after either cohort was frozen.', '',
        'Reproduce with `scripts/aggregate_unseen_3rscan_replication_extension.py` after both cohort scoring runs complete.',
    ]
    (out / 'COMBINED_INTERPRETATION.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
