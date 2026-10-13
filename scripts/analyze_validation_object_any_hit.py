"""Reaggregate full-validation linked-instance coverage at object/environment level."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent.parent
    run_dir = root / 'outputs/reports/20261008_bufferx_validation'
    vitg_dir = root / 'outputs/reports/20261006_vitg_pose_locked_motion_validation'
    out = root / 'outputs/reports/20261009_validation_object_any_hit'
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(run_dir / 'run_manifest.csv')
    if len(manifest) != 110 or manifest.run_id.duplicated().any():
        raise ValueError('requires all 110 unique validation relations')
    calibration = json.loads((root / 'outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json').read_text())
    threshold = float(calibration['threshold_m'])
    fraction = float(calibration['dino_raw_budget'] / calibration['raw_queries'])
    chunks = []

    for relation in manifest.itertuples(index=False):
        dino_path = vitg_dir / 'relations' / relation.run_id / 'query_diagnostics.csv'
        dino_all = pd.read_csv(dino_path)
        dino = dino_all[dino_all.arm.eq('global_ransac')].sort_values('query_index').reset_index(drop=True)
        if dino.query_index.duplicated().any():
            raise ValueError(f'duplicate DINO query index in {relation.run_id}')
        query = dino[['query_index', 'moved_rigid', 'official_rigid_reference_instance_id']].copy()
        query['environment_id'] = relation.environment_id
        query['run_id'] = relation.run_id
        query['vitg_global_ransac'] = (
            dino.moved_rigid.astype(bool)
            & dino.consensus_inlier.astype(bool)
            & dino.official_rigid_instance_correct.astype(bool)
        )

        transfer_path = run_dir / 'pose_conditioned_transfer' / f'{relation.run_id}.csv.gz'
        transfer = pd.read_csv(transfer_path)
        nearest = transfer[transfer.arm.eq('bufferx_pose_nearest_patch')].sort_values('query_index').reset_index(drop=True)
        restricted = transfer[transfer.arm.eq('bufferx_pose_vitg_k20')].sort_values('query_index').reset_index(drop=True)
        if len(nearest) != len(query) or not np.array_equal(nearest.query_index.to_numpy(), query.query_index.to_numpy()):
            raise ValueError(f'query support mismatch in {relation.run_id}')
        restricted = restricted.sort_values('query_index').reset_index(drop=True)
        if len(restricted) != len(query) or not np.array_equal(restricted.query_index.to_numpy(), query.query_index.to_numpy()):
            raise ValueError(f'top-20 query support mismatch in {relation.run_id}')
        if not np.array_equal(nearest.moved_rigid.to_numpy(bool), dino.moved_rigid.to_numpy(bool)):
            raise ValueError(f'moved-object denominator mismatch in {relation.run_id}')
        query['bufferx_pose_nearest_patch'] = nearest.moved_rigid_correct.to_numpy(bool)
        query['bufferx_pose_vitg_k20'] = restricted.moved_rigid_correct.to_numpy(bool)

        # Reconstruct the unchanged source-frozen distance/rank selectors from
        # the saved label-free nearest-patch residual and the official labels.
        resid = nearest.residual_m.to_numpy(float)
        close = resid <= .1
        query['source_frozen_geometry'] = nearest.moved_rigid_correct.to_numpy(bool) & (resid <= threshold)
        n_accept = int(np.floor(fraction * len(query)))
        eligible = np.flatnonzero(close)
        order = eligible[np.lexsort((eligible, resid[eligible]))]
        rank_accept = np.zeros(len(query), dtype=bool)
        rank_accept[order[:n_accept]] = True
        query['source_frozen_rank'] = nearest.moved_rigid_correct.to_numpy(bool) & rank_accept
        query = query[query.moved_rigid.astype(bool) & (query.official_rigid_reference_instance_id > 0)]
        if len(query):
            chunks.append(query)
        print(f'{relation.run_id} {len(query)} mapped query patches', flush=True)

    queries = pd.concat(chunks, ignore_index=True)
    arms = ['vitg_global_ransac', 'bufferx_pose_nearest_patch', 'bufferx_pose_vitg_k20', 'source_frozen_geometry', 'source_frozen_rank']
    long = queries.melt(
        id_vars=['environment_id', 'run_id', 'query_index', 'official_rigid_reference_instance_id'],
        value_vars=arms, var_name='arm', value_name='correct_query',
    )
    # The persistent reference object ID merges rescans of one environment;
    # any-hit success is the target-level transfer event.
    object_relation = long.groupby(
        ['environment_id', 'run_id', 'official_rigid_reference_instance_id', 'arm'], as_index=False
    ).correct_query.max()
    relation_scores = object_relation.groupby(
        ['environment_id', 'run_id', 'arm'], as_index=False
    ).correct_query.agg(['sum', 'count', 'mean'])
    relation_scores = relation_scores.rename(columns={
        'sum': 'object_hits', 'count': 'eligible_objects', 'mean': 'relation_object_any_hit',
    })
    relation_scores.to_csv(out / 'per_relation.csv', index=False)
    relation_macro = relation_scores.groupby(['environment_id', 'arm'], as_index=False).relation_object_any_hit.mean()
    relation_macro.to_csv(out / 'per_environment_relation_macro.csv', index=False)

    object_env = object_relation.groupby(
        ['environment_id', 'official_rigid_reference_instance_id', 'arm'], as_index=False
    ).correct_query.max()
    object_env.to_csv(out / 'object_environment_outcomes.csv', index=False)
    env = object_env.groupby(['environment_id', 'arm']).correct_query.agg(['sum', 'count', 'mean']).reset_index()
    env = env.rename(columns={'sum': 'objects_any_hit', 'count': 'mapped_objects', 'mean': 'object_any_hit_coverage'})
    env.to_csv(out / 'per_environment.csv', index=False)
    env_macro = env.groupby('arm').object_any_hit_coverage.mean().to_dict()

    wide = env.pivot(index='environment_id', columns='arm', values='object_any_hit_coverage')
    pairs = [
        ('vitg_global_ransac', 'bufferx_pose_nearest_patch'),
        ('vitg_global_ransac', 'bufferx_pose_vitg_k20'),
        ('vitg_global_ransac', 'source_frozen_geometry'),
        ('vitg_global_ransac', 'source_frozen_rank'),
        ('bufferx_pose_vitg_k20', 'bufferx_pose_nearest_patch'),
        ('bufferx_pose_vitg_k20', 'source_frozen_geometry'),
        ('bufferx_pose_vitg_k20', 'source_frozen_rank'),
    ]
    rng = np.random.default_rng(10092029)
    effects = []
    for reference, control in pairs:
        pair = wide[[reference, control]].dropna()
        delta = (pair[reference] - pair[control]).to_numpy()
        draws = delta[rng.integers(0, len(delta), (50000, len(delta)))].mean(axis=1)
        effects.append(dict(
            reference_arm=reference, control_arm=control, environments=len(delta),
            mean_difference=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
            bonferroni_ci95=np.quantile(draws, [.05/len(pairs), 1-.05/len(pairs)]).tolist(),
            positive_environments=int((delta > 1e-12).sum()),
            tied_environments=int((np.abs(delta) <= 1e-12).sum()),
            negative_environments=int((delta < -1e-12).sum()),
        ))
    relation_wide = relation_macro.pivot(index='environment_id', columns='arm', values='relation_object_any_hit')
    relation_effects = []
    for reference, control in pairs:
        pair = relation_wide[[reference, control]].dropna()
        delta = (pair[reference] - pair[control]).to_numpy()
        draws = delta[rng.integers(0, len(delta), (50000, len(delta)))].mean(axis=1)
        relation_effects.append(dict(
            reference_arm=reference, control_arm=control, environments=len(delta),
            mean_difference=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
            bonferroni_ci95=np.quantile(draws, [.05/len(pairs), 1-.05/len(pairs)]).tolist(),
            positive_environments=int((delta > 1e-12).sum()),
            tied_environments=int((np.abs(delta) <= 1e-12).sum()),
            negative_environments=int((delta < -1e-12).sum()),
        ))
    pooled = object_env.groupby('arm').correct_query.agg(['sum', 'count', 'mean']).to_dict('index')
    summary = dict(
        relations=int(manifest.run_id.nunique()), environments=int(env.environment_id.nunique()),
        reference_object_environment_pairs=int(object_env.groupby(['environment_id', 'official_rigid_reference_instance_id']).ngroups),
        object_environment_outcomes=pooled, effects=effects,
        relation_object_observations={arm: int(row['count']) for arm, row in object_relation.groupby('arm').correct_query.agg(['count']).iterrows()},
        relation_macro_effects=relation_effects,
        calibration_sha256=hashlib.sha256((root / 'outputs/reports/20261008_bufferx_source16/frozen_source_threshold.json').read_bytes()).hexdigest(),
        threshold_m=threshold, source_rank_fraction=fraction,
        environment_macro_coverage=env_macro,
        scope=('Post-hoc object-level any-hit reaggregation on all official validation relations. A target object is covered if any valid query patch maps correctly to its official persistent reference ID across the observed rescans in that reference environment. This differs from query-weighted coverage and is not independent-site evidence.'),
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = [
        '# Object-level any-hit coverage on full 3RScan validation', '',
        'This post-hoc reaggregation uses all 110 relations across 47 reference environments and merges rescans within each environment using the official persistent reference object ID. An object counts as covered if any valid query patch across its observed rescans is assigned to the correct linked reference instance. Forty-two environments contain supported linked-object queries, yielding 129 object-environment pairs. This measures whether a pipeline ever recovers an object in that environment, not the fraction of query patches transferred correctly.', '',
        '| Policy | Object-environment hits / eligible objects | Object-weighted coverage | Environment-macro coverage |',
        '|---|---:|---:|---:|',
    ]
    for arm, label in [
        ('vitg_global_ransac', 'ViT-G RANSAC'),
        ('bufferx_pose_nearest_patch', 'BUFFER-X pose, nearest patch'),
        ('bufferx_pose_vitg_k20', 'BUFFER-X pose, ViT-G top-20'),
        ('source_frozen_geometry', 'Source-frozen distance'),
        ('source_frozen_rank', 'Source-frozen rank'),
    ]:
        x = pooled[arm]
        lines.append(f"| {label} | {int(x['sum'])}/{int(x['count'])} | {100*x['mean']:.2f}% | {100*env_macro[arm]:.2f}% |")
    lines += ['', '## Environment-merged any-hit contrasts', '', 'Paired environment-macro differences for the environment-merged any-hit endpoint (95% intervals are pointwise; seven-contrast Bonferroni intervals are in `summary.json`):', '', '| Reference | Control | Environments | Difference | 95% environment-bootstrap CI |', '|---|---|---:|---:|---:|']
    labels = {'vitg_global_ransac':'ViT-G RANSAC','bufferx_pose_nearest_patch':'BUFFER-X nearest patch','bufferx_pose_vitg_k20':'BUFFER-X, ViT-G top-20','source_frozen_geometry':'Source-frozen distance','source_frozen_rank':'Source-frozen rank'}
    for row in effects:
        lo, hi = row['ci95']
        lines.append(f"| {labels[row['reference_arm']]} | {labels[row['control_arm']]} | {row['environments']} | {100*row['mean_difference']:+.2f} pp | [{100*lo:+.2f},{100*hi:+.2f}] pp |")
    lines += ['', '## Per-rescan object coverage sensitivity', '', 'The main any-hit measure merges all observed rescans within a reference environment. To check sensitivity to the number of rescan opportunities, we also score each reference/rescan relation separately, compute its fraction of linked objects hit, and macro-average relations within each environment before the cluster bootstrap. This includes repeated sightings of one reference object as separate relation-object attempts.', '', 'Paired environment-macro differences (95% intervals are pointwise; seven-contrast Bonferroni intervals are in `summary.json`):', '', '| Reference | Control | Environments | Difference | 95% environment-bootstrap CI |', '|---|---|---:|---:|---:|']
    for row in relation_effects:
        lo, hi = row['ci95']
        lines.append(f"| {labels[row['reference_arm']]} | {labels[row['control_arm']]} | {row['environments']} | {100*row['mean_difference']:+.2f} pp | [{100*lo:+.2f},{100*hi:+.2f}] pp |")
    lines += ['', 'The environment-merged estimates above answer whether an object is ever correctly matched across its observed rescans. This relation-level sensitivity avoids collapsing multiple rescans into one chance to hit. Both are exploratory post-hoc aggregations and neither replaces the declared query-level outcomes.', '', 'Per-relation object rows are in `per_relation.csv`; environment-macro relation scores are in `per_environment_relation_macro.csv`. Reproduce with `scripts/analyze_validation_object_any_hit.py`.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
