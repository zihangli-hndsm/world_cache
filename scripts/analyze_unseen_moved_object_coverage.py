"""Exploratory object-level audit of frozen policies on the 16 unseen relations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate_3rscan_multimotion import (
    annotation_labels, load_feature_cache, relation_record, rigid_object_mapping,
)


ARMS = ['bufferx_pose_vitg_k20', 'source_frozen_geometry', 'source_frozen_rank']


def main():
    root = Path(__file__).resolve().parent.parent
    first = root / 'outputs/reports/20261008_unseen_train_replication'
    second = root / 'outputs/reports/20261009_unseen_train_replication_extension'
    data = Path.home() / 'data/worldcache_datasets/3rscan'
    metadata = json.loads((data / '3RScan.json').read_text())
    rows = []

    for out in [first, second]:
        frozen = json.loads((out / 'frozen_population.json').read_text())
        for relation in frozen['population']:
            relation_dir = out / 'relations' / relation['run_id']
            features = load_feature_cache(
                out / 'descriptor_candidates' / f"{relation['run_id']}.npz",
                relation['run_id'], relation['reference_scan'], relation['rescan_scan'],
                Path('.'), 'giant', (2, 20, 40), 224,
            )
            if features is None:
                raise FileNotFoundError(f"missing cached features for {relation['run_id']}")
            labels = annotation_labels(data, features)
            scan = relation_record(metadata, relation['reference_scan'], relation['rescan_scan'])['scan']
            mapping = rigid_object_mapping(scan)
            obj = labels['query_object_id']
            sem = labels['query_semantic_id']
            ref_obj = labels['reference_object_id']
            evaluable = (obj != 0) & (sem != 0)
            assignments = np.load(relation_dir / 'frozen_policy_assignments.npz')

            for arm in ARMS:
                selected = assignments[f'{arm}_selected']
                assigned = assignments[f'{arm}_assigned'].astype(bool)
                object_rows = []
                for query_id, expected_ref_id in mapping.items():
                    idx = np.flatnonzero(evaluable & (obj == query_id))
                    if not len(idx):
                        continue
                    correct = assigned[idx] & (ref_obj[selected[idx]] == expected_ref_id)
                    accepted = assigned[idx]
                    object_rows.append((int(correct.any()), int(correct.sum()), int(accepted.sum()), len(idx)))
                if not object_rows:
                    continue
                objects = np.asarray(object_rows, dtype=np.int64)
                rows.append(dict(
                    run_id=relation['run_id'], environment_id=relation['environment_id'], arm=arm,
                    observed_mapped_objects=len(objects), objects_with_any_correct=int(objects[:, 0].sum()),
                    query_points=int(objects[:, 3].sum()), correct_queries=int(objects[:, 1].sum()),
                    accepted_queries=int(objects[:, 2].sum()),
                ))

    frame = pd.DataFrame(rows)
    frame.to_csv(second / 'object_level_per_relation.csv', index=False)
    env = frame.groupby(['environment_id', 'arm']).sum(numeric_only=True)
    env['object_any_hit_coverage'] = env.objects_with_any_correct / env.observed_mapped_objects
    env['query_correct_coverage'] = env.correct_queries / env.query_points
    env['query_precision'] = env.correct_queries / env.accepted_queries.replace(0, np.nan)
    env.to_csv(second / 'object_level_per_environment.csv')

    rng = np.random.default_rng(10092027)
    effects = []
    for metric in ['object_any_hit_coverage', 'query_correct_coverage', 'query_precision']:
        wide = env[metric].unstack('arm')
        for control in ['source_frozen_geometry', 'source_frozen_rank']:
            pair = wide[[control, 'bufferx_pose_vitg_k20']].dropna()
            delta = (pair.bufferx_pose_vitg_k20 - pair[control]).to_numpy()
            if not len(delta):
                continue
            boot = delta[rng.integers(0, len(delta), (50000, len(delta)))].mean(axis=1)
            loo = [float(np.delete(delta, i).mean()) for i in range(len(delta))] if len(delta) > 1 else []
            effects.append(dict(
                metric=metric, control=control, environments=len(delta),
                mean_difference=float(delta.mean()),
                ci95=np.quantile(boot, [.025, .975]).tolist(),
                bonferroni_ci95=np.quantile(boot, [.05 / 12, 1 - .05 / 12]).tolist(),
                positive_environments=int((delta > 1e-12).sum()),
                tied_environments=int((np.abs(delta) <= 1e-12).sum()),
                negative_environments=int((delta < -1e-12).sum()),
                leave_one_environment_out_range=[min(loo), max(loo)] if loo else [],
            ))
    summary = dict(
        relations=int(frame.run_id.nunique()),
        environments=int(frame.environment_id.nunique()),
        total_observed_mapped_objects=int(frame[frame.arm.eq('bufferx_pose_vitg_k20')].observed_mapped_objects.sum()),
        pooled={arm: row.to_dict() for arm, row in frame.groupby('arm').sum(numeric_only=True).iterrows()},
        scope=('Exploratory post-hoc object-level reaggregation of the same 16 frozen relations. '
               'Objects are included when at least one valid query patch is observed; this does not add independent data.'),
        effects=effects,
    )
    (second / 'object_level_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    precision_effects = {row['control']: row for row in effects if row['metric'] == 'object_any_hit_coverage'}
    lines = [
        '# Object-level check of the moved-instance coverage signal', '',
        'This post-hoc reaggregation uses the same 16 frozen relations and the same source-frozen assignments. It counts a linked rigid instance as covered if at least one valid query patch is assigned correctly. Only 29 linked instances across 13 environments have observed query support, so this is low-power and not independent evidence.', '',
        '| Policy | Instances with ≥1 correct query | Correct query patches / all linked-object patches |',
        '|---|---:|---:|',
    ]
    for arm, label in [
        ('bufferx_pose_vitg_k20', 'ViT-G + BUFFER-X pose'),
        ('source_frozen_geometry', 'Frozen distance'),
        ('source_frozen_rank', 'Frozen rank'),
    ]:
        row = summary['pooled'][arm]
        lines.append(f"| {label} | {int(row['objects_with_any_correct'])}/29 | {int(row['correct_queries'])}/1367 |")
    for control, label in [('source_frozen_geometry', 'distance'), ('source_frozen_rank', 'rank')]:
        row = precision_effects[control]
        lo, hi = row['ci95']
        pos_noun = 'environment' if row['positive_environments'] == 1 else 'environments'
        tie_noun = 'environment' if row['tied_environments'] == 1 else 'environments'
        neg_noun = 'environment' if row['negative_environments'] == 1 else 'environments'
        pos_verb = 'favors' if row['positive_environments'] == 1 else 'favor'
        neg_verb = 'favors' if row['negative_environments'] == 1 else 'favor'
        lines.append(f"\nEnvironment-macro object-any-hit coverage versus {label}: {100*row['mean_difference']:+.2f} pp, 95% cluster-bootstrap CI [{100*lo:+.2f},{100*hi:+.2f}] pp; {row['positive_environments']} {pos_noun} {pos_verb} ViT-G, {row['tied_environments']} {tie_noun} tie, {row['negative_environments']} {neg_noun} {neg_verb} the control.")
    lines += [
        '',
        'The query-patch moved-instance coverage gain therefore mostly reflects additional correct patches within instances already covered by geometry. Against distance, only one environment adds an instance-level hit; the object-level interval includes zero. This weakens the interpretation of the query-weighted secondary signal and leaves no robust moved-instance coverage result.', '',
        'Per-relation and per-environment rows, all paired intervals, and reproduction: `object_level_per_relation.csv`, `object_level_per_environment.csv`, `object_level_summary.json`, and `scripts/analyze_unseen_moved_object_coverage.py`.',
    ]
    (second / 'OBJECT_LEVEL_INTERPRETATION.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
