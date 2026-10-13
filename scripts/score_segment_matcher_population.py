"""Post-hoc segment scoring after all anonymous predictions are persisted.

Shared scan-local IDs use the existing semantic-consistent fallback; explicit
rigid links take precedence and metadata ambiguity groups are excluded.
Metrics remain distinct from the manuscript's fixed-patch denominators.
"""
import hashlib
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from evaluate_3rscan_instance_correspondence import read_annotated_mesh


def semantics(root, scan):
    _, ids, _, categories, labels = read_annotated_mesh(root, scan)
    result = {}
    for object_id in np.unique(ids):
        values, counts = np.unique(categories[ids == object_id], return_counts=True)
        result[int(object_id)] = int(values[counts.argmax()])
    return result, labels


def identity_links(metadata, reference, query, ref_names, query_names):
    relation = next(r for r in metadata if r['reference'] == reference)
    record = next(r for r in relation['scans'] if r['reference'] == query)
    ambiguous = {int(item[k]) for group in relation.get('ambiguity', []) for item in group
                 for k in ['instance_source', 'instance_target'] if k in item}
    links = {}
    for item in record.get('rigid', []):
        if not isinstance(item, dict):
            continue
        source, target = int(item['instance_rescan']), int(item['instance_reference'])
        if source in ambiguous or target in ambiguous:
            continue
        if source in links and links[source] != target:
            raise ValueError('conflicting rigid identity mapping')
        links[source] = target
    explicit = set(links)
    for object_id in set(ref_names) & set(query_names):
        if object_id not in ambiguous and ref_names[object_id] == query_names[object_id]:
            links.setdefault(object_id, object_id)
    return links, explicit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-preprocessing', choices=['official-demo', 'official-dataset'], default='official-demo')
    parser.add_argument('--output-root', type=Path, default=Path('outputs/reports/20261008_segmast3r_feasibility'))
    parser.add_argument('--bootstrap-resamples', type=int, default=1_000_000)
    parser.add_argument('--comparison-family-size', type=int, default=96)
    args = parser.parse_args()
    if args.bootstrap_resamples < 20_000 or args.comparison_family_size < 1:
        raise ValueError('use at least 20,000 bootstrap resamples and a positive effect-family size')
    output = args.output_root
    frozen = json.loads((output / 'frozen_segment_population.json').read_text())
    manifest_name = 'segment_inference_manifest.json' if args.input_preprocessing == 'official-demo' else 'segment_inference_manifest_dataset.json'
    segment_manifest = json.loads((output / manifest_name).read_text())
    dino_manifest = json.loads((output / 'dino_inference_manifest.json').read_text())
    expected_runs = {r['run_id'] for r in frozen['population']}
    expected_pairs = sum(min(len(r['reference_frames']), len(r['rescan_frames'])) for r in frozen['population'])
    for manifest in [segment_manifest, dino_manifest]:
        if len(manifest) != len(expected_runs) or {r['run_id'] for r in manifest} != expected_runs:
            raise ValueError('every frozen attempt must be present before cohort scoring')
        if not all(r['status'] == 'complete' for r in manifest):
            raise ValueError('failed attempt requires explicit zero-acceptance scoring adapter; do not omit it')
    root = Path('data/3rscan')
    metadata = json.loads((root / '3RScan.json').read_text())
    rows = []
    for r in frozen['population']:
        directory = output / 'uniform20_inputs' / r['run_id']
        inputs = json.loads((directory / 'manifest.json').read_text())
        variant = 'segment_predictions' if args.input_preprocessing == 'official-demo' else 'segment_predictions_dataset'
        seg = json.loads((output / variant / r['run_id'] / 'inference_summary.json').read_text())
        dino = json.loads((output / 'dino_segment_predictions' / r['run_id'] / 'inference_summary.json').read_text())
        if seg['identity_targets_read'] or dino['identity_targets_read']:
            raise ValueError('identity target leaked into matching')
        pair_count = min(len(inputs['records']['reference']), len(inputs['records']['rescan']))
        if len(seg['records']) != pair_count or len(dino['records']) != pair_count:
            raise ValueError('prediction records must cover every input frame pair')
        # Verify every frozen anonymous prediction before reading identity IDs.
        assignments = []
        for ref, query, sm, dm in zip(inputs['records']['reference'], inputs['records']['rescan'], seg['records'], dino['records']):
            frame_pair = (ref['frame_id'], query['frame_id'])
            if frame_pair != (sm['reference_frame'], sm['query_frame']) or frame_pair != (dm['reference_frame'], dm['query_frame']):
                raise ValueError('frame mismatch')
            for record in [ref, query]:
                if hashlib.sha256(Path(record['input']).read_bytes()).hexdigest() != record['input_sha256']:
                    raise ValueError('input changed after prediction')
            with np.load(sm['predictions']) as p:
                forward = p['reference_to_query'].copy()
            if forward.shape != (ref['masks'],) or np.any(forward < -1) or np.any(forward >= query['masks']):
                raise ValueError('segment assignment outside anonymous mask support')
            selected_seg = np.full(query['masks'], -1, np.int64)
            valid = forward >= 0
            if len(np.unique(forward[valid])) != int(valid.sum()):
                raise ValueError('official mutual assignments must be injective')
            selected_seg[forward[valid]] = np.flatnonzero(valid)
            with np.load(dm['predictions']) as p:
                scores, selected_dino, mutual = p['scores'].copy(), p['query_to_reference'].copy(), p['mutual_assigned'].copy()
            if scores.shape != (query['masks'], ref['masks']):
                raise ValueError('anonymous candidate support mismatch')
            if selected_dino.shape != (query['masks'],) or mutual.shape != selected_dino.shape:
                raise ValueError('DINO assignment shape mismatch')
            if np.any(selected_dino < -1) or np.any(selected_dino >= ref['masks']) or not np.isfinite(scores).all():
                raise ValueError('invalid DINO anonymous assignment or score')
            if np.any(mutual & (selected_dino < 0)):
                raise ValueError('mutual acceptance requires an assigned reference mask')
            nn = selected_dino >= 0
            equal = np.zeros(len(nn), bool)
            budget = int(valid.sum())
            if budget:
                eligible = np.flatnonzero(nn)
                confidence = scores[eligible, selected_dino[eligible]]
                order = np.lexsort((eligible, -confidence))
                equal[eligible[order[:budget]]] = True
            if equal.sum() != budget:
                raise ValueError('raw per-frame budget not reproduced')
            # Standard exact-cardinality assignment: dummy blocks permit
            # unmatched masks while retaining precisely the same raw budget
            # and the SegMASt3R one-to-one constraint. No labels set costs.
            qcount, rcount = scores.shape
            optimal = np.full(qcount, -1, np.int64)
            if budget:
                cost = np.zeros((qcount + rcount - budget,) * 2)
                cost[:qcount, :rcount] = -scores
                cost[qcount:, rcount:] = np.inf
                rr, cc = linear_sum_assignment(cost)
                valid_pairs = (rr < qcount) & (cc < rcount)
                optimal[rr[valid_pairs]] = cc[valid_pairs]
                if valid_pairs.sum() != budget:
                    raise ValueError('one-to-one cardinality budget mismatch')
            arms = {'segmast3r': (selected_seg, selected_seg >= 0),
                    'dino_nn': (selected_dino, nn), 'dino_mutual': (selected_dino, mutual),
                    'dino_equal_segment_budget': (selected_dino, equal),
                    'dino_equal_budget_one_to_one': (optimal, optimal >= 0)}
            diagnostics = output / 'diagnostic_assignments' / args.input_preprocessing / r['run_id']
            diagnostics.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(diagnostics / f'ref{frame_pair[0]:06d}_query{frame_pair[1]:06d}.npz',
                                **{f'{arm}_{key}': value for arm, pair in arms.items()
                                   for key, value in zip(['selected', 'assigned'], pair)})
            assignments.append((ref, query, arms))
        ref_sem, ref_names = semantics(root, r['reference_scan'])
        query_sem, query_names = semantics(root, r['rescan_scan'])
        links, explicit = identity_links(metadata, r['reference_scan'], r['rescan_scan'], ref_names, query_names)
        if len(set(links.values())) != len(links):
            raise ValueError('linked identity mapping has target collisions; audit before scoring')
        for ref, query, arms in assignments:
            ref_ids, query_ids = np.load(ref['identity_targets']), np.load(query['identity_targets'])
            for record, ids in [(ref, ref_ids), (query, query_ids)]:
                if ids.shape != (record['masks'],) or len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
                    raise ValueError('identity targets must map uniquely to anonymous masks')
            qs = np.array([query_sem.get(int(x), 0) for x in query_ids])
            rs = np.array([ref_sem.get(int(x), 0) for x in ref_ids])
            expected = np.array([links.get(int(x), -1) for x in query_ids])
            semantic_evaluable = qs > 0
            linked = semantic_evaluable & (expected > 0)
            present = linked & np.isin(expected, ref_ids)
            things_present = present & ~np.isin(qs, [1, 2, 22])
            for arm, (selected, assigned) in arms.items():
                mapped = np.full(len(selected), -1, np.int64)
                selected_sem = np.zeros(len(selected), np.int64)
                if assigned.any():
                    mapped[assigned] = ref_ids[selected[assigned]]
                    selected_sem[assigned] = rs[selected[assigned]]
                physical_correct = assigned & linked & (mapped == expected)
                semantic_correct = assigned & semantic_evaluable & (selected_sem == qs)
                absence_correct = (physical_correct & present) | (linked & ~present & ~assigned)
                rows.append(dict(run_id=r['run_id'], environment_id=r['environment_id'], arm=arm,
                                 reference_frame=ref['frame_id'], query_frame=query['frame_id'],
                                 raw_queries=len(query_ids), raw_assigned=int(assigned.sum()),
                                 semantic_queries=int(semantic_evaluable.sum()), semantic_assigned=int((assigned & semantic_evaluable).sum()),
                                 semantic_correct=int(semantic_correct.sum()), linked_queries=int(linked.sum()),
                                 linked_assigned=int((assigned & linked).sum()), linked_correct=int(physical_correct.sum()),
                                 linked_semantic_correct=int((semantic_correct & linked).sum()),
                                 linked_present_queries=int(present.sum()), linked_present_correct=int((physical_correct & present).sum()),
                                 linked_absence_correct=int(absence_correct.sum()), things_present_queries=int(things_present.sum()),
                                 things_present_correct=int((physical_correct & things_present).sum()),
                                 explicit_link_queries=int((linked & np.isin(query_ids, list(explicit))).sum())))
    frame = pd.DataFrame(rows)
    if len(frame) != expected_pairs * 5:
        raise ValueError('full frozen frame-pair population must contain all five arms')
    destination = output / ('segment_comparison' if args.input_preprocessing == 'official-demo' else 'segment_comparison_dataset')
    destination.mkdir(exist_ok=True)
    frame.to_csv(destination / 'per_frame.csv', index=False)
    counts = [c for c in frame.columns if c not in ['run_id', 'environment_id', 'arm', 'reference_frame', 'query_frame']]
    env = frame.groupby(['environment_id', 'arm'])[counts].sum()
    metrics = {'semantic_coverage': ('semantic_correct', 'semantic_queries'),
               'semantic_precision': ('semantic_correct', 'semantic_assigned'),
               'linked_coverage': ('linked_correct', 'linked_queries'),
               'linked_precision': ('linked_correct', 'linked_assigned'),
               'linked_semantic_precision': ('linked_semantic_correct', 'linked_assigned'),
               'linked_present_coverage': ('linked_present_correct', 'linked_present_queries'),
               'linked_with_absence_accuracy': ('linked_absence_correct', 'linked_queries'),
               'things_present_coverage': ('things_present_correct', 'things_present_queries')}
    for metric, (num, den) in metrics.items():
        env[metric] = env[num] / env[den].replace(0, np.nan)
    env.to_csv(destination / 'per_environment.csv')
    effects = []
    rng = np.random.default_rng(10082026)
    comparisons = args.comparison_family_size
    for metric in metrics:
        wide = env[metric].unstack('arm')
        for arm in ['dino_nn', 'dino_mutual', 'dino_equal_segment_budget', 'dino_equal_budget_one_to_one']:
            pair = wide[['segmast3r', arm]].dropna()
            delta = (pair.segmast3r - pair[arm]).to_numpy()
            if not len(delta):
                continue
            boot = np.empty(args.bootstrap_resamples, dtype=np.float64)
            batch_size = min(20_000, args.bootstrap_resamples)
            for start in range(0, args.bootstrap_resamples, batch_size):
                stop = min(start + batch_size, args.bootstrap_resamples)
                sampled = rng.integers(0, len(delta), (stop - start, len(delta)))
                boot[start:stop] = delta[sampled].mean(1)
            effects.append(dict(metric=metric, comparator=arm, environments=len(delta),
                                paired_segmast3r_mean=float(pair.segmast3r.mean()),
                                paired_comparator_mean=float(pair[arm].mean()),
                                segmast3r_minus_dino=float(delta.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                                bonferroni_ci95=np.quantile(boot, [.05/(2*comparisons), 1-.05/(2*comparisons)]).tolist()))
    summary = dict(relations=len(expected_runs), frame_pairs=expected_pairs, environments=len({r['environment_id'] for r in frozen['population']}),
                   input_preprocessing=args.input_preprocessing, bonferroni_effect_family=comparisons,
                   bootstrap_resamples=args.bootstrap_resamples,
                   comparison_family_definition='64 exploratory 8-group metric/control/preprocessing outcomes plus 32 full-cohort outcomes across eight metrics and four DINO controls',
                   macro_means=env[list(metrics)].groupby('arm').mean().to_dict('index'),
                   pooled=frame.groupby('arm')[counts].sum().to_dict('index'), effects=effects,
                   population_sha256=hashlib.sha256((output / 'frozen_segment_population.json').read_bytes()).hexdigest(),
                   scope='Segmentation-assisted native3RScan comparison; privileged mesh masks/poses shared by all arms. Linked annotation identity, not certified physical identity. Released-reference clusters; site independence unproven. Uniform available frames paired by chronological rank. Different denominator from fixed-patch paper. Equal-budget diagnostic uses SegMASt3R raw counts; no semantic/identity labels select accepts.')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['macro_means'], indent=2))


if __name__ == '__main__':
    main()
