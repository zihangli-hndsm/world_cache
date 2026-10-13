#!/usr/bin/env python3
"""Propagate source-group threshold-fit uncertainty into a frozen-gate transfer.

This is a post-hoc diagnostic on the first/next eight explored 3RScan train
groups. It independently resamples source groups for cutoff fitting and target
groups for transfer uncertainty; it is not independent-site validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_dino_multiview_source_gate import (
    ROOT,
    fit_threshold,
    grouped_metrics,
    load_split,
)


def fit_threshold_vectorized(relations: list[dict], target_precision: float) -> tuple[float | None, dict | None]:
    """Equivalent to fit_threshold, with group-level score-count vectorization."""
    candidates = np.unique(np.concatenate([r['scores'][r['valid']] for r in relations]))
    if not len(candidates):
        return None, None
    accepted_by_group, correct_by_group = [], []
    queries_by_group = []
    for relation in relations:
        valid = relation['valid']
        scores = relation['scores'][valid]
        correct = relation['correct'][valid].astype(np.int64)
        order = np.argsort(scores, kind='stable')
        scores = scores[order]
        correct = correct[order]
        start = np.searchsorted(scores, candidates, side='left')
        accepted = len(scores) - start
        prefix_correct = np.concatenate(([0], np.cumsum(correct)))
        correct_accepted = int(correct.sum()) - prefix_correct[start]
        accepted_by_group.append(accepted)
        correct_by_group.append(correct_accepted)
        queries_by_group.append(int(relation['queries']))

    accepted = np.stack(accepted_by_group)
    correct = np.stack(correct_by_group)
    precision = np.divide(correct, accepted, out=np.full(correct.shape, np.nan, dtype=float), where=accepted > 0)
    macro_precision = np.nanmean(precision, axis=0)
    macro_coverage = np.mean(correct / np.asarray(queries_by_group, dtype=float)[:, None], axis=0)
    feasible = np.flatnonzero((macro_precision >= target_precision) & (accepted.sum(axis=0) > 0))
    if not len(feasible):
        return None, None
    # Match the established rule: maximize macro correct coverage; break ties
    # toward the higher, more selective similarity cutoff.
    best = max(feasible.tolist(), key=lambda i: (macro_coverage[i], candidates[i]))
    threshold = float(candidates[best])
    metrics, _ = grouped_metrics(relations, threshold)
    return threshold, metrics


def relation_metrics(relation: dict, threshold: float) -> dict:
    accepted = relation['valid'] & (relation['scores'] >= threshold)
    n_accept = int(accepted.sum())
    correct = int(np.sum(accepted & relation['correct']))
    return {
        'accepted': n_accept,
        'correct': correct,
        'queries': int(relation['queries']),
        'precision': correct / n_accept if n_accept else np.nan,
        'coverage': correct / int(relation['queries']),
    }


def qinterval(values: list[float] | np.ndarray, qlo: float = .025, qhi: float = .975) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return None
    return [float(x) for x in np.quantile(array, [qlo, qhi])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-input-root', type=Path, default=ROOT / 'outputs/reports/20261009_unseen_oasis_equal_budget')
    parser.add_argument('--source-prediction-root', type=Path, default=ROOT / 'outputs/reports/20261009_dino_multiview_source_bank_train8')
    parser.add_argument('--target-input-root', type=Path, default=ROOT / 'outputs/reports/20261009_dino_object_view_budget_curve_extension')
    parser.add_argument('--target-prediction-root', type=Path, default=ROOT / 'outputs/reports/20261009_dino_multiview_source_bank_extension')
    parser.add_argument('--target-baseline-summary', type=Path, default=ROOT / 'outputs/reports/20261009_dino_multiview_source_bank_extension/summary.json')
    parser.add_argument('--target-macro-precision', type=float, default=.90)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/reports/20261009_dino_source_gate_nested_bootstrap_train8_extension8')
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError('use at least 100 bootstrap draws')

    source = load_split(args.source_input_root, args.source_prediction_root)
    target = load_split(args.target_input_root, args.target_prediction_root)
    source_groups = {r['environment_id'] for r in source}
    target_groups = {r['environment_id'] for r in target}
    if source_groups & target_groups:
        raise ValueError(f'source/target reference groups overlap: {sorted(source_groups & target_groups)}')
    fixed_threshold, fixed_metrics = fit_threshold(source, args.target_macro_precision)
    vector_threshold, vector_metrics = fit_threshold_vectorized(source, args.target_macro_precision)
    if fixed_threshold is None or vector_threshold != fixed_threshold:
        raise RuntimeError(f'vectorized threshold differs from established fit: {fixed_threshold} vs {vector_threshold}')
    if fixed_metrics is None or not np.isclose(fixed_metrics['macro_coverage'], vector_metrics['macro_coverage']):
        raise RuntimeError('vectorized source-fit metrics differ from established implementation')

    baseline_summary = json.loads(args.target_baseline_summary.read_text())
    baseline_by_group = {r['environment_id']: r['remind_one_to_one'] for r in baseline_summary['reference_groups']}
    if set(baseline_by_group) != target_groups:
        raise ValueError('target REMIND groups do not match DINO prediction groups')

    fixed_target_rows = [relation_metrics(r, fixed_threshold) for r in target]
    fixed_target_precision = np.asarray([r['precision'] for r in fixed_target_rows], dtype=float)
    fixed_target_coverage = np.asarray([r['coverage'] for r in fixed_target_rows], dtype=float)
    fixed_target_baseline_precision = np.asarray([baseline_by_group[r['environment_id']]['precision'] for r in target])
    fixed_target_baseline_coverage = np.asarray([baseline_by_group[r['environment_id']]['coverage'] for r in target])

    rng = np.random.default_rng(args.seed)
    source_thresholds: list[float] = []
    fit_failures = 0
    fit_only_precision, fit_only_coverage = [], []
    nested_precision, nested_coverage = [], []
    nested_precision_delta, nested_coverage_delta = [], []
    nested_accept_rate, nested_target_90 = [], []
    nested_precision_groups, nested_paired_precision_groups = [], []

    source_n, target_n = len(source), len(target)
    for draw in range(args.draws):
        source_ix = rng.integers(0, source_n, size=source_n)
        sampled_source = []
        for replicate, index in enumerate(source_ix):
            relation = dict(source[int(index)])
            relation['environment_id'] = f'source_boot_{replicate}'
            sampled_source.append(relation)
        threshold, _ = fit_threshold_vectorized(sampled_source, args.target_macro_precision)
        if threshold is None:
            fit_failures += 1
            continue
        source_thresholds.append(threshold)

        fixed_metrics_rows = [relation_metrics(r, threshold) for r in target]
        p_fixed = np.asarray([r['precision'] for r in fixed_metrics_rows], dtype=float)
        c_fixed = np.asarray([r['coverage'] for r in fixed_metrics_rows], dtype=float)
        fit_only_precision.append(float(np.nanmean(p_fixed)))
        fit_only_coverage.append(float(np.mean(c_fixed)))

        target_ix = rng.integers(0, target_n, size=target_n)
        gate_rows = [relation_metrics(target[int(i)], threshold) for i in target_ix]
        remind_rows = [baseline_by_group[target[int(i)]['environment_id']] for i in target_ix]
        p_gate = np.asarray([r['precision'] for r in gate_rows], dtype=float)
        c_gate = np.asarray([r['coverage'] for r in gate_rows], dtype=float)
        p_remind = np.asarray([r['precision'] for r in remind_rows], dtype=float)
        c_remind = np.asarray([r['coverage'] for r in remind_rows], dtype=float)
        nested_precision.append(float(np.nanmean(p_gate)))
        nested_coverage.append(float(np.mean(c_gate)))
        nested_accept_rate.append(float(sum(r['accepted'] for r in gate_rows) / sum(r['queries'] for r in gate_rows)))
        nested_target_90.append(bool(np.nanmean(p_gate) >= args.target_macro_precision))
        paired_p = np.isfinite(p_gate) & np.isfinite(p_remind)
        nested_precision_groups.append(int(np.isfinite(p_gate).sum()))
        nested_paired_precision_groups.append(int(paired_p.sum()))
        nested_precision_delta.append(float(np.mean((p_gate - p_remind)[paired_p])) if paired_p.any() else np.nan)
        nested_coverage_delta.append(float(np.mean(c_gate - c_remind)))

    if not source_thresholds:
        raise RuntimeError('no source bootstrap draw produced a feasible cutoff')
    result = {
        'status': 'complete',
        'scope': 'post-hoc nested source/target reference-group bootstrap for the existing train8-to-extension8 all-view DINO gate',
        'selection_rule': {
            'source_target_macro_precision': args.target_macro_precision,
            'objective': 'maximize source macro correct coverage among cutoffs meeting the source macro precision floor',
            'tie_break': 'higher cutoff',
            'source_groups': len(source),
            'target_groups': len(target),
            'fixed_full_source_threshold': fixed_threshold,
        },
        'fixed_source_fit': {
            'threshold': fixed_threshold,
            'source_metrics': fixed_metrics,
            'target_metrics': grouped_metrics(target, fixed_threshold)[0],
        },
        'nested_bootstrap': {
            'draws_requested': args.draws,
            'draws_feasible': len(source_thresholds),
            'source_fit_failure_rate': fit_failures / args.draws,
            'seed': args.seed,
            'unit': 'reference group; source and target groups resampled independently',
            'source_threshold_percentile_95': qinterval(source_thresholds),
            'fixed_target_population_metrics_with_source_fit_variation': {
                'macro_precision_percentile_95': qinterval(fit_only_precision),
                'macro_correct_coverage_percentile_95': qinterval(fit_only_coverage),
                'target_macro_precision_at_least_source_floor_fraction': float(np.mean(np.asarray(fit_only_precision) >= args.target_macro_precision)),
            },
            'source_and_target_population_percentile_95': {
                'target_macro_precision': qinterval(nested_precision),
                'target_macro_correct_coverage': qinterval(nested_coverage),
                'paired_macro_precision_difference_vs_remind': qinterval(nested_precision_delta),
                'paired_macro_correct_coverage_difference_vs_remind': qinterval(nested_coverage_delta),
                'target_macro_precision_at_least_source_floor_fraction': float(np.mean(nested_target_90)),
                'pooled_query_accept_rate': qinterval(nested_accept_rate),
                'groups_with_gate_accepts': qinterval(nested_precision_groups),
                'groups_with_paired_precision': qinterval(nested_paired_precision_groups),
                'precision_macro_definition': 'mean over resampled target groups with at least one gate accept; paired precision difference uses groups where both arms have accepts',
            },
        },
        'limits': [
            'This is same-dataset post-hoc analysis; the next-eight cohort was selected after first-cohort outcomes were seen.',
            'The bootstrap resamples only eight reference groups per split and cannot establish cross-site or distribution-free calibration.',
            'The gate uses annotation-derived mask inputs; its target labels are used only for evaluation, not threshold fitting.',
            'The paired REMIND comparison is not equal-budget; acceptance rates are reported separately.',
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'nested_bootstrap.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output_dir / 'INTERPRETATION.md').write_text(
        '# Nested uncertainty for the source-frozen all-view gate\n\n'
        f"The full source fit reproduces cutoff {fixed_threshold:.6f}. A nested reference-group bootstrap refits the cutoff on resampled source groups and independently resamples the eight target groups. {fit_failures}/{args.draws} source resamples had no cutoff meeting the 90% macro precision floor.\n\n"
        f"Cutoff 95% percentile range: {result['nested_bootstrap']['source_threshold_percentile_95']}. With target groups held fixed, threshold-fit variation gives macro precision interval {result['nested_bootstrap']['fixed_target_population_metrics_with_source_fit_variation']['macro_precision_percentile_95']} and correct-coverage interval {result['nested_bootstrap']['fixed_target_population_metrics_with_source_fit_variation']['macro_correct_coverage_percentile_95']}.\n\n"
        f"Resampling both source and target groups gives target macro precision interval {result['nested_bootstrap']['source_and_target_population_percentile_95']['target_macro_precision']}, correct-coverage interval {result['nested_bootstrap']['source_and_target_population_percentile_95']['target_macro_correct_coverage']}, paired precision difference versus REMIND {result['nested_bootstrap']['source_and_target_population_percentile_95']['paired_macro_precision_difference_vs_remind']}, and paired coverage difference {result['nested_bootstrap']['source_and_target_population_percentile_95']['paired_macro_correct_coverage_difference_vs_remind']}.\n\n"
        'These are exploratory uncertainty estimates on eight already examined groups per split, not external confirmation or a distribution-free precision guarantee. See `nested_bootstrap.json` for draws and exact metrics.\n'
    )
    print(json.dumps({
        'draws': args.draws,
        'fit_failures': fit_failures,
        'threshold_interval': result['nested_bootstrap']['source_threshold_percentile_95'],
        'fit_only_precision': result['nested_bootstrap']['fixed_target_population_metrics_with_source_fit_variation']['macro_precision_percentile_95'],
        'nested_precision': result['nested_bootstrap']['source_and_target_population_percentile_95']['target_macro_precision'],
        'nested_precision_delta': result['nested_bootstrap']['source_and_target_population_percentile_95']['paired_macro_precision_difference_vs_remind'],
        'nested_coverage_delta': result['nested_bootstrap']['source_and_target_population_percentile_95']['paired_macro_correct_coverage_difference_vs_remind'],
    }, indent=2))


if __name__ == '__main__':
    main()
