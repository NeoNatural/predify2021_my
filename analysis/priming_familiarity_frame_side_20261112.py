"""Test temporal separation and four-way competition on existing Conv5 maps."""

from __future__ import annotations

import csv

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from priming_familiarity_20261112 import DECAY, checked
from priming_familiarity_partition_20261112 import HISTORIES, fold3
from priming_familiarity_report_20261112 import paired_auc, write
from priming_sharpen_search_20261112 import DATASETS, DOC, save
from report_hebb_post_pool_20261111 import estimate


COMPARISONS = (
    ('four_mean', 'two_mean'),
    ('four_side_max', 'two_max'),
    ('four_max', 'four_side_max'),
    ('four_max', 'two_max'),
    ('two_max', 'two_mean'),
    ('four_max', 'four_mean'),
)


@torch.inference_mode()
def readouts(model, target, preview):
    assert target.shape[-2:] == preview.shape[-2:] == (14, 14)
    halves = preview.split(7, dim=-1)
    two = torch.stack([model.memory(half) for half in halves])
    four = []
    for half in halves:
        first, second = half.unbind(-4)
        four.append(torch.stack((2 * DECAY * model.coactivation(first), 2 * model.coactivation(second))))
    four = torch.stack(four)  # Side, then frame.
    reconstructed = four.mean(dim=1)
    torch.testing.assert_close(reconstructed, two, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(reconstructed.mean(0), model.memory(preview), atol=2e-7, rtol=2e-5)
    encoded = model.encode(target)
    outcomes, winners = {}, {}
    for query, z in (('native_target', encoded), ('fixed_exact_target', encoded[:, :1])):
        h = model.reference(z)
        a = model.read_encoded(z, h, two)
        b = model.read_encoded(z, h, four)
        side_mean = b.mean(dim=1)
        best = b.amax(dim=(0, 1))
        assert torch.all(side_mean <= a + 2e-6)
        assert torch.all(side_mean.amax(0) <= best)
        assert torch.count_nonzero(model.read_encoded(z, h, torch.zeros_like(two))) == 0
        result = dict(two_mean=a.mean(0), two_max=a.amax(0), four_mean=side_mean.mean(0),
                      four_side_max=side_mean.amax(0), four_max=best)
        outcomes[query] = {arm: v.numpy() for arm, v in result.items()}
        # Split votes evenly if branches tie within the FP32 tolerance.
        top = (best - b).abs() <= 1e-7
        winners[query] = (top.sum(0) / top.sum((0, 1))).numpy()
    for arm, value in outcomes['native_target'].items():
        np.testing.assert_allclose(value[:, 0], outcomes['fixed_exact_target'][arm][:, 0], atol=2e-6, rtol=2e-5)
    return outcomes, winners, float((reconstructed - two).abs().max())


def p90(values, counts):
    """Linear-interpolated empirical 90th percentile in every concept resample."""
    assert np.all(counts.sum(1) == len(values))
    order = np.argsort(values)
    sorted_values = values[order].astype(np.float64)
    cumulative = counts[:, order].cumsum(1)
    rank = .9 * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    a = sorted_values[(cumulative <= lower).sum(1)]
    b = sorted_values[(cumulative <= upper).sum(1)]
    result = a + (b - a) * (rank - lower)
    np.testing.assert_allclose(result[0], np.quantile(values, .9), atol=1e-7)
    return result


def previous_results():
    with (DOC / 'chain/familiarity_partition_fold3.csv').open(newline='', encoding='utf-8') as f:
        return {(r['dataset'], r['query'], r['arm'], r['history']): r for r in csv.DictReader(f)
                if not r['reference'] and r['metric'] == 'auc'}


def evaluate(dataset, registration, previous, rows):
    model, target, preview, paired = fold3(dataset, registration)
    values, winners, matrix_error = readouts(model, target, preview)
    w = paired.concept_weights
    max_replay_error = 0.
    for query, arms in values.items():
        def emit(arm, ref, history, metric, v):
            rows.append(dict(dataset=dataset, query=query, arm=arm, reference=ref,
                             history=history, metric=metric, **estimate(v)))

        for h in (1, 2):
            negative = HISTORIES[h]
            auc = {arm: paired_auc(scores[:, 0], scores[:, h], w) for arm, scores in arms.items()}
            for arm, stats in auc.items():
                emit(arm, '', negative, 'auc', stats)
            for arm, ref in COMPARISONS:
                emit(arm, ref, negative, 'auc', auc[arm] - auc[ref])
            for arm, prior in (('two_mean', 'average'), ('two_max', 'competition')):
                observed = estimate(auc[arm])
                saved = previous[dataset, query, prior, negative]
                for key, estimate_value in observed.items():
                    error = abs(estimate_value - float(saved[key]))
                    assert error < 1e-10
                    max_replay_error = max(max_replay_error, error)

        tail = {(arm, h): p90(scores[:, h], paired.counts)
                for arm, scores in arms.items() for h in range(len(HISTORIES))}
        for h, history in enumerate(HISTORIES):
            for arm in arms:
                emit(arm, '', history, 'score_p90', tail[arm, h])
            for arm, ref in COMPARISONS:
                emit(arm, ref, history, 'score_p90', tail[arm, h] - tail[ref, h])
            for j in range(2):
                emit('four_max', '', history, f'frame_{j + 1}_win_pct', 100 * (w @ winners[query][j, :, h]))
        for arm, ref in COMPARISONS:
            boost = arms[arm] - arms[ref]
            for h in (1, 2):
                emit(arm, ref, HISTORIES[h], 'exact_minus_control_mean_boost', w @ (boost[:, 0] - boost[:, h]))
    return dict(targets=len(target), matrix_max_abs_error=matrix_error, prior_auc_max_abs_error=max_replay_error,
                cleared_memory_zero=True)


def main():
    reg = checked()
    previous = previous_results()
    rows = []
    audit = {d: evaluate(d, reg['datasets'][d], previous, rows) for d in DATASETS}
    write('frame_side_fold3', rows)
    save(DOC / 'chain/familiarity_frame_side_audit.json', audit)


if __name__ == '__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        main()
