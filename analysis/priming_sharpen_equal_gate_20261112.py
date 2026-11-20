"""Replay the frozen target-only update with the same gate for each paired target."""

from __future__ import annotations

import csv

import numpy as np
from threadpoolctl import threadpool_limits

from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_corroboration_20261112 import direction, update
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_same_concept_eval_20261112 import csv_output, load
from priming_sharpen_search_20261112 import DATASETS, DOC, control_gate, transform
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def recorded_results():
    path = DOC / 'chain' / 'same_concept_contrasts.csv'
    with path.open(newline='', encoding='utf-8') as f:
        return {(r['dataset'], r['method'], r['metric']): r for r in csv.DictReader(f)
                if r['comparison'] == 'history' and r['history'] == 'exact' and r['reference'] == 'same_concept'}


def replay(x, d, gate, cfg):
    z = transform(x, d, gate, cfg['mixing'], cfg['suppression']).astype(x.dtype)
    expected_mean = x.mean(1) * (1 - cfg['suppression'] * gate)
    np.testing.assert_allclose(z.mean(1), expected_mean, rtol=3e-6, atol=1e-8)
    assert np.array_equal(z > 0, x > 0)
    return z


def main():
    protocol = frozen()
    cfg = protocol['configs']['target_only']
    assert cfg['mode'] == 'product' and cfg['weight'] == 0
    kernel = weights()
    recorded = recorded_results()
    rows, summary = [], []
    for dataset in DATASETS:
        bank, head, cuts = fitted(dataset, kernel)
        fc2, p, labels, stimuli = load(dataset)
        raw = fc2[:, :2, protocol['t']].transpose(1, 0, 2)
        paired = PairedConcepts(labels, np.array([r['concept'] for r in stimuli['targets']]))
        queries = [bank.query(x, p[:, i]) for i, x in enumerate(raw)]
        exact, same = (control_gate(q['match'], cuts[cfg['quantile']]) for q in queries)
        middle = (exact + same) / 2
        original = [update(bank, x, q, cuts, cfg) for x, q in zip(raw, queries)]
        directions = [direction(bank, x, q, cfg) for x, q in zip(raw, queries)]
        for x, d, g, y in zip(raw, directions, (exact, same), original):
            np.testing.assert_array_equal(replay(x, d, g, cfg), y)

        gates = dict(original=(exact, same), equal_midpoint=(middle, middle),
                     equal_exact=(exact, exact), equal_same_concept=(same, same))
        outputs = dict(chain_only=raw, original=original)
        for name in ('equal_midpoint', 'equal_exact', 'equal_same_concept'):
            outputs[name] = [replay(x, d, g, cfg) for x, d, g in zip(raw, directions, gates[name])]
        contrasts = {}
        for name, (a, b) in outputs.items():
            results = [metrics(z, head, paired) for z in (a, b)]
            contrasts[name] = contrast(*results)
            for i, (history, z, values) in enumerate(zip(('exact', 'same_concept'), (a, b), results)):
                gate = float(paired.concept_weights[0] @ gates[name][i]) if name in gates else 0.
                correct = int((head.decision_function(z).argmax(1) == labels).sum())
                summary.append(dict(dataset=dataset, condition=name, history=history, mean_gate=gate,
                                    correct=correct, n=len(labels), **{k: float(values[k][0])
                                    for k in ('mean', 'within', 'between', 'fisher', 'margin', 'auc', 'ba')}))

        for name, old in (('chain_only', 'chain_only'), ('original', 'target_only')):
            for metric, values in contrasts[name].items():
                observed = estimate(values)
                expected = recorded[dataset, old, metric]
                for key in observed:
                    np.testing.assert_allclose(observed[key], float(expected[key]), atol=1e-7, rtol=1e-6)

        for name in ('equal_midpoint', 'equal_exact', 'equal_same_concept'):
            common = contrasts[name]
            unequal_gate = {k: v - common[k] for k, v in contrasts['original'].items()}
            shared_gate_response = {k: v - contrasts['chain_only'][k] for k, v in common.items()}
            for metric, values in contrasts['original'].items():
                np.testing.assert_allclose(values, contrasts['chain_only'][metric] + shared_gate_response[metric]
                                           + unequal_gate[metric], rtol=1e-10, atol=1e-12)
            contrasts[f'unequal_gate_at_{name}'] = unequal_gate
            contrasts[f'shared_gate_response_at_{name}'] = shared_gate_response
        for name, results in contrasts.items():
            rows.extend(dict(dataset=dataset, condition=name, metric=k, **estimate(v)) for k, v in results.items())

    csv_output('equal_gate_contrasts', rows)
    csv_output('equal_gate_summary', summary)
    print('EQUAL GATE', len(rows), 'paired estimates', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
