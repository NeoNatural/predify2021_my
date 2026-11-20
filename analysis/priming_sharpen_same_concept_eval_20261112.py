"""Compare the two frozen directions and the four matched preview histories."""

from __future__ import annotations

import argparse
import csv

import numpy as np
from threadpoolctl import threadpool_limits

from priming_sharpen_collect_20261112 import digest, vanilla
from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_corroboration_20261112 import update
from priming_sharpen_object_precision_20261112 import encode_tiles, fitted, tiles
from priming_sharpen_same_concept_data_20261112 import HISTORIES, PLAN, SCOPE, frozen, renderer
from priming_sharpen_search_20261112 import DATASETS, DOC, OUT, control_gate, read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def load(dataset):
    root = OUT / dataset
    path = root / f'chain_{SCOPE}.npz'
    audit = read(path.with_suffix('.json'))
    plan = read(root / f'{SCOPE}_manifest.json')
    assert audit['protocol_sha256'] == digest(PLAN) and audit['feature_sha256'] == digest(path)
    assert audit['deterministic_fp32'] and tuple(audit['histories']) == HISTORIES
    with np.load(path) as f:
        fc2, preview, index, labels = (f[key] for key in ('fc2', 'preview', 'index', 'label'))
    np.testing.assert_array_equal(index, [r['index'] for r in plan['targets']])
    np.testing.assert_array_equal(labels, [r['label'] for r in plan['targets']])
    assert fc2.shape == (len(labels), len(HISTORIES), 15, 4096)
    assert preview.shape == (len(labels), len(HISTORIES) - 1, 4, 4096)
    return fc2, preview, labels, plan


def csv_output(name, rows):
    path = DOC / 'chain' / f'{SCOPE}_{name}.csv'
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def evaluate(plan, kernel):
    rows, summaries, gates = [], [], []
    comparisons = plan['comparisons']
    for dataset in DATASETS:
        bank, head, cuts = fitted(dataset, kernel)
        fc2, p, labels, stimuli = load(dataset)
        raw = fc2[:, :, plan['t']]
        paired = PairedConcepts(labels, np.array([r['concept'] for r in stimuli['targets']]))
        query = [bank.query(raw[:, i], p[:, i]) for i in range(3)]
        cut = cuts[plan['configs']['geometric']['quantile']]
        gate_metrics = {}
        for h, q in zip(HISTORIES, query):
            gate = control_gate(q['match'], cut)
            gate_metrics[h] = dict(match=paired.concept_weights @ q['match'],
                                   gate=paired.concept_weights @ gate,
                                   active_pct=100 * (paired.concept_weights @ (gate > 0)))
            gates.extend(dict(dataset=dataset, history=h, reference='', cut=cut, metric=key, **estimate(value))
                         for key, value in gate_metrics[h].items())
        for a, b in comparisons:
            if b != 'none':
                gates.extend(dict(dataset=dataset, history=a, reference=b, cut=cut, metric=key,
                                  **estimate(gate_metrics[a][key] - gate_metrics[b][key])) for key in gate_metrics[a])

        outputs = {'chain_only': dict(zip(HISTORIES, raw.transpose(1, 0, 2)))}
        for name, cfg in plan['configs'].items():
            y = [update(bank, raw[:, i], q, cuts, cfg) for i, q in enumerate(query)]
            for i, z in enumerate(y):
                assert np.array_equal(z > 0, raw[:, i] > 0)
                expected_mean = raw[:, i].mean(1) * (1 - cfg['suppression'] * control_gate(query[i]['match'], cut))
                np.testing.assert_allclose(z.mean(1), expected_mean, rtol=3e-6, atol=1e-8)
            outputs[name] = dict(zip(HISTORIES, [*y, raw[:, 3]]))
        for h in HISTORIES:
            np.testing.assert_allclose(outputs['target_only'][h].mean(1), outputs['geometric'][h].mean(1),
                                       rtol=3e-6, atol=1e-8)
        scores = {m: {h: metrics(x, head, paired) for h, x in values.items()} for m, values in outputs.items()}

        def write(kind, method, history, ref, changes):
            rows.extend(dict(dataset=dataset, comparison=kind, method=method, history=history, reference=ref,
                             metric=key, **estimate(val)) for key, val in changes.items())

        for method, histories in scores.items():
            for h, result in histories.items():
                summary = {key: float(result[key][0]) for key in ('mean', 'within', 'between', 'fisher', 'ba', 'auc', 'margin')}
                correct = int((head.decision_function(outputs[method][h]).argmax(1) == labels).sum())
                summaries.append(dict(dataset=dataset, method=method, history=h, n=len(labels), correct=correct, **summary))
            for a, b in comparisons:
                write('history', method, a, b, contrast(histories[a], histories[b]))
            if method != 'chain_only':
                for h in HISTORIES[:-1]:
                    write('gain_vs_chain', method, h, 'chain_only', contrast(histories[h], scores['chain_only'][h]))

        g, t = scores['geometric'], scores['target_only']
        for h in HISTORIES[:-1]:
            write('direction', 'geometric', h, 'target_only', contrast(g[h], t[h]))
        for a, b in comparisons:
            c_g, c_t = contrast(g[a], g[b]), contrast(t[a], t[b])
            write('direction_by_history', 'geometric_minus_target_only', a, b,
                  {key: c_g[key] - c_t[key] for key in c_g})
    csv_output('contrasts', rows)
    csv_output('summary', summaries)
    csv_output('gates', gates)
    print('PAIRED EVALUATION', len(rows), 'contrasts', flush=True)


def audit_online(plan, kernel):
    from hebb_chain_20261111 import Controls, Experiment, old
    from priming_sharpen_chain_20261112 import simulate

    parts = old.initialize()
    model, buffers = vanilla()
    controls = Controls(tuple(plan['hebbian_lambdas']))
    result = {}
    for dataset in DATASETS:
        bank, _, cuts = fitted(dataset, kernel)
        fc2, preview, _, stimuli = load(dataset)
        render = renderer(stimuli)
        run = Experiment(dataset, parts)
        run.fit()
        details = {}
        try:
            for k, h in enumerate(HISTORIES[:-1]):
                raw_t9 = fc2[:, k, plan['t']]
                scores = bank.query(raw_t9, preview[:, k])['match']
                i = int(scores.argmax())
                row = stimuli['targets'][i]
                image, frames = render(row)
                vectors = encode_tiles(model, buffers, tiles(frames[k].cpu().numpy()))
                np.testing.assert_array_equal(vectors, preview[i, k])
                details[h] = dict(target=row['index'], steps=15, directions={})
                for name, cfg in plan['configs'].items():
                    def edit(x, _):
                        memory = np.broadcast_to(vectors, (len(x), *vectors.shape))
                        return update(bank, x, bank.query(x, memory), cuts, cfg)

                    observed = simulate(run, image, frames[k], controls, active=True, edit=edit)[0]
                    np.testing.assert_array_equal(observed['fc2'], fc2[i, k])
                    expected = edit(fc2[i, k], None)
                    np.testing.assert_allclose(observed['terminal'], expected, rtol=1e-5, atol=1e-5)
                    details[h]['directions'][name] = float(np.max(np.abs(expected - observed['terminal'])))
            image, _ = render(stimuli['targets'][0])
            none = simulate(run, image, [run.c.grey] * 2, controls, active=False)[0]
            np.testing.assert_array_equal(none['fc2'], fc2[0, 3])
            details['no_preview_identical'] = True
        finally:
            run.detach()
        result[dataset] = details
    save(DOC / 'chain' / f'{SCOPE}_online.json', result)
    print('ONLINE AUDIT', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        action = audit_online if args.online else evaluate
        action(frozen(), weights())
