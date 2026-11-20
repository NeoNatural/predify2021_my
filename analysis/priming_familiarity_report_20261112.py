"""Paired concept statistics for the frozen Conv5 familiarity experiment."""

from __future__ import annotations

import csv

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from priming_familiarity_20261112 import SHUFFLES, checked, gate
from priming_familiarity_verify_20261112 import SCOPES, apply_gate, archive, target_direction
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import HISTORIES, frozen
from priming_sharpen_search_20261112 import DATASETS, DOC, save, unit
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


WINDOWS = {**{f't{i}': slice(i, i + 1) for i in range(15)}, 'middle': slice(5, 10), 'late': slice(10, 15)}


def write(name, rows):
    dest = DOC / 'chain' / f'familiarity_{name}.csv'
    tmp = dest.with_suffix('.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dest)


def measured(x, head, paired):
    n, steps, units = x.shape
    v = unit(x.reshape(-1, units).astype(np.float64)).reshape(n, steps, units).transpose(1, 0, 2)
    result = paired.geometry(v @ v.transpose(0, 2, 1), equal_class=True)
    result['mean'] = paired.concept_weights @ x.mean(-1)
    scores = head.decision_function(x.reshape(-1, units)).reshape(n, steps, -1)
    readout = [paired.readout(scores[:, t:t + 1], fc=True) for t in range(steps)]
    for metric in ('ba', 'auc', 'margin'):
        result[metric] = np.column_stack([r[metric] for r in readout])
    return result, scores[:, 9].argmax(1) == paired.labels


def differences(a, b, window):
    result = {}
    for metric, v in a.items():
        left, right = v[:, window], b[metric][:, window]
        if metric in ('mean', 'within', 'between', 'fisher'):
            ratio = left / right
            change = 100 * (1 - ratio) if metric == 'mean' else 100 * (ratio - 1)
        else:
            change = (left - right) * (100 if metric in ('ba', 'auc') else 1)
        result['suppression' if metric == 'mean' else metric] = change.mean(1)
    return result


def predicted_change(a, b):
    return dict(corrected=int((a & ~b).sum()), lost=int((~a & b).sum()),
                both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()))


def fc_statistics(data, bank, head, cfg, paired, row, absolute, contrasts):
    x = data['fc2']
    n, _, t, d = x.shape
    base, hits, directions = {}, {}, {}
    for h, name in enumerate(HISTORIES):
        raw = x[:, h]
        base[name], hits[name] = measured(raw, head, paired)
        directions[name] = target_direction(bank, raw.reshape(n * t, d), cfg)
    results = {'base': base, 'clear': base}
    correctness = {'base': hits, 'clear': hits}
    shuffle_results = []
    method_gates = [('old', data['gate_old']), ('new', data['gate_new'])]
    method_gates.extend((str(seed), data['gate_shuffled'][:, :, i]) for i, seed in enumerate(SHUFFLES))
    for method, g in method_gates:
        values, flags = {}, {}
        for h, name in enumerate(HISTORIES):
            if name == 'none':
                assert np.all(g[:, h] == 0)
                values[name], flags[name] = base[name], hits[name]
                continue
            raw = x[:, h].reshape(n * t, d)
            gate_values = g[:, h].reshape(-1)
            treatment = apply_gate(raw, directions[name], gate_values, cfg)
            assert np.array_equal(treatment > 0, raw > 0)
            np.testing.assert_allclose(treatment.mean(1), raw.mean(1) * (1 - cfg['suppression'] * gate_values),
                                       atol=1e-8, rtol=3e-6)
            values[name], flags[name] = measured(treatment.reshape(n, t, d), head, paired)
        if method in ('old', 'new'):
            results[method] = values
            correctness[method] = flags
        else:
            shuffle_results.append(values)
    results['shuffled'] = {
        h: {k: np.mean([r[h][k] for r in shuffle_results], axis=0) for k in base[h]} for h in HISTORIES
    }
    for w, interval in WINDOWS.items():
        for method, measurements in results.items():
            for h, metrics in measurements.items():
                absolute.extend(dict(**row, window=w, method=method, history=h, metric=k, **estimate(v[:, interval].mean(1)),
                                     shuffle_sd=float(np.std([s[h][k][0, interval].mean() for s in shuffle_results]))
                                     if method == 'shuffled' else '') for k, v in metrics.items())
            for ref in HISTORIES[1:]:
                test = differences(measurements['exact'], measurements[ref], interval)
                spread = {}
                if method == 'shuffled':
                    per_seed = [differences(s['exact'], s[ref], interval) for s in shuffle_results]
                    test = {k: np.mean([s[k] for s in per_seed], axis=0) for k in test}
                    spread = {k: float(np.std([s[k][0] for s in per_seed])) for k in test}
                contrasts.extend(dict(**row, window=w, condition=method, reference=method, history='exact',
                                      reference_history=ref, metric=k, **estimate(v), shuffle_sd=spread.get(k, ''))
                                 for k, v in test.items())
        for a, b in (('new', 'old'), ('new', 'base'), ('old', 'base'), ('new', 'clear'), ('new', 'shuffled')):
            for h in HISTORIES:
                change = differences(results[a][h], results[b][h], interval)
                spread = {}
                if b == 'shuffled':
                    per_seed = [differences(results[a][h], s[h], interval) for s in shuffle_results]
                    change = {k: np.mean([s[k] for s in per_seed], axis=0) for k in change}
                    spread = {k: float(np.std([s[k][0] for s in per_seed])) for k in change}
                contrasts.extend(dict(**row, window=w, condition=a, reference=b, history=h,
                                      reference_history=h, metric=k, **estimate(v), shuffle_sd=spread.get(k, ''))
                                 for k, v in change.items())
    return dict(n=n, correct={m: {h: int(v.sum()) for h, v in r.items()} for m, r in correctness.items()},
                new_vs_old={h: predicted_change(correctness['new'][h], correctness['old'][h]) for h in HISTORIES},
                new_vs_base={h: predicted_change(correctness['new'][h], correctness['base'][h]) for h in HISTORIES},
                new_exact_vs_other={h: predicted_change(correctness['new']['exact'], correctness['new'][h]) for h in HISTORIES[1:]})


def paired_auc(exact, control, weights):
    compare = (exact[:, None] > control[None, :]) + .5 * (exact[:, None] == control[None, :])
    return ((weights @ compare) * weights).sum(1)


def gate_statistics(data, paired, row, cal, out):
    zero = np.zeros_like(data['gate_new'])
    fixed_target = np.concatenate([data['score_fixed_target'], zero[:, :1]], axis=1)
    sources = {
        'old': (data['score_old'][None], data['gate_old'][None]),
        'new': (data['score_new'][None], data['gate_new'][None]),
        'clear': (zero[None], zero[None]),
        'shuffled': (data['score_shuffled'].transpose(2, 0, 1, 3), data['gate_shuffled'].transpose(2, 0, 1, 3)),
        'new_fixed_target': (fixed_target[None], gate(fixed_target, cal['threshold'], cal['slope'])[None]),
    }
    w = paired.concept_weights

    def emit(method, window, h, reference, metric, values):
        v = np.array(values)
        spread = float(v[:, 0].std()) if len(v) > 1 else ''
        out.append(dict(**row, method=method, window=window, history=h, reference=reference, metric=metric,
                        **estimate(v.mean(0)), shuffle_sd=spread))

    for method, (raw, gates) in sources.items():
        for window, interval in WINDOWS.items():
            for h, name in enumerate(HISTORIES):
                g = gates[:, :, h, interval]
                for metric, v in (('mean_gate', g), ('trigger_pct', 100 * (g > 0)), ('saturation_pct', 100 * (g >= 1))):
                    emit(method, window, name, '', metric, [w @ a.mean(1) for a in v])
                if window == 't9':
                    for q in (10, 50, 90):
                        value = float(np.mean(np.percentile(g[:, :, 0], q, axis=1)))
                        out.append(dict(**row, method=method, window=window, history=name, reference='',
                                        metric=f'q{q}', estimate=value, ci_low='', ci_high='', shuffle_sd=''))
            for h in (1, 2):
                for label, arr in (('score_auc', raw), ('gate_auc', gates)):
                    values = [np.mean([paired_auc(a[:, 0, t], a[:, h, t], w)
                                       for t in range(interval.start, interval.stop)], axis=0) for a in arr]
                    emit(method, window, 'exact', HISTORIES[h], label, values)
        if method in ('new', 'old'):
            for slot in sorted(np.unique(data['slot'])):
                use = data['slot'] == slot
                for h in (1, 2):
                    actual = np.r_[np.ones(use.sum()), np.zeros(use.sum())]
                    score = np.r_[raw[0, use, 0, 9], raw[0, use, h, 9]]
                    out.append(dict(**row, method=method, window=f't9_slot{slot}', history='exact', reference=HISTORIES[h],
                                    metric='score_auc', estimate=float(roc_auc_score(actual, score)),
                                    ci_low='', ci_high='', shuffle_sd=''))


def plot(rows, gates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = {'base': 'Chain', 'old': 'Old gate', 'new': 'Conv5 gate', 'shuffled': 'Shuffled memory'}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout='constrained')
    for ds, line in zip(DATASETS, axes):
        for ax, metric, name in zip(line, ('mean_gate', 'fisher', 'margin'), ('Exact gate', 'Exact vs same: Fisher (%)',
                                                                             'Exact vs same: SVM margin')):
            for method, label in labels.items():
                values = []
                for t in range(15):
                    if metric == 'mean_gate':
                        if method == 'base':
                            continue
                        match = [r for r in gates if r['dataset'] == ds and r['scope'] == 'fresh' and r['method'] == method
                                 and r['history'] == 'exact' and r['window'] == f't{t}' and r['metric'] == metric]
                    else:
                        match = [r for r in rows if r['dataset'] == ds and r['scope'] == 'fresh' and r['condition'] == method
                                 and r['reference'] == method and r['reference_history'] == 'same_concept'
                                 and r['window'] == f't{t}' and r['metric'] == metric]
                    assert len(match) == 1
                    values.append(match[0]['estimate'])
                if values:
                    ax.plot(range(15), values, label=label)
            ax.axvline(9, color='.65', linestyle=':', linewidth=1)
            ax.set(xlabel='Outer step', title=f'{ds} · {name}')
            ax.spines[['right', 'top']].set_visible(False)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.savefig(DOC / 'chain/familiarity_timecourse.png', dpi=150)
    plt.close(fig)


def main():
    reg = checked()
    cfg = frozen()['configs']['target_only']
    kernel = weights()
    absolute, contrasts, gates, predictions, state = [], [], [], {}, {}
    for dataset in DATASETS:
        bank, head, _ = fitted(dataset, kernel)
        cal = reg['datasets'][dataset]['calibration']
        for scope in SCOPES:
            data = archive(dataset, scope)
            paired = PairedConcepts(data['label'], data['concept'])
            row = dict(dataset=dataset, scope=scope)
            key = f'{dataset}_{scope}'
            gate_statistics(data, paired, row, cal, gates)
            predictions[key] = fc_statistics(data, bank, head, cfg, paired, row, absolute, contrasts)
            state[key] = dict(targets=len(data['label']), memory_norm=data['diagnostics'][:, :, 0].mean(0).tolist(),
                              fraction_active=data['diagnostics'][:, :, 1].mean(0).tolist())
            print('STATISTICS', dataset, scope, flush=True)
    write('absolute', absolute)
    write('contrasts', contrasts)
    write('gates', gates)
    save(DOC / 'chain/familiarity_predictions.json', predictions)
    save(DOC / 'chain/familiarity_state.json', state)
    plot(contrasts, gates)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
