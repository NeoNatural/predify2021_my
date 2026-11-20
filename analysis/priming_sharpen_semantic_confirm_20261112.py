"""Evaluate the frozen semantic mechanism against both unprimed baselines."""

from __future__ import annotations

import argparse
import csv

from priming_sharpen_collect_20261112 import digest
from priming_sharpen_confirm_20261112 import contrast, metrics, read_features
from priming_sharpen_search_20261112 import Candidate, DATASETS, DOC, OUT, control_gate, read, save, transform
from priming_sharpen_teacher_20261112 import models, weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate
import numpy as np
from threadpoolctl import threadpool_limits


def states(bank, target, preview, cut, config):
    q = [bank.query(target[:, i], preview[:, i]) for i in range(2)]
    g = [control_gate(a['match'], cut) for a in q]
    x = [target[:, i] for i in range(3)]
    d = [bank.direction(x[i], q[i], config.mechanism, config.preview_weight) for i in range(2)]
    exact, different = [transform(x[i], d[i], g[i], config.mixing, config.suppression) for i in range(2)]
    same_mass = transform(x[0], d[0], g[0], 0., config.suppression)
    np.testing.assert_allclose(exact.mean(1), same_mass.mean(1), rtol=3e-6, atol=1e-8)
    assert np.array_equal(exact > 0, x[0] > 0)
    assert np.all(exact.mean(1) <= x[0].mean(1) + 1e-7)
    rng = np.random.default_rng(20261112)
    order = rng.permutation(len(x[0]))
    partner = np.empty_like(order)
    partner[order] = np.roll(order, 1)
    shuffled = bank.query(x[0], preview[partner, 0])
    g_random = control_gate(shuffled['match'], cut)
    baseline_direction = bank.direction(x[0], q[0], config.mechanism, 0.)
    cases = dict(exact=exact, different=different, none=x[2], raw_exact=x[0], raw_different=x[1], uniform=same_mass,
                 shuffled=transform(x[0], d[0], g_random, config.mixing, config.suppression),
                 constant_gate=transform(x[0], d[0], np.full(len(g[0]), g[0].mean()), config.mixing, config.suppression),
                 ridge_only=transform(x[0], baseline_direction, g[0], config.mixing, config.suppression))
    return {k: x.astype(target.dtype, copy=False) for k, x in cases.items()}, g


def confirm(dataset, source, scope, config, kernel):
    bank, head, cuts, _ = models(dataset, source, kernel)
    data = read_features(dataset, source, scope)
    s = PairedConcepts(data['y'], data['concept'])
    values, gates = states(bank, data['target_fc2'], data['preview_fc2'], cuts[config.quantile], config)
    outcomes = {k: metrics(x, head, s) for k, x in values.items()}
    result = []
    for test, ref in (('exact', 'none'), ('exact', 'different'), ('exact', 'raw_exact'), ('different', 'none'),
                      ('exact', 'uniform'), ('exact', 'shuffled'), ('exact', 'constant_gate'), ('exact', 'ridge_only'),
                      ('raw_exact', 'none'), ('raw_exact', 'raw_different')):
        for metric, delta in contrast(outcomes[test], outcomes[ref]).items():
            result.append(dict(dataset=dataset, source=source, scope=scope, test=test, reference=ref, metric=metric,
                               **estimate(delta)))
    info = dict(targets=len(data['y']), concepts=len(np.unique(data['concept'])), exact_gate=float(gates[0].mean()),
                different_gate=float(gates[1].mean()), fixed_head_classes=head.classes_.tolist(),
                absolute={k: {m: float(v[0]) for m, v in d.items()} for k, d in outcomes.items()})
    return result, info


def frozen_profile(scope):
    path = OUT / ('chain_lower_gain_frozen.json' if scope == 'replication' else 'vanilla_semantic_frozen.json')
    frozen = read(path)
    if scope == 'replication':
        assert digest(DOC / 'chain/guard_exploration.json') == frozen['exploration_sha256']
    return path, frozen, Candidate(**next(iter(frozen['candidates'].values()))['parameters'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=('vanilla', 'chain'), default='vanilla')
    p.add_argument('--scope', default='reserve')
    args = p.parse_args()
    frozen_path, frozen, c = frozen_profile(args.scope)
    kernel = weights()
    records, details = [], {}
    for dataset in DATASETS:
        rows, details[dataset] = confirm(dataset, args.source, args.scope, c, kernel)
        records.extend(rows)
        for row in rows:
            if row['test'] == 'exact' and row['reference'] in ('none', 'different') and row['metric'] in ('suppression', 'fisher', 'ba', 'auc', 'margin'):
                print(dataset, row['reference'], row['metric'],
                      *(round(row[k], 5) for k in ('estimate', 'ci_low', 'ci_high')), flush=True)
    dest = DOC / args.source
    save(dest / f'semantic_{args.scope}_details.json', dict(frozen=frozen, frozen_sha256=digest(frozen_path), datasets=details))
    with (dest / f'semantic_{args.scope}_contrasts.csv').open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(records[0]))
        out.writeheader()
        out.writerows(records)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
