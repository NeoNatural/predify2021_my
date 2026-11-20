"""Test target-defined sharpening gated by genuine preview matches."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from itertools import product
from pathlib import Path

from priming_sharpen_search_20261112 import (Association, Candidate, DATASETS, DOC, OUT, build, control_gate, evaluate,
                                            inputs, rank, read, save, shape, transform)
from priming_sharpen_confirm_20261112 import contrast, metrics, read_features
from priming_sharpen_collect_20261112 import digest
from report_hebb_post_pool_20261111 import PairedConcepts, estimate
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits


class TargetDirection:
    def __init__(self, bank):
        self.bank = bank
        self.prototype_mean = bank.prototypes.mean(0)
        self.prototype_scale = np.maximum(bank.prototypes.std(0), .15)

    def query(self, x, p):
        return self.bank.query(x, p)

    def direction(self, x, q, kind, strength):
        if kind == 'exemplar':
            return softmax(strength * q['item_target'], axis=-1) @ self.bank.values
        if kind == 'class':
            return softmax(strength * q['target_class'], axis=-1) @ self.bank.prototypes
        probabilities = softmax(10 * q['target_class'], axis=-1)
        evidence = probabilities @ self.bank.prototypes - self.prototype_mean
        if kind == 'residual':
            return shape(np.maximum(shape(x) + strength * evidence, 0))
        if kind == 'gain':
            return shape(x * np.exp(strength * np.tanh(evidence / self.prototype_scale)))
        if kind == 'power':
            return shape(np.power(shape(x), strength))
        raise ValueError(kind)


def grid():
    strength = {'exemplar': (5., 12., 24.), 'class': (5., 10., 20.), 'residual': (.5, 1., 2.),
                'gain': (.25, .65, 1.25), 'power': (1.3, 2.)}
    return [Candidate(kind, q, a, s, w) for kind, levels in strength.items() for q, a, s, w in
            product((.25, .5, .75, .9), (.06, .12, .24, .4, .6), (.005, .015, .035), levels)]


def screen(source):
    candidates = grid()
    all_rows = {}
    aux = {}
    for dataset in DATASETS:
        x, p, y, folds, slots = inputs(dataset, source, 'fc2')
        bank, head, cuts, training = build(x, p, y, folds, slots)
        selected = folds == 3
        all_rows[dataset], details = evaluate(x[selected], p[selected], y[selected], TargetDirection(bank), head, cuts, candidates)
        aux[dataset] = dict(training=training, fold3=details)
        print('GATED SCREEN', source, dataset, len(candidates), 'match AUC', round(details['auc'], 4), flush=True)
    rows = [dict(index=i, parameters=asdict(c), **rank([all_rows[d][i] for d in DATASETS]))
            for i, c in enumerate(candidates)]
    rows.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    passed = [r for r in rows if r['passed']]
    chosen = {}
    for family in ('exemplar', 'class', 'residual', 'gain', 'power'):
        candidate = next((r for r in passed if r['parameters']['mechanism'] == family), None)
        if candidate:
            chosen[family] = candidate
    frozen = dict(source=source, candidates=dict(sorted(chosen.items(), key=lambda kv: kv[1]['score'], reverse=True)[:2]),
                  fit=[1, 2], screen=3, candidates_searched=len(candidates), passed=len(passed),
                  primary='positive suppression/Fisher/margin/AUC and BA nondecreasing vs Different and no preview',
                  scope='secondary fold4', code_sha256=digest(Path(__file__)))
    path = OUT / f'{source}_gated_frozen.json'
    if path.exists():
        assert read(path) == frozen
    else:
        save(path, frozen)
    save(DOC / source / 'gated_screen.json', dict(frozen=frozen, diagnostics=aux, ranking=rows, records=all_rows))
    print('GATED FROZEN', source, frozen['candidates'], flush=True)
    return frozen


def confirm(source, scope, frozen):
    if not frozen['candidates']:
        raise RuntimeError('No passing screening candidate')
    all_rows, details = [], {}
    for dataset in DATASETS:
        x, p, y, fold, slots = inputs(dataset, source, 'fc2')
        bank, head, cuts, _ = build(x, p, y, fold, slots)
        direction = TargetDirection(bank)
        z = read_features(dataset, source, scope)
        s = PairedConcepts(z['y'], z['concept'])
        target, preview = z['target_fc2'], z['preview_fc2']
        x = [target[:, i] for i in range(3)]
        q = [direction.query(x[i], preview[:, i]) for i in range(2)]
        rng = np.random.default_rng(20261112)
        order = rng.permutation(len(target))
        partner = np.empty_like(order)
        partner[order] = np.roll(order, 1)
        shuffle = direction.query(x[0], preview[partner, 0])
        details[dataset] = {}
        for family, row in frozen['candidates'].items():
            c = Candidate(**row['parameters'])
            gates = [control_gate(a['match'], cuts[c.quantile]) for a in q]
            vectors = [direction.direction(t, a, c.mechanism, c.preview_weight) for t, a in zip(x, q)]
            paired = [transform(x[i], vectors[i], gates[i], c.mixing, c.suppression) for i in range(2)]
            uniform = transform(x[0], vectors[0], gates[0], 0., c.suppression)
            np.testing.assert_allclose(uniform.mean(1), paired[0].mean(1), rtol=3e-6, atol=1e-8)
            null = transform(x[0], vectors[0], control_gate(shuffle['match'], cuts[c.quantile]), c.mixing, c.suppression)
            constant = transform(x[0], vectors[0], np.full(len(target), gates[0].mean()), c.mixing, c.suppression)
            reps = dict(exact=paired[0], different=paired[1], none=x[2], raw_exact=x[0], raw_different=x[1],
                        uniform=uniform, shuffled=null, constant_gate=constant)
            stats = {k: metrics(v, head, s) for k, v in reps.items()}
            details[dataset][family] = dict(n=len(x[0]), means=[float(g.mean()) for g in gates],
                       absolute={k: {m: float(v[0]) for m, v in t.items()} for k, t in stats.items()})
            for test, ref in (('exact', 'none'), ('exact', 'different'), ('exact', 'raw_exact'), ('different', 'none'),
                              ('exact', 'uniform'), ('exact', 'shuffled'), ('exact', 'constant_gate'),
                              ('raw_exact', 'none'), ('raw_exact', 'raw_different')):
                for metric, values in contrast(stats[test], stats[ref]).items():
                    row = dict(dataset=dataset, source=source, scope=scope, family=family, test=test, reference=ref,
                               metric=metric, **estimate(values))
                    all_rows.append(row)
                    if test == 'exact' and ref in ('none', 'different') and metric in ('suppression', 'fisher', 'ba', 'auc', 'margin'):
                        print(dataset, family, ref, metric, *(round(row[k], 5) for k in ('estimate', 'ci_low', 'ci_high')), flush=True)
    dest = DOC / source
    save(dest / f'gated_{scope}_details.json', dict(frozen=frozen, datasets=details))
    with (dest / f'gated_{scope}_contrasts.csv').open('w', encoding='utf-8', newline='') as f:
        csvout = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        csvout.writeheader()
        csvout.writerows(all_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='vanilla')
    p.add_argument('--scope', choices=('confirm', 'secondary'), default='secondary')
    p.add_argument('--confirm', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        if args.confirm:
            frozen = read(OUT / f'{args.source}_gated_frozen.json')
            confirm(args.source, args.scope, frozen)
        else:
            screen(args.source)


if __name__ == '__main__':
    main()
