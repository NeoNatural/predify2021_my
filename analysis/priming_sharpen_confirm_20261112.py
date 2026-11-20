"""Freeze the screened mechanisms and quantify independent paired sharpening."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace

from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import (Candidate, DATASETS, DOC, OUT, ROOT, build, control_gate, inputs, read, save,
                                            transform, unit)
import numpy as np
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def freeze(source):
    path = DOC / source / 'fc2_screen.json'
    grid = read(path)['ranking']
    selected = {}
    for mechanism in ('class', 'exemplar'):
        row = next((r for r in grid if r['passed'] and r['parameters']['mechanism'] == mechanism), None)
        if row is not None:
            selected[mechanism] = dict(index=row['index'], **row['parameters'])
    if not selected:
        raise RuntimeError('No training-side candidate meets the directional joint criterion')
    frozen = dict(source=source, screen_sha256=digest(path), candidates=selected, fit_folds=[1, 2], selection_fold=3,
                  test='fold0 excluding target concepts from the prior chain confirmation',
                  readout='fixed independent StandardScaler + LinearSVC fitted on the untreated target FC2 from folds1/2',
                  primary='Exact vs both Different and no preview; direction of mean/Fisher/margin/AUC positive, BA nondecreasing',
                  interval='4000 paired concept draws; descriptive percentile intervals; eight primary paired contrasts')
    dest = OUT / f'{source}_frozen.json'
    if dest.exists():
        assert read(dest) == frozen, 'Frozen candidate changed'
    else:
        save(dest, frozen)
    return frozen


def read_features(dataset, source, scope):
    folder = OUT / dataset
    path = folder / f'{source}_{scope}.npz'
    audit = read(folder / f'{source}_{scope}_manifest.json')
    assert digest(path) == audit['feature_sha256']
    if 'stimulus_manifest_sha256' in audit:
        assert digest(folder / f'{scope}_manifest.json') == audit['stimulus_manifest_sha256']
    with np.load(path) as z:
        values = {k: z[k] for k in z.files}
    assert np.array_equal(values['row'], [r['dataset_index'] for r in audit['trials']])
    return values


def run(x, preview, bank, cut, candidate):
    q = bank.query(x, preview)
    gate = control_gate(q['match'], cut)
    d = bank.direction(x, q, candidate.mechanism, candidate.preview_weight)
    return transform(x, d, gate, candidate.mixing, candidate.suppression), q, gate


def metrics(x, head, s):
    u = unit(x.astype(np.float64))
    geom = s.geometry((u @ u.T)[None], equal_class=True)
    result = {k: v[:, 0] for k, v in geom.items()}
    result['mean'] = s.concept_weights @ x.mean(1)
    result.update(s.readout(head.decision_function(x)[:, None], fc=True))
    return result


def contrast(a, b):
    return dict(suppression=100 * (1 - a['mean'] / b['mean']),
                within=100 * (a['within'] / b['within'] - 1), between=100 * (a['between'] / b['between'] - 1),
                fisher=100 * (a['fisher'] / b['fisher'] - 1), margin=a['margin'] - b['margin'],
                ba=100 * (a['ba'] - b['ba']), auc=100 * (a['auc'] - b['auc']))


def evaluate(dataset, source, scope, candidates):
    training = inputs(dataset, source, 'fc2')
    bank, head, cuts, _ = build(*training)
    z = read_features(dataset, source, scope)
    x, p = z['target_fc2'], z['preview_fc2']
    s = PairedConcepts(z['y'], z['concept'], 4000)
    raw = [metrics(x[:, i], head, s) for i in range(3)]
    rng = np.random.default_rng(20261112)
    permutation = rng.permutation(len(x))
    partner = np.empty_like(permutation)
    partner[permutation] = np.roll(permutation, 1)
    records, info = [], {}
    for family, spec in candidates.items():
        c = Candidate(**{k: v for k, v in spec.items() if k != 'index'})
        cut = cuts[c.quantile]
        exact, q, gate = run(x[:, 0], p[:, 0], bank, cut, c)
        different, qd, gate_d = run(x[:, 1], p[:, 1], bank, cut, c)
        direction = bank.direction(x[:, 0], q, c.mechanism, c.preview_weight)
        uniform = transform(x[:, 0], direction, gate, 0, c.suppression)
        target_only = transform(x[:, 0], bank.direction(x[:, 0], q, c.mechanism, 0), gate, c.mixing, c.suppression)
        shuffled, qs, _ = run(x[:, 0], p[partner, 0], bank, cut, c)
        shuffled_same_gate = transform(x[:, 0], bank.direction(x[:, 0], qs, c.mechanism, c.preview_weight),
                                       gate, c.mixing, c.suppression)
        np.testing.assert_allclose(exact.mean(1), uniform.mean(1), rtol=3e-6, atol=1e-8)
        np.testing.assert_allclose(exact.mean(1), shuffled_same_gate.mean(1), rtol=3e-6, atol=1e-8)
        stats = dict(exact=metrics(exact, head, s), different=metrics(different, head, s), none=raw[2], raw_exact=raw[0],
                     uniform=metrics(uniform, head, s), target_only=metrics(target_only, head, s),
                     shuffled=metrics(shuffled, head, s), shuffled_same_gate=metrics(shuffled_same_gate, head, s),
                     raw_different=raw[1])
        for test, ref in (('exact', 'none'), ('exact', 'different'), ('exact', 'raw_exact'), ('different', 'none'),
                          ('exact', 'uniform'), ('exact', 'target_only'), ('exact', 'shuffled'),
                          ('exact', 'shuffled_same_gate'), ('raw_exact', 'none'), ('raw_exact', 'raw_different')):
            for metric, delta in contrast(stats[test], stats[ref]).items():
                records.append(dict(dataset=dataset, source=source, scope=scope, family=family, test=test,
                                    reference=ref, metric=metric, **estimate(delta)))
        info[family] = dict(n=len(x), concepts=len(np.unique(z['concept'])), cut=cut,
                            exact_gate=float(gate.mean()), different_gate=float(gate_d.mean()),
                            exact_match_slot_accuracy=float((q['slot'] == z['slot']).mean()),
                            exact_match=float(q['match'].mean()), different_match=float(qd['match'].mean()),
                            readouts={k: {metric: float(v[0]) for metric, v in d.items()} for k, d in stats.items()})
    return records, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='vanilla')
    parser.add_argument('--scope', choices=('confirm', 'secondary'), default='confirm')
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    fixed = freeze(args.source)
    print('FROZEN', fixed['candidates'], flush=True)
    if args.freeze_only:
        return
    records, details = [], {}
    for dataset in DATASETS:
        rows, details[dataset] = evaluate(dataset, args.source, args.scope, fixed['candidates'])
        records.extend(rows)
        for row in rows:
            if row['test'] == 'exact' and row['reference'] in ('none', 'different') and row['metric'] in ('suppression', 'fisher', 'margin', 'ba', 'auc'):
                print(row['dataset'], row['family'], row['reference'], row['metric'],
                      *(round(row[k], 5) for k in ('estimate', 'ci_low', 'ci_high')), flush=True)
    dest = DOC / args.source
    save(dest / f'{args.scope}_details.json', dict(frozen=fixed, datasets=details))
    with (dest / f'{args.scope}_contrasts.csv').open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(records[0]))
        out.writeheader()
        out.writerows(records)


if __name__ == '__main__':
    main()
