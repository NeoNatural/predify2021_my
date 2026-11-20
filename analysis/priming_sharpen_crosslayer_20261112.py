"""Explore and prospectively test FC1 evidence as a supplement to FC2 semantic gain."""

from __future__ import annotations

import argparse
import csv
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from priming_sharpen_collect_20261112 import digest
from priming_sharpen_confirm_20261112 import contrast as paired_contrast, metrics as paired_metrics, read_features
from priming_sharpen_search_20261112 import DATASETS, DOC, OUT, contrast, control_gate, inputs, metrics, rank, read, save, shape, transform
from priming_sharpen_teacher_20261112 import models, weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def load_models(dataset, kernel):
    bank, fixed, cuts, (x2, previews, labels, folds) = models(dataset, 'chain', kernel)
    x1, _, y1, folds1, _ = inputs(dataset, 'chain', 'fc1')
    assert np.array_equal(labels, y1) and np.array_equal(folds, folds1)
    fit = (folds == 1) | (folds == 2)
    auxiliary = make_pipeline(StandardScaler(), LogisticRegression(C=.03, max_iter=1000, random_state=20261112))
    auxiliary.fit(shape(x1[fit, 2]), labels[fit])
    return bank, fixed, cuts, auxiliary, (x1, x2, previews, labels, folds)


def parameters():
    names = ('mode', 'weight', 'quantile', 'mixing', 'suppression')
    return [dict(zip(names, values, strict=True)) for values in product(
        ('linear', 'product', 'agreement'), (.25, .5, .75), (.25, .5, .75), (.15, .3, .45, .6), (.003, .008, .015))]


def cache(bank, aux, x1, x2, previews):
    return [(bank.query(x2[:, i], previews[:, i]), aux.predict_proba(shape(x1[:, i]))) for i in range(2)]


def directions(bank, x, teacher, auxiliary, mode, weight):
    if mode == 'product':
        p = np.maximum(teacher, 1e-10) ** (1 - weight) * np.maximum(auxiliary, 1e-10) ** weight
        p /= p.sum(1, keepdims=True)
    else:
        p = (1 - weight) * teacher + weight * auxiliary
    evidence = (p @ bank.bank.prototypes - bank.mean) / bank.std
    reshaped = shape(x * np.exp(.7 * np.tanh(evidence)))
    if mode == 'agreement':
        use = .35 + .65 * (teacher.argmax(1) == auxiliary.argmax(1))[:, None]
        reshaped = (1 - use) * shape(x) + use * reshaped
    return reshaped


def intervention(x2, queries, bank, cuts, cfg, directions_cache):
    adjusted = []
    for i, (q, p) in enumerate(queries):
        key = (i, cfg['mode'], cfg['weight'])
        if key not in directions_cache:
            directions_cache[key] = directions(bank, x2[:, i], q['teacher'], p, cfg['mode'], cfg['weight'])
        g = control_gate(q['match'], cuts[cfg['quantile']])
        adjusted.append(transform(x2[:, i], directions_cache[key], g, cfg['mixing'], cfg['suppression']).astype(x2.dtype))
    return adjusted


def evaluate(fitted, data, grid):
    bank, fixed, cuts, aux, _ = fitted
    x1, x2, p, y = data
    q = cache(bank, aux, x1, x2, p)
    d = {}
    baseline = metrics(x2[:, 2], fixed.decision_function(x2[:, 2]), y)
    results = []
    for cfg in grid:
        e, v = [metrics(z, fixed.decision_function(z), y) for z in intervention(x2, q, bank, cuts, cfg, d)]
        results.append(dict(exact_v_none=contrast(e, baseline), exact_v_different=contrast(e, v)))
    return results


def screen():
    grid = parameters()
    kernel = weights()
    records = {}
    for ds in DATASETS:
        fitted = load_models(ds, kernel)
        x1, x2, p, y, folds = fitted[-1]
        use = folds == 3
        external = read_features(ds, 'chain', 'reserve')
        records[ds + '_fold3'] = evaluate(fitted, (x1[use], x2[use], p[use], y[use]), grid)
        records[ds + '_reserve'] = evaluate(fitted, tuple(external[k] for k in ('target_fc1', 'target_fc2', 'preview_fc2', 'y')), grid)
        print('CROSS LAYER', ds, len(grid), flush=True)
    ranked = [dict(index=i, config=c, **rank([v[i] for v in records.values()])) for i, c in enumerate(grid)]
    ranked.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    report = DOC / 'chain/crosslayer_exploration.json'
    save(report, dict(tested=len(grid), ranking=ranked, datasets={k: [v[r['index']] for r in ranked[:5]] for k, v in records.items()}))
    if ranked[0]['passed']:
        path = OUT / 'chain_crosslayer_frozen.json'
        frozen = dict(choice=ranked[0], screen_sha256=digest(report),
                      replication_manifests={d: digest(OUT / d / 'replication_manifest.json') for d in DATASETS})
        if path.exists():
            assert read(path) == frozen
        else:
            save(path, frozen)
    print(ranked[:3], flush=True)


def confirm():
    frozen = read(OUT / 'chain_crosslayer_frozen.json')
    assert digest(DOC / 'chain/crosslayer_exploration.json') == frozen['screen_sha256']
    cfg = frozen['choice']['config']
    kernel = weights()
    rows = []
    for ds in DATASETS:
        assert digest(OUT / ds / 'replication_manifest.json') == frozen['replication_manifests'][ds]
        bank, fixed, cuts, aux, _ = load_models(ds, kernel)
        z = read_features(ds, 'chain', 'replication')
        x2 = z['target_fc2']
        q = cache(bank, aux, z['target_fc1'], x2, z['preview_fc2'])
        a, b = intervention(x2, q, bank, cuts, cfg, {})
        for i, adjusted in enumerate((a, b)):
            assert np.array_equal(adjusted > 0, x2[:, i] > 0)
            assert np.all(adjusted.mean(1) <= x2[:, i].mean(1) + 1e-7)
        samples = PairedConcepts(z['y'], z['concept'])
        exact, different, none = [paired_metrics(v, fixed, samples) for v in (a, b, x2[:, 2])]
        for ref, v in (('different', different), ('none', none)):
            for metric, change in paired_contrast(exact, v).items():
                row = dict(dataset=ds, reference=ref, metric=metric, **estimate(change))
                rows.append(row)
                print(row, flush=True)
    with (DOC / 'chain/crosslayer_replication.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirm', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        confirm() if args.confirm else screen()
