"""Explore preview-supported control of the frozen object gain direction."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from priming_sharpen_collect_20261112 import digest
from priming_sharpen_confirm_20261112 import contrast as paired_contrast, metrics as paired_metrics
from priming_sharpen_object_precision_20261112 import features, fitted
from priming_sharpen_search_20261112 import DATASETS, DOC, OUT, contrast, control_gate, metrics, rank, read, save, shape, transform
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def choices():
    kinds = [('linear', 0.), ('product', 0.)]
    kinds.extend((mode, slope) for mode in ('overconfidence', 'corroboration') for slope in (2., 4., 6.))
    return [dict(mode=mode, slope=slope, weight=w, quantile=q, mixing=a, suppression=s)
            for (mode, slope), w, q, a, s in product(kinds, (.25, .5, .75), (.25, .5), (.3, .45, .6), (.003, .008))]


def direction(bank, x, q, cfg):
    target, preview = q['teacher'], q['preview_posterior']
    w = cfg['weight']
    if cfg['mode'] == 'product':
        pooled = np.maximum(target, 1e-8) ** (1 - w) * np.maximum(preview, 1e-8) ** w
        pooled /= pooled.sum(1, keepdims=True)
    else:
        pooled = (1 - w) * target + w * preview
    evidence = (pooled @ bank.bank.prototypes - bank.mean) / bank.std
    modified = shape(x * np.exp(.7 * np.tanh(evidence)))
    if cfg['mode'] in ('overconfidence', 'corroboration'):
        winner = target.argmax(1)
        support = preview[np.arange(len(x)), winner]
        excessive = np.maximum(target.max(1) - support, 0)
        reliability = np.clip(1 - cfg['slope'] * excessive, .05, 1.)
        if cfg['mode'] == 'corroboration':
            reliability *= np.where(winner == preview.argmax(1), 1., .35)
        modified = (1 - reliability[:, None]) * shape(x) + reliability[:, None] * modified
    return modified


def update(bank, x, q, cuts, cfg, d=None):
    g = control_gate(q['match'], cuts[cfg['quantile']])
    d = direction(bank, x, q, cfg) if d is None else d
    y = transform(x, d, g, cfg['mixing'], cfg['suppression']).astype(x.dtype)
    return y


def screen_one(x, p, y, bank, head, cuts, grid):
    queries = [bank.query(x[:, i], p[:, i]) for i in range(2)]
    base = metrics(x[:, 2], head.decision_function(x[:, 2]), y)
    cached = {}
    rows = []
    for c in grid:
        key = (c['mode'], c['slope'], c['weight'])
        if key not in cached:
            cached[key] = [direction(bank, x[:, i], queries[i], c) for i in range(2)]
        e, d = [metrics(v, head.decision_function(v), y) for v in
                (update(bank, x[:, i], queries[i], cuts, c, cached[key][i]) for i in range(2))]
        rows.append(dict(exact_v_none=contrast(e, base), exact_v_different=contrast(e, d)))
    return rows


def screen():
    kernel = weights()
    grid = choices()
    scores = {}
    for dataset in DATASETS:
        bank, head, cuts = fitted(dataset, kernel)
        x, p, y, meta = features(dataset, 'train')
        fold3 = np.array([r['fold'] == 3 for r in meta])
        cohorts = {'fold3': (x[fold3], p[fold3], y[fold3])}
        cohorts.update({scope: features(dataset, scope)[:3] for scope in ('reserve', 'replication', 'confirmation')})
        for scope, (x, p, y) in cohorts.items():
            scores[f'{dataset}_{scope}'] = screen_one(x, p, y, bank, head, cuts, grid)
            print('CORROBORATION', dataset, scope, len(grid), flush=True)
    ranking = [dict(index=i, config=c, **rank([r[i] for r in scores.values()])) for i, c in enumerate(grid)]
    ranking.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    path = DOC / 'chain/object_corroboration_exploration.json'
    save(path, dict(ranking=ranking, shortlisted={k: [v[r['index']] for r in ranking[:8]] for k, v in scores.items()},
                    note='Both previously verified image cohorts were used for exploratory choice. An image-disjoint prospective run is required.'))
    print('CORROBORATION BEST', ranking[:5], flush=True)


def freeze():
    source = DOC / 'chain/object_corroboration_exploration.json'
    row = next(x for x in read(source)['ranking'] if x['passed'])
    protocol = dict(config=row['config'], index=row['index'], exploration_sha256=digest(source),
                    scope='validation', comparison=['different', 'none'], t=9,
                    primary='positive suppression, Fisher, margin and AUC; nondecreasing BA in both datasets',
                    stimulus_sha256={ds: digest(OUT / ds / 'validation_manifest.json') for ds in DATASETS},
                    source_sha256=digest(Path(__file__)))
    path = OUT / 'chain_object_corroboration_frozen.json'
    if path.exists():
        assert read(path) == protocol
    else:
        save(path, protocol)
    print(protocol, flush=True)
    return protocol


def verify():
    protocol = freeze()
    cfg = protocol['config']
    kernel = weights()
    table = []
    for dataset in DATASETS:
        bank, head, cuts = fitted(dataset, kernel)
        x, p, labels, meta = features(dataset, 'validation')
        output = [update(bank, x[:, i], bank.query(x[:, i], p[:, i]), cuts, cfg) for i in range(2)]
        for i, y in enumerate(output):
            assert np.array_equal(y > 0, x[:, i] > 0)
            assert (y.mean(1) <= x[:, i].mean(1) + 1e-7).all()
        paired = PairedConcepts(labels, np.array([r['concept'] for r in meta]))
        e, d, none = [paired_metrics(v, head, paired) for v in (*output, x[:, 2])]
        for ref, baseline in (('different', d), ('none', none)):
            for name, delta in paired_contrast(e, baseline).items():
                row = dict(dataset=dataset, reference=ref, metric=name, **estimate(delta))
                table.append(row)
                print(row, flush=True)
    with (DOC / 'chain/object_corroboration_validation.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--verify', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        if args.freeze:
            freeze()
        elif args.verify:
            verify()
        else:
            screen()
