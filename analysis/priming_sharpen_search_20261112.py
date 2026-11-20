"""Compare preview retrieval and shape suppression on frozen native FC representations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import product
import json
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'

import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'Hebbian-CNN/Log/priming_sharpen_20261112'
DOC = ROOT / 'docs/priming_sharpen_20261112'
DATASETS = ('imagenet', 'things')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def shape(x):
    return x / np.maximum(x.mean(-1, keepdims=True), 1e-8)


class Ridge:
    def __init__(self, x, y, strength):
        self.center = x.mean(0)
        self.offset = y.mean(0)
        z = unit(x - self.center)
        self.dual = np.linalg.solve(z @ z.T + strength * np.eye(len(z), dtype=z.dtype), y - self.offset)
        self.keys = z

    def __call__(self, x):
        return (unit(x - self.center) @ self.keys.T) @ self.dual + self.offset


@dataclass(frozen=True)
class Candidate:
    mechanism: str
    quantile: float
    mixing: float
    suppression: float
    preview_weight: float


class Association:
    def __init__(self, x, p, y, strength=.18):
        self.target_center = shape(x).mean(0)
        self.preview_center = shape(p).mean(0)
        self.target_keys = unit(shape(x) - self.target_center)
        self.preview_keys = unit(shape(p) - self.preview_center)
        self.values = shape(x)
        self.estimate = Ridge(shape(p), self.target_keys, strength)
        self.align = Ridge(shape(p), self.values, strength)
        labels = np.eye(int(y.max()) + 1, dtype=np.float32)[y]
        self.preview_class = Ridge(shape(p), labels, strength)
        self.target_class = Ridge(shape(x), labels, strength)
        self.prototypes = (labels.T @ self.values) / labels.sum(0)[:, None]

    def similarity(self, x, p):
        target = unit(shape(x) - self.target_center)
        prediction = unit(self.estimate(shape(p)))
        return (target[:, None, :] * prediction).sum(-1)

    def query(self, x, p):
        score = self.similarity(x, p)
        attention = softmax(18 * score, axis=1)
        xp, pp = shape(x), shape(p)
        frame_class = self.preview_class(pp)
        prior_class = np.einsum('bf,bfc->bc', attention, frame_class)
        target_class = self.target_class(xp)
        item_target = unit(xp - self.target_center) @ self.target_keys.T
        item_preview = np.einsum('bf,bfn->bn', attention, unit(pp - self.preview_center) @ self.preview_keys.T)
        projected = (attention[:, :, None] * self.align(pp)).sum(1)
        projected = shape(np.maximum(projected, 0))
        return dict(match=score.max(1), slot=score.argmax(1), target_class=target_class, prior_class=prior_class,
                    item_target=item_target, item_preview=item_preview, aligned=projected)

    def direction(self, x, query, kind, weight):
        if kind == 'power':
            return shape(np.power(shape(x), weight))
        if kind == 'aligned':
            return query['aligned']
        if kind == 'exemplar':
            logits = 12 * (query['item_target'] + weight * query['item_preview']) / (1 + weight)
            return softmax(logits, axis=-1) @ self.values
        if kind == 'class':
            logits = 9 * (query['target_class'] + weight * query['prior_class']) / (1 + weight)
            return softmax(logits, axis=-1) @ self.prototypes
        raise ValueError(kind)


def control_gate(scores, cut):
    return np.clip((scores - cut) / .20, 0., 1.)


def transform(x, q, gate, mixing, suppression):
    a = mixing * gate[:, None]
    mass = x.mean(1, keepdims=True) * (1 - suppression * gate[:, None])
    return mass * ((1 - a) * shape(x) + a * q)


def metrics(x, scores, y):
    v = unit(x.astype(np.float64))
    classes = np.unique(y)
    centers = np.stack([v[y == c].mean(0) for c in classes])
    within = np.mean([np.square(v[y == c] - centers[i]).sum(1).mean() for i, c in enumerate(classes)])
    between = np.square(centers - centers.mean(0)).sum(1).mean()
    margin = scores[np.arange(len(y)), y] - np.where(np.eye(len(classes), dtype=bool)[y], -np.inf, scores).max(1)
    auc = np.mean([roc_auc_score(y == c, scores[:, c]) for c in classes])
    return dict(mean=float(x.mean()), within=float(within), between=float(between), fisher=float(between / within),
                ba=float(balanced_accuracy_score(y, scores.argmax(1))), margin=float(margin.mean()), auc=float(auc))


def contrast(a, b):
    return dict(suppression=100 * (1 - a['mean'] / b['mean']), fisher=100 * (a['fisher'] / b['fisher'] - 1),
                margin=a['margin'] - b['margin'], ba=100 * (a['ba'] - b['ba']), auc=100 * (a['auc'] - b['auc']))


def inputs(dataset, source, layer):
    prior = ROOT / 'Hebbian-CNN/Log/fc_preview_readout_refit_20261105' / dataset / 'inputs/features.npz'
    if source == 'vanilla':
        with np.load(prior) as f:
            target, preview, y, row = (f[key] for key in ('target_' + layer, 'preview_' + layer, 'y', 'row'))
        target = np.stack((target,) * 3, axis=1)
    else:
        with np.load(OUT / dataset / f'{source}.npz') as f:
            target, preview, y, row = (f[key] for key in ('target_' + layer, 'preview_' + layer, 'y', 'row'))
    manifest = read(ROOT / 'docs/conv_fc_transfer_20261106' / f'{dataset}_manifest.json')
    lookup = {t['dataset_index']: t for t in manifest['training']}
    keep = np.array([r in lookup for r in row])
    fold = np.array([lookup[r]['fold'] for r in row[keep]])
    slots = np.array([lookup[r]['slot'] for r in row[keep]])
    if preview.shape[2] == 2:
        slots //= 2
    assert np.array_equal(y[keep], [lookup[r]['category_index'] for r in row[keep]])
    assert all(len(np.unique(fold[np.isin(row[keep], [t['dataset_index'] for t in manifest['training'] if t['concept'] == c])])) == 1
               for c in {t['concept'] for t in manifest['training']})
    return target[keep], preview[keep], y[keep], fold, slots


def build(target, preview, y, fold, slots):
    masks = [fold == i for i in (1, 2)]
    fit = masks[0] | masks[1]
    paired = preview[np.arange(len(preview)), 0, slots]
    diff_scores = []
    exact_scores = []
    for train, test in (masks, masks[::-1]):
        bank = Association(target[train, 2], paired[train], y[train])
        exact_scores.extend(bank.similarity(target[test, 0], preview[test, 0]).max(1))
        diff_scores.extend(bank.similarity(target[test, 1], preview[test, 1]).max(1))
    bank = Association(target[fit, 2], paired[fit], y[fit])
    head = make_pipeline(StandardScaler(), LinearSVC(C=.03, dual='auto', max_iter=30000, random_state=20261112))
    head.fit(target[fit, 2], y[fit])
    cuts = {q: float(np.quantile(diff_scores, q)) for q in (.25, .5, .75, .9)}
    return bank, head, cuts, dict(fit=int(fit.sum()), crossfit_match_auc=float(roc_auc_score(
        np.repeat([1, 0], [len(exact_scores), len(diff_scores)]), np.r_[exact_scores, diff_scores])), cuts=cuts)


def candidates():
    for mechanism, weights in (('power', (1.3, 2.)), ('aligned', (0.,)), ('exemplar', (.3, 1., 2.)),
                               ('class', (.3, 1., 2., 4.))):
        for q, a, s, w in product((.25, .5, .75, .9), (.12, .3, .55, .8), (.015, .045, .09), weights):
            yield Candidate(mechanism, q, a, s, w)


def evaluate(target, preview, y, bank, head, cuts, grid):
    x = [target[:, i] for i in range(3)]
    q = [bank.query(x[i], preview[:, i]) for i in range(2)]
    anchor = metrics(x[2], head.decision_function(x[2]), y)
    raw = [metrics(z, head.decision_function(z), y) for z in x]
    vectors = {}
    results = []
    for c in grid:
        key = (c.mechanism, c.preview_weight)
        if key not in vectors:
            vectors[key] = [bank.direction(z, qq, *key) for z, qq in zip(x, q)]
        z = [transform(x[i], vectors[key][i], control_gate(q[i]['match'], cuts[c.quantile]), c.mixing, c.suppression)
             for i in range(2)]
        e, d = [metrics(v, head.decision_function(v), y) for v in z]
        results.append(dict(parameters=asdict(c), exact=e, different=d, none=anchor, exact_v_none=contrast(e, anchor),
                            exact_v_different=contrast(e, d), different_v_none=contrast(d, anchor),
                            exact_v_raw=contrast(e, raw[0])))
    return results, dict(raw=raw, slot=[qq['slot'].tolist() for qq in q], match=[qq['match'].tolist() for qq in q],
                         auc=float(roc_auc_score(np.repeat([1, 0], len(y)), np.r_[q[0]['match'], q[1]['match']])))


def rank(rows):
    deltas = [row[key] for row in rows for key in ('exact_v_none', 'exact_v_different')]
    minima = {k: min(d[k] for d in deltas) for k in ('suppression', 'fisher', 'margin', 'ba', 'auc')}
    passed = min(minima[k] for k in ('suppression', 'fisher', 'margin')) > 0 and minima['ba'] >= 0 and minima['auc'] > 0
    objective = min(minima['suppression'] / .2, minima['fisher'] / 2, minima['margin'] / .05,
                    minima['ba'] / 2 + .5, minima['auc'] / 2)
    return dict(passed=passed, score=float(objective), minimum=minima)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='vanilla')
    p.add_argument('--layer', choices=('fc1', 'fc2'), default='fc2')
    args = p.parse_args()
    grid = list(candidates())
    records = {}
    details = {}
    with threadpool_limits(limits=4):
        for dataset in DATASETS:
            x, v, y, folds, slots = inputs(dataset, args.source, args.layer)
            bank, head, cuts, detail = build(x, v, y, folds, slots)
            test = folds == 3
            records[dataset], aux = evaluate(x[test], v[test], y[test], bank, head, cuts, grid)
            details[dataset] = dict(**detail, fold3=aux, exact_slot_accuracy=float(np.mean(np.array(aux['slot'][0]) == slots[test])))
            print(dataset, 'tested', len(records[dataset]), 'cases', 'matching AUC', round(aux['auc'], 4),
                  'exact slot', round(details[dataset]['exact_slot_accuracy'], 4), flush=True)
    ranking = [dict(index=i, parameters=asdict(c), **rank([records[d][i] for d in DATASETS])) for i, c in enumerate(grid)]
    ranking.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    folder = DOC / args.source
    save(folder / f'{args.layer}_screen.json', dict(protocol=dict(fit=[1, 2], screen=3, grid_count=len(grid),
         primary='Exact vs Different and No preview; positive suppression/Fisher/margin/AUC, BA nondecreasing'),
         summaries=details, ranking=ranking, records=records))
    with (folder / f'{args.layer}_ranking.csv').open('w', encoding='utf-8', newline='') as f:
        fields = ('index', 'mechanism', 'quantile', 'mixing', 'suppression', 'preview_weight', 'passed', 'score',
                  'min_suppression', 'min_fisher', 'min_margin', 'min_ba', 'min_auc')
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        for row in ranking:
            out.writerow(dict(index=row['index'], **row['parameters'], passed=row['passed'], score=row['score'],
                              **{'min_' + k: v for k, v in row['minimum'].items()}))
    for r in ranking[:5]:
        print(r, flush=True)


if __name__ == '__main__':
    main()
