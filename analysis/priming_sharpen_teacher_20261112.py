"""Test whether independent pretrained semantic evidence preserves the FC readout."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product

from priming_sharpen_search_20261112 import Candidate, DATASETS, DOC, OUT, ROOT, build, evaluate, inputs, rank, read, save, shape
from priming_sharpen_confirm_20261112 import read_features
from priming_sharpen_collect_20261112 import CHECKPOINT
from predify_experiments.paper_stimuli import imagenet_supercategory
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch


class SemanticEvidence:
    def __init__(self, bank, dataset, weights, x, y):
        self.bank = bank
        self.w, self.b = weights
        self.mean = bank.prototypes.mean(0)
        self.std = np.maximum(bank.prototypes.std(0), .15)
        categories = read(ROOT / 'docs/conv_fc_transfer_20261106' / f'{dataset}_manifest.json')['categories']
        canonical = [imagenet_supercategory(i) for i in range(len(self.b))]
        self.columns = [np.where(np.array(canonical, dtype=object) == cat)[0] for cat in categories]
        assert all(len(c) for c in self.columns)
        self.teacher = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000))
        self.teacher.fit(self.native_scores(x), y)

    def native_scores(self, x):
        logits = x @ self.w.T + self.b
        return np.stack([logsumexp(logits[:, col], axis=1) - np.log(len(col)) for col in self.columns], 1)

    def query(self, x, p):
        q = self.bank.query(x, p)
        q['teacher'] = self.teacher.predict_proba(self.native_scores(x))
        return q

    def direction(self, x, q, mechanism, weight):
        ridge = softmax(10 * q['target_class'], axis=1)
        teacher = q['teacher']
        probabilities = (1 - weight) * ridge + weight * teacher
        prototype = probabilities @ self.bank.prototypes
        evidence = prototype - self.mean
        if mechanism == 'semantic_residual':
            return shape(np.maximum(shape(x) + evidence, 0))
        if mechanism == 'semantic_gain':
            return shape(x * np.exp(.7 * np.tanh(evidence / self.std)))
        if mechanism == 'semantic_prototype':
            return prototype
        if mechanism == 'semantic_consensus':
            soft = np.sqrt(ridge * teacher)
            soft = soft / np.maximum(soft.sum(1, keepdims=True), 1e-12)
            confidence = .25 + .75 * (ridge.argmax(1) == teacher.argmax(1))[:, None]
            d = (soft @ self.bank.prototypes - self.mean) * confidence
            return shape(np.maximum(shape(x) + weight * d, 0))
        raise ValueError(mechanism)


def weights():
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    return tuple(state[f'classifier.6.{k}'].numpy().copy() for k in ('weight', 'bias'))


def models(dataset, source, frozen_weights):
    training = inputs(dataset, source, 'fc2')
    x, p, y, folds, slots = training
    bank, head, cuts, audit = build(*training)
    use = (folds == 1) | (folds == 2)
    return SemanticEvidence(bank, dataset, frozen_weights, x[use, 2], y[use]), head, cuts, (x, p, y, folds)


def grid():
    kinds = ('semantic_residual', 'semantic_gain', 'semantic_prototype', 'semantic_consensus')
    for kind, cut, mix, budget, evidence in product(kinds, (.25, .5, .75, .9), (.12, .24, .4, .6), (.003, .008, .015), (.25, .5, .75, 1.)):
        yield Candidate(kind, cut, mix, budget, evidence)


def main():
    source = 'vanilla'
    frozen_weights = weights()
    choices = list(grid())
    results = {}
    for dataset in DATASETS:
        bank, head, cuts, (x, p, y, fold) = models(dataset, source, frozen_weights)
        test = fold == 3
        splits = {'fold3': (x[test], p[test], y[test])}
        for scope in ('confirm', 'secondary'):
            z = read_features(dataset, source, scope)
            splits[scope] = (z['target_fc2'], z['preview_fc2'], z['y'])
        for key, (x, p, y) in splits.items():
            post = bank.teacher.predict_proba(bank.native_scores(x[:, 2]))
            print('TEACHER', dataset, key, 'coarse accuracy', float(np.mean(post.argmax(1) == y)), flush=True)
            results[dataset + '_' + key], _ = evaluate(x, p, y, bank, head, cuts, choices)
    summary = [dict(index=i, parameters=asdict(c), **rank([r[i] for r in results.values()])) for i, c in enumerate(choices)]
    summary.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    options = [r for r in summary if r['passed']]
    frozen = dict(candidates={r['parameters']['mechanism']: r for r in options[:1]},
                  fits=[1, 2], examined=['fold3', 'fold0_half', 'fold4'],
                  future_test='new concepts in all visual roles', grid=len(choices), passing=len(options))
    save(DOC / 'vanilla' / 'semantic_screen.json', dict(frozen=frozen, ranking=summary,
         selected={k: [rows[o['index']] for o in options[:5]] for k, rows in results.items()}))
    if options:
        save(OUT / 'vanilla_semantic_frozen.json', frozen)
    print('SEMANTIC', frozen, summary[:4], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
