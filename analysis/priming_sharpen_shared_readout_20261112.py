"""Compare frozen and shared refitted readouts on identical reserve activations."""

from __future__ import annotations

import argparse
import csv

from priming_sharpen_confirm_20261112 import contrast, metrics, read_features
from priming_sharpen_search_20261112 import Association, DATASETS, DOC, inputs
from priming_sharpen_semantic_confirm_20261112 import frozen_profile, states
from priming_sharpen_teacher_20261112 import SemanticEvidence, models, weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


def train_shared(dataset, source, kernel, cut, cfg):
    target, preview, labels, fold, slots = inputs(dataset, source, 'fc2')
    tr = []
    y = []
    for other, own in ((1, 2), (2, 1)):
        fit = fold == other
        test = fold == own
        x, p = target[fit, 2], preview[np.where(fit)[0], 0, slots[fit]]
        bank = Association(x, p, labels[fit])
        semantic = SemanticEvidence(bank, dataset, kernel, x, labels[fit])
        values, _ = states(semantic, target[test], preview[test], cut, cfg)
        tr.extend(values[history] for history in ('exact', 'different', 'none'))
        y.extend([labels[test]] * 3)
    head = make_pipeline(StandardScaler(), LinearSVC(C=.03, dual='auto', max_iter=30000, random_state=20261112))
    head.fit(np.concatenate(tr), np.concatenate(y))
    return head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=('vanilla', 'chain'), default='vanilla')
    p.add_argument('--scope', choices=('reserve', 'replication'), default='reserve')
    args = p.parse_args()
    _, _, cfg = frozen_profile(args.scope)
    kernel = weights()
    rows = []
    for dataset in DATASETS:
        bank, head, cuts, _ = models(dataset, args.source, kernel)
        data = read_features(dataset, args.source, args.scope)
        samples = PairedConcepts(data['y'], data['concept'])
        cut = cuts[cfg.quantile]
        shared = train_shared(dataset, args.source, kernel, cut, cfg)
        values, _ = states(bank, data['target_fc2'], data['preview_fc2'], cut, cfg)
        for name, evaluator in (('fixed', head), ('shared_refit', shared)):
            m = {key: metrics(values[key], evaluator, samples) for key in ('exact', 'different', 'none')}
            for baseline in ('different', 'none'):
                for metric, delta in contrast(m['exact'], m[baseline]).items():
                    if metric in ('ba', 'auc', 'margin'):
                        rows.append(dict(dataset=dataset, source=args.source, head=name, baseline=baseline, metric=metric,
                                         **estimate(delta)))
                        print(rows[-1], flush=True)
    name = 'semantic_readout_comparison.csv' if args.scope == 'reserve' else 'semantic_replication_readouts.csv'
    with (DOC / args.source / name).open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
