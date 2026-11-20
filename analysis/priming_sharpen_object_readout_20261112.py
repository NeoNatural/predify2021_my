"""Evaluate a single common readout fitted on cross-fitted object-memory trajectories."""

from __future__ import annotations

import argparse
import csv

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_object_memory_20261112 import ObjectEvidence
from priming_sharpen_object_precision_20261112 import adjusted, features, fitted
from priming_sharpen_search_20261112 import Association, Candidate, DATASETS, DOC, OUT, read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def shared(dataset, kernel, cuts, cfg):
    x, p, y, rows = features(dataset, 'train')
    folds = np.array([r['fold'] for r in rows])
    slot = np.array([r['slot'] for r in rows])
    feats, labels = [], []
    for training_fold, scoring_fold in ((1, 2), (2, 1)):
        train = np.flatnonzero(folds == training_fold)
        score = folds == scoring_fold
        x0, y0 = x[train, 2], y[train]
        bank = Association(x0, p[train, 0, slot[train]], y0)
        semantic = ObjectEvidence(bank, dataset, kernel, x0, y0)
        feats.extend(adjusted(semantic, x[score, i], p[score, i], cuts, cfg) for i in range(2))
        feats.append(x[score, 2])
        labels.extend([y[score]] * 3)
    model = make_pipeline(StandardScaler(), LinearSVC(C=.03, dual='auto', max_iter=30000, random_state=20261112))
    model.fit(np.concatenate(feats), np.concatenate(labels))
    return model


def evaluate(scope, kernel, cfg):
    output = []
    details = {}
    for ds in DATASETS:
        bank, fixed, cuts = fitted(ds, kernel)
        refit = shared(ds, kernel, cuts, cfg)
        x, p, y, rows = features(ds, scope)
        e, d = [adjusted(bank, x[:, i], p[:, i], cuts, cfg) for i in range(2)]
        conditions = dict(exact=e, different=d, none=x[:, 2], raw_exact=x[:, 0], raw_different=x[:, 1])
        samples = PairedConcepts(y, np.array([r['concept'] for r in rows]))
        details[ds] = {}
        for name, model in (('fixed', fixed), ('shared_refit', refit)):
            results = {k: metrics(v, model, samples) for k, v in conditions.items()}
            correct = {k: model.predict(v) == y for k, v in conditions.items()}
            details[ds][name] = dict(correct={k: int(v.sum()) for k, v in correct.items()},
                                     exact_vs_none_recovered=[rows[i]['dataset_index'] for i in np.where(correct['exact'] & ~correct['none'])[0]],
                                     exact_vs_none_lost=[rows[i]['dataset_index'] for i in np.where(~correct['exact'] & correct['none'])[0]],
                                     raw_vs_none_recovered=[rows[i]['dataset_index'] for i in np.where(correct['raw_exact'] & ~correct['none'])[0]],
                                     raw_vs_none_lost=[rows[i]['dataset_index'] for i in np.where(~correct['raw_exact'] & correct['none'])[0]])
            for case, ref in (('exact', 'none'), ('exact', 'different'), ('exact', 'raw_exact'), ('raw_exact', 'none')):
                for key, diff in contrast(results[case], results[ref]).items():
                    if key in ('ba', 'auc', 'margin'):
                        row = dict(dataset=ds, head=name, case=case, reference=ref, metric=key, **estimate(diff))
                        output.append(row)
                        if case == 'exact' and ref in ('none', 'different'):
                            print(scope, row, flush=True)
    save(DOC / 'chain' / f'object_memory_{scope}_readout_details.json', details)
    with (DOC / 'chain' / f'object_memory_{scope}_readouts.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('replication', 'confirmation'), default='confirmation')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        frozen = read(OUT / 'chain_object_memory_frozen.json')
        evaluate(args.scope, weights(), Candidate(**frozen['choice']['parameters']))
