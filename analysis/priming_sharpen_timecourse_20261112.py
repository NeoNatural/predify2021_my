"""Summarize frozen shaping effects across all 15 actual Predify target steps."""

from __future__ import annotations

import argparse
import csv

from priming_sharpen_confirm_20261112 import contrast as paired_contrast, metrics as paired_metrics, read_features
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import Candidate, DATASETS, DOC, OUT, contrast, control_gate, metrics, read, transform
from priming_sharpen_semantic_confirm_20261112 import frozen_profile
from priming_sharpen_teacher_20261112 import models, weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


def timecourse(dataset, kernel, config, scope='reserve', object_memory=False):
    data = read_features(dataset, 'chain', scope)
    x, y = data['trajectory_fc2'], data['y']
    if object_memory:
        from priming_sharpen_object_memory_20261112 import data as object_data, fitted
        bank, head, cuts, _, _ = fitted(dataset, kernel)
        _, p, object_labels, _ = object_data(dataset, scope)
        np.testing.assert_array_equal(y, object_labels)
    else:
        bank, head, cuts, _ = models(dataset, 'chain', kernel)
        p = data['preview_fc2']
    records = []
    late = {k: [] for k in ('exact', 'different', 'none')}
    for t in range(x.shape[2]):
        raw = [x[:, j, t] for j in range(3)]
        adjusted = []
        for i in range(2):
            q = bank.query(raw[i], p[:, i])
            g = control_gate(q['match'], cuts[config.quantile])
            d = bank.direction(raw[i], q, config.mechanism, config.preview_weight)
            adjusted.append(transform(raw[i], d, g, config.mixing, config.suppression).astype(x.dtype))
        banked = {'exact': adjusted[0], 'different': adjusted[1], 'none': raw[2], 'raw_exact': raw[0], 'raw_different': raw[1]}
        if t >= 5:
            for k in late:
                late[k].append(banked[k])
        values = {k: metrics(a, head.decision_function(a), y) for k, a in banked.items()}
        for case, baseline in (('exact', 'different'), ('exact', 'none'), ('raw_exact', 'raw_different'), ('raw_exact', 'none')):
            a, b = values[case], values[baseline]
            records.append(dict(dataset=dataset, t=t, case=case, baseline=baseline, **contrast(a, b)))
    return records, pooled(dataset, data, late)


def pooled(dataset, data, late):
    folder = OUT / dataset
    train = folder / 'chain.npz'
    meta = read(folder / 'chain_manifest.json')
    assert digest(train) == meta['feature_sha256']
    folds = np.array([row['fold'] for row in meta['trials']])
    use = (folds == 1) | (folds == 2)
    head = make_pipeline(StandardScaler(), LinearSVC(C=.03, dual='auto', max_iter=30000, random_state=20261112))
    with np.load(train) as z:
        head.fit(z['trajectory_fc2'][use, 2, 5:].mean(1), z['y'][use])
    s = PairedConcepts(data['y'], data['concept'])
    stats = {k: paired_metrics(np.mean(frames, axis=0), head, s) for k, frames in late.items()}
    return [dict(dataset=dataset, case='exact', baseline=baseline, metric=metric, **estimate(delta))
            for baseline in ('different', 'none') for metric, delta in paired_contrast(stats['exact'], stats[baseline]).items()]


def draw(path, rows, treatment='Semantic'):
    names = [('suppression', 'Mean activity reduction (%)'), ('fisher', 'Fisher change (%)'),
             ('auc', 'Macro AUC change (pp)'), ('margin', 'True-class margin change')]
    styles = [('exact', 'none', '#007671', '-', f'{treatment} · no preview'),
              ('exact', 'different', '#405eaa', '-', f'{treatment} · different'),
              ('raw_exact', 'none', '#6b747c', '--', 'Hebbian chain · no preview'),
              ('raw_exact', 'raw_different', '#bd7663', ':', 'Hebbian chain · different')]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, layout='constrained')
    for i, dataset in enumerate(DATASETS):
        for j, (metric, title) in enumerate(names):
            ax = axes[i, j]
            ax.axhline(0, color='#d6dce0', lw=.8)
            ax.axvline(9, color='#d6dce0', lw=.8)
            for case, baseline, color, ls, label in styles:
                use = [r for r in rows if r['dataset'] == dataset and r['case'] == case and r['baseline'] == baseline]
                ax.plot([r['t'] for r in use], [r[metric] for r in use], color=color, ls=ls, label=label, lw=1.6)
            ax.set_title(title, fontsize=10)
            ax.spines[['right', 'top']].set_visible(False)
            ax.set_xticks([0, 4, 9, 14])
            if i == 1:
                ax.set_xlabel('Target step')
            if j == 0:
                ax.set_ylabel('ImageNet' if i == 0 else 'THINGS')
    axes[0, 1].legend(frameon=False, fontsize=7, loc='best')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=('reserve', 'replication', 'confirmation'), default='reserve')
    p.add_argument('--object-memory', action='store_true')
    args = p.parse_args()
    if args.object_memory:
        if args.scope == 'reserve':
            p.error('--object-memory requires a replication or confirmation cohort')
        cfg = Candidate(**read(OUT / 'chain_object_memory_frozen.json')['choice']['parameters'])
        prefix = f'object_memory_{args.scope}'
    else:
        if args.scope == 'confirmation':
            p.error('The confirmation cohort has an object-memory protocol')
        _, _, cfg = frozen_profile(args.scope)
        prefix = 'semantic' if args.scope == 'reserve' else 'semantic_replication'
    kernel = weights()
    both = [timecourse(dataset, kernel, cfg, args.scope, args.object_memory) for dataset in DATASETS]
    rows = [r for steps, _ in both for r in steps]
    pooling = [r for _, paired in both for r in paired]
    folder = DOC / 'chain'
    with (folder / f'{prefix}_timecourse.csv').open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    with (folder / f'{prefix}_pooled_readout.csv').open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=list(pooling[0]))
        out.writeheader()
        out.writerows(pooling)
    draw(folder / f'{prefix}_timecourse.png', rows, 'Object memory' if args.object_memory else 'Semantic')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
