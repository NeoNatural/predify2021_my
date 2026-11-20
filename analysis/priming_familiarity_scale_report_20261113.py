"""Summarize training gaps and held-out paired learning curves."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from priming_familiarity_report_20261112 import paired_auc
from priming_familiarity_scale_data_20261113 import DATASETS, LEVELS, MANIFEST, SEEDS
from priming_familiarity_scale_fit_20261113 import ARMS, PROTOCOL, SELECTION, protocol, run_path, weights_for
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import DOC, read, save
from report_hebb_post_pool_20261111 import estimate


OUT = DOC / 'learning_curve_20261113'
SPLITS = ('train', 'validation', 'test')
NEGATIVES = ('same_concept', 'different')


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{name}.csv'
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def empirical_auc(v, w):
    return [paired_auc(v[:, 0], v[:, i], w) for i in (1, 2)]


def cached_result(path, model_sha):
    info = read(path.with_suffix('.json'))
    raw = path.with_suffix('.npz')
    test = path.with_suffix('.test.npz')
    test_audit = read(path.with_suffix('.test.json'))
    assert info['model_sha256'] == model_sha == test_audit['model_sha256'] == digest(path.with_suffix('.pth'))
    assert test_audit['selection_sha256'] == digest(SELECTION) and digest(test) == test_audit['score_sha256']
    assert info['protocol_sha256'] == digest(PROTOCOL) and digest(raw) == info['score_sha256']
    with np.load(raw) as a, np.load(test) as b:
        return info, {'train': a['train'], 'validation': a['validation'], 'test': b['scores'],
                      'local_validation': a['local_validation'], 'local_test': b['local']}


def row_summary(values):
    return estimate(values.mean(0)) | {'seed_sd': float(values[:, 0].std(ddof=1)),
                                       'seed_min': float(values[:, 0].min()),
                                       'seed_max': float(values[:, 0].max())}


def summarize(ds, plan, selected, summaries, growth, individual):
    views = {}
    samples = plan['splits']
    stable = {}
    val_test_weights = {s: weights_for(samples[s]) for s in SPLITS[1:]}
    for level in LEVELS:
        training = [r for r in samples['train'] if r['rank'] < level]
        weights = {'train': weights_for(training), **val_test_weights}
        for arm in ARMS:
            records = []
            for seed in SEEDS:
                path = run_path(ds, arm, level, seed)
                info, scores = cached_result(path, selected['model_sha256'][ds][path.name])
                assert info['n'] == len(training)
                d = {}
                for split, data in scores.items():
                    code = split.removeprefix('local_')
                    d[split] = empirical_auc(data, weights[code])
                    for name, values in zip(NEGATIVES, d[split]):
                        individual.append(dict(dataset=ds, arm=arm, level=level, train_n=len(training),
                                               seed=seed, split=split, negative=name, auc=values[0],
                                               best_epoch=info['best_epoch'], stopped=info['stopped'],
                                               trainable=info['trainable']))
                records.append(d)
            for split in records[0]:
                for i, negative in enumerate(NEGATIVES):
                    v = np.stack([r[split][i] for r in records])
                    views[arm, level, split, negative] = v
                    summaries.append(dict(dataset=ds, arm=arm, level=level, train_n=len(training), split=split,
                                          negative=negative, **row_summary(v)))
    for arm in ARMS:
        for level, ref in ((LEVELS[1], LEVELS[0]), (LEVELS[2], LEVELS[0]), (LEVELS[2], LEVELS[1])):
            for split in ('validation', 'test', 'local_validation', 'local_test'):
                for negative in NEGATIVES:
                    delta = views[arm, level, split, negative] - views[arm, ref, split, negative]
                    growth.append(dict(dataset=ds, arm=arm, reference_arm=arm, level=level, reference_level=ref,
                                       split=split, negative=negative, **row_summary(delta)))
    for level in LEVELS:
        for split in ('validation', 'test', 'local_validation', 'local_test'):
            for negative in NEGATIVES:
                a, b = (views[arm, level, split, negative] for arm in ARMS)
                growth.append(dict(dataset=ds, arm=ARMS[0], reference_arm=ARMS[1], level=level, reference_level=level,
                                   split=split, negative=negative, **row_summary(a - b)))
    # Test the size selected on validation against the same arm's small-size baseline.
    choice = selected['chosen'][ds]
    arm, level = choice['arm'], choice['level']
    score = views[arm, level, 'test', 'same_concept']
    difference = score - views[arm, LEVELS[0], 'test', 'same_concept']
    confidence = row_summary(difference)
    stable = dict(arm=arm, level=level, test_auc=row_summary(score), improvement_over_small=confidence,
                  evidence_of_growth=bool(level > LEVELS[0] and confidence['ci_low'] > 0 and row_summary(score)['ci_low'] > .5))
    return stable


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.5))
    marks = {'train': ('#6C6C6C', ':'), 'validation': ('#296493', '--'), 'test': ('#B24D20', '-')}
    for ds, axes_row in zip(DATASETS, axes):
        for arm, ax in zip(ARMS, axes_row):
            for split, (color, line) in marks.items():
                results = [r for r in rows if r['dataset'] == ds and r['arm'] == arm and r['split'] == split
                           and r['negative'] == 'same_concept']
                x = [r['train_n'] for r in results]
                y = [r['estimate'] for r in results]
                ax.plot(x, y, color=color, linestyle=line, marker='o')
                if split == 'test':
                    ax.fill_between(x, [r['ci_low'] for r in results], [r['ci_high'] for r in results],
                                    color=color, alpha=.12, linewidth=0)
            ax.axhline(.5, color='.75', linewidth=.8)
            ax.set(xticks=x, ylim=(.40, 1.01), title=f'{ds.upper() if ds == "things" else "ImageNet"} · {arm.replace("_", " ")}')
            ax.spines[['right', 'top']].set_visible(False)
        axes_row[0].set_ylabel('Exact vs same concept AUC')
    fig.legend(handles=[Line2D([], [], color=color, linestyle=line, label=s.capitalize())
                        for s, (color, line) in marks.items()], loc='upper center', ncol=3, frameon=False)
    fig.supxlabel('Distinct training images', y=.02)
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(OUT / 'learning_curve.png', dpi=170)
    plt.close(fig)


def main():
    protocol()
    selection = read(SELECTION)
    assert selection['protocol_sha256'] == digest(PROTOCOL)
    data = read(MANIFEST)
    summaries, growth, individual = [], [], []
    support = {ds: summarize(ds, data[ds], selection, summaries, growth, individual) for ds in DATASETS}
    write('learning_curve', summaries)
    write('paired_changes', growth)
    write('runs', individual)
    save(OUT / 'heldout_summary.json', support)
    plot(summaries)
    print(support, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
