"""Paired t9 differences between real, cleared, shuffled and original gates."""

import csv

import numpy as np
from threadpoolctl import threadpool_limits

from priming_familiarity_20261112 import checked
from priming_familiarity_report_20261112 import paired_auc, write
from priming_familiarity_verify_20261112 import SCOPES, archive
from priming_sharpen_same_concept_data_20261112 import HISTORIES
from priming_sharpen_search_20261112 import DATASETS, DOC
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    folder = DOC / 'chain'
    with (folder / 'familiarity_contrasts.csv').open(newline='', encoding='utf-8') as f:
        outcomes = {(r['dataset'], r['condition'], r['metric'], r['window']): float(r['estimate'])
                    for r in csv.DictReader(f) if r['scope'] == 'fresh' and r['condition'] == r['reference']
                    and r['reference_history'] == 'same_concept'}
    with (folder / 'familiarity_gates.csv').open(newline='', encoding='utf-8') as f:
        gates = {(r['dataset'], r['method'], 'mean_gate', r['window']): float(r['estimate'])
                 for r in csv.DictReader(f) if r['scope'] == 'fresh' and r['history'] == 'exact'
                 and r['metric'] == 'mean_gate' and not r['reference']}
    curves = {'base': ('Four-layer baseline', '#555555', '--'), 'old': ('Original gate', '#1260A0', '-'),
              'new': ('Conv5 familiarity gate', '#B74D12', '-'), 'shuffled': ('Shuffled memory', '#547E73', ':')}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    steps = range(15)
    for ds, line in zip(DATASETS, axes):
        for ax, metric, title in zip(line, ('mean_gate', 'fisher', 'margin'),
                                     ('Exact gate', 'Exact vs same concept · Fisher (%)',
                                      'Exact vs same concept · SVM margin')):
            for method, (_, color, style) in curves.items():
                if metric == 'mean_gate' and method == 'base':
                    continue
                data = gates if metric == 'mean_gate' else outcomes
                values = [data[ds, method, metric, f't{t}'] for t in steps]
                ax.plot(steps, values, color=color, linestyle=style, linewidth=1.8)
            ax.axvline(9, color='.65', linestyle=':', linewidth=1)
            ax.set(title=title, xticks=(0, 3, 6, 9, 12, 14))
            ax.spines[['right', 'top']].set_visible(False)
        line[0].set_ylabel('ImageNet' if ds == 'imagenet' else 'THINGS')
    legend = [Line2D([], [], color=color, linestyle=style, label=label)
              for label, color, style in curves.values()]
    fig.legend(handles=legend, ncols=4, loc='upper center', bbox_to_anchor=(.5, .99), frameon=False)
    fig.supxlabel('Outer step', y=.02)
    fig.tight_layout(rect=(0, .04, 1, .92))
    fig.savefig(folder / 'familiarity_timecourse.png', dpi=150)
    plt.close(fig)


def main():
    checked()
    rows = []
    for dataset in DATASETS:
        for scope in SCOPES:
            d = archive(dataset, scope)
            weights = PairedConcepts(d['label'], d['concept']).concept_weights
            for kind in ('score', 'gate'):
                real = d[f'{kind}_new'][..., 9]
                old = d[f'{kind}_old'][..., 9]
                shuffled = d[f'{kind}_shuffled'][..., 9].transpose(2, 0, 1)
                for h in (1, 2):
                    original_auc = paired_auc(old[:, 0], old[:, h], weights)
                    real_auc = paired_auc(real[:, 0], real[:, h], weights)
                    scrambled_auc = np.stack([paired_auc(v[:, 0], v[:, h], weights) for v in shuffled])
                    comparisons = {'old': real_auc - original_auc, 'clear': real_auc - .5,
                                   'shuffled': real_auc - scrambled_auc.mean(0)}
                    for ref, change in comparisons.items():
                        rows.append(dict(dataset=dataset, scope=scope, metric=f'{kind}_auc',
                                         comparison=f'new_minus_{ref}', negative=HISTORIES[h], **estimate(change)))
    write('memory_contrasts', rows)
    plot()


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
