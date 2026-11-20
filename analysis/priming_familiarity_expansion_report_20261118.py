"""Report paired familiarity AUCs, within-category effects and training gaps."""
import argparse
import csv

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from priming_familiarity_20261112 import Familiarity
from priming_familiarity_expansion_data_20261118 import CONDITIONS, DATASETS, DOC, MANIFEST, ROOT, register
from priming_familiarity_expansion_fit_20261118 import baseline_path, evaluate, load_features, model_path
from priming_familiarity_scale_data_20261113 import SEEDS
from priming_familiarity_scale_report_20261113 import row_summary
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import read
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def weights(rows):
    return PairedConcepts(np.array([r['label'] for r in rows]), np.array([r['concept'] for r in rows])).concept_weights


def auc(exact, negative, weights):
    order = np.argsort(negative, kind='stable')
    negative = negative[order]
    left = np.searchsorted(negative, exact, side='left')
    right = np.searchsorted(negative, exact, side='right')
    sums = np.cumsum(weights[:, order], axis=1)
    sums = np.pad(sums, ((0, 0), (1, 0)))
    return (.5 * (sums[:, left] + sums[:, right]) * weights).sum(axis=1)


def write(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_scores(dataset, condition, seed, sets):
    folder = ROOT / dataset
    tag = f'{condition}_{seed}'
    path = baseline_path(dataset, seed) if condition == 'A' else folder / tag
    with np.load(path.with_suffix('.npz')) as file:
        scores = {split: file[split] for split in ('train', 'validation')}
    for split in ('legacy', 'fresh'):
        if split in sets:
            with np.load(folder / f'{tag}.{split}.npz') as file:
                np.testing.assert_array_equal(file['index'], [r['index'] for r in sets[split]])
                assert file['selection_sha256'].item() == digest(folder / 'selection.json')
                scores[split] = file['scores']
    for split, value in scores.items():
        assert value.shape == (len(sets[split]), 3) and np.isfinite(value).all()
    return scores


def summarize(dataset, plan, baseline, preliminary):
    overall, per_seed, categories = [], [], []
    sets = {s: baseline['splits'][s] for s in ('train', 'validation')}
    if not preliminary:
        sets['legacy'] = baseline['splits']['test']
        if min(plan['fresh_coverage'].values()) >= 2:
            sets['fresh'] = plan['fresh']
    conditions = CONDITIONS[:3] if preliminary else CONDITIONS
    comparisons = [(cond, 'A') for cond in conditions[1:]] + [('B', 'C')]
    samples = {s: weights(rows) for s, rows in sets.items()}
    # Keep all comparisons paired within a split. Draw the disjoint training and
    # validation sets independently when estimating the uncertainty in their gap.
    rng = np.random.default_rng(2026111818)
    valid = samples['validation']
    samples['validation'] = valid[np.r_[0, rng.permutation(np.arange(1, len(valid)))]]
    values, group_values = {}, {}
    names = baseline['categories']
    groups = {split: [(name, ids, weights([rows[i] for i in ids])) for group, name in enumerate(names)
                      if len(ids := [i for i, row in enumerate(rows) if row['label'] == group]) > 0]
              for split, rows in sets.items() if split != 'train'}
    for condition in conditions:
        scores = [load_scores(dataset, condition, seed, sets) for seed in SEEDS]
        for i, negative in enumerate(('same_concept', 'different'), 1):
            for split, w in samples.items():
                values[condition, split, negative] = np.stack([auc(s[split][:, 0], s[split][:, i], w) for s in scores])
                for name, ids, group_weights in groups.get(split, []):
                    group_values[condition, split, negative, name] = np.stack(
                        [auc(s[split][ids, 0], s[split][ids, i], group_weights) for s in scores])
            values[condition, 'gap', negative] = (
                values[condition, 'train', negative] - values[condition, 'validation', negative])

    def emit(condition, reference, split, negative, values):
        row = dict(dataset=dataset, condition=condition, reference=reference, split=split, negative=negative)
        overall.append(dict(**row, **row_summary(values)))
        for seed, points in zip(SEEDS, values, strict=True):
            per_seed.append(dict(**row, seed=seed, **estimate(points)))

    for condition in conditions:
        for split in (*sets, 'gap'):
            for negative in ('same_concept', 'different'):
                emit(condition, '', split, negative, values[condition, split, negative])
    for condition, reference in comparisons:
        for split in (*sets, 'gap'):
            for negative in ('same_concept', 'different'):
                emit(condition, reference, split, negative, values[condition, split, negative] - values[reference, split, negative])
                for name, _, _ in groups.get(split, []):
                    a = group_values[condition, split, negative, name]
                    b = group_values[reference, split, negative, name]
                    categories.append(dict(dataset=dataset, condition=condition, reference=reference, split=split,
                                           negative=negative, category=name, **row_summary(a-b)))
    return overall, per_seed, categories


def query_check(dataset, plan, baseline):
    folder = ROOT / dataset
    selection = read(folder / 'selection.json')['models']
    sets = {'legacy': baseline['splits']['test'], 'fresh': plan['fresh']}
    comparisons = [(cond, 'A') for cond in CONDITIONS[1:]] + [('B', 'C')]
    table = []
    for split, rows in sets.items():
        if split == 'fresh' and min(plan['fresh_coverage'].values()) < 2:
            continue
        inputs = load_features(dataset, split, rows, legacy=split == 'legacy')
        w = weights(rows)
        views = {}
        for condition in CONDITIONS:
            for seed in SEEDS:
                tag = f'{condition}_{seed}'
                path = model_path(dataset, condition, seed)
                assert selection[tag] == digest(path)
                model = Familiarity().cuda().eval()
                model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
                shared = evaluate(model, inputs)
                with np.load(folder / f'{tag}.{split}.npz') as file:
                    native = file['scores']
                    assert file['selection_sha256'].item() == digest(folder / 'selection.json')
                views[condition, seed] = np.stack([auc(s[:, 0], s[:, 1], w) for s in (shared, native)])
        for condition, reference in comparisons:
            effects = np.stack([views[condition, seed] - views[reference, seed] for seed in SEEDS])
            for query, values in [('shared', effects[:, 0]), ('native', effects[:, 1]),
                                  ('native_minus_shared', effects[:, 1] - effects[:, 0])]:
                table.append(dict(dataset=dataset, split=split, condition=condition, reference=reference,
                                  query=query, **row_summary(values)))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--preliminary', action='store_true')
    parser.add_argument('--query-check', action='store_true')
    args = parser.parse_args()
    base, plans = read(MANIFEST), register()
    chosen = (args.dataset,) if args.dataset else DATASETS
    if args.query_check:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        rows = [row for ds in chosen for row in query_check(ds, plans[ds], base[ds])]
        suffix = f'_{args.dataset}' if args.dataset else ''
        write(DOC / f'query_check{suffix}.csv', rows)
        return
    all_rows = [summarize(ds, plans[ds], base[ds], args.preliminary) for ds in chosen]
    suffix = ''.join(f'_{label}' for label in (args.dataset, 'ABC' if args.preliminary else None) if label)
    DOC.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(('summary', 'per_seed', 'by_category'), zip(*all_rows), strict=True):
        write(DOC / f'{name}{suffix}.csv', [row for group in rows for row in group])
    for row in (r for group in all_rows for r in group[0]):
        if row['negative'] == 'same_concept' and row['reference'] and row['split'] != 'train':
            print(row['dataset'], row['condition'] + '-' + row['reference'], row['split'],
                  *(round(100 * row[k], 2) for k in ('estimate', 'ci_low', 'ci_high')), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
