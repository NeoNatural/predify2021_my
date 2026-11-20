"""Require a candidate to survive multiple exploration splits before a new reservation."""

from __future__ import annotations

from dataclasses import asdict, replace

from priming_sharpen_confirm_20261112 import read_features
from priming_sharpen_gated_20261112 import TargetDirection, grid
from priming_sharpen_search_20261112 import DATASETS, DOC, OUT, build, evaluate, inputs, rank, save, shape
import numpy as np
from threadpoolctl import threadpool_limits


class ConfidenceDirection(TargetDirection):
    def direction(self, x, q, kind, strength):
        name, _, modifier = kind.partition('_')
        baseline = super().direction(x, q, name, strength)
        if not modifier:
            return baseline
        class_scores = q['target_class']
        top_two = np.sort(class_scores, axis=1)[:, -2:]
        separation = np.diff(top_two, axis=1)
        if modifier == 'mild':
            strength = .40 + .60 * np.clip(separation / .60, 0, 1)
        elif modifier == 'strict':
            strength = .12 + .88 * np.clip(separation / .60, 0, 1)
        else:
            raise ValueError(modifier)
        return (1 - strength) * shape(x) + strength * baseline


def candidates():
    base = grid()
    focused = [c for c in base if c.mechanism in ('class', 'exemplar', 'residual', 'gain')
               and c.quantile in (.25, .5, .75) and c.suppression in (.005, .015) and c.mixing in (.24, .4, .6)]
    return [*base, *[replace(c, mechanism=f'{c.mechanism}_{mode}') for c in focused for mode in ('mild', 'strict')]]


def screen():
    source = 'vanilla'
    grid = candidates()
    records = {}
    diag = {}
    for dataset in DATASETS:
        x, p, y, fold, slots = inputs(dataset, source, 'fc2')
        bank, head, cuts, _ = build(x, p, y, fold, slots)
        bank = ConfidenceDirection(bank)
        mask = fold == 3
        data = {'fold3': (x[mask], p[mask], y[mask])}
        for scope in ('confirm', 'secondary'):
            z = read_features(dataset, source, scope)
            data[scope] = (z['target_fc2'], z['preview_fc2'], z['y'])
        for split, rows in data.items():
            key = f'{dataset}_{split}'
            records[key], diag[key] = evaluate(*rows, bank, head, cuts, grid)
            print('ROBUST SEARCH', key, len(grid), flush=True)
    result = [dict(index=i, parameters=asdict(c), **rank([records[key][i] for key in records])) for i, c in enumerate(grid)]
    result.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    finalists = {}
    for row in result:
        if not row['passed']:
            break
        base = row['parameters']['mechanism'].split('_')[0]
        finalists.setdefault(base, row)
    frozen = dict(candidates=dict(list(finalists.items())[:2]), fitting=[1, 2],
                  exploration=['fold3', 'first half of fold0 from previous prospective trial', 'fold4'],
                  reserved='remaining half of formal fold0, previously used only by the earlier Hebbian mechanism',
                  grid=len(grid), count_passing=sum(r['passed'] for r in result),
                  selected_effects={k: {g: rows[r['index']] for g, rows in records.items()} for k, r in finalists.items()})
    save(DOC / source / 'robust_screen.json', dict(ranking=result, diagnostics=diag, frozen=frozen))
    if frozen['candidates']:
        path = OUT / 'vanilla_robust_frozen.json'
        if path.exists():
            raise FileExistsError(path)
        save(path, frozen)
    print('ROBUST CANDIDATES', frozen['count_passing'], result[:4], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        screen()
