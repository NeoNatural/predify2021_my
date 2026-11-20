"""Explore label-free decision guards after a semantic FC2 update."""

from __future__ import annotations

from itertools import product

from priming_sharpen_collect_20261112 import digest
from priming_sharpen_confirm_20261112 import read_features
from priming_sharpen_search_20261112 import DATASETS, DOC, OUT, contrast, control_gate, metrics, rank, read, save, transform
from priming_sharpen_teacher_20261112 import models, weights
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


def grid():
    for q, a, s, gate, (mode, retain) in product((.25, .5, .75), (.30, .45, .60), (.003, .008),
           ('linear', 'sqrt', 'binary'), (('none', 1.), ('any_flip', 0.), ('any_flip', .25),
                                        ('unsupported_flip', 0.), ('unsupported_flip', .25))):
        yield dict(quantile=q, mix=a, suppression=s, gate=gate, guard=mode, retain=retain)


def fit_guards(x, y, fold):
    result = []
    for i in (1, 2):
        model = make_pipeline(StandardScaler(), LinearSVC(C=.03, dual='auto', max_iter=30000, random_state=20261112))
        model.fit(x[fold == i], y[fold == i])
        result.append(model)
    return result


def evaluate(dataset, data, bank, head, guards, cuts, choices):
    target, preview, y = data
    x = [target[:, i] for i in range(3)]
    q = [bank.query(x[i], preview[:, i]) for i in range(2)]
    direction = [bank.direction(x[i], q[i], 'semantic_gain', 1.) for i in range(2)]
    initial = [np.stack([g.predict(a) for g in guards]) for a in x[:2]]
    reference = metrics(x[2], head.decision_function(x[2]), y)
    result = []
    cache = {}
    for cfg in choices:
        key = tuple(cfg[k] for k in ('quantile', 'mix', 'suppression', 'gate'))
        if key not in cache:
            cache[key] = []
            for i in range(2):
                gate = control_gate(q[i]['match'], cuts[cfg['quantile']])
                if cfg['gate'] == 'sqrt':
                    gate = np.sqrt(gate)
                if cfg['gate'] == 'binary':
                    gate = (gate > 0).astype(gate.dtype)
                a = transform(x[i], direction[i], gate, cfg['mix'], cfg['suppression'])
                uniform = transform(x[i], direction[i], gate, 0, cfg['suppression'])
                proposed = np.stack([g.predict(a) for g in guards])
                flips = proposed != initial[i]
                unsupported = flips & (proposed != q[i]['teacher'].argmax(1))
                cache[key].append((a, uniform, dict(none=np.zeros(len(y), dtype=bool),
                                                    any_flip=flips.any(0), unsupported_flip=unsupported.any(0))))
        outputs, counts = [], []
        for a, uniform, mask in cache[key]:
            limited = mask[cfg['guard']]
            output = np.where(limited[:, None], uniform + cfg['retain'] * (a - uniform), a)
            np.testing.assert_allclose(output.mean(1), uniform.mean(1), rtol=3e-6, atol=1e-8)
            outputs.append(metrics(output, head.decision_function(output), y))
            counts.append(int(limited.sum()))
        e, d = outputs
        result.append(dict(exact_v_none=contrast(e, reference), exact_v_different=contrast(e, d), limited=counts))
    print('GUARD', dataset, len(y), len(choices), flush=True)
    return result


def main():
    kernel = weights()
    choices = list(grid())
    result = {}
    for source in ('vanilla', 'chain'):
        for dataset in DATASETS:
            bank, head, cuts, (x, p, y, fold) = models(dataset, source, kernel)
            guards = fit_guards(x[:, 2], y, fold)
            held = fold == 3
            external = read_features(dataset, source, 'reserve')
            pairs = {'fold3': (x[held], p[held], y[held]),
                     'reserve': (external['target_fc2'], external['preview_fc2'], external['y'])}
            for cohort, data in pairs.items():
                key = f'{source}_{dataset}_{cohort}'
                result[key] = evaluate(key, data, bank, head, guards, cuts, choices)
    ranking = [dict(index=i, config=cfg, **rank([r[i] for k, r in result.items() if k.startswith('chain_')]))
               for i, cfg in enumerate(choices)]
    ranking.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    independent = [dict(index=i, config=cfg, **rank([r[i] for r in result.values()])) for i, cfg in enumerate(choices)]
    independent.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    save(DOC / 'chain/guard_exploration.json', dict(tested=len(choices), chain=ranking, both_backbones=independent,
         datasets={k: [r[i['index']] for i in ranking[:6]] for k, r in result.items()},
         boundary='Reserve results were used for this exploration. These candidates require new confirmation.'))
    print('GUARD BEST ACTUAL', ranking[:3], flush=True)
    print('GUARD BEST BOTH', independent[:3], flush=True)


def freeze_lower_gain():
    source = DOC / 'chain/guard_exploration.json'
    pick = next(r for r in read(source)['chain'] if r['passed'] and r['config']['guard'] == 'none'
                and r['config']['gate'] == 'linear')
    row = pick['config']
    frozen = dict(candidates={'semantic_gain': dict(parameters=dict(mechanism='semantic_gain', quantile=row['quantile'],
                              mixing=row['mix'], suppression=row['suppression'], preview_weight=1.))},
                  exploration_sha256=digest(source), exploration=pick,
                  confirmation='prospective targets and preview stimuli separate from the first reserve')
    path = OUT / 'chain_lower_gain_frozen.json'
    if path.exists():
        assert read(path) == frozen
    else:
        save(path, frozen)
    print(frozen, flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-lower-gain', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.freeze_lower_gain:
            freeze_lower_gain()
        else:
            main()
