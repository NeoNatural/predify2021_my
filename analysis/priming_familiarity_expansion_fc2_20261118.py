"""Recheck FC2 for conditions that improve held-out familiarity discrimination."""
import argparse
import csv

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from priming_familiarity_20261112 import calibrate, gate
from priming_familiarity_expansion_collect_20261118 import path_for
from priming_familiarity_expansion_data_20261118 import DATASETS, DOC, MANIFEST, ROOT, register
from priming_familiarity_expansion_report_20261118 import load_scores, write
from priming_familiarity_scale_collect_20261113 import SHARD
from priming_familiarity_scale_data_20261113 import HISTORIES, SEEDS
from priming_familiarity_scale_fc2_20261113 import data_fc2, old_gate, preview_objects
from priming_familiarity_scale_report_20261113 import row_summary
from priming_familiarity_verify_20261112 import apply_gate, target_direction
from priming_sharpen_collect_20261112 import digest, vanilla
from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_search_20261112 import read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts


def eligible_conditions(dataset):
    with (DOC / 'summary.csv').open(encoding='utf-8-sig', newline='') as file:
        estimates = {(r['condition'], r['split']): r for r in csv.DictReader(file)
                     if r['dataset'] == dataset and r['reference'] == 'A' and r['negative'] == 'same_concept'}
    return [condition for condition in ('B', 'C', 'D', 'E')
            if (condition, 'fresh') in estimates
            and float(estimates[condition, 'validation']['estimate']) > 0
            and float(estimates[condition, 'fresh']['ci_low']) > 0]


def native_responses(dataset, split, rows):
    if split == 'legacy':
        return data_fc2(dataset, 'test', rows)
    values = []
    for start in range(0, len(rows), SHARD):
        with np.load(path_for(dataset, 'fresh', start)) as file:
            np.testing.assert_array_equal(file['index'], [r['index'] for r in rows[start:start + SHARD]])
            values.append(file['native_fc2'])
    return np.concatenate(values)


def experiment(dataset):
    conditions = eligible_conditions(dataset)
    selection_path = ROOT / dataset / 'selection.json'
    audit = dict(conditions=conditions, selection_sha256=digest(selection_path),
                 criterion='positive shared-query validation point estimate and positive lower 95% bound on fresh native AUC')
    if not conditions:
        save(DOC / f'fc2_{dataset}_eligibility.json', audit)
        print('FC2: no qualifying condition', dataset, flush=True)
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, buffers = vanilla()
    bank, head, cuts = fitted(dataset, weights())
    params = frozen()['configs']['target_only']
    setup = read(MANIFEST)[dataset]['splits']
    sets = {s: setup[s] for s in ('train', 'validation')}
    sets.update(fresh=register()[dataset]['fresh'], legacy=setup['test'])
    val = data_fc2(dataset, 'validation', sets['validation'])
    reference = old_gate(bank, np.repeat(val[:, None], 3, axis=1),
                         preview_objects(sets['validation'], model, buffers), cuts[.5])
    scores = {(condition, seed): load_scores(dataset, condition, seed, sets)
              for condition in ('A', *conditions) for seed in SEEDS}
    calibrations = {key: calibrate(value['validation'], reference) for key, value in scores.items()}
    audit['calibrations'] = {f'{c}_{s}': calibration for (c, s), calibration in calibrations.items()}
    rows = []
    for split in ('fresh', 'legacy'):
        x = native_responses(dataset, split, sets[split])
        paired = PairedConcepts(np.array([r['label'] for r in sets[split]]),
                                np.array([r['concept'] for r in sets[split]]))
        directions = [target_direction(bank, x[:, h], params) for h in range(3)]
        measured = {}
        for (condition, seed), raw in scores.items():
            cal = calibrations[condition, seed]
            strength = gate(raw[split], cal['threshold'], cal['slope'])
            measured[condition, seed] = [metrics(apply_gate(x[:, h], directions[h], strength[:, h], params),
                                                  head, paired) for h in range(3)]
        for condition in conditions:
            for h in (1, 2):
                effects = {}
                for seed in SEEDS:
                    a = measured['A', seed]
                    b = measured[condition, seed]
                    for kind, lhs, rhs in [('history', contrast(b[0], b[h]), contrast(a[0], a[h])),
                                           ('absolute_exact', b[0], a[0]),
                                           ('absolute_other', b[h], a[h])]:
                        for metric, value in lhs.items():
                            effects.setdefault((kind, metric), []).append(value - rhs[metric])
                rows.extend(dict(dataset=dataset, condition=condition, reference='A', split=split,
                                 negative=HISTORIES[h], effect=kind, metric=metric,
                                 **row_summary(np.stack(values))) for (kind, metric), values in effects.items())
    write(DOC / f'fc2_{dataset}.csv', rows)
    save(DOC / f'fc2_{dataset}_eligibility.json', audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        experiment(args.dataset)
