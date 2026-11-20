"""Decompose the recorded FC2 Fisher contrast using paired cached responses."""

from __future__ import annotations

import csv

import numpy as np
from threadpoolctl import threadpool_limits

from priming_familiarity_20261112 import gate
from priming_familiarity_scale_data_20261113 import MANIFEST, ROOT, SEEDS
from priming_familiarity_scale_fc2_20261113 import data_fc2
from priming_familiarity_scale_fit_20261113 import run_path
from priming_familiarity_scale_report_20261113 import OUT
from priming_familiarity_verify_20261112 import apply_gate, target_direction
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import PLAN
from priming_sharpen_search_20261112 import read, unit
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def contrast(exact, different):
    return 100 * (exact / different - 1)


def main():
    manifest = read(MANIFEST)
    calibration = read(OUT / 'fc2_encoder_control_calibration.json')
    config = read(PLAN)['configs']['target_only']
    with (OUT / 'fc2_encoder_control.csv').open(encoding='utf-8', newline='') as stream:
        reference = list(csv.DictReader(stream))
    kernel = weights()
    records = []
    max_reproduction_error = 0.
    for dataset in ('things', 'imagenet'):
        rows = manifest[dataset]['splits']['test']
        samples = PairedConcepts(
            np.array([row['label'] for row in rows]),
            np.array([row['concept'] for row in rows]),
        )
        x = data_fc2(dataset, 'test', rows)
        bank, _, _ = fitted(dataset, kernel)
        directions = {h: target_direction(bank, x[:, h], config) for h in (0, 2)}
        components = {}
        for seed in SEEDS:
            fisher = {}
            for arm, label in (('trainable', 'selected'), ('extended_frozen', 'extended_frozen')):
                if arm == 'trainable':
                    path = run_path(dataset, 'trainable_encoder', 16, seed).with_suffix('.native_test.npz')
                    with np.load(path) as archive:
                        scores = archive['scores']
                else:
                    with np.load(ROOT / dataset / 'fc2_encoder_control_frozen_scores.npz') as archive:
                        scores = archive[f'frozen_test_{seed}']
                cal = calibration[dataset]['calibrations'][f'{label}_{seed}']
                gates = gate(scores, cal['threshold'], cal['slope'])
                for h, history in ((0, 'exact'), (2, 'different')):
                    output = apply_gate(x[:, h], directions[h], gates[:, h], config)
                    normalized = unit(output.astype(np.float64))
                    values = samples.geometry((normalized @ normalized.T)[None], equal_class=True)['fisher'][:, 0]
                    fisher[arm, history] = values
                    expected = next(r for r in reference if r['dataset'] == dataset
                                    and r['comparison'] == 'absolute' and r['metric'] == 'fisher'
                                    and r['history'] == history and r['condition'] == arm
                                    and r['seed'] == str(seed))
                    for key, observed in estimate(values).items():
                        error = abs(observed - float(expected[key]))
                        max_reproduction_error = max(max_reproduction_error, error)
                        np.testing.assert_allclose(observed, float(expected[key]), atol=2e-8, rtol=2e-7)

            e0, d0 = fisher['trainable', 'exact'], fisher['trainable', 'different']
            e1, d1 = fisher['extended_frozen', 'exact'], fisher['extended_frozen', 'different']
            original = contrast(e0, d0)
            frozen = contrast(e1, d1)
            numerator_first = contrast(e1, d0) - original
            numerator_last = frozen - contrast(e0, d1)
            denominator_first = contrast(e0, d1) - original
            denominator_last = frozen - contrast(e1, d0)
            numerator = (numerator_first + numerator_last) / 2
            denominator = (denominator_first + denominator_last) / 2
            net = frozen - original
            np.testing.assert_allclose(numerator + denominator, net, atol=1e-12, rtol=1e-10)
            seed_values = dict(
                trainable_contrast=original,
                frozen_contrast=frozen,
                numerator_contribution=numerator,
                denominator_contribution=denominator,
                net_change=net,
                numerator_first=numerator_first,
                numerator_last=numerator_last,
                denominator_first=denominator_first,
                denominator_last=denominator_last,
                trainable_exact_frozen_different=contrast(e0, d1),
                frozen_exact_trainable_different=contrast(e1, d0),
            )
            for name, values in seed_values.items():
                components.setdefault(name, []).append(values)
                records.append(dict(dataset=dataset, component=name, seed=seed, **estimate(values)))

        for name, by_seed in components.items():
            values = np.mean(by_seed, axis=0)
            result = estimate(values)
            records.append(dict(dataset=dataset, component=name, seed='mean', **result))
            if name in ('numerator_contribution', 'denominator_contribution', 'net_change'):
                print(dataset, name, result, flush=True)
            if name == 'net_change':
                expected = next(r for r in reference if r['dataset'] == dataset
                                and r['comparison'] == 'history_contrast_difference'
                                and r['reference_history'] == 'different' and r['metric'] == 'fisher'
                                and r['seed'] == 'mean')
                for key, observed in result.items():
                    np.testing.assert_allclose(observed, float(expected[key]), atol=2e-7, rtol=2e-7)

    destination = OUT / 'fisher_ratio_decomposition.csv'
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('Maximum absolute reproduction error:', max_reproduction_error, flush=True)
    print(destination, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
