"""Replay cached t9 FC2 features with counterfactual different-history gates."""

from __future__ import annotations

import csv

import numpy as np
from threadpoolctl import threadpool_limits

from priming_familiarity_scale_encoder_fc2_control_20261113 import (
    ARMS, DATASETS, LEVELS, MANIFEST, OUT, ROOT, SEEDS, SELECTION, SHARD,
    apply_gate, digest, emit, estimate, fitted, frozen, gate, pairs, read,
    run_path, shard_path, target_direction, weights, write,
)
from priming_sharpen_search_20261112 import unit


TRAINABLE, FROZEN = ARMS
T, F = 'trainable_native', 'frozen_native'
TC, FC, FM = 'trainable_cap', 'frozen_cap', 'frozen_cap_equal_mean'
TL, FL = 'trainable_equal_mean_low_cap', 'frozen_equal_mean_low_cap'
CAP = .95
LOW_CAP = .80
PERMUTATIONS = 256
PERMUTATION_SEED = 20261114


def cached_fc2(dataset, rows):
    audit = read(ROOT / dataset / 'native_test_audit.json')
    assert audit['selection_sha256'] == digest(SELECTION)
    features = []
    for start in range(0, len(rows), SHARD):
        path = shard_path(dataset, 'test', start)
        assert digest(path) == audit['shards'][str(start)]
        with np.load(path) as f:
            np.testing.assert_array_equal(f['index'], [r['index'] for r in rows[start:start + SHARD]])
            features.append(f['native_fc2'][:, [0, 2]])
    return np.concatenate(features)


def cached_gates(dataset, calibration, n):
    result = []
    with np.load(ROOT / dataset / 'fc2_encoder_control_frozen_scores.npz') as f:
        for seed in SEEDS:
            path = run_path(dataset, 'trainable_encoder', LEVELS[-1], seed).with_suffix('.native_test.npz')
            with np.load(path) as t:
                scores = (t['scores'], f[f'frozen_test_{seed}'])
            arms = {}
            for arm, group, score in zip(ARMS, ('selected', FROZEN), scores):
                assert score.shape == (n, 3) and np.isfinite(score).all()
                params = calibration[f'{group}_{seed}']
                arms[arm] = gate(score[:, [0, 2]], params['threshold'], params['slope'])
            result.append(arms)
    return result


def cap_at_mean(g, w, cap):
    """Lower the upper endpoint and redistribute mass to active gates."""
    target = w @ g
    assert target < cap * w[g > 0].sum()
    lo, hi = 1., cap / g[g > 0].min()
    for _ in range(60):
        mid = (lo + hi) / 2
        if w @ np.minimum(cap, mid * g) < target:
            lo = mid
        else:
            hi = mid
    result = np.minimum(cap, (lo + hi) / 2 * g)
    np.testing.assert_allclose(w @ result, target, atol=1e-12)
    np.testing.assert_array_equal(result > 0, g > 0)
    return result


class Geometry:
    """Compute the original equal-class geometry using concept-level Gram matrices."""

    def __init__(self, rows, sample):
        concepts = np.array([(r['label'], r['concept']) for r in rows], dtype=str)
        _, inverse = np.unique(concepts, axis=0, return_inverse=True)
        members = inverse[None, :] == np.arange(inverse.max() + 1)[:, None]
        self.pool = members / members.sum(axis=1, keepdims=True)
        class_mass = sample.class_weights @ members.T
        class_pair = np.einsum('bci,bcj->bij', class_mass, class_mass, optimize=True)
        class_pair /= len(sample.classes)
        global_mass = class_mass.mean(axis=1)
        global_pair = global_mass[:, :, None] * global_mass[:, None, :]
        self.pair = np.stack((class_pair, class_pair - global_pair)).reshape(2, len(class_mass), -1)
        self.weights = sample.class_weights.mean(axis=1)

    def measure(self, vectors):
        u = unit(np.stack(vectors).astype(np.float64))
        centers = self.pool @ u
        gram = centers @ centers.swapaxes(-1, -2)
        center_norm, between = self.pair @ gram.reshape(len(vectors), -1).T
        image_norm = self.weights @ np.square(u).sum(axis=-1).T
        within = np.maximum(image_norm - center_norm, 0)
        between = np.maximum(between, 0)
        return dict(within=within, between=between, fisher=between / np.maximum(within, 1e-20))


def shuffled_indices(sample, rng):
    # Classwise permutations also preserve the original class-balanced gate mean.
    w = sample.concept_weights[0]
    out = np.tile(np.arange(len(w)), (PERMUTATIONS, 1))
    for mask in sample.members:
        idx = np.flatnonzero(mask)
        np.testing.assert_allclose(w[idx], w[idx[0]], atol=1e-14)
        out[:, idx] = rng.permuted(np.broadcast_to(idx, (PERMUTATIONS, len(idx))), axis=1)
    return out


def evaluate_permutations(gates, permutations, adjust, geometry):
    result = {}
    for name, g in gates.items():
        vectors = g[permutations]
        fisher_sum = reciprocal_sum = 0.
        for start in range(0, PERMUTATIONS, 16):
            batch = vectors[start:start + 16]
            fisher = geometry.measure([adjust(v) for v in batch])['fisher']
            fisher_sum = fisher_sum + fisher.sum(axis=1)
            reciprocal_sum = reciprocal_sum + (1 / fisher).sum(axis=1)
        result[name] = dict(fisher=fisher_sum / PERMUTATIONS, inverse=reciprocal_sum / PERMUTATIONS)
    return result


def gate_stats(sample, g):
    w = sample.concept_weights
    return dict(mean_gate=w @ g, trigger=100 * (w @ (g > 0)), saturation=100 * (w @ (g == 1)),
                at_least_095=100 * (w @ (g >= CAP)), at_least_080=100 * (w @ (g >= LOW_CAP)))


def history(exact, inverse):
    return 100 * (exact * inverse - 1)


def contrasts(exact, different, perm):
    inv = {k: 1 / v['fisher'] for k, v in different.items()}
    e0, e1 = exact[TRAINABLE]['fisher'], exact[FROZEN]['fisher']
    h00, h01 = history(e0, inv[T]), history(e0, inv[F])
    h10, h11 = history(e1, inv[T]), history(e1, inv[F])
    restore, reverse = h10 - h11, h00 - h01
    change = h11 - h00
    numerator = ((h11 - h01) + (h10 - h00)) / 2
    denominator = -(restore + reverse) / 2
    np.testing.assert_allclose(numerator + denominator, change, atol=1e-12)

    # Averaging over Exact gates keeps the control decomposition symmetric as well.
    exact_mean = (e0 + e1) / 2
    h = {k: history(exact_mean, i) for k, i in inv.items()}
    shuffled = {k: history(exact_mean, v['inverse']) for k, v in perm.items()}
    saturation = (h[F] - h[FC]) - (h[T] - h[TC])
    strength = h[FC] - h[FM]
    equal_mean = h[FM] - h[TC]
    remaining_distribution = shuffled[FM] - shuffled[TC]
    remaining_assignment = equal_mean - remaining_distribution
    np.testing.assert_allclose(saturation + strength + equal_mean, denominator, atol=1e-12)
    np.testing.assert_allclose(remaining_distribution + remaining_assignment, equal_mean, atol=1e-12)
    return dict(
        trainable_native=h00, frozen_native=h11, frozen_exact_trainable_different=h10,
        trainable_exact_frozen_different=h01, net_change=change,
        exact_contribution=numerator, different_contribution=denominator,
        restored_on_frozen=restore, lost_on_trainable=reverse,
        saturation_path=saturation, mean_strength_path=strength, equal_mean_unsaturated=equal_mean,
        equal_mean_low_cap=h[FL] - h[TL], low_cap_change=(h[FL] - h[TL]) - equal_mean,
        remaining_distribution=remaining_distribution, remaining_assignment=remaining_assignment,
        native_distribution=shuffled[F] - shuffled[T],
        native_assignment=denominator - (shuffled[F] - shuffled[T]),
        trainable_native_assignment=h[T] - shuffled[T],
        frozen_native_assignment=h[F] - shuffled[F],
    )


def check_old(baseline, dataset, seed, exact, different, gstats, effects):
    for arm, condition in ((TRAINABLE, T), (FROZEN, F)):
        for hist, values in (('exact', exact[arm]), ('different', different[condition])):
            for metric, samples in values.items():
                ref = baseline[dataset, seed, hist, '', metric, 'absolute', arm]
                compare_estimate(samples, ref)
        for metric in ('mean_gate', 'trigger', 'saturation'):
            ref = baseline[dataset, seed, 'different', '', metric, 'gate', arm]
            compare_estimate(gstats[condition][metric], ref)
    ref = baseline[dataset, seed, 'exact', 'different', 'fisher', 'history_contrast_difference', FROZEN]
    compare_estimate(effects['net_change'], ref)


def compare_estimate(samples, ref):
    for k, v in estimate(samples).items():
        np.testing.assert_allclose(v, float(ref[k]), atol=2e-6, rtol=2e-6)


def old_results():
    result = {}
    for name in ('fc2_encoder_control', 'fc2_encoder_control_gates'):
        with (OUT / f'{name}.csv').open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if 'scope' in row and (row['scope'] != 'test' or row['reference']):
                    continue
                key = (row['dataset'], row['seed'], row['history'], row.get('reference_history', ''),
                       row['metric'], row.get('comparison', 'gate'), row['condition'])
                result[key] = row
    return result


def experiment(dataset, rows, calibration, kernel, cfg, baseline):
    sample = pairs(rows)
    geometry = Geometry(rows, sample)
    x = cached_fc2(dataset, rows)
    bank, _, _ = fitted(dataset, kernel)
    directions = [target_direction(bank, x[:, h], cfg) for h in range(x.shape[1])]
    w = sample.concept_weights[0]
    rng = np.random.default_rng(PERMUTATION_SEED)
    outputs, statistics, absolute = [], [], []
    for seed, gates in zip(SEEDS, cached_gates(dataset, calibration, len(rows))):
        gt, gf = [gates[arm][:, 1].astype(np.float64) for arm in ARMS]
        gc_t, gc_f = cap_at_mean(gt, w, CAP), cap_at_mean(gf, w, CAP)
        assert w @ gt < w @ gf
        gm = gc_f * ((w @ gt) / (w @ gc_f))
        np.testing.assert_allclose(w @ gm, w @ gc_t, atol=1e-12)
        controls = {T: gt, F: gf, TC: gc_t, FC: gc_f, FM: gm,
                    TL: cap_at_mean(gc_t, w, LOW_CAP), FL: cap_at_mean(gm, w, LOW_CAP)}

        def adjust(g):
            return apply_gate(x[:, 1], directions[1], g, cfg)

        exact_vectors = [apply_gate(x[:, 0], directions[0], gates[arm][:, 0], cfg) for arm in ARMS]
        exact = {arm: {k: v[:, j] for k, v in geometry.measure(exact_vectors).items()}
                 for j, arm in enumerate(ARMS)}
        different = {name: {k: v[:, 0] for k, v in geometry.measure([adjust(g)]).items()}
                     for name, g in controls.items()}
        u = unit(adjust(gt).astype(np.float64))
        reference = sample.geometry((u @ u.T)[None], equal_class=True)
        for metric, value in reference.items():
            np.testing.assert_allclose(different[T][metric], value[:, 0], atol=1e-12)

        # Compare native and equal-mean gates with the same within-class assignments.
        perm = evaluate_permutations({k: controls[k] for k in (T, F, TC, FM)},
                                     shuffled_indices(sample, rng), adjust, geometry)
        effects = contrasts(exact, different, perm)
        gstats = {name: gate_stats(sample, g) for name, g in controls.items()}
        check_old(baseline, dataset, str(seed), exact, different, gstats, effects)
        outputs.append(effects)
        statistics.append(gstats)
        absolute.append({**different, **{f'{k}_permuted': {'fisher': v['fisher']} for k, v in perm.items()}})
        print(dataset, seed, 'cached gate replay verified', flush=True)
    return outputs, statistics, absolute


def main():
    plan = read(MANIFEST)
    cal = read(OUT / 'fc2_encoder_control_calibration.json')
    cfg = frozen()['configs']['target_only']
    assert cfg['weight'] == 0 and cfg['mixing'] == .45 and cfg['suppression'] == .003
    kernel = weights()
    baseline = old_results()
    effects_table, gates_table, geometry_table = [], [], []
    for ds in DATASETS:
        effects, stats, geo = experiment(ds, plan[ds]['splits']['test'], cal[ds]['calibrations'], kernel, cfg, baseline)
        for name in effects[0]:
            unit_name = '%' if name in (T, F, 'frozen_exact_trainable_different',
                                        'trainable_exact_frozen_different') else 'pp'
            emit(effects_table, dict(dataset=ds, effect=name, unit=unit_name), [v[name] for v in effects])
        for table, values in ((gates_table, stats), (geometry_table, geo)):
            for condition, measured in values[0].items():
                for metric in measured:
                    unit_name = '%' if metric != 'mean_gate' and table is gates_table else 'native'
                    emit(table, dict(dataset=ds, condition=condition, metric=metric, unit=unit_name),
                         [v[condition][metric] for v in values])
    write('fc2_gate_swap_effects', effects_table)
    write('fc2_gate_swap_gates', gates_table)
    write('fc2_gate_swap_geometry', geometry_table)


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        main()
