"""Compare the two 8x encoders at FC2 using frozen checkpoints and cached features."""

from __future__ import annotations

import csv

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from priming_familiarity_20261112 import BATCH, Familiarity, gate, intensity
from priming_familiarity_scale_collect_20261113 import SHARD, shard_path
from priming_familiarity_scale_data_20261113 import HISTORIES, LEVELS, MANIFEST, ROOT, SEEDS
from priming_familiarity_scale_fit_20261113 import SELECTION, aucs, load_data, prediction, protocol, run_path
from priming_familiarity_scale_frozen_control_20261113 import FOLDER, FROZEN, PLAN
from priming_familiarity_scale_report_20261113 import OUT, write
from priming_familiarity_verify_20261112 import apply_gate, target_direction
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_search_20261112 import read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


DATASETS = ('things', 'imagenet')
ARMS = ('trainable', 'extended_frozen')
METRICS = ('mean', 'within', 'between', 'fisher', 'margin', 'auc', 'ba')
HISTORY_METRICS = ('suppression', 'within', 'between', 'fisher', 'margin', 'auc', 'ba')


def calibrate_from_stats(values, reference):
    # The original calibration only uses these three statistics of the reference gate.
    thresholds = np.unique(np.r_[0., np.quantile(values[:, 1:], np.linspace(0, 1, 41))])
    weights = np.array([2., .75, .35])
    slopes = np.geomspace(1, 512, 81)
    candidates = []
    for threshold in thresholds:
        for slope in slopes:
            stats = intensity(gate(values, threshold, slope))
            candidates.append((float(np.abs(stats - reference) @ weights), float(threshold), float(slope)))
    error, threshold, slope = min(candidates)
    values = gate(values, threshold, slope)
    return dict(error=error, threshold=threshold, slope=slope, new_intensity=intensity(values).tolist(),
                new_history_mean=values.mean(0).tolist())


def pairs(rows):
    return PairedConcepts(np.array([r['label'] for r in rows]), np.array([r['concept'] for r in rows]))


def test_cache(dataset, rows):
    audit = read(ROOT / dataset / 'native_test_audit.json')
    assert audit['selection_sha256'] == digest(SELECTION)
    arrays = []
    for start in range(0, len(rows), SHARD):
        path = shard_path(dataset, 'test', start)
        assert digest(path) == audit['shards'][str(start)]
        with np.load(path) as f:
            np.testing.assert_array_equal(f['index'], [r['index'] for r in rows[start:start + SHARD]])
            arrays.append([f[k] for k in ('native_query', 'memory', 'native_fc2')])
    q, p, fc2 = (np.concatenate(items) for items in zip(*arrays))
    return torch.from_numpy(q).cuda(), torch.from_numpy(p).cuda(), fc2


def mean_seeds(values):
    return np.stack(values).mean(0)


def emit(table, meta, values):
    table.append(dict(**meta, seed='mean', **estimate(mean_seeds(values))))
    table.extend(dict(**meta, seed=seed, **estimate(v)) for seed, v in zip(SEEDS, values))


def gate_results(dataset, scope, vectors, paired, table):
    w = paired.concept_weights
    for metric in ('mean_gate', 'trigger', 'saturation'):
        matrices = {}
        for arm, by_seed in vectors.items():
            if metric == 'mean_gate':
                v = by_seed
            elif metric == 'trigger':
                v = [100 * (g > 0) for g in by_seed]
            else:
                v = [100 * (g >= 1) for g in by_seed]
            matrices[arm] = [w @ g for g in v]
        for h, history in enumerate(HISTORIES):
            common = dict(dataset=dataset, scope=scope, history=history, metric=metric)
            for arm in ARMS:
                emit(table, dict(**common, condition=arm, reference='', unit='fraction' if metric == 'mean_gate' else '%'),
                     [v[:, h] for v in matrices[arm]])
            a, b = (matrices[k] for k in ('extended_frozen', 'trainable'))
            emit(table, dict(**common, condition='extended_frozen', reference='trainable',
                             unit='fraction_difference' if metric == 'mean_gate' else 'pp'),
                 [x[:, h] - y[:, h] for x, y in zip(a, b)])


def fc2_results(ds, outcomes, table):
    for h, history in enumerate(HISTORIES):
        for metric in METRICS:
            percent = metric in ('ba', 'auc')
            common = dict(dataset=ds, history=history, reference_history='', metric=metric)
            for arm in ARMS:
                emit(table, dict(**common, comparison='absolute', condition=arm, reference='',
                                 unit='%' if percent else 'native'),
                     [(100 if percent else 1) * v[h][metric] for v in outcomes[arm]])
            a, b = ([v[h][metric] for v in outcomes[k]] for k in ('extended_frozen', 'trainable'))
            emit(table, dict(**common, comparison='same_history_difference', condition='extended_frozen',
                             reference='trainable', unit='pp' if percent else 'native_difference'),
                 [(100 if percent else 1) * (x - y) for x, y in zip(a, b)])
            if metric in ('mean', 'within', 'between', 'fisher'):
                emit(table, dict(**common, comparison='same_history_relative_change', condition='extended_frozen',
                                 reference='trainable', unit='%_relative'),
                     [100 * (x / y - 1) for x, y in zip(a, b)])
    for h, ref in enumerate(HISTORIES[1:], 1):
        effects = {arm: [contrast(v[0], v[h]) for v in outcomes[arm]] for arm in ARMS}
        for metric in HISTORY_METRICS:
            unit = 'pp' if metric in ('ba', 'auc') else 'native_difference' if metric == 'margin' else '%_relative'
            common = dict(dataset=ds, history='exact', reference_history=ref, metric=metric)
            for arm in ARMS:
                emit(table, dict(**common, comparison='history_contrast', condition=arm, reference='', unit=unit),
                     [v[metric] for v in effects[arm]])
            emit(table, dict(**common, comparison='history_contrast_difference', condition='extended_frozen',
                             reference='trainable', unit='pp_of_relative_effect' if unit == '%_relative' else unit),
                 [a[metric] - b[metric] for a, b in zip(effects['extended_frozen'], effects['trainable'])])


def previous_contrasts():
    with (OUT / 'fc2_test.csv').open(newline='', encoding='utf-8') as f:
        return {(r['dataset'], r['condition'], r['reference'], r['history'], r['metric']): r for r in csv.DictReader(f)}


def verify_prior(ds, name, outcomes, prior):
    for h, ref in enumerate(HISTORIES[1:], 1):
        effects = [contrast(v[0], v[h]) for v in outcomes]
        for metric in HISTORY_METRICS:
            original = prior[ds, name, '', ref, metric]
            observed = estimate(mean_seeds([v[metric] for v in effects]))
            for key, value in observed.items():
                np.testing.assert_allclose(value, float(original[key]), atol=2e-6, rtol=2e-6)


def ba_audit(ds, small, large, prior, table):
    old = [np.stack([v['ba'] for v in seed]) for seed in small]
    new = [np.stack([v['ba'] for v in seed]) for seed in large]
    change = [b - a for a, b in zip(old, new)]
    for h, history in enumerate(HISTORIES):
        common = dict(dataset=ds, history=history, reference_history='', metric='ba')
        for name, source in (('1x', old), ('8x', new)):
            emit(table, dict(**common, comparison='absolute', condition=name, reference='', unit='%'),
                 [100 * v[h] for v in source])
        emit(table, dict(**common, comparison='same_history_difference', condition='8x', reference='1x', unit='pp'),
             [100 * v[h] for v in change])
    for h, history in enumerate(HISTORIES[1:], 1):
        values = [100 * (v[0] - v[h]) for v in change]
        expected = prior[ds, 'selected', 'small', history, 'ba']
        for key, value in estimate(mean_seeds(values)).items():
            np.testing.assert_allclose(value, float(expected[key]), atol=2e-6)
        emit(table, dict(dataset=ds, history='exact', reference_history=history, metric='ba',
                         comparison='history_contrast_difference', condition='8x', reference='1x', unit='pp'), values)


def main():
    protocol()
    selection = read(SELECTION)
    extension = read(FROZEN)
    assert extension['protocol_sha256'] == digest(PLAN)
    assert read(PLAN)['primary_selection_sha256'] == digest(SELECTION)
    cfg = frozen()['configs']['target_only']
    assert cfg['weight'] == 0 and cfg['mixing'] == .45 and cfg['suppression'] == .003
    manifests = read(MANIFEST)
    prior = previous_contrasts()
    kernel = weights()
    ftable, gtable, correction, reports = [], [], [], {}
    for ds in DATASETS:
        data = manifests[ds]['splits']
        choice = selection['chosen'][ds]
        assert (choice['arm'], choice['level']) == ('trainable_encoder', LEVELS[-1])
        cal_path = OUT / f'{ds}_fc2_calibration.json'
        archived = read(cal_path)
        assert archived['selection_sha256'] == digest(SELECTION)
        calibrations = archived['calibrations']
        anchor = calibrations[f'selected_{SEEDS[0]}']['old_intensity']
        assert all(r['old_intensity'] == anchor for r in calibrations.values())
        val_q, val_p, val_hash = load_data(ds, 'validation', data['validation'])
        val_pair = pairs(data['validation'])
        val_gates = {arm: [] for arm in ARMS}
        test_gates = {arm: [] for arm in (*ARMS, 'small')}
        config, models = {}, {}
        scores_to_save = {}
        for seed in SEEDS:
            extended = extension['datasets'][ds][str(seed)]
            model_path = FOLDER / f'{ds}_{seed}.pth'
            assert digest(model_path) == extended['model_sha256']
            model = Familiarity().cuda().eval()
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
            models[seed] = model.requires_grad_(False)
            val, _ = prediction(model, val_q, val_p)
            val = val.cpu().numpy()
            np.testing.assert_allclose(aucs(val, val_pair.concept_weights[:1]), extended['validation_auc'], atol=1e-8)
            config[f'extended_frozen_{seed}'] = calibrate_from_stats(val, anchor)
            c = config[f'extended_frozen_{seed}']
            val_gates['extended_frozen'].append(gate(val, c['threshold'], c['slope']))
            scores_to_save[f'frozen_validation_{seed}'] = val
            for group, level in (('small', LEVELS[0]), ('selected', LEVELS[-1])):
                path = run_path(ds, 'trainable_encoder', level, seed)
                record = read(path.with_suffix('.json'))
                assert record['val_cache'] == val_hash and digest(path.with_suffix('.npz')) == record['score_sha256']
                assert record['model_sha256'] == selection['model_sha256'][ds][path.name]
                with np.load(path.with_suffix('.npz')) as f:
                    original = f['validation']
                c = calibrate_from_stats(original, anchor)
                for k, value in c.items():
                    np.testing.assert_allclose(value, calibrations[f'{group}_{seed}'][k], atol=1e-10, rtol=1e-10)
                config[f'{group}_{seed}'] = c
                if group == 'selected':
                    val_gates['trainable'].append(gate(original, c['threshold'], c['slope']))
        reports[ds] = dict(old_validation_intensity=anchor, old_calibration_sha256=digest(cal_path),
                           validation_cache=val_hash, models=extension['datasets'][ds], calibrations=config)
        save(OUT / 'fc2_encoder_control_calibration.json', reports)
        del val_q, val_p

        q, p, x = test_cache(ds, data['test'])
        test_pair = pairs(data['test'])
        prior_auc = {}
        with (OUT / 'frozen_budget_test.csv').open(newline='', encoding='utf-8') as f:
            prior_auc = {r['negative']: float(r['estimate']) for r in csv.DictReader(f)
                         if r['dataset'] == ds and r['arm'] == 'extended_frozen' and not r['reference']}
        reproduced = []
        for seed in SEEDS:
            with torch.inference_mode():
                observed = torch.cat([models[seed](a, b) for a, b in zip(q.split(BATCH), p.split(BATCH))]).cpu().numpy()
            reproduced.append(aucs(observed, test_pair.concept_weights[:1]))
            scores_to_save[f'frozen_test_{seed}'] = observed
            c = config[f'extended_frozen_{seed}']
            test_gates['extended_frozen'].append(gate(observed, c['threshold'], c['slope']))
            for arm, label, level in (('small', 'small', LEVELS[0]), ('trainable', 'selected', LEVELS[-1])):
                path = run_path(ds, 'trainable_encoder', level, seed).with_suffix('.native_test.npz')
                with np.load(path) as f:
                    observed = f['scores']
                c = config[f'{label}_{seed}']
                test_gates[arm].append(gate(observed, c['threshold'], c['slope']))
        np.testing.assert_allclose(np.mean(reproduced, axis=0), [prior_auc[k] for k in HISTORIES[1:]], atol=1e-8)
        np.savez(ROOT / ds / 'fc2_encoder_control_frozen_scores.npz', **scores_to_save)
        gate_results(ds, 'validation', val_gates, val_pair, gtable)
        gate_results(ds, 'test', {k: test_gates[k] for k in ARMS}, test_pair, gtable)
        bank, head, _ = fitted(ds, kernel)
        directions = [target_direction(bank, x[:, h], cfg) for h in range(3)]
        outcomes = {}
        for arm, gates in test_gates.items():
            outcomes[arm] = [[metrics(apply_gate(x[:, h], directions[h], g[:, h], cfg), head, test_pair)
                              for h in range(3)] for g in gates]
        verify_prior(ds, 'selected', outcomes['trainable'], prior)
        verify_prior(ds, 'small', outcomes['small'], prior)
        fc2_results(ds, outcomes, ftable)
        ba_audit(ds, outcomes['small'], outcomes['trainable'], prior, correction)
        print('CACHE CONTROL', ds, len(x), flush=True)
    write('fc2_encoder_control', ftable)
    write('fc2_encoder_control_gates', gtable)
    write('fc2_ba_history_audit', correction)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        main()
