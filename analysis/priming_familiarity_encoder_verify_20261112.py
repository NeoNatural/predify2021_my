"""Replay encoder-only comparisons on the previously archived cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from hebb_chain_20261111 import Controls, Experiment, old
from priming_familiarity_20261112 import SHUFFLES, gate, shuffled
from priming_familiarity_data_20261112 import OUT, holdout_stimuli, trial
from priming_familiarity_encoder_20261112 import ARMS, FOLDER, RESULT, check_result, load_model
from priming_familiarity_report_20261112 import WINDOWS, differences, measured, paired_auc, predicted_change, write
from priming_familiarity_verify_20261112 import SCOPES, apply_gate, archive, split_source, target_direction
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import HISTORIES, frozen
from priming_sharpen_search_20261112 import DOC, read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def get_models(ds):
    base = torch.load(OUT / ds / 'circuit.pth', weights_only=True, map_location='cpu')
    return [load_model(base | torch.load(FOLDER / ds / f'{arm}.pth', weights_only=True, map_location='cpu'))
            for arm in ARMS]


@torch.inference_mode()
def raw_scores(models, conv, memory):
    target, preview = (torch.from_numpy(np.stack(a)).cuda() for a in (conv, memory))
    result, scrambled = [], []
    for model in models:
        z = model.encode(target)
        h = model.reference(z)
        a = model.memory(preview)
        zero = model.read_encoded(z, h, torch.zeros_like(a[:, None]))
        assert torch.count_nonzero(zero) == 0
        result.append(model.read_encoded(z, h, a[:, None]).cpu().numpy())
        random = torch.stack([shuffled(a, seed) for seed in SHUFFLES])
        scrambled.append(model.read_encoded(z, h, random[:, :, None]).cpu().numpy())
    return np.stack(result), np.stack(scrambled)


def collect(ds, scope, spec, parts, models, controls):
    dest = FOLDER / ds / f'{scope}.npz'
    audit = dest.with_suffix('.json')
    version = digest(RESULT)
    source = digest(Path(__file__))
    if dest.exists() and audit.exists():
        assert read(audit) == dict(frozen_sha256=version, source_sha256=source, feature_sha256=digest(dest))
        return
    prior = archive(ds, scope)
    plan, _, _ = split_source(ds, scope)
    np.testing.assert_array_equal(prior['index'], [r['index'] for r in plan['targets']])
    run = Experiment(ds, parts)
    run.fit()
    scores, shuffles = [], []
    try:
        for j, (_, x, p) in enumerate(holdout_stimuli(plan)):
            conv, memory, fc, _ = zip(*(trial(run, x, frames, controls) for frames in p))
            np.testing.assert_array_equal(np.stack(fc), prior['fc2'][j, :3])
            observed, scrambled = raw_scores(models, conv, memory)
            np.testing.assert_allclose(observed[0], prior['score_new'][j, :3], atol=2e-6, rtol=2e-5)
            scores.append(observed)
            shuffles.append(scrambled)
            if (j + 1) % 8 == 0 or j + 1 == len(plan['targets']):
                print('OLD IMAGES', ds, scope, j + 1, '/', len(plan['targets']), flush=True)
    finally:
        run.detach()
    scores = np.stack(scores)
    shuffles = np.stack(shuffles)
    gates, shuffled_gates = [], []
    for i, arm in enumerate(ARMS):
        cal = spec['models'][arm]['calibration']
        params = cal['threshold'], cal['slope']
        gates.append(gate(scores[:, i], *params))
        shuffled_gates.append(gate(shuffles[:, i], *params))
    np.testing.assert_allclose(gates[0], prior['gate_new'][:, :3], atol=5e-5, rtol=2e-5)
    gates[0] = prior['gate_new'][:, :3]
    tmp = dest.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, scores=scores, gates=np.stack(gates, axis=1), shuffled_scores=shuffles,
                 shuffled_gates=np.stack(shuffled_gates, axis=1), index=prior['index'])
    tmp.replace(dest)
    save(audit, dict(frozen_sha256=version, source_sha256=source, feature_sha256=digest(dest)))


def read_replay(ds, scope):
    dest = FOLDER / ds / f'{scope}.npz'
    assert read(dest.with_suffix('.json')) == dict(frozen_sha256=digest(RESULT), source_sha256=digest(Path(__file__)),
                                                  feature_sha256=digest(dest))
    with np.load(dest) as f:
        return {k: f[k] for k in f.files}


def gate_stats(replay, prior, paired, row, gate_rows):
    w = paired.concept_weights
    for i, arm in enumerate(ARMS):
        raw = replay['scores'][:, i, :, 9]
        g = replay['gates'][:, i, :, 9]
        random_raw = replay['shuffled_scores'][:, i, :, :, 9].transpose(1, 0, 2)
        random_gate = replay['shuffled_gates'][:, i, :, :, 9].transpose(1, 0, 2)
        emit_gates(arm, g, raw, row, w, gate_rows)
        for h in (1, 2):
            real = paired_auc(raw[:, 0], raw[:, h], w)
            random = np.stack([paired_auc(r[:, 0], r[:, h], w) for r in random_raw])
            emit_gate(arm, 'exact', HISTORIES[h], 'raw_auc_minus_shuffled', real - random.mean(0), row, gate_rows)
            emit_gate(arm, 'exact', HISTORIES[h], 'raw_auc_minus_clear', real - .5, row, gate_rows)
        for h, name in enumerate(HISTORIES[:3]):
            for metric, values in (('shuffled_mean_gate', random_gate[:, :, h]),
                                   ('shuffled_trigger_pct', 100 * (random_gate[:, :, h] > 0))):
                emit_gate(arm, name, '', metric, w @ values.mean(0), row, gate_rows)
    emit_gates('old', prior['gate_old'][:, :3, 9], prior['score_old'][:, :3, 9], row, w, gate_rows)


def emit_gate(arm, history, reference, metric, value, row, table):
    table.append(dict(**row, arm=arm, history=history, reference=reference, metric=metric, **estimate(value)))


def emit_gates(arm, g, raw, row, w, out):
    for h, name in enumerate(HISTORIES[:3]):
        for metric, values in (('mean_gate', g[:, h]), ('trigger_pct', 100 * (g[:, h] > 0)),
                               ('saturation_pct', 100 * (g[:, h] >= 1))):
            emit_gate(arm, name, '', metric, w @ values, row, out)
    for h in (1, 2):
        for metric, data in (('raw_auc', raw), ('gate_auc', g)):
            emit_gate(arm, 'exact', HISTORIES[h], metric, paired_auc(data[:, 0], data[:, h], w), row, out)


def report(spec, cfg):
    table, gate_rows, predictions = [], [], {}
    kernel = weights()
    comparisons = (('continued', 'checkpoint'), ('paired', 'checkpoint'), ('paired', 'continued'),
                   ('checkpoint', 'old'), ('continued', 'old'), ('paired', 'old'),
                   ('checkpoint', 'base'), ('continued', 'base'), ('paired', 'base'))
    for ds, registered in spec['datasets'].items():
        if not registered['eligible']:
            continue
        bank, head, _ = fitted(ds, kernel)
        for scope in SCOPES:
            prior = archive(ds, scope)
            replay = read_replay(ds, scope)
            np.testing.assert_array_equal(prior['index'], replay['index'])
            paired = PairedConcepts(prior['label'], prior['concept'])
            row = dict(dataset=ds, scope=scope)
            gate_stats(replay, prior, paired, row, gate_rows)
            raw = prior['fc2']
            n, _, steps, units = raw.shape
            data, hits = {'base': {}}, {'base': {}}
            directions = []
            for h, name in enumerate(HISTORIES):
                data['base'][name], hits['base'][name] = measured(raw[:, h], head, paired)
                if h < 3:
                    directions.append(target_direction(bank, raw[:, h].reshape(n * steps, units), cfg))
            treatments = {'old': prior['gate_old'][:, :3]}
            treatments.update({arm: replay['gates'][:, i] for i, arm in enumerate(ARMS)})
            for arm, values in treatments.items():
                data[arm], hits[arm] = {'none': data['base']['none']}, {'none': hits['base']['none']}
                for h, name in enumerate(HISTORIES[:3]):
                    x = raw[:, h].reshape(n * steps, units)
                    g = values[:, h].reshape(-1)
                    z = apply_gate(x, directions[h], g, cfg)
                    np.testing.assert_array_equal(z > 0, x > 0)
                    np.testing.assert_allclose(z.mean(1), x.mean(1) * (1 - cfg['suppression'] * g), atol=1e-8, rtol=3e-6)
                    data[arm][name], hits[arm][name] = measured(z.reshape(n, steps, units), head, paired)
            predictions[f'{ds}_{scope}'] = dict(n=n,
                correct={k: {h: int(v.sum()) for h, v in x.items()} for k, x in hits.items()},
                paired_vs_continued={h: predicted_change(hits['paired'][h], hits['continued'][h]) for h in HISTORIES[:3]},
                paired_vs_checkpoint={h: predicted_change(hits['paired'][h], hits['checkpoint'][h]) for h in HISTORIES[:3]})
            for win, interval in WINDOWS.items():
                changes = {}
                for arm, results in data.items():
                    for ref in HISTORIES[1:]:
                        v = differences(results['exact'], results[ref], interval)
                        changes[arm, ref] = v
                        emit_fc(arm, arm, 'exact', ref, 'history', v, row, win, table)
                for a, b in comparisons:
                    for h in HISTORIES[:3]:
                        v = differences(data[a][h], data[b][h], interval)
                        emit_fc(a, b, h, h, 'arm', v, row, win, table)
                    for ref in HISTORIES[1:]:
                        left, right = changes[a, ref], changes[b, ref]
                        diff = {k: left[k] - right[k] for k in left}
                        emit_fc(a, b, 'exact', ref, 'arm_by_history', diff, row, win, table)
            print('REPLAY STATS', ds, scope, flush=True)
    write('encoder_fc2', table)
    write('encoder_gates', gate_rows)
    save(DOC / 'chain/familiarity_encoder_predictions.json', predictions)


def emit_fc(arm, ref, history, reference_history, comparison, values, row, win, table):
    table.extend(dict(**row, window=win, comparison=comparison, arm=arm, reference=ref,
                      history=history, reference_history=reference_history, metric=k, **estimate(v))
                 for k, v in values.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('collect', 'report'))
    args = parser.parse_args()
    spec = check_result()
    eligible = {ds: v for ds, v in spec['datasets'].items() if v['eligible']}
    if not eligible:
        print('No dataset passed both fold 3 probes.', flush=True)
        return
    cfg = frozen()
    if args.phase == 'collect':
        parts = old.initialize()
        controls = Controls(tuple(cfg['hebbian_lambdas']))
        for ds, registered in eligible.items():
            models = get_models(ds)
            for scope in SCOPES:
                collect(ds, scope, registered, parts, models, controls)
    else:
        report(spec, cfg['configs']['target_only'])


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
