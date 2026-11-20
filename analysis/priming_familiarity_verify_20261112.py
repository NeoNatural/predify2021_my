"""Replay frozen familiarity gates on development and fresh paired four-layer trials."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from hebb_chain_20261111 import Controls, Experiment, old
from priming_familiarity_20261112 import OUT, SHUFFLES, PreviewMemory, checked, gate, load_circuit, shuffled
from priming_familiarity_data_20261112 import holdout_stimuli, paths, trial
from priming_sharpen_collect_20261112 import digest, vanilla
from priming_sharpen_corroboration_20261112 import direction
from priming_sharpen_object_precision_20261112 import encode_tiles, fitted, tiles
from priming_sharpen_same_concept_data_20261112 import HISTORIES, frozen, renderer
from priming_sharpen_same_concept_eval_20261112 import load
from priming_sharpen_search_20261112 import DATASETS, DOC, control_gate, read, save, transform
from priming_sharpen_teacher_20261112 import weights


SCOPES = ('development', 'fresh')
METHODS = ('base', 'old', 'new', 'clear', 'shuffled')


def apply_gate(x, d, values, cfg):
    output = transform(x, d, values, cfg['mixing'], cfg['suppression']).astype(x.dtype)
    output[values == 0] = x[values == 0]
    return output


def target_direction(bank, x, cfg):
    teacher = bank.teacher.predict_proba(bank.native_scores(x))
    return direction(bank, x, dict(teacher=teacher, preview_posterior=teacher), cfg)


def split_source(dataset, scope):
    if scope == 'development':
        x, p, _, plan = load(dataset)
        return plan, x, p
    return read(OUT / dataset / 'fresh_stimuli.json'), None, None


@torch.no_grad()
def memories(model, conv, previews):
    active = len(previews)
    all_features = torch.from_numpy(np.stack(conv)).cuda()
    encoded = model.encode(all_features)
    reference = model.reference(encoded)
    state = PreviewMemory(model)
    values, controls, same, lifecycle = [], [], [], []
    for h in range(active):
        state.reset()
        assert state.writes == 0 and torch.count_nonzero(state.value) == 0
        p = torch.from_numpy(previews[h]).cuda()
        for i, frame in enumerate(p, 1):
            state.write(frame)
            assert state.writes == i
        state.freeze()
        a = state.value
        assert bool((a >= 0).all())
        before = a.clone()
        ref = reference[h]
        z = encoded[h]
        real = model.read_encoded(z, ref, a)
        torch.testing.assert_close(real[9], state.read(all_features[h, 9]), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(real[9], model(all_features[h, 9], p), rtol=2e-5, atol=2e-6)
        zero = model.read_encoded(z, ref, torch.zeros_like(a))
        assert torch.count_nonzero(zero) == 0
        scramble = torch.stack([shuffled(a, seed) for seed in SHUFFLES])
        torch.testing.assert_close(scramble.flatten(1).sort().values, a.flatten().sort().values.expand(len(SHUFFLES), -1),
                                   rtol=0, atol=0)
        random = model.read_encoded(z, ref, scramble[:, None])
        assert torch.equal(state.value, before)
        values.append(real)
        controls.append(random)
        same.append(model.read_encoded(encoded[0], reference[0], a))
        lifecycle.append([float(a.norm()), float((z > 0).float().mean())])
    state.reset()
    state.freeze()
    blank = state.read(all_features[-1])
    assert torch.count_nonzero(blank) == 0
    return (torch.stack([*values, blank]).cpu().numpy(),
            torch.stack([*controls, blank.expand(len(SHUFFLES), -1)]).cpu().numpy(),
            torch.stack(same).cpu().numpy(), np.asarray(lifecycle))


def collect(dataset, scope, parts, config, calibration, kernel):
    dest, audit = paths(dataset, scope)
    frozen_sha = digest(OUT / 'frozen.json')
    if dest.exists() and audit.exists():
        metadata = read(audit)
        assert metadata['feature_sha256'] == digest(dest) and metadata['frozen_sha256'] == frozen_sha
        return
    plan, old_fc2, previous_preview = split_source(dataset, scope)
    model = load_circuit(dataset)
    bank, _, cuts = fitted(dataset, kernel)
    controls = Controls(tuple(config['hebbian_lambdas']))
    run = Experiment(dataset, parts)
    run.fit()
    if scope == 'fresh':
        vgg, buffers = vanilla()
    x_values, real, random, same, diagnostics, old_p = [], [], [], [], [], []
    try:
        for j, (row, x, p) in enumerate(holdout_stimuli(plan)):
            observed = [trial(run, x, frames, controls) for frames in p]
            observed.append(trial(run, x, [run.c.grey] * 2, controls, active=False))
            conv, preview, fc2, _ = zip(*observed)
            f = np.stack(fc2)
            if old_fc2 is not None:
                np.testing.assert_array_equal(f, old_fc2[j])
            else:
                pixel = p.cpu().numpy()
                patches = [tile for frames in pixel for tile in tiles(frames)]
                old_p.append(encode_tiles(vgg, buffers, patches).reshape(3, 4, 4096))
            scores, scrambled, fixed_target, details = memories(model, conv, preview[:3])
            real.append(scores)
            random.append(scrambled)
            same.append(fixed_target)
            diagnostics.append(details)
            x_values.append(f)
            if (j + 1) % 8 == 0 or j + 1 == len(plan['targets']):
                print('VERIFY', scope, dataset, j + 1, '/', len(plan['targets']), flush=True)
    finally:
        run.detach()
    fc2 = np.stack(x_values)
    n, _, steps, units = fc2.shape
    p = previous_preview if previous_preview is not None else np.stack(old_p)
    old_score = np.zeros((n, 4, steps), dtype=np.float32)
    for h in range(3):
        x = fc2[:, h].reshape(n * steps, units)
        frames = np.repeat(p[:, h], steps, axis=0)
        old_score[:, h] = bank.bank.similarity(x, frames).max(1).reshape(n, steps)
    old_gate = control_gate(old_score, cuts[.5])
    old_gate[:, 3] = 0
    cal = (calibration['threshold'], calibration['slope'])
    new_gate = gate(np.stack(real), *cal)
    shuffled_gate = gate(np.stack(random), *cal)
    assert (new_gate[:, 3] == 0).all() and (shuffled_gate[:, 3] == 0).all()
    tmp = dest.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, fc2=fc2, score_old=old_score, score_new=np.stack(real), score_shuffled=np.stack(random),
                 score_fixed_target=np.stack(same), gate_old=old_gate, gate_new=new_gate, gate_shuffled=shuffled_gate,
                 diagnostics=np.stack(diagnostics), label=[r['label'] for r in plan['targets']],
                 concept=[r['concept'] for r in plan['targets']], index=[r['index'] for r in plan['targets']],
                 slot=[r['slot'] for r in plan['targets']])
    tmp.replace(dest)
    save(audit, dict(scope=scope, feature_sha256=digest(dest), frozen_sha256=frozen_sha, targets=n, steps=steps,
                     collect_source_sha256=digest(Path(__file__)), histories=HISTORIES, state_checks='all trials'))


def archive(dataset, scope):
    path, audit = paths(dataset, scope)
    metadata = read(audit)
    assert metadata['feature_sha256'] == digest(path) and metadata['frozen_sha256'] == digest(OUT / 'frozen.json')
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def online(parts, registration, config, kernel):
    output = {}
    for dataset in DATASETS:
        plan, _, _ = split_source(dataset, 'fresh')
        rendered = renderer(plan)
        data = archive(dataset, 'fresh')
        bank, _, _ = fitted(dataset, kernel)
        model = load_circuit(dataset)
        cfg = config['configs']['target_only']
        controls = Controls(tuple(config['hebbian_lambdas']))
        calibration = registration['datasets'][dataset]['calibration']
        cal = calibration['threshold'], calibration['slope']
        run = Experiment(dataset, parts)
        run.fit()
        checked_trials = {}
        try:
            for h in range(4):
                i = int(data['gate_new'][:, h, 9].argmax())
                row = plan['targets'][i]
                target, frames = rendered(row)
                state = PreviewMemory(model)
                g_seen = []

                def edit(native, conv, memory):
                    if not state.frozen:
                        for frame in memory:
                            state.write(frame)
                        state.freeze()
                    assert state.writes == 2
                    before = state.value._version
                    g = gate(state.read(conv).cpu().numpy(), *cal)
                    g_seen.append(float(g[0]))
                    assert before == state.value._version
                    x = native.cpu().numpy()
                    d = target_direction(bank, x, cfg)
                    return torch.from_numpy(apply_gate(x, d, g, cfg)).to(native)

                p = frames[h] if h < 3 else [run.c.grey] * 2
                _, _, observed, terminal = trial(run, target, p, controls, active=h < 3, edit=edit)
                x = data['fc2'][i, h]
                np.testing.assert_array_equal(x, observed)
                if h < 3:
                    np.testing.assert_allclose(g_seen, data['gate_new'][i, h], atol=1e-5, rtol=1e-5)
                    expected = apply_gate(x, target_direction(bank, x, cfg), data['gate_new'][i, h], cfg)
                    np.testing.assert_allclose(terminal, expected, atol=1e-5, rtol=1e-5)
                    error = float(np.max(np.abs(terminal - expected)))
                else:
                    assert not g_seen and terminal is None
                    error = 0.
                checked_trials[HISTORIES[h]] = dict(target=row['index'], steps=15, terminal_error=error,
                                                     raw_fc2_identical=True)
        finally:
            run.detach()
        output[dataset] = checked_trials
    save(DOC / 'chain/familiarity_online.json', output)
    print('ONLINE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('collect', 'online'))
    args = parser.parse_args()
    registration = checked()
    config = frozen()
    parts = old.initialize()
    kernel = weights()
    if args.phase == 'collect':
        for scope in SCOPES:
            for ds in DATASETS:
                collect(ds, scope, parts, config, registration['datasets'][ds]['calibration'], kernel)
    else:
        online(parts, registration, config, kernel)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
