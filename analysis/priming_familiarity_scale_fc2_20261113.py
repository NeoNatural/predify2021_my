"""Check FC2 only where the validation-selected model shows held-out growth."""

from __future__ import annotations

import argparse

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from hebb_chain_20261111 import Controls, Experiment, old
from priming_familiarity_20261112 import Familiarity, calibrate, gate
from priming_familiarity_data_20261112 import trial
from priming_familiarity_scale_collect_20261113 import SHARD, shard_path
from priming_familiarity_scale_data_20261113 import HISTORIES, LEVELS, MANIFEST, SEEDS, Renderer
from priming_familiarity_scale_fit_20261113 import SELECTION, protocol, run_path
from priming_familiarity_scale_report_20261113 import OUT, write
from priming_familiarity_verify_20261112 import apply_gate, target_direction
from priming_sharpen_collect_20261112 import digest, vanilla
from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_object_precision_20261112 import encode_tiles, fitted, tiles
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_search_20261112 import control_gate, read, save
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def data_fc2(dataset, split, rows):
    key = 'fc2' if split == 'validation' else 'native_fc2'
    out = []
    for start in range(0, len(rows), SHARD):
        with np.load(shard_path(dataset, split, start)) as f:
            np.testing.assert_array_equal(f['index'], [r['index'] for r in rows[start:start + SHARD]])
            out.append(f[key])
    return np.concatenate(out)


def preview_objects(rows, model, buffers):
    render = Renderer(rows)
    indices, patches, memo = [], [], {}
    for row in rows:
        _, first, last = render(row)
        trial = []
        for frame in last:
            group = []
            for patch in tiles(torch.stack((first, frame)).cpu().numpy()):
                key = patch.tobytes()
                if key not in memo:
                    memo[key] = len(patches)
                    patches.append(patch)
                group.append(memo[key])
            trial.append(group)
        indices.append(trial)
    return encode_tiles(model, buffers, patches)[np.array(indices)]


def old_gate(bank, x, p, cut):
    return np.column_stack([control_gate(bank.bank.similarity(x[:, i], p[:, i]).max(1), cut) for i in range(3)])


def main():
    protocol()
    support = read(OUT / 'heldout_summary.json')
    native = read(OUT / 'native_heldout_summary.json')
    selected = read(SELECTION)
    eligible = [ds for ds, info in support.items() if info['evidence_of_growth'] and native[ds]['evidence_of_growth']]
    if not eligible:
        print('No validation-selected configuration met the held-out growth criterion.', flush=True)
        return
    cfg = frozen()['configs']['target_only']
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    kernel = weights()
    model, buffers = vanilla()
    plan = read(MANIFEST)
    table = []
    for ds in eligible:
        data = plan[ds]['splits']
        bank, head, cuts = fitted(ds, kernel)
        val = data_fc2(ds, 'validation', data['validation'])
        val_old = old_gate(bank, np.repeat(val[:, None], 3, axis=1),
                           preview_objects(data['validation'], model, buffers), cuts[.5])
        arm, level = (selected['chosen'][ds][key] for key in ('arm', 'level'))
        calibration = {}
        for name, size in (('small', LEVELS[0]), ('selected', level)):
            for seed in SEEDS:
                path = run_path(ds, arm, size, seed)
                with np.load(path.with_suffix('.npz')) as f:
                    calibration[name, seed] = calibrate(f['validation'], val_old)
        save(OUT / f'{ds}_fc2_calibration.json', dict(selection_sha256=digest(SELECTION),
             folds='new validation only; shared t9 target for old calibration gate',
             calibrations={f'{name}_{seed}': v for (name, seed), v in calibration.items()}))
        x = data_fc2(ds, 'test', data['test'])
        p = preview_objects(data['test'], model, buffers)
        s = PairedConcepts(np.array([r['label'] for r in data['test']]), np.array([r['concept'] for r in data['test']]))
        d = [target_direction(bank, x[:, i], cfg) for i in range(3)]

        def measured(g):
            return [metrics(apply_gate(x[:, i], d[i], g[:, i], cfg), head, s) for i in range(3)]

        outcomes = dict(base=[[metrics(x[:, i], head, s) for i in range(3)]],
                        old_gate=[measured(old_gate(bank, x, p, cuts[.5]))], small=[], selected=[])
        for name, size in (('small', LEVELS[0]), ('selected', level)):
            for seed in SEEDS:
                path = run_path(ds, arm, size, seed)
                with np.load(path.with_suffix('.native_test.npz')) as f:
                    cal = calibration[name, seed]
                    outcomes[name].append(measured(gate(f['scores'], cal['threshold'], cal['slope'])))
        for h in (1, 2):
            effects = {}
            for name, methods in outcomes.items():
                reps = [contrast(v[0], v[h]) for v in methods]
                effects[name] = {k: np.mean([v[k] for v in reps], axis=0) for k in reps[0]}
                table.extend(dict(dataset=ds, condition=name, reference='', history=HISTORIES[h], metric=k,
                                  **estimate(v)) for k, v in effects[name].items())
            for ref in ('base', 'old_gate', 'small'):
                table.extend(dict(dataset=ds, condition='selected', reference=ref, history=HISTORIES[h], metric=k,
                                  **estimate(v - effects[ref][k])) for k, v in effects['selected'].items())
    write('fc2_test', table)


def online():
    protocol()
    selected = read(SELECTION)
    plans = read(MANIFEST)
    cfg = frozen()
    controls = Controls(tuple(cfg['hebbian_lambdas']))
    parts = old.initialize()
    kernel = weights()
    seed = SEEDS[0]
    report = {}
    for ds in selected['chosen']:
        calibration_path = OUT / f'{ds}_fc2_calibration.json'
        if not calibration_path.exists():
            continue
        cal = read(calibration_path)
        assert cal['selection_sha256'] == digest(SELECTION)
        g = cal['calibrations'][f'selected_{seed}']
        calibration = g['threshold'], g['slope']
        pick = selected['chosen'][ds]
        path = run_path(ds, pick['arm'], pick['level'], seed)
        model_path = path.with_suffix('.pth')
        assert digest(model_path) == selected['model_sha256'][ds][path.name]
        model = Familiarity().cuda().eval()
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        model.requires_grad_(False)
        with np.load(path.with_suffix('.native_test.npz')) as f:
            native_scores = f['scores']
        rows = plans[ds]['splits']['test']
        cache = data_fc2(ds, 'test', rows)
        strengths = gate(native_scores, *calibration)
        separation = strengths[:, 0] - strengths[:, 1:].mean(1)
        i = int(separation.argmax())
        assert separation[i] > 0
        image, first, last = Renderer(rows)(rows[i])
        bank, _, _ = fitted(ds, kernel)
        params = cfg['configs']['target_only']
        run = Experiment(ds, parts)
        run.fit()
        histories = {}
        try:
            for h, name in enumerate(HISTORIES):
                gates = []

                def edit(x, conv, memory):
                    strength = gate(model(conv, memory[None]).cpu().numpy(), *calibration)
                    gates.append(float(strength[0]))
                    response = x.cpu().numpy()
                    edited = apply_gate(response, target_direction(bank, response, params), strength, params)
                    return torch.from_numpy(edited).to(x)

                conv, p, raw, received = trial(run, image, (first, last[h]), controls, edit=edit)
                with torch.inference_mode():
                    scores = model(torch.from_numpy(conv).cuda(), torch.from_numpy(p).cuda()).cpu().numpy()
                expected = apply_gate(raw, target_direction(bank, raw, params), gate(scores, *calibration), params)
                np.testing.assert_allclose(received, expected, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(raw[9], cache[i, h], atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(scores[9], native_scores[i, h], atol=2e-5, rtol=2e-5)
                histories[name] = dict(steps=15, gate_t9=gates[9],
                                       terminal_max_abs_error=float(np.abs(received - expected).max()),
                                       batched_fc2_max_abs_error=float(np.abs(raw[9] - cache[i, h]).max()))
        finally:
            run.detach()
        report[ds] = dict(target=rows[i]['index'], seed=seed, histories=histories)
    save(OUT / 'fc2_online.json', report)
    print('FC2 ONLINE', report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        (online if args.online else main)()
