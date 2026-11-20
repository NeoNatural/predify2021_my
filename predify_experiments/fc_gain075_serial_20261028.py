"""FC1 corr gain .75 k32/k128: identical SVM and true serial FC2 evaluation."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from . import fc_serial_20261028 as serial
from . import fc_lateral_study_20261028 as core
from . import fc_finite_steps_20261028 as finite
from .fc_serial_svm_completion_20261028 import OUT as PREVIOUS_SVM, validate_original
from .fc_lateral_vanilla_20261028 import ROOT, sha, write_json

OUT = ROOT / 'Hebbian-CNN/Log/fc_gain075_serial_20261028'
DOC = ROOT / 'docs/fc_gain075_serial_20261028'
GRAPHS = ('k32', 'k128')
TIMES = (1, 5, 20)
STAGES = ('fc1_output', 'fc2_input', 'fc2_output', 'fc2_local')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parameters(graph):
    assert graph in GRAPHS
    return (dict(rule='corr', graph=graph, mode='hard', gain=.75, budget=1.),
            dict(rule='corr', graph='k32', mode='hard', gain=.75, budget=1.))


def conditions():
    rows = []
    def add(name, m1='off', t1=0, m2='off', t2=0, calibration='none'):
        rows.append(dict(name=name, m1=m1, t1=t1, m2=m2, t2=t2, calibration=calibration))
    add('baseline')
    for t in TIMES:
        add(f'fc1_hard_{t}', 'hard', t)
        add(f'fc2_hard_{t}', m2='hard', t2=t, calibration='frozen')
        for cal in ('frozen', 'matched'):
            add(f'both_hard_hard_{t}_{t}_{cal}', 'hard', t, 'hard', t, cal)
    for cal in ('frozen', 'matched'):
        add(f'gain1_hard_5_{cal}', 'gain_hard', 5, 'hard', 5, cal)
        add(f'gain2_hard_5_{cal}', 'hard', 5, 'gain_hard', 5, cal)
        add(f'gain_both_hard_5_{cal}', 'gain_hard', 5, 'gain_hard', 5, cal)
    add('inhibition_only_hard_5', 'inhib_hard', 5, 'inhib_hard', 5, 'matched')
    add('fc1_inhibition_hard_5', 'inhib_hard', 5)
    add('fc2_inhibition_hard_5', m2='inhib_hard', t2=5, calibration='frozen')
    add('fc1_gain_5', 'gain_hard', 5)
    assert len(rows) == 23 and len({r['name'] for r in rows}) == 23
    return rows


def feature_keys(graph, c):
    mode, t = c['m1'], c['t1']
    # Only truly graph-independent features share fitted readouts.
    tag = f'{mode}_{t}' if mode in ('off', 'inhib_hard') else f'{graph}_{mode}_{t}'
    incoming = 'fc2_input__' + tag
    if c['m2'] == 'off':
        final = incoming
    elif mode == 'off' or c['name'] == 'inhibition_only_hard_5':
        final = 'fc2_output__' + c['name']
    else:
        final = f"fc2_output__{graph}_{c['name']}"
    return dict(fc1_output='fc1__' + tag, fc2_input=incoming, fc2_output=final)


def validate(protocol):
    original = validate_original(protocol)
    assert sha(__file__) == protocol['source_sha256']
    for p, h in protocol['dependencies'].items():
        assert sha(ROOT / p) == h
    for d, inputs in protocol['inputs'].items():
        for p, h in inputs.items():
            assert sha(serial.DATA / d / p) == h
    return original


def freeze():
    old = read(serial.OUT / 'protocol.json')
    dependencies = dict(old['dependencies'])
    dependencies.update({p:sha(ROOT / p) for p in (
        'predify_experiments/fc_serial_20261028.py',
        'predify_experiments/fc_serial_svm_completion_20261028.py',
        'docs/fc_finite_steps_20261028/grid_fold_metrics.csv',
        'docs/fc_serial_20261028/layer_summary.json',
        'docs/fc_serial_20261028/layer_fold_metrics.csv')})
    protocol = dict(version=1, source_sha256=sha(__file__),
        serial_protocol_sha256=sha(serial.OUT / 'protocol.json'), dependencies=dependencies,
        inputs=old['inputs'], graphs=GRAPHS, times=TIMES, conditions=conditions(),
        fc1='positive regularized correlation; mutual k32 or k128; gain .75; hard adaptive gate; budget1',
        fc2=parameters('k32')[1], division_strength=2, relaxation=.5,
        architecture=old['architecture'],
        calibration=old['calibration'],
        primary_goal='activation down + normalized within down + Fisher up, separately per layer; SVM functional check only',
        primary_comparison='5 steps each; 1 and20 steps sensitivity; all comparisons fixed before this run',
        controls='FC1/FC2-only; inhibition-only; mean-matched FC1 gain + FC2 Hebb; FC1 Hebb + FC2 gain; gains both; FC1 gain-only',
        control_target='per-image gain controls target corresponding k32/k128 gain.75 hard5 response, never old gain.25 response',
        svm=dict(model='StandardScaler + LinearSVC', C=1, dual='auto', max_iter=30000, random_state=core.SEED,
                 sklearn_version=sklearn.__version__, preprocessing='train-only class/concept weighted scaler and classifier; physical amplitudes retained'),
        selection='none; no parameter/step choice from SVM or outer test outcomes; reused exploratory cohorts',
        previous_svm_complete_hashes={f'{d}/{f}':sha(PREVIOUS_SVM / d / f'fold{f}' / 'complete.json') for d in ('imagenet', 'things') for f in range(5)})
    OUT.mkdir(exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        assert read(p) == json.loads(json.dumps(protocol)), 'Frozen experiment changed'
    else:
        write_json(p, protocol)
        (OUT / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    validate(protocol)
    return protocol


def run_fold(dataset, fold, x1, x2, y, concepts, folds, heads, fine, w3, b3, w6, b6, protocol, grid):
    dest = OUT / dataset / f'fold{fold}'
    if (dest / 'complete.json').exists():
        print('EXISTS', dataset, fold, flush=True)
        return
    started = time.monotonic()
    tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
    assert not set(concepts[tr]) & set(concepts[te])
    order, n = np.concatenate((tr, te)), len(tr)
    yt = torch.as_tensor(y[te], device=serial.DEVICE)
    previous = PREVIOUS_SVM / dataset / f'fold{fold}'
    assert sha(previous / 'complete.json') == protocol['previous_svm_complete_hashes'][f'{dataset}/{fold}']
    for p, h in read(previous / 'complete.json')['files'].items():
        assert sha(previous / p) == h
    previous_result = read(previous / 'results.json')
    assert previous_result['train_rows'] == tr.tolist() and previous_result['test_rows'] == te.tolist()
    with (previous / 'svm_models.pckl').open('rb') as f:
        old_models = pickle.load(f)
    reused_keys = ('fc1__off_0', 'fc2_input__off_0', 'fc1__inhib_hard_5',
                   'fc2_input__inhib_hard_5', 'fc2_output__inhibition_only_hard_5')
    existing = {k:old_models[k] for k in reused_keys}
    for t in TIMES:
        k = f'fc2_output__fc2_hard_{t}'
        existing[k] = old_models[k]
    del old_models
    model_bank, score_bank, metric_bank, features, feature_hashes = {}, {}, {}, {}, {}
    graphs, trained, reused = {}, 0, 0
    replay_grid_error = replay_svm_error = 0.
    bridges = (serial.bridge(x1[te], w3, b3) - x2[te]).abs().max()
    assert float(bridges) < 1e-4

    def characterize(key, r, layer):
        nonlocal trained, reused, replay_svm_error
        if key in metric_bank:
            assert np.array_equal(r[n:].cpu().numpy(), features[key])
            return copy.deepcopy(metric_bank[key])
        reference = x1[te] if layer == 'fc1' else x2[te]
        test = r[n:]
        m = finite.evaluate(test, reference, yt)
        m.update(core.geometry(test, yt, detailed=True))
        if key in existing:
            model = existing[key]
            scores = model.decision_function(test.cpu().numpy())
            reused += 1
        else:
            scores, model = core.svm_fit(r[:n], test, y[tr], concepts[tr])
            trained += 1
        assert model[-1].C == 1 and model[-1].random_state == core.SEED
        assert int(model[-1].n_iter_) < model[-1].max_iter
        m.update(core.classification(scores, y[te]))
        m['frozen_svm'] = core.classification(heads[layer][fold].decision_function(test.cpu().numpy()), y[te])
        m['original_imagenet_top1_pct'] = m['original_imagenet_top5_pct'] = None
        if dataset == 'imagenet' and layer == 'fc2':
            logits = test @ w6.T + b6
            target = torch.as_tensor(fine[te], device=serial.DEVICE)
            m['original_imagenet_top1_pct'] = 100 * float((logits.argmax(1) == target).float().mean())
            m['original_imagenet_top5_pct'] = 100 * float((logits.topk(5, dim=1).indices == target[:, None]).any(1).float().mean())
        if key in existing:
            error = float(np.abs(scores - old_scores[key]).max())
            replay_svm_error = max(replay_svm_error, error)
            assert error < 1e-8
        feature_hashes[key] = sha_array(r[:n].cpu().numpy())
        model_bank[key], score_bank[key] = model, scores
        metric_bank[key], features[key] = copy.deepcopy(m), test.cpu().numpy()
        return m

    with np.load(previous / 'scores.npz') as old_scores:
        for graph in GRAPHS:
            c1, c2 = parameters(graph)
            runner = serial.SerialRunner(x1[order], x2[order], n, c1, c2, w3, b3)
            base1 = characterize('fc1__off_0', runner.x1, 'fc1')
            base2 = characterize('fc2_input__off_0', runner.x2, 'fc2')
            results = {}
            for i, condition in enumerate(conditions()):
                name = condition['name']
                stage = runner.stage(condition['m1'], condition['t1'])
                final = runner.output(condition)
                keys = feature_keys(graph, condition)
                values = dict(fc1_output=stage['fc1'], fc2_input=stage['fc2_input'], fc2_output=final)
                metrics = {}
                for which in ('fc1_output', 'fc2_input', 'fc2_output'):
                    layer = 'fc1' if which == 'fc1_output' else 'fc2'
                    m = characterize(keys[which], values[which], layer)
                    finite.add_effects(m, base1 if layer == 'fc1' else base2)
                    metrics[which] = m
                local = copy.deepcopy(metrics['fc2_output'])
                local.update(finite.evaluate(final[n:], stage['fc2_input'][n:], yt))
                finite.add_effects(local, metrics['fc2_input'])
                metrics['fc2_local'] = local
                if condition['m2'] in ('hard', 'inhib_hard'):
                    assert local['max_support_ratio'] <= 1.00001
                if condition['m1'] == 'hard' and condition['m2'] == 'off':
                    want = grid[(graph, condition['t1'])]
                    for field in ('activation', 'within', 'between', 'fisher'):
                        error = abs(metrics['fc1_output'][field] - float(want[field]))
                        replay_grid_error = max(replay_grid_error, error)
                        assert error < 1e-5, (dataset, fold, graph, field, error)
                if condition['m2'] == 'gain_hard':
                    canonical = runner.output(dict(m1='hard', t1=5, m2='hard', t2=5, calibration=condition['calibration']))
                    assert torch.allclose(final.mean(1), canonical.mean(1), atol=2e-6, rtol=1e-5)
                if condition['m1'] == 'gain_hard':
                    assert torch.allclose(stage['fc1'].mean(1), runner.stage('hard', 5)['fc1'].mean(1), atol=2e-6, rtol=1e-5)
                assert torch.isfinite(final).all() and bool((final >= 0).all())
                results[name] = dict(condition=condition, feature_keys=keys, metrics=metrics)
                if i % 8 == 7:
                    print('CONDITION', dataset, fold, graph, i + 1, '/23', 'SVM new/reused', trained, reused,
                          'seconds', round(time.monotonic() - started, 1), flush=True)
            graphs[graph] = dict(params1=c1, params2=c2, results=results,
                matched_models={str(k):dict(scale=float(v['scale']), graph_statistics=v['graph_statistics']) for k,v in runner.models2.items()})
            print('GRAPH DONE', dataset, fold, graph, 'seconds', round(time.monotonic() - started, 1), flush=True)
            del runner
    assert len(features) == 49 and trained == 41 and reused == 8
    write_json(dest / 'results.json', dict(dataset=dataset, fold=fold, train_rows=tr.tolist(), test_rows=te.tolist(),
        graphs=graphs, unique_readouts=len(features), trained=trained, reused=reused,
        fc1_grid_replay_max_error=replay_grid_error, old_svm_score_max_error=replay_svm_error,
        bridge_baseline_max_error=float(bridges), train_representation_sha256=feature_hashes))
    np.savez_compressed(dest / 'representations.npz', row=te, y=y[te], concept=concepts[te], **features)
    np.savez_compressed(dest / 'scores.npz', row=te, y=y[te], **score_bank)
    with (dest / 'svm_models.pckl').open('wb') as f:
        pickle.dump(model_bank, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(dest / 'complete.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}, seconds=time.monotonic() - started))
    print('FOLD DONE', dataset, fold, 'SVM new/reused', trained, reused, 'grid replay', replay_grid_error,
          'SVM replay', replay_svm_error, 'seconds', round(time.monotonic() - started, 1), flush=True)


def sha_array(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()


def run(dataset, requested_folds):
    protocol = freeze()
    original = validate(protocol)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    weights = torch.load(original['checkpoint'], map_location='cpu', weights_only=True)
    w3, b3 = (weights[f'classifier.3.{k}'].to(serial.DEVICE) for k in ('weight', 'bias'))
    w6, b6 = (weights[f'classifier.6.{k}'].to(serial.DEVICE) for k in ('weight', 'bias'))
    del weights
    with np.load(serial.DATA / dataset / 'activations.npz') as z:
        x1, x2 = (torch.as_tensor(z[k], device=serial.DEVICE) for k in ('fc1', 'fc2'))
        y, concepts, folds = z['category'], z['concept'], z['fold']
    with (serial.DATA / dataset / 'baseline_models.pckl').open('rb') as f:
        heads = pickle.load(f)['models']
    manifest = read(serial.DATA / dataset / 'manifest.json')
    fine = np.array([r['imagenet_label'] for r in manifest['rows']])
    with (ROOT / 'docs/fc_finite_steps_20261028/grid_fold_metrics.csv').open(encoding='utf-8-sig') as f:
        previous_grid = [r for r in csv.DictReader(f) if r['dataset'] == dataset and r['layer'] == 'fc1'
                         and r['rule'] == 'corr' and r['mode'] == 'hard' and float(r['gain']) == .75
                         and float(r['budget']) == 1. and int(r['step']) in TIMES and r['graph'] in GRAPHS]
    with torch.inference_mode():
        for fold in requested_folds:
            grid = {(r['graph'], int(r['step'])):r for r in previous_grid if int(r['fold']) == fold}
            assert len(grid) == 6
            run_fold(dataset, fold, x1, x2, y, concepts, folds, heads, fine, w3, b3, w6, b6, protocol, grid)
    validate(protocol)
    print('SHARD COMPLETE', dataset, requested_folds, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--dataset', choices=('imagenet', 'things'))
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    if args.action == 'freeze':
        print('FROZEN', len(freeze()['conditions']), 'conditions per graph')
    else:
        for dataset in ([args.dataset] if args.dataset else ['imagenet', 'things']):
            run(dataset, args.folds)
