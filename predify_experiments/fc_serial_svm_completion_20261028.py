"""Complete identical train-only SVM readouts for all serial FC stages."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

from . import fc_serial_20261028 as serial
from . import fc_lateral_study_20261028 as core
from .fc_lateral_vanilla_20261028 import ROOT, sha, write_json

OUT = serial.OUT / 'svm_layer_completion_20261028'
STAGES = ('fc1_output', 'fc2_input', 'fc2_output')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def feature_keys(condition):
    tag = f"{condition['m1']}_{condition['t1']}"
    incoming = 'fc2_input__' + tag
    return dict(fc1_output='fc1__' + tag, fc2_input=incoming,
                fc2_output=incoming if condition['m2'] == 'off' else 'fc2_output__' + condition['name'])


def validate_original(protocol):
    assert sha(serial.OUT / 'protocol.json') == protocol['serial_protocol_sha256']
    original = read(serial.OUT / 'protocol.json')
    assert sha(ROOT / 'predify_experiments/fc_serial_20261028.py') == original['source_sha256']
    for p, h in {**original['dependencies'], **original['original_model_sources']}.items():
        assert sha(ROOT / p) == h
    assert sha(original['checkpoint']) == original['checkpoint_sha256']
    return original


def freeze():
    original = read(serial.OUT / 'protocol.json')
    protocol = dict(version=1, scope='all 57 original conditions; FC1, actual FC2 input, final FC2; reuse identical features across downstream settings',
        request='complete FC1 with the identical SVM settings; SVM primary on both layers; sharpening primary objective; original model unchanged',
        selection='no new hyperparameter, feature, condition, or horizon selection',
        classifier=dict(type='StandardScaler + LinearSVC', C=1, dual='auto', max_iter=30000,
                        random_state=core.SEED, sklearn_version=sklearn.__version__,
                        weighting='same class/concept-balanced training sample weights for scaler and LinearSVC',
                        fitting='training rows only; original feature amplitudes retained; ConvergenceWarning treated as error'),
        serial_protocol_sha256=sha(serial.OUT / 'protocol.json'),
        source_sha256=sha(__file__),
        inputs=original['inputs'],
        serial_fold_complete_hashes={f'{d}/{f}':sha(serial.OUT / d / f'fold{f}' / 'complete.json') for d in ('imagenet', 'things') for f in range(5)})
    OUT.mkdir(exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        assert read(p) == protocol, 'Supplement source/protocol changed'
    else:
        write_json(p, protocol)
        (OUT / 'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    validate_original(protocol)
    return protocol


def run_fold(dataset, fold, x1, x2, y, concepts, folds, heads, w, b, protocol):
    dest = OUT / dataset / f'fold{fold}'
    if (dest / 'complete.json').exists():
        print('EXISTS', dataset, fold, flush=True)
        return
    started = time.monotonic()
    prior = serial.OUT / dataset / f'fold{fold}'
    assert sha(prior / 'complete.json') == protocol['serial_fold_complete_hashes'][f'{dataset}/{fold}']
    complete = read(prior / 'complete.json')
    for p, digest in complete['files'].items():
        assert sha(prior / p) == digest
    source = read(prior / 'results.json')
    tr, te = np.array(source['train_rows']), np.array(source['test_rows'])
    assert np.array_equal(tr, np.flatnonzero(folds != fold))
    assert np.array_equal(te, np.flatnonzero(folds == fold))
    assert not set(concepts[tr]) & set(concepts[te])
    order = np.concatenate((tr, te))
    n = len(tr)
    runner = serial.SerialRunner(x1[order], x2[order], n, source['params1'], source['params2'], w, b)
    with (prior / 'svm_models.pckl').open('rb') as f:
        old_models = pickle.load(f)
    existing = {'fc1__off_0': heads['fc1'][fold], 'fc2_input__off_0': heads['fc2'][fold]}
    for name, model in old_models.items():
        existing[feature_keys(source['results'][name]['condition'])['fc2_output']] = model
    fitted, scores, metrics, mapping = {}, {}, {}, {}
    trained = reused = 0
    replay_error = score_error = 0.

    def characterize(key, representation):
        nonlocal trained, reused
        if key in metrics:
            return metrics[key]
        if key in existing:
            model = existing[key]
            result = model.decision_function(representation[n:].cpu().numpy())
            reused += 1
        else:
            result, model = core.svm_fit(representation[:n], representation[n:], y[tr], concepts[tr])
            trained += 1
        assert result.shape == (len(te), 5) and np.isfinite(result).all()
        assert model[-1].C == 1 and model[-1].random_state == core.SEED
        assert int(model[-1].n_iter_) < model[-1].max_iter
        fitted[key], scores[key] = model, result
        metrics[key] = core.classification(result, y[te])
        return metrics[key]

    with np.load(prior / 'intermediate_stages.npz') as stages, np.load(prior / 'final_fc2.npz') as final, np.load(prior / 'readout_scores.npz') as old_scores:
        assert np.array_equal(final['row'], te) and np.array_equal(stages['row'], te)
        for i, condition in enumerate(serial.conditions()):
            name = condition['name']
            tag = f"{condition['m1']}_{condition['t1']}"
            stage = runner.stage(condition['m1'], condition['t1'])
            output = runner.output(condition)
            values = dict(fc1_output=stage['fc1'], fc2_input=stage['fc2_input'], fc2_output=output)
            references = dict(fc1_output=stages['fc1_' + tag], fc2_input=stages['input2_' + tag], fc2_output=final[name])
            keys = feature_keys(condition)
            for which in STAGES:
                error = float(np.abs(values[which][n:].cpu().numpy() - references[which]).max())
                replay_error = max(replay_error, error)
                assert error < 1e-6, (dataset, fold, name, which, error)
                characterize(keys[which], values[which])
            mapping[name] = keys
            if 'refit_svm' in source['results'][name]['metrics']:
                result = scores[keys['fc2_output']]
                ref = old_scores[name + '_svm'] if name != 'baseline' else heads['fc2'][fold].decision_function(final[name])
                error = float(np.abs(result - ref).max())
                score_error = max(score_error, error)
                assert error < 1e-8
            if i % 12 == 11:
                print('SVM', dataset, fold, i + 1, '/57', 'trained', trained, 'reused', reused,
                      'seconds', round(time.monotonic() - started, 1), flush=True)
    results = {name:{stage:metrics[key] for stage,key in keys.items()} for name,keys in mapping.items()}
    write_json(dest / 'results.json', dict(dataset=dataset, fold=fold, train_rows=tr.tolist(), test_rows=te.tolist(),
        results=results, feature_keys=mapping, unique_readouts=len(metrics), trained=trained, reused=reused,
        replay_max_error=replay_error, existing_svm_score_max_error=score_error,
        protocol_sha256=sha(OUT / 'protocol.json')))
    np.savez_compressed(dest / 'scores.npz', row=te, y=y[te], **scores)
    with (dest / 'svm_models.pckl').open('wb') as f:
        pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(dest / 'complete.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
        files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}, seconds=time.monotonic() - started))
    print('DONE', dataset, fold, 'unique', len(metrics), 'trained', trained, 'reused', reused,
          'replay', replay_error, 'existing_svm_error', score_error,
          'seconds', round(time.monotonic() - started, 1), flush=True)


def run(dataset):
    protocol = freeze()
    original = validate_original(protocol)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for p, digest in protocol['inputs'][dataset].items():
        assert sha(serial.DATA / dataset / p) == digest
    weights = torch.load(original['checkpoint'], map_location='cpu', weights_only=True)
    w, b = (weights[f'classifier.3.{k}'].to(serial.DEVICE) for k in ('weight', 'bias'))
    del weights
    with np.load(serial.DATA / dataset / 'activations.npz') as z:
        x1 = torch.as_tensor(z['fc1'], device=serial.DEVICE)
        x2 = torch.as_tensor(z['fc2'], device=serial.DEVICE)
        y, concepts, folds = z['category'], z['concept'], z['fold']
    with (serial.DATA / dataset / 'baseline_models.pckl').open('rb') as f:
        heads = pickle.load(f)['models']
    with torch.inference_mode():
        for fold in range(5):
            run_fold(dataset, fold, x1, x2, y, concepts, folds, heads, w, b, protocol)
    validate_original(protocol)
    for p, digest in protocol['inputs'][dataset].items():
        assert sha(serial.DATA / dataset / p) == digest
    print('DATASET COMPLETE', dataset, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--dataset', choices=('imagenet', 'things'))
    args = parser.parse_args()
    if args.action == 'freeze':
        print('FROZEN', freeze()['classifier'])
    else:
        for dataset in ([args.dataset] if args.dataset else ['imagenet', 'things']):
            run(dataset)
