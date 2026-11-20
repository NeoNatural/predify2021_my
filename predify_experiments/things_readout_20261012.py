"""THINGS replication of the readout/initialization study; separate frozen artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from .supplementary import ROOT, PAPER, load, save, digest
from .things_readout_data import ThingsImages, prepare_manifest, DATA_ROOT

OUT = ROOT / 'Hebbian-CNN/Log/things_readout_20261012'
DOC = ROOT / 'docs/things_readout_20261012'
SEED = 20261013
STEPS = 15
IDENTITIES = ('exact', 'same_synset', 'same_category', 'different_category', 'neutral')
CONDITIONS = ('none_random', 'none_feedforward') + tuple('carry_' + x for x in IDENTITIES)
TRAIN_CONDITIONS = ('none_random', 'none_feedforward', 'carry_neutral')
NOISES = {'train': (1201, 1202, 1203), 'dev': (1211, 1212, 1213), 'test': (1221, 1222, 1223)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != value:
            raise ValueError(f'Refusing to replace frozen specification: {path}')
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def budget_index(cumulative_calls, budget):
    indices = np.flatnonzero(np.asarray(cumulative_calls) == budget)
    if len(indices) != 1:
        raise ValueError(f'Budget {budget} not represented uniquely')
    return int(indices[0])


def prepare():
    return prepare_manifest(OUT, DOC, SEED, NOISES, CONDITIONS, TRAIN_CONDITIONS)


def extract(split, limit=None, smoke=False):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torchvision.models import VGG16_Weights
    from .config import load_experiment_config
    from .paper_parafoveal import _FeatureRecorder, _add_legacy_module_path
    from .hebb_vgg_priming import _build_model
    from .paper_stimuli import make_paper_canvas
    from .sequence import reset_hebbian_state, run_recurrent_sequence
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf8'))
    frozen_path = OUT / 'frozen_readout.json'
    if split == 'test':
        frozen = json.loads(frozen_path.read_text())
        assert sha(OUT / 'readouts.pckl') == frozen['readouts_sha256']
        assert sha(OUT / 'manifest.json') == frozen['manifest_sha256']
        assert sha(DOC / '预先方案.md') == frozen['protocol_sha256']
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    _add_legacy_module_path(ROOT)
    from Hebb_Priming_Utils import apply_input_degradation
    config = load_experiment_config(ROOT / 'configs/paper_memory_carrier_calibrated.toml').with_overrides(
        imagenet_root='C:/Users/liang/Documents/ImageNet', device='cuda')
    config.validate_for_run()
    torch.manual_seed(SEED)
    model, layers, _ = _build_model(config, torch.device('cuda'), torch)
    recorder = _FeatureRecorder(model, layers, torch)
    dataset = ThingsImages(manifest, VGG16_Weights.IMAGENET1K_V1.transforms())
    directory = OUT / ('smoke' if smoke else split)
    directory.mkdir(parents=True, exist_ok=True)
    source_hash = sha(__file__)
    dependencies = ['predify_experiments/things_readout_data.py', 'predify_experiments/things_metadata.py',
                    'predify_experiments/sequence.py', 'predify_experiments/paper_stimuli.py',
                    'predify_experiments/hebb_vgg_priming.py', 'predify_experiments/paper_parafoveal.py',
                    'Hebbian-CNN/Hebb_Priming_Utils.py', 'Hebbian-CNN/Hebbian_VGG_Lib.py',
                    'predify2021/model_factory/get_model.py', 'predify2021/model_factory/pvgg16_shared.py']
    dependency_hashes = {p:sha(ROOT/p) for p in dependencies}
    immutable_json(OUT / 'implementation.json', dict(runner_sha256=source_hash, dependencies=dependency_hashes))
    for p in dependencies:
        copy = OUT / 'source_snapshot' / p
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((ROOT/p).read_bytes())
    immutable_json(directory / 'provenance.json', dict(source_sha256=source_hash, manifest_sha256=sha(OUT / 'manifest.json'),
        dataset='THINGS', dataset_root=str(DATA_ROOT), implementation=dependency_hashes,
        torch=torch.__version__, numpy=np.__version__, device=torch.cuda.get_device_name(), config=json.loads(json.dumps(asdict(config), default=str)),
        pcoder_weights={p.name: sha(p) for p in config.paths.pcoder_weight_dir.glob('pvgg16*pmodule.pth')},
        legacy_readout_sha256=sha(PAPER / 'shared_decoder/analysis/shared_decoders.pckl')))
    (directory / f'source_{source_hash[:12]}.py').write_bytes(Path(__file__).read_bytes())
    calls = [0]
    handle = model.backbone.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    def set_init(random_init):
        model.random_init = random_init
        for j in range(1, 6):
            getattr(model, f'pcoder{j}').random_init = random_init
    def canvas(values):
        return make_paper_canvas(values, object_scale=.15, horizontal_offset=.175, background_value=.5)
    def execute(trial, condition, target, images, waste_preview=False):
        carry = condition.startswith('carry_')
        identity = condition.removeprefix('carry_') if carry else 'exact'
        random_init = condition != 'none_feedforward'
        reset_hebbian_state(layers); model.reset(); set_init(random_init)
        for layer in layers:
            layer.update_enabled = False
        calls[0] = 0
        torch.cuda.synchronize()
        start = time.perf_counter()
        preview_indices = list(trial['previews'])
        preview_indices[trial['slot']] = trial['primes'][identity]
        if carry or waste_preview:
            for stage in range(2):
                torch.manual_seed((SEED + trial['dataset_index'] * 1009 + stage * 100003) % (2**31-1))
                preview = canvas([(images[i], side) for i, side in zip(preview_indices[stage*2:stage*2+2], ('left', 'right')) if i is not None])
                run_recurrent_sequence(model, preview[None], 2, hebbian_layers=layers,
                                       commit_update=False, preserve_model_state=carry and stage > 0)
        torch.cuda.synchronize()
        preview_seconds = time.perf_counter() - start
        preview_calls = calls[0]
        torch.manual_seed((SEED + trial['dataset_index'] * 1009 + 2 * 100003) % (2**31-1))
        features, logits, totals, elapsed = [], [], [], []
        def after_step(t, output):
            features.append(recorder.snapshot()['fc2'])
            logits.append(output.detach().cpu().numpy()[0])
            totals.append(calls[0])
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
        run_recurrent_sequence(model, target[None], STEPS, hebbian_layers=layers, commit_update=False,
                               preserve_model_state=carry, after_step=after_step)
        expected = (5 if carry else 0) + np.arange(1, STEPS+1) + (0 if carry or not random_init else 1)
        if not waste_preview:
            assert np.array_equal(totals, expected), (condition, totals)
        return dict(condition=condition, fc2=np.stack(features), logits=np.stack(logits),
                    total_backbone_calls=totals, target_backbone_calls=[c-preview_calls for c in totals],
                    preview_calls=preview_calls, elapsed_seconds=elapsed, preview_seconds=preview_seconds,
                    preview_indices=preview_indices if carry or waste_preview else [],
                    target_hash=digest(target), random_init=random_init)
    trials = manifest[split][:limit] if limit else manifest[split]
    conditions = TRAIN_CONDITIONS if split == 'train' and not smoke else CONDITIONS
    noises = manifest['noise_seeds'][split][:1] if smoke else manifest['noise_seeds'][split]
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for ordinal, trial in enumerate(trials):
                index = trial['dataset_index']
                missing = [s for s in noises if not (directory / f'image_{index}_seed_{s}.pckl').exists()]
                if not missing:
                    continue
                image = dataset[index][0].cuda()
                indices = set(trial['previews']) | {x for x in trial['primes'].values() if x is not None}
                images = {i: dataset[i][0].cuda() for i in indices}
                for noise in missing:
                    noise_seed = SEED + noise * 1000003 + index * 1009
                    target = canvas([(apply_input_degradation(image, mode='predify_noise', level=.5, seed=noise_seed), 'center')])
                    records = [execute(trial, c, target, images) for c in conditions]
                    assert len({r['target_hash'] for r in records}) == 1
                    assert all(np.isfinite(r['fc2']).all() and np.isfinite(r['logits']).all() for r in records)
                    for r in records:
                        budget_index(r['total_backbone_calls'], 9)
                        budget_index(r['total_backbone_calls'], 15)
                    if smoke:
                        for condition in ('none_random', 'none_feedforward'):
                            control = execute(trial, condition, target, images, waste_preview=True)
                            original = next(r for r in records if r['condition'] == condition)
                            assert np.array_equal(original['logits'], control['logits'])
                            assert np.array_equal(original['fc2'], control['fc2'])
                    save(directory / f'image_{index}_seed_{noise}.pckl', dict(trial=trial, noise=noise, noise_seed=noise_seed,
                         split=split, records=records, source_sha256=source_hash))
                print(f'{split} {ordinal+1}/{len(trials)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    finally:
        recorder.close(); handle.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'train', 'dev', 'test', 'smoke'))
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        extract('dev' if args.action == 'smoke' else args.action,
                limit=5 if args.action == 'smoke' else args.limit, smoke=args.action == 'smoke')


if __name__ == '__main__':
    main()
