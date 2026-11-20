"""Paired supplementary priming experiment, with resumable per-image records."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'Hebbian-CNN/Log/paper_parafoveal_v1'
OUT = ROOT / 'Hebbian-CNN/Log/supplementary_20261009'
IDENTITIES = ('exact', 'same_synset', 'same_category', 'different_category', 'neutral')
CARRIERS = ('hebb_only', 'pcoder_carry', 'hebb_and_pcoder')
NOISE_SEEDS = (202610091, 202610092, 202610093)
CONFIRMATION_NOISE_SEEDS = (202610191, 202610192, 202610193)
SEED = 20261009


def load(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def make_manifest(labels):
    from .paper_stimuli import SUPER_CATEGORIES, imagenet_supercategory
    used = set()
    sources = []
    for path in PAPER.rglob('*.pckl'):
        if 'analysis' in path.parts or 'decoding' in path.parts:
            continue
        result = load(path)
        if isinstance(result, dict) and 'dataset_indices' in result:
            used.update(map(int, result['dataset_indices']))
            sources.append(str(path.relative_to(ROOT)))
    # Also exclude previous full-size/foveated benchmark targets.
    sample = load(ROOT / 'Hebbian-CNN/Log/DatasetSample/N_Back-1.pckl')
    used.update(map(int, sample['out_list']))
    rng = random.Random(SEED)
    pools = {c: {} for c in SUPER_CATEGORIES}
    for i, label in enumerate(labels):
        c = imagenet_supercategory(label)
        if c and i not in used:
            pools[c].setdefault(int(label), []).append(i)
    selected = {}
    inventory = []
    for c, by_label in pools.items():
        synsets = list(by_label)
        rng.shuffle(synsets)
        for values in by_label.values():
            rng.shuffle(values)
        indices = []
        while any(by_label.values()):
            for label in synsets:
                if by_label[label]:
                    indices.append(by_label[label].pop())
        if len(indices) < 80:
            raise ValueError(f'{c}: only {len(indices)} unused images; need 80')
        selected[c] = indices
        inventory.append(dict(category=c, synsets=len(synsets), unused_images=len(indices)))
    explore = [selected[c][i] for i in range(40) for c in SUPER_CATEGORIES]
    confirm = [selected[c][i] for i in range(40, 80) for c in SUPER_CATEGORIES]
    # Confirmation images cannot appear as previews in exploration.
    forbidden = set(confirm) | set(explore)
    by_label = {}
    by_category = {c: [] for c in SUPER_CATEGORIES}
    for i, label in enumerate(labels):
        c = imagenet_supercategory(label)
        if c and i not in forbidden:
            by_label.setdefault(int(label), []).append(i)
            by_category[c].append(i)
    trials = []
    for ordinal, index in enumerate(explore):
        label = int(labels[index])
        c = imagenet_supercategory(label)
        trial_rng = random.Random(SEED + index)
        distractors = [trial_rng.choice(by_category[x]) for x in SUPER_CATEGORIES if x != c]
        trial_rng.shuffle(distractors)
        slot = 2 + ordinal % 2
        same_category = [x for x in by_category[c] if int(labels[x]) != label]
        trials.append(dict(dataset_index=index, imagenet_label=label,
                           category_index=SUPER_CATEGORIES.index(c), supercategory=c,
                           previews=distractors, slot=slot,
                           primes=dict(exact=index, same_synset=trial_rng.choice(by_label[label]),
                                       same_category=trial_rng.choice(same_category),
                                       different_category=distractors[slot], neutral=None)))
    return dict(seed=SEED, noise_seeds=NOISE_SEEDS, exploration=trials,
                confirmation_indices=confirm, excluded_previous_images=sorted(used),
                exclusion_sources=sources, inventory=inventory,
                scope='image-disjoint exploration/confirmation; synsets may overlap frozen readout training')


def confirmation_trials(labels, manifest):
    from .paper_stimuli import SUPER_CATEGORIES, imagenet_supercategory
    forbidden = set(manifest['confirmation_indices'])
    pools = {c: [] for c in SUPER_CATEGORIES}
    for i,label in enumerate(labels):
        c = imagenet_supercategory(label)
        if c and i not in forbidden:
            pools[c].append(i)
    trials=[]
    for ordinal,index in enumerate(manifest['confirmation_indices']):
        c=imagenet_supercategory(labels[index])
        rng=random.Random(SEED+index)
        previews=[rng.choice(pools[x]) for x in SUPER_CATEGORIES if x!=c]
        rng.shuffle(previews)
        slot=2+ordinal%2
        trials.append(dict(dataset_index=index,imagenet_label=int(labels[index]),
            category_index=SUPER_CATEGORIES.index(c),supercategory=c,previews=previews,
            slot=slot,primes=dict(exact=index,different_category=previews[slot])))
    return trials


def run(args):
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from torchvision.datasets import ImageNet
    from torchvision.models import VGG16_Weights
    from .config import load_experiment_config
    from .paper_parafoveal import _FeatureRecorder, _add_legacy_module_path, _recovery_metrics
    from .hebb_vgg_priming import _build_model
    from .paper_stimuli import make_paper_canvas
    from .sequence import reset_hebbian_state, run_recurrent_sequence
    config = load_experiment_config(ROOT / 'configs/paper_memory_carrier_calibrated.toml').with_overrides(
        imagenet_root=args.imagenet_root, device='cuda')
    config.validate_for_run()
    _add_legacy_module_path(ROOT)
    from Hebb_Priming_Utils import apply_input_degradation
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dataset = ImageNet(str(args.imagenet_root), 'val', transform=VGG16_Weights.IMAGENET1K_V1.transforms())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.pckl'
    if manifest_path.exists():
        manifest = load(manifest_path)
    else:
        manifest = make_manifest(dataset.targets)
        save(manifest_path, manifest)
        (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('INVENTORY', manifest['inventory'], flush=True)
    if args.manifest_only:
        return
    torch.manual_seed(SEED)
    model, layers, _ = _build_model(config, torch.device('cuda'), torch)
    recorder = _FeatureRecorder(model, layers, torch)
    decoder_path = PAPER / 'shared_decoder/analysis/shared_decoders.pckl'
    decoders = load(decoder_path)['classifiers']
    weights = list(config.paths.pcoder_weight_dir.glob('pvgg16*pmodule.pth'))
    provenance = dict(config=asdict(config), torch=torch.__version__, numpy=np.__version__,
                      device=torch.cuda.get_device_name(0), deterministic=True,
                      weights={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in weights},
                      readout_sha256=hashlib.sha256(decoder_path.read_bytes()).hexdigest(),
                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      conditions=17, reference='fixed clean FF and clean recurrent t9 per image',
                      started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    provenance_path = OUT / ('provenance_stage_e_occlusion.json' if args.occlusion else 'provenance_stage_d.json' if args.confirmation else 'provenance_stage_c.json' if args.stage_c else 'provenance.json')
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str), encoding='utf-8')
    (OUT / ('source_' + provenance['code_sha256'][:12] + '.py')).write_bytes(Path(__file__).read_bytes())
    def canvas(values):
        return make_paper_canvas(values, object_scale=.15, horizontal_offset=.175, background_value=.5)
    def seed_for(index, noise, stage):
        value = (SEED + index * 1009 + noise * 37 + stage * 100003) % (2**31 - 1)
        torch.manual_seed(value)
        return value
    def capture_reference(clean, index, random_init, steps):
        reset_hebbian_state(layers)
        for layer in layers[2:]:
            layer.coeff = .05
            layer.inh_c = 4
        model.random_init = random_init
        for j in range(1, 6):
            getattr(model, f'pcoder{j}').random_init = random_init
        seed_for(index, 0, 90)
        run_recurrent_sequence(model, clean[None], steps, hebbian_layers=layers, commit_update=False)
        return [getattr(model, f'pcoder{j}').rep.detach().clone() for j in range(1, 6)]
    def execute(trial, noise, carrier, identity, clean, degraded, images, refs, scale=1., inhibition=4):
        reset_hebbian_state(layers)
        model.reset()
        for layer in layers[2:]:
            layer.coeff = .05 * scale
            layer.inh_c = inhibition
        has_hebb = carrier in ('hebb_only', 'hebb_and_pcoder')
        has_pc = carrier in ('pcoder_carry', 'hebb_and_pcoder')
        previews = list(trial['previews'])
        previews[trial['slot']] = trial['primes'][identity]
        sequence_hashes = []
        seeds = []
        if carrier != 'target_only':
            for stage in range(2):
                seed = seed_for(trial['dataset_index'], noise, stage)
                seeds.append(seed)
                values = [(images[i], side) for i, side in zip(previews[2*stage:2*stage+2], ('left','right')) if i is not None]
                preview = canvas(values)
                sequence_hashes.append(digest(preview))
                run_recurrent_sequence(model, preview[None], 2, hebbian_layers=layers,
                                       commit_update=has_hebb, preserve_model_state=has_pc and stage > 0)
        seeds.append(seed_for(trial['dataset_index'], noise, 2))
        feature_values = {x: [] for x in recorder.cache}
        logits_values, recovery_values, errors, activation_values = [], [], [], []
        def after_step(t, output):
            logits_values.append(output.detach().cpu().numpy()[0])
            snapshot = recorder.snapshot()
            for name, value in snapshot.items():
                feature_values.setdefault(name, []).append(value)
            pcs = [getattr(model, f'pcoder{j}') for j in range(1, 6)]
            recovery_values.append(np.stack([np.stack([_recovery_metrics(pc.rep, ref[k], torch).cpu().numpy() for k, pc in enumerate(pcs)]) for ref in refs]))
            errors.append([float(pc.prediction_error.detach()) for pc in pcs])
            activation_values.append([float(pc.rep.detach().mean()) for pc in pcs])
        started = time.perf_counter()
        run_recurrent_sequence(model, degraded[None], 10, hebbian_layers=layers,
                               commit_update=False, preserve_model_state=has_pc, after_step=after_step)
        features = {x: np.stack(v) for x, v in feature_values.items()}
        scores = {}
        for name, values in features.items():
            scores[name] = decoders[('predify_noise', name, 9)].decision_function(values.astype(np.float32))
        return dict(carrier=carrier, identity=identity, logits=np.stack(logits_values), features=features,
                    scores=scores, recovery=np.stack(recovery_values), internal_error=np.asarray(errors),
                    activation=np.asarray(activation_values), seconds=time.perf_counter()-started,
                    preview_indices=previews if carrier != 'target_only' else [],
                    preview_hashes=sequence_hashes, stage_seeds=seeds,
                    target_hash=digest(degraded), scale=scale, inhibition=inhibition)
    frozen = json.loads(args.confirmation.read_text(encoding='utf-8')) if args.confirmation else None
    directory = OUT / ('stage_e_occlusion' if args.occlusion else 'stage_d' if args.confirmation else 'stage_c' if args.stage_c else 'pilot' if args.pilot else 'stage_b')
    trials = confirmation_trials(dataset.targets,manifest) if args.confirmation else manifest['exploration'][:50] if args.stage_c else manifest['exploration'][:5] if args.pilot else manifest['exploration']
    noises = CONFIRMATION_NOISE_SEEDS if args.confirmation else NOISE_SEEDS[:1] if args.pilot else NOISE_SEEDS
    if args.confirmation:
        (OUT/'frozen_confirmation.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    started = time.time()
    try:
        with torch.no_grad():
            for ordinal, trial in enumerate(trials):
                missing = [s for s in noises if not (directory / f"image_{trial['dataset_index']}_seed_{s}.pckl").exists()]
                if not missing:
                    continue
                index = trial['dataset_index']
                image = dataset[index][0].cuda()
                clean = canvas(((image, 'center'),))
                refs = [capture_reference(clean, index, False, 1), capture_reference(clean, index, True, 10)]
                for j in range(1, 6):
                    getattr(model, f'pcoder{j}').random_init = True
                model.random_init = True
                image_indices = set(trial['previews']) | {x for x in trial['primes'].values() if x is not None}
                images = {i: dataset[i][0].cuda() for i in image_indices}
                for noise in missing:
                    degradation_mode='fractal_occlusion' if args.occlusion else 'predify_noise'
                    degradation_level=.3 if args.occlusion else .5
                    actual_degradation_seed=noise+index*1009 if args.occlusion else noise
                    degraded = canvas(((apply_input_degradation(image, mode=degradation_mode, level=degradation_level, seed=actual_degradation_seed), 'center'),))
                    conditions = [(c, identity) for c in CARRIERS for identity in IDENTITIES] + [('none','exact'),('target_only','exact')]
                    if args.pilot:
                        conditions += [('none',x) for x in IDENTITIES[1:]]
                    if args.confirmation:
                        records = [execute(trial,noise,frozen['carrier'],identity,clean,degraded,images,refs,
                                           scale=frozen['scale'],inhibition=frozen['inhibition']) for identity in ('exact','different_category')]
                        records.append(execute(trial,noise,'none','exact',clean,degraded,images,refs))
                    elif args.stage_c:
                        baseline = load(OUT / 'stage_b' / f'image_{index}_seed_{noise}.pckl')
                        records = [r for r in baseline['records'] if r['identity'] in ('exact','different_category')]
                        for scale, inhibition in ((.25,4), (1.,0), (0.,4)):
                            for c in ('hebb_only','hebb_and_pcoder'):
                                for identity in ('exact','different_category'):
                                    records.append(execute(trial, noise, c, identity, clean, degraded, images, refs,
                                                           scale=scale,inhibition=inhibition))
                        for r in records:
                            if r['scale']==0 and r['carrier'] in ('hebb_only','hebb_and_pcoder'):
                                carrier = 'none' if r['carrier']=='hebb_only' else 'pcoder_carry'
                                other = next(x for x in baseline['records'] if x['carrier']==carrier and x['identity']==('exact' if carrier=='none' else r['identity']))
                                assert np.array_equal(r['logits'],other['logits']), 'zero channel modulation is not matched to its baseline'
                    else:
                        records = [execute(trial, noise, c, identity, clean, degraded, images, refs) for c, identity in conditions]
                    if args.pilot:
                        anchor = next(r for r in records if r['carrier']=='none' and r['identity']=='exact')
                        for r in records:
                            if r['carrier'] in ('none','target_only'):
                                assert np.array_equal(anchor['logits'],r['logits']), 'no-memory logits depend on preview'
                                for layer in anchor['features']:
                                    assert np.array_equal(anchor['features'][layer],r['features'][layer]), 'no-memory features depend on preview'
                    assert len({r['target_hash'] for r in records}) == 1
                    assert all(np.isfinite(r['logits']).all() and np.isfinite(r['recovery']).all() for r in records)
                    save(directory / f'image_{index}_seed_{noise}.pckl', dict(trial=trial, noise_seed=noise, records=records,
                         degradation_mode=degradation_mode,degradation_level=degradation_level,degradation_seed=actual_degradation_seed,
                         clean_hash=digest(clean), reference_features='full spatial FF/t9 tensors shared within image'))
                elapsed = time.time()-started
                print(f'COMPLETE image {ordinal+1}/{len(trials)} index={index} elapsed={elapsed:.1f}s', flush=True)
    finally:
        recorder.close()
    print('DONE', directory, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-root', type=Path, default=Path('C:/Users/liang/Documents/ImageNet'))
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--stage-c', action='store_true')
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--occlusion', action='store_true', help='Conditional external validation of the frozen noise readout')
    args = parser.parse_args()
    if args.occlusion and not args.confirmation:
        parser.error('--occlusion requires the frozen --confirmation specification')
    run(args)


if __name__ == '__main__':
    main()
