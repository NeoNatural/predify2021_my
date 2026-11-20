"""Record full-frame Conv5 responses for the familiarity-gate experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hebb_chain_20261111 import Controls, Experiment, PREVIEW, old
from priming_sharpen_collect_20261112 import ARCHIVE, digest, vanilla
from priming_sharpen_object_precision_20261112 import encode_tiles, tiles
from priming_sharpen_reserve_20261112 import NATIVE, csv_rows, pools
from priming_sharpen_same_concept_data_20261112 import HISTORIES, PREVIOUS, frozen, renderer
from priming_sharpen_same_concept_eval_20261112 import load
from priming_sharpen_search_20261112 import DATASETS, OUT, read, save


OUT = OUT / 'familiarity'
PRECEDING = (*PREVIOUS, 'same_concept')
TRAIN_HISTORIES = ('exact', 'same_concept', 'different_category')


def paths(dataset, scope):
    folder = OUT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{scope}.npz', folder / f'{scope}_manifest.json'


def source_manifest(dataset):
    return read(ARCHIVE / dataset / 'manifest.json')


def prepare_fresh(dataset):
    dest = OUT / dataset / 'fresh_stimuli.json'
    if dest.exists():
        return read(dest)
    archive = source_manifest(dataset)
    old_images = list(archive['images'].values())
    previous_paths = [OUT.parent / dataset / f'{scope}_manifest.json' for scope in PRECEDING]
    previous = [read(p) for p in previous_paths]
    earlier = [row for plan in previous for row in (*plan['targets'], *plan['distractors'].values())]
    forbidden = {r['sha256'] for r in (*old_images, *earlier)}
    old_concepts = {r['concept'] for r in old_images}
    seen = {r['concept'] for r in earlier}
    fit = {archive['images'][str(i)]['concept'] for r in archive['trials'] if r['fold'] in (1, 2, 3)
           for i in (r['dataset_index'], *r['previews'], *r['primes'].values())}
    review = csv_rows(NATIVE / dataset / 'concept_review.csv')
    older = {r['concept']: r['category'] for r in review if r['fold'] in ('0', '4') and r['concept'] not in fit}
    plants = {r['concept'] for r in csv_rows(NATIVE / 'inventory/concepts.csv') if r['manual_exclusive'] == 'plant'} - fit
    available = pools(dataset, fit, 'familiarity', plants=plants, additional_concepts=older)
    rng = np.random.default_rng(2026111217)
    targets, distractors = [], {}
    categories = archive['categories']
    for label, name in enumerate(categories):
        concepts = sorted(c for c, group in available.items() if len(group) >= 3 and group[0][2] == name)
        concepts = sorted(rng.permutation(concepts).tolist(), key=lambda c: (c in old_concepts or c in seen))
        selected = 0
        for concept in concepts:
            images = []
            accepted = set()
            for k in rng.permutation(len(available[concept])):
                i, path, _ = available[concept][k]
                if not path.is_file():
                    continue
                sha = digest(path)
                if sha not in forbidden and sha not in accepted:
                    images.append(dict(index=i, path=str(path), sha256=sha, concept=concept, label=label))
                    accepted.add(sha)
                if len(images) == 3:
                    break
            if len(images) < 3:
                continue
            forbidden.update(r['sha256'] for r in images)
            targets.extend(images[:2])
            distractors[concept] = images[2]
            selected += 1
            if selected == 4:
                break
        if selected < 4:
            raise RuntimeError(f'{dataset} {name}: only {selected} concepts with three unused images')
    per_category = 8
    for j, row in enumerate(targets):
        rank = j % per_category
        other = [i for i in range(len(categories)) if i != row['label']]
        shift = 1 + rank % (len(categories) - 1)
        different = ((row['label'] + shift) % len(categories)) * per_category + rank
        background = [str(rng.choice(sorted(c for c, r in distractors.items() if r['label'] == label)))
                      for label in rng.permutation(other)[:3]]
        assert targets[j ^ 1]['concept'] == row['concept'] and targets[different]['label'] != row['label']
        row.update(slot=2 + (rank // 2) % 2, backgrounds=background,
                   same_concept=targets[j ^ 1]['index'], different=targets[different]['index'])
    assert not set(distractors) & fit
    expected = sorted((r['index'], r['slot']) for r in targets)
    assert all(sorted((r[h], r['slot']) for r in targets) == expected for h in ('same_concept', 'different'))
    result = dict(dataset=dataset, categories=categories, targets=targets, distractors=distractors,
                  prior_sha256={p.stem: digest(p) for p in previous_paths},
                  reused_concepts=sorted(set(distractors) & (old_concepts | seen)))
    save(dest, result)
    print('FRESH', dataset, len(targets), result['reused_concepts'], flush=True)
    return result


@torch.no_grad()
def trial(run, x, preview, controls, *, steps=15, active=True, edit=None):
    run.reset()
    run.controls(PREVIEW if active else Controls())
    conv = run.layers[1]
    fc = run.layers[3]
    memory_maps = []
    for j, frame in enumerate(preview, 1):
        run.step(frame, 2)
        if active:
            memory_maps.append(conv.record['signed'][0].to(torch.float16))
            for layer in run.layers:
                layer.write(layer.record['signed'])
            assert all(layer.memory.writes == j for layer in run.layers)
    for layer in run.layers:
        layer.memory.freeze()
    assert all(layer.memory.writes == (2 if active else 0) for layer in run.layers)
    versions = run.versions()
    run.controls(controls if active else Controls())
    conv_maps, fc2, received = [], [], []
    hooks = []
    if edit is not None and active:
        def apply(_, __, native):
            return edit(native, conv.record['signed'].to(torch.float16), torch.stack(memory_maps))
        hooks.append(run.pipe[4].register_forward_hook(apply))
        hooks.append(run.pipe[6].register_forward_pre_hook(lambda _, args: received.append(args[0][0].detach().clone())))

    def capture(t, _):
        if steps == 15 or t == steps - 1:
            conv_maps.append(conv.record['signed'][0].to(torch.float16))
            fc2.append(fc.record['signed'][0])
    try:
        run.step(x, steps, capture)
    finally:
        for hook in hooks:
            hook.remove()
    assert versions == run.versions()
    maps = torch.stack(conv_maps).cpu().numpy()
    memory = torch.stack(memory_maps).cpu().numpy() if active else np.zeros((2, *maps.shape[1:]), np.float16)
    terminal = torch.stack(received).cpu().numpy() if received else None
    return maps, memory, torch.stack(fc2).cpu().numpy(), terminal


def training_stimuli(dataset, rows):
    folder = ARCHIVE / dataset / 'formal'
    for row in rows:
        with np.load(folder / f"stimuli_{row['dataset_index']}.npz") as data:
            preview = np.array([[data[f'{h}_preview{j}'] for j in range(2)] for h in TRAIN_HISTORIES])
            x = data['target_0']
        yield row, torch.from_numpy(x).cuda(), torch.from_numpy(preview).cuda()


def holdout_stimuli(plan):
    render = renderer(plan)
    for row in plan['targets']:
        yield row, *render(row)


def collect_train(dataset, parts, controls):
    dest, meta = paths(dataset, 'train')
    if dest.exists() and meta.exists():
        assert read(meta)['feature_sha256'] == digest(dest)
        return
    reference = OUT.parent / dataset
    rows = read(reference / 'chain_manifest.json')['trials']
    with np.load(reference / 'chain.npz') as f:
        known = f['target_fc2']
    with np.load(reference / 'object_memory_fp32_train.npz') as f:
        previous_p = f['preview']
    run = Experiment(dataset, parts)
    run.fit()
    model, buffers = vanilla()
    maps, memory, fc2, hard_p = [], [], [], []
    try:
        for j, (row, x, p) in enumerate(training_stimuli(dataset, rows)):
            result = [trial(run, x, frames, controls, steps=10) for frames in p]
            target_maps, preview_maps, target_fc2, _ = zip(*result)
            f = np.concatenate(target_fc2)
            np.testing.assert_array_equal(f[[0, 2]], known[j, :2])
            maps.append(np.concatenate(target_maps))
            memory.append(np.stack(preview_maps))
            fc2.append(f)
            hard_p.append(tiles(p[1].cpu().numpy()))
            if (j + 1) % 8 == 0:
                print('TRAIN MAPS', dataset, j + 1, '/', len(rows), flush=True)
        hard = encode_tiles(model, buffers, [tile for group in hard_p for tile in group]).reshape(len(rows), 4, 4096)
    finally:
        run.detach()
    old_preview = np.stack((previous_p[:, 0], hard, previous_p[:, 1]), 1)
    tmp = dest.with_suffix('.tmp')
    with tmp.open('wb') as out:
        np.savez(out, conv=np.stack(maps), memory=np.stack(memory), fc2=np.stack(fc2), old_preview=old_preview,
                 folds=np.array([r['fold'] for r in rows]), rows=np.array([r['dataset_index'] for r in rows]))
    tmp.replace(dest)
    save(meta, dict(feature_sha256=digest(dest), histories=TRAIN_HISTORIES, step=9,
                    trial_sha256=digest(reference / 'chain_manifest.json'),
                    source_sha256=digest(Path(__file__)), rows=len(rows)))
    print('TRAIN SAVED', dataset, len(rows), np.shape(maps), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'train'))
    args = parser.parse_args()
    if args.phase == 'prepare':
        for ds in DATASETS:
            paths(ds, 'fresh')
            prepare_fresh(ds)
    else:
        protocol = frozen()
        parts = old.initialize()
        controls = Controls(tuple(protocol['hebbian_lambdas']))
        for ds in DATASETS:
            collect_train(ds, parts, controls)


if __name__ == '__main__':
    main()
