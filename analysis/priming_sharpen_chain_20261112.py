"""Record native FC trajectories of the previously frozen four-layer chain."""

from __future__ import annotations

import argparse
from collections import defaultdict

from priming_sharpen_collect_20261112 import digest, images, trials
from priming_sharpen_reserve_20261112 import manifest, renderer
from priming_sharpen_search_20261112 import DATASETS, OUT, read, save
from hebb_chain_20261111 import Experiment, PREVIEW, Controls, old
from hebb_chain_20261111 import OUT as CHAIN
import numpy as np
import torch


@torch.no_grad()
def simulate(run, x, previews, controls, active, edit=None):
    run.reset()
    run.controls(PREVIEW if active else Controls())
    fc = run.layers[2:]
    frames = {name: [] for name in ('fc1', 'fc2')}
    for preview in previews:
        run.step(preview, 2)
        for name, layer in zip(frames, fc):
            frames[name].append(layer.record['signed'][0].cpu().numpy())
        if active:
            for layer in run.layers:
                layer.write(layer.record['signed'])
    for layer in run.layers:
        layer.memory.freeze()
        assert layer.memory.writes == (2 if active else 0)
    versions = run.versions()
    run.controls(controls if active else Controls())
    target = {name: [] for name in frames}
    hooks = []
    if edit is not None and active:
        memory = np.stack(frames['fc2'])[None]

        def apply(_, __, native):
            treated = edit(native.cpu().numpy(), memory)
            return torch.from_numpy(treated).to(native)

        target['terminal'] = []
        hooks.append(run.pipe[4].register_forward_hook(apply))
        hooks.append(run.pipe[6].register_forward_pre_hook(lambda _, args: target['terminal'].append(args[0][0].detach().clone())))

    def capture(_, __):
        for name, layer in zip(target, fc):
            target[name].append(layer.record['signed'][0])

    try:
        run.step(x, 15, capture)
    finally:
        for hook in hooks:
            hook.remove()
    assert versions == run.versions()
    native = {name: torch.stack(value).cpu().numpy() for name, value in target.items()}
    return native, {k: np.stack(v) for k, v in frames.items()}


def subset(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['fold'], r['concept']].append(r)
    return [r for group in groups.values() for r in sorted(group, key=lambda r: r['dataset_index'])[:2]]


def collect(dataset, scope, parts, selection):
    folder = OUT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / ('chain.npz' if scope == 'train' else f'chain_{scope}.npz')
    meta = dest.with_name(dest.stem + '_manifest.json')
    if dest.exists() and meta.exists():
        assert digest(dest) == read(meta)['feature_sha256']
        return
    run = Experiment(dataset, parts)
    run.fit()
    controls = Controls(tuple(selection['suppression']['lam']))
    prospective = scope in ('reserve', 'replication', 'confirmation', 'validation')
    if prospective:
        plan = manifest(dataset, scope)
        render = renderer(plan)
        rows = [dict(r, dataset_index=r['index'], category_index=r['label']) for r in plan['targets']]
    else:
        rows = trials(dataset, scope)
    if scope == 'train':
        rows = subset(rows)
    data = defaultdict(list)
    hashes = []
    for i, row in enumerate(rows):
        if prospective:
            image, preview = render(row)
            source_hash = row['sha256']
        else:
            image, preview, source_hash = images(dataset, row)
        x = torch.from_numpy(image).cuda()
        p = torch.from_numpy(preview).cuda()
        histories = [simulate(run, x, frames, controls, active=True) for frames in p]
        unprimed = simulate(run, x, [run.c.grey, run.c.grey], controls, active=False)[0]
        for name in ('fc1', 'fc2'):
            trajectory = np.stack([histories[0][0][name], histories[1][0][name], unprimed[name]])
            assert np.isfinite(trajectory).all() and (trajectory.mean(-1) > 0).all()
            data['trajectory_' + name].append(trajectory)
            data['target_' + name].append(trajectory[:, 9])
            data['preview_' + name].append(np.stack([item[1][name] for item in histories]))
        hashes.append(source_hash)
        if (i + 1) % 12 == 0 or i + 1 == len(rows):
            print('CHAIN', scope, dataset, i + 1, '/', len(rows), flush=True)
    tmp = dest.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **{k: np.stack(v) for k, v in data.items()}, y=np.array([r['category_index'] for r in rows]),
                 row=np.array([r['dataset_index'] for r in rows]), concept=np.array([r['concept'] for r in rows]),
                 slot=np.array([r['slot'] // 2 for r in rows]))
    tmp.replace(dest)
    provenance = dict(trials=rows, archived_stimuli_sha256=hashes, feature_sha256=digest(dest),
                      original_chain_selection_sha256=digest(CHAIN / 'selection.json'), actual_target_step=9)
    if prospective:
        provenance['stimulus_manifest_sha256'] = digest(folder / f'{scope}_manifest.json')
    save(meta, provenance)
    run.detach()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=('train', 'confirm', 'secondary', 'reserve', 'replication', 'confirmation', 'validation'), default='train')
    args = p.parse_args()
    frozen = {'reserve': 'vanilla_semantic_frozen.json', 'replication': 'chain_lower_gain_frozen.json',
              'confirmation': 'chain_object_memory_confirmation_protocol.json',
              'validation': 'chain_object_corroboration_frozen.json'}.get(args.scope, 'chain_frozen.json')
    if args.scope != 'train' and not (OUT / frozen).is_file():
        raise RuntimeError('Freeze candidates before encoding confirmation')
    selection = read(CHAIN / 'selection.json')
    parts = old.initialize()
    for dataset in DATASETS:
        collect(dataset, args.scope, parts, selection)


if __name__ == '__main__':
    main()
