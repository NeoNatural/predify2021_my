"""Collect four paired preview histories with a fixed four-layer Hebbian chain."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hebb_chain_20261111 import Controls, Experiment, old
from hebb_chain_20261111 import OUT as CHAIN
from priming_sharpen_chain_20261112 import simulate
from priming_sharpen_collect_20261112 import ARCHIVE, CHECKPOINT, digest, vanilla
from priming_sharpen_object_precision_20261112 import encode_tiles, tiles
from priming_sharpen_reserve_20261112 import NATIVE, csv_rows, pools
from priming_sharpen_search_20261112 import DATASETS, OUT, read, save


HISTORIES = ('exact', 'same_concept', 'different', 'none')
PREVIOUS = ('reserve', 'replication', 'confirmation', 'validation')
SCOPE = 'same_concept'
PLAN = OUT / f'{SCOPE}_frozen.json'


def manifest(dataset):
    path = OUT / dataset / f'{SCOPE}_manifest.json'
    if path.exists():
        return read(path)
    source = ARCHIVE / dataset / 'manifest.json'
    old_manifest = read(source)
    archived = list(old_manifest['images'].values())
    prior_paths = [OUT / dataset / f'{scope}_manifest.json' for scope in PREVIOUS]
    earlier = [r for p in prior_paths for plan in (read(p),)
               for r in (*plan['targets'], *plan['distractors'].values())]
    forbidden = {r['sha256'] for r in (*archived, *earlier)}
    archived_concepts = {r['concept'] for r in archived}
    earlier_concepts = {r['concept'] for r in earlier}
    fit_concepts = {old_manifest['images'][str(i)]['concept'] for r in old_manifest['trials']
                    if r['fold'] in (1, 2, 3) for i in (r['dataset_index'], *r['previews'], *r['primes'].values())}
    plants = {r['concept'] for r in csv_rows(NATIVE / 'inventory/concepts.csv')
              if r['manual_exclusive'] == 'plant'} - fit_concepts if dataset == 'things' else set()
    older_heldout = {r['concept']: r['category'] for r in csv_rows(NATIVE / dataset / 'concept_review.csv')
                     if r['fold'] in ('0', '4') and r['concept'] not in fit_concepts} if dataset == 'things' else {}
    excluded = archived_concepts - plants - older_heldout.keys()
    available = pools(dataset, excluded, SCOPE, plants=plants, additional_concepts=older_heldout)
    rng = np.random.default_rng(2026111207)
    targets, distractors = [], {}
    categories = old_manifest['categories']
    per_category = 16
    for label, category in enumerate(categories):
        eligible = sorted(c for c, group in available.items() if len(group) >= 5 and group[0][2] == category)
        ordered = sorted(rng.permutation(eligible).tolist(), key=lambda c: (c in archived_concepts, c in earlier_concepts))
        selected = 0
        for concept in ordered:
            group = available[concept]
            images = []
            for k in rng.permutation(len(group)):
                index, source_path, _ = group[k]
                if not source_path.is_file():
                    continue
                sha = digest(source_path)
                if sha in forbidden:
                    continue
                forbidden.add(sha)
                images.append(dict(index=index, path=str(source_path), sha256=sha, concept=concept, label=label))
                if len(images) == 5:
                    break
            if len(images) != 5:
                continue
            targets.extend(images[:4])
            distractors[concept] = images[4]
            selected += 1
            if selected == 4:
                break
        if selected != 4:
            raise RuntimeError(f'{dataset} / {category}: only {selected} concepts with five new images')

    for j, row in enumerate(targets):
        rank = j % per_category
        shift = 1 + (rank // 2) % (len(categories) - 1)
        different_label = (row['label'] + shift) % len(categories)
        sibling = targets[j ^ 2]
        unrelated = targets[different_label * per_category + rank]
        assert sibling['concept'] == row['concept'] and sibling['sha256'] != row['sha256']
        assert unrelated['label'] != row['label']
        other_labels = [i for i in range(len(categories)) if i != row['label']]
        backgrounds = [str(rng.choice(sorted(c for c, item in distractors.items() if item['label'] == label)))
                       for label in rng.permutation(other_labels)[:3]]
        row.update(slot=2 + j % 2, backgrounds=backgrounds, same_concept=sibling['index'], different=unrelated['index'])

    # Both comparison histories use precisely the same foreground images and spatial frequencies as Exact.
    expected = sorted((r['index'], r['slot']) for r in targets)
    for h in HISTORIES[1:3]:
        assert sorted((r[h], r['slot']) for r in targets) == expected
    assert not set(distractors) & fit_concepts
    result = dict(dataset=dataset, categories=categories, targets=targets, distractors=distractors,
                  source_manifest_sha256=digest(source), prior_manifest_sha256={p.stem: digest(p) for p in prior_paths},
                  reused_concepts=sorted(set(distractors) & (earlier_concepts | archived_concepts)),
                  older_nonfit_concepts=sorted(set(distractors) & archived_concepts))
    save(path, result)
    print(dataset, len(targets), 'new targets;', len(distractors), 'new background images;',
          'reused held-out concepts:', result['reused_concepts'], flush=True)
    return result


def protocol():
    previous = OUT / 'chain_object_corroboration_frozen.json'
    config = read(previous)['config']
    selection = CHAIN / 'selection.json'
    source_files = ['priming_sharpen_corroboration_20261112.py', 'priming_sharpen_object_precision_20261112.py',
                    'priming_sharpen_chain_20261112.py', 'hebb_chain_20261111.py',
                    'priming_sharpen_search_20261112.py', 'priming_sharpen_same_concept_data_20261112.py']
    training_files = ('chain.npz', 'chain_manifest.json', 'object_memory_fp32_train.npz', 'object_memory_fp32_train.json')
    return dict(
        histories=HISTORIES, t=9, steps=15, hebbian_lambdas=read(selection)['suppression']['lam'],
        configs={'target_only': dict(config, weight=0.), 'geometric': config},
        source_sha256={name: digest(Path(__file__).with_name(name)) for name in source_files},
        chain_selection_sha256=digest(selection), previous_config_sha256=digest(previous),
        stimulus_sha256={ds: digest(OUT / ds / f'{SCOPE}_manifest.json') for ds in DATASETS},
        train_sha256={ds: {name: digest(OUT / ds / name) for name in training_files} for ds in DATASETS},
        checkpoint_sha256=digest(CHECKPOINT),
        comparisons=[['exact', 'same_concept'], ['same_concept', 'different'], ['exact', 'different'],
                     ['exact', 'none'], ['same_concept', 'none'], ['different', 'none']],
        bootstrap='4000 paired concept draws stratified by category',
    )


def frozen():
    registered = read(PLAN)
    assert registered == protocol() | {'histories': list(HISTORIES)}
    return registered


def renderer(plan):
    from PIL import Image
    from torchvision.models import VGG16_Weights
    from predify_experiments.native_categories_20261029 import FilteredInput

    canvas = FilteredInput(torch).canvas
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()

    def pixels(row):
        source = Path(row['path'])
        assert digest(source) == row['sha256']
        with Image.open(source) as im:
            return transform(im.convert('RGB')).cuda()

    targets = {r['index']: pixels(r) for r in plan['targets']}
    backgrounds = {c: pixels(r) for c, r in plan['distractors'].items()}

    def render(row):
        x = targets[row['index']]
        central = canvas([(x, 'center')])
        previews = []
        for i in (row['index'], row['same_concept'], row['different']):
            objects = [backgrounds[c] for c in row['backgrounds']]
            objects.insert(row['slot'], targets[i])
            previews.append(torch.stack([canvas(list(zip(objects[k:k + 2], ('left', 'right'))), preview=True)
                                         for k in (0, 2)]))
        return central, torch.stack(previews)

    return render


def collect(dataset, parts, controls):
    folder = OUT / dataset
    dest = folder / f'chain_{SCOPE}.npz'
    audit = dest.with_suffix('.json')
    if dest.exists() and audit.exists():
        cached = read(audit)
        assert cached['protocol_sha256'] == digest(PLAN) and cached['feature_sha256'] == digest(dest)
        return
    plan = read(folder / f'{SCOPE}_manifest.json')
    render = renderer(plan)
    run = Experiment(dataset, parts)
    run.fit()
    model, buffers = vanilla()
    patch_ids, patches, cache, trajectories = [], [], {}, []
    try:
        for j, row in enumerate(plan['targets'], 1):
            x, previews = render(row)
            rendered_tiles = [tiles(frames) for frames in previews.cpu().numpy()]
            for candidate in rendered_tiles[1:]:
                unchanged = [np.array_equal(a, b) for a, b in zip(rendered_tiles[0], candidate)]
                assert unchanged == [k != row['slot'] for k in range(4)]
            row_patches = []
            for group in rendered_tiles:
                ids = []
                for tile in group:
                    key = tile.tobytes()
                    if key not in cache:
                        cache[key] = len(patches)
                        patches.append(tile)
                    ids.append(cache[key])
                row_patches.append(ids)
            patch_ids.append(row_patches)
            active = [simulate(run, x, p, controls, active=True)[0]['fc2'] for p in previews]
            none = simulate(run, x, [run.c.grey] * 2, controls, active=False)[0]['fc2']
            trajectories.append(np.stack([*active, none]))
            if j % 8 == 0:
                print('COLLECT', dataset, j, '/', len(plan['targets']), flush=True)
        patch_vectors = encode_tiles(model, buffers, patches)
    finally:
        run.detach()
    data = np.stack(trajectories)
    assert np.isfinite(data).all() and np.all(data.mean(-1) > 0)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    tmp = dest.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, fc2=data, preview=patch_vectors[np.array(patch_ids)],
                 index=[r['index'] for r in plan['targets']], label=[r['label'] for r in plan['targets']])
    tmp.replace(dest)
    save(audit, dict(protocol_sha256=digest(PLAN), feature_sha256=digest(dest), histories=HISTORIES,
                     targets=len(data), unique_preview_tiles=len(patches), deterministic_fp32=True))
    print('ARCHIVED', dataset, len(data), len(patches), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'collect'))
    args = parser.parse_args()
    if args.phase == 'freeze':
        for dataset in DATASETS:
            manifest(dataset)
        if PLAN.exists():
            frozen()
        else:
            save(PLAN, protocol())
        print('FROZEN', digest(PLAN), flush=True)
        return
    registration = frozen()
    parts = old.initialize()
    controls = Controls(tuple(registration['hebbian_lambdas']))
    for dataset in DATASETS:
        collect(dataset, parts, controls)


if __name__ == '__main__':
    main()
