"""Reserve new concepts in every visual role for the final sharpening replication."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

from priming_sharpen_collect_20261112 import ARCHIVE, CHECKPOINT, digest, vanilla, vanilla_trial
from priming_sharpen_search_20261112 import DATASETS, OUT, ROOT, read, save
import numpy as np


NATIVE = ROOT / 'docs/native_categories_readouts_20261029'
THINGS = Path('C:/Users/liang/Documents/ChatGPT/THINGS数据集测试predify_my/data/THINGS/images')
PLANT_RESERVE = {'bonsai', 'bush', 'hedge', 'weed'}
PLANT_REPLICATION = {'aloe', 'flower', 'plant', 'tree'}
PLANT_CONFIRMATION = {'flower', 'plant', 'tree', 'hedge'}
REUSE_CONFIRMATION = PLANT_CONFIRMATION | {'raft', 'wagon', 'ferry', 'hearse'}
PLANT_VALIDATION = {'bonsai', 'bush', 'weed', 'flower'}
REUSE_VALIDATION = PLANT_VALIDATION | {'cruise_ship', 'ship', 'skateboard', 'raft', 'kayak'}


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pools(dataset, used, scope, *, plants=None, additional_concepts=None):
    accepted = {r['concept']: r['category'] for r in csv_rows(NATIVE / dataset / 'concept_review.csv')
                if r['split'] == 'reserve'}
    if additional_concepts is not None:
        accepted.update(additional_concepts)
    if dataset == 'imagenet':
        from torchvision.datasets import ImageNet
        val = ImageNet('C:/Users/liang/Documents/ImageNet', split='val')
        files = ((index, Path(p), val.wnids[label]) for index, (p, label) in enumerate(val.samples))
    else:
        metadata = {r['concept']: r for r in csv_rows(NATIVE / 'inventory/concepts.csv')}
        if plants is None:
            plants = {'reserve': PLANT_RESERVE, 'replication': PLANT_REPLICATION,
                      'confirmation': PLANT_CONFIRMATION, 'validation': PLANT_VALIDATION}[scope]
        for c in plants:
            assert metadata[c]['manual_exclusive'] == 'plant' and c not in used
            accepted[c] = 'plants'
        rows = csv_rows(NATIVE / 'inventory/images.csv')
        files = ((int(r['index']), THINGS / r['relative_path'], r['concept']) for r in rows
                 if r['eligible'] == 'True' and r['historical_image'] == 'False')
    result = defaultdict(list)
    for index, path, concept in files:
        if concept in accepted and concept not in used:
            result[concept].append((index, path, accepted[concept]))
    return result


def manifest(dataset, scope='reserve'):
    path = OUT / dataset / f'{scope}_manifest.json'
    if path.exists():
        return read(path)
    source = ARCHIVE / dataset / 'manifest.json'
    old = read(source)
    previous = []
    archived = list(old['images'].values())
    known_hashes = {r['sha256'] for r in archived}
    if scope == 'replication':
        prior = manifest(dataset)
        previous = prior['targets'] + list(prior['distractors'].values())
        if dataset == 'things':
            formal = {str(idx) for r in old['trials'] for idx in
                      (r['dataset_index'], *r['previews'], *r['primes'].values())}
            assert all(old['images'][idx]['concept'] != 'aloe' for idx in formal)
            archived = [r for r in archived if r['concept'] != 'aloe']
    preceding = {'confirmation': ('reserve', 'replication'), 'validation': ('reserve', 'replication', 'confirmation')}.get(scope, ())
    if preceding:
        for prior_scope in preceding:
            prior = manifest(dataset, prior_scope)
            previous.extend(prior['targets'] + list(prior['distractors'].values()))
    forbidden = {r['sha256'] for r in [*archived, *previous]}
    used = {r['concept'] for r in [*archived, *previous]}
    past_concepts = {r['concept'] for r in previous}
    if dataset == 'things' and preceding:
        eligible_reuse = REUSE_CONFIRMATION if scope == 'confirmation' else REUSE_VALIDATION
        assert not eligible_reuse & {r['concept'] for r in archived}
        used -= eligible_reuse
    groups = pools(dataset, used, scope)
    rng = np.random.default_rng({'reserve': 2026111203, 'replication': 2026111204,
                                 'confirmation': 2026111205, 'validation': 2026111206}[scope])
    group_size = 5 if preceding else 7
    selected = {}
    rows = []
    for label, category in enumerate(old['categories']):
        available = sorted(c for c, items in groups.items() if len(items) >= group_size and items[0][2] == category)
        assert len(available) >= 4, (dataset, category, len(available))
        count = 0
        concepts = list(rng.permutation(available))
        if preceding:
            concepts.sort(key=lambda c: c in past_concepts)
        for c in concepts:
            group = groups[c]
            accepted = []
            order = list(rng.permutation(len(group)))
            if dataset == 'things' and c == 'aloe':
                order.sort(key=lambda j: digest(group[j][1]) in known_hashes)
            for j in order:
                i, p, _ = group[j]
                if not p.is_file():
                    continue
                sha = digest(p)
                if sha not in forbidden:
                    accepted.append(dict(index=i, path=str(p), sha256=sha, concept=str(c), label=label))
                    forbidden.add(sha)
                if len(accepted) == group_size:
                    break
            if len(accepted) < group_size:
                continue
            selected[str(c)] = accepted[-1]
            rows.extend(accepted[:-1])
            count += 1
            if count == 4:
                break
        assert count == 4, (dataset, category, count)
    for j, row in enumerate(rows):
        candidate_labels = [label for label in range(len(old['categories'])) if label != row['label']]
        if len(candidate_labels) == 3:
            candidate_labels.append(candidate_labels[(j // 4) % 3])
        used = set()
        previews = []
        for label in candidate_labels:
            c = str(rng.choice(sorted(k for k, r in selected.items() if r['label'] == label and k not in used)))
            used.add(c)
            previews.append(c)
        rng.shuffle(previews)
        row.update(slot=2 + j % 2, distractors=previews)
    reserved = dict(dataset=dataset, categories=old['categories'], source_manifest_sha256=digest(source),
                    targets=rows, distractors=selected,
                    plants_note='THINGS plant concepts use native manual exclusive labels and broaden prior plant subtype exclusions')
    if scope == 'replication':
        reserved.update(previous_reserve_sha256=digest(OUT / dataset / 'reserve_manifest.json'),
                        historical_development_reuse=[r['index'] for r in [*rows, *selected.values()]
                                                      if r['sha256'] in known_hashes])
    if preceding:
        reserved.update(previous_manifest_sha256={s: digest(OUT / dataset / f'{s}_manifest.json') for s in preceding},
                        reused_concepts=sorted(set(selected) & past_concepts), target_images_per_concept=group_size - 1)
    save(path, reserved)
    return reserved


def renderer(plan):
    from PIL import Image
    import torch
    from torchvision.models import VGG16_Weights
    from predify_experiments.native_categories_20261029 import FilteredInput
    f = FilteredInput(torch)
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()

    def pixels(record):
        p = Path(record['path'])
        assert digest(p) == record['sha256']
        with Image.open(p) as im:
            return transform(im.convert('RGB')).cuda()

    background = {c: pixels(p) for c, p in plan['distractors'].items()}

    def render(row):
        x = pixels(row)
        central = f.canvas([(x, 'center')]).cpu().numpy()
        negative = [background[c] for c in row['distractors']]
        positive = list(negative)
        positive[row['slot']] = x
        preview = np.stack([[f.canvas(list(zip(p[k:k+2], ('left', 'right'))), preview=True).cpu().numpy()
                             for k in (0, 2)] for p in (positive, negative)])
        return central, preview

    return render


def encode(dataset, plan, model, buffers):
    path = OUT / dataset / 'vanilla_reserve.npz'
    note = path.with_name('vanilla_reserve_manifest.json')
    if path.exists() and note.exists():
        assert digest(path) == read(note)['feature_sha256']
        return
    render = renderer(plan)
    features = defaultdict(list)
    for j, row in enumerate(plan['targets']):
        central, preview = render(row)
        for name, a in vanilla_trial(model, buffers, central, preview).items():
            features[name].append(a)
        if (j + 1) % 24 == 0 or j + 1 == len(plan['targets']):
            print('RESERVE', dataset, j + 1, '/', len(plan['targets']), flush=True)
    rows = plan['targets']
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as out:
        np.savez(out, **{k: np.stack(v) for k, v in features.items()},
                 y=np.array([r['label'] for r in rows]), row=np.array([r['index'] for r in rows]),
                 concept=np.array([r['concept'] for r in rows]), slot=np.array([r['slot'] // 2 for r in rows]))
    tmp.replace(path)
    save(note, dict(feature_sha256=digest(path), checkpoint_sha256=digest(CHECKPOINT),
                    reserve_manifest_sha256=digest(OUT / dataset / 'reserve_manifest.json'),
                    trials=[dict(dataset_index=r['index']) for r in rows]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encode', action='store_true')
    p.add_argument('--scope', choices=('reserve', 'replication', 'confirmation', 'validation'), default='reserve')
    args = p.parse_args()
    plans = {dataset: manifest(dataset, args.scope) for dataset in DATASETS}
    print({d: {c: 4 for c in plan['categories']} for d, plan in plans.items()}, flush=True)
    if args.encode:
        if args.scope != 'reserve':
            p.error('The replication is collected through the four-layer chain')
        if not any((OUT / name).is_file() for name in ('vanilla_semantic_frozen.json', 'vanilla_robust_frozen.json')):
            raise RuntimeError('Freeze a candidate before encoding the out-of-cohort reserve')
        model, buffers = vanilla()
        for dataset, plan in plans.items():
            encode(dataset, plan, model, buffers)


if __name__ == '__main__':
    main()
