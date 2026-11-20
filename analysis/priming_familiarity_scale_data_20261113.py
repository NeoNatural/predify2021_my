"""Register nested training and concept-disjoint validation/test stimuli."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from priming_familiarity_data_20261112 import OUT, PRECEDING, source_manifest
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_reserve_20261112 import NATIVE, THINGS, csv_rows
from priming_sharpen_search_20261112 import DATASETS, read, save


ROOT = OUT / 'learning_curve_20261113'
LEVELS = (2, 8, 16)  # Concepts per category, eight targets each.
SEEDS = (2026111301, 2026111302, 2026111303)
SPLITS = ('train', 'validation', 'test')
HISTORIES = ('exact', 'same_concept', 'different')
MANIFEST = ROOT / 'stimuli.json'


def earlier(dataset, archive):
    records = list(archive['images'].values())
    sources = [OUT.parent / dataset / f'{scope}_manifest.json' for scope in PRECEDING]
    sources.append(OUT / dataset / 'fresh_stimuli.json')
    for path in sources:
        plan = read(path)
        records.extend(plan['targets'] + list(plan['distractors'].values()))
    return {r['concept'] for r in records}, {r['sha256'] for r in records}


def candidates(dataset):
    categories = {r['concept']: r['category'] for r in csv_rows(NATIVE / dataset / 'concept_review.csv')}
    if dataset == 'imagenet':
        from torchvision.datasets import ImageNet

        val = ImageNet('C:/Users/liang/Documents/ImageNet', split='val')
        rows = ((i, val.wnids[label], Path(path), None, False) for i, (path, label) in enumerate(val.samples))
    else:
        categories.update({r['concept']: 'plants' for r in csv_rows(NATIVE / 'inventory/concepts.csv')
                           if r['manual_exclusive'] == 'plant'})
        rows = ((int(r['index']), r['concept'], THINGS / r['relative_path'], r['sha256'],
                 r['historical_image'] == 'True') for r in csv_rows(NATIVE / 'inventory/images.csv')
                if r['eligible'] == 'True')
    result = defaultdict(list)
    for i, concept, path, sha, historical in rows:
        if concept in categories:
            result[concept].append((i, path, sha, historical))
    return categories, result


def assign_previews(rows, rng):
    by_concept = defaultdict(list)
    for r in rows:
        by_concept[r['concept']].append(r)
    peers = {}
    for group in by_concept.values():
        for a, b in zip(group, group[1:] + group[:1]):
            peers[a['index']] = b['index']
    for j, r in enumerate(rows):
        cutoff = next((n for n in LEVELS if r['rank'] < n), LEVELS[-1])
        pool = [x for x in rows if x['label'] != r['label'] and x['rank'] < cutoff]
        labels = sorted({x['label'] for x in pool})
        label_order = rng.permutation(labels)
        backgrounds = [int(rng.choice([x['index'] for x in pool if x['label'] == k]))
                       for k in label_order[:3]]
        different = int(rng.choice([x['index'] for x in pool if x['index'] not in backgrounds]))
        r.update(same_concept=peers[r['index']], different=different, backgrounds=backgrounds,
                 side=(j + r['label']) % 2, contrast=(.4, .5, .6)[(j + 2 * r['label']) % 3])


def partition(dataset):
    archive = source_manifest(dataset)
    exposed, previous_hash = earlier(dataset, archive)
    prior_fit = {archive['images'][str(i)]['concept'] for r in archive['trials'] if r['fold'] in (1, 2, 3)
                 for i in (r['dataset_index'], *r['previews'], *r['primes'].values())}
    category, pool = candidates(dataset)
    rng = np.random.default_rng(2026111317)
    rows = {split: [] for split in SPLITS}
    taken_concepts, taken_hashes = set(), set()
    for split, n_concepts, per_concept in (('test', 6, 4), ('validation', 6, 4), ('train', 16, 8)):
        for label, name in enumerate(archive['categories']):
            n_images = 2 if (dataset, split, name) == ('things', 'test', 'plants') else per_concept
            eligible = sorted(c for c, group in pool.items() if category[c] == name and len(group) >= n_images
                              and c not in taken_concepts and (split != 'test' or c not in prior_fit))
            ordered = rng.permutation(eligible).tolist()
            if split != 'train':
                ordered.sort(key=lambda c: (c in exposed, len(pool[c]) >= 8))
            selected = 0
            for concept in ordered:
                group = pool[concept]
                accepted = []
                selected_hashes = set()
                for j in rng.permutation(len(group)):
                    i, path, sha, historical = group[j]
                    if split != 'train' and (historical or sha in previous_hash):
                        continue
                    if not path.is_file():
                        continue
                    actual = digest(path)
                    if sha is not None and sha != actual:
                        continue
                    if actual in taken_hashes or actual in selected_hashes:
                        continue
                    if split != 'train' and actual in previous_hash:
                        continue
                    selected_hashes.add(actual)
                    accepted.append(dict(index=i, path=str(path), sha256=actual, concept=concept, label=label,
                                         rank=selected, previously_exposed=concept in exposed))
                    if len(accepted) == n_images:
                        break
                if len(accepted) < n_images:
                    continue
                rows[split].extend(accepted)
                taken_concepts.add(concept)
                taken_hashes.update(selected_hashes)
                selected += 1
                if selected == n_concepts:
                    break
            if selected != n_concepts:
                raise RuntimeError(f'{dataset} {split} {name}: {selected}/{n_concepts} eligible concepts')
        rows[split].sort(key=lambda r: (r['rank'], r['label']))
        assign_previews(rows[split], rng)
        lookup = {r['index']: r for r in rows[split]}
        assert len(lookup) == len(rows[split])
        for r in rows[split]:
            cutoff = next(n for n in LEVELS if r['rank'] < n)
            assert all(lookup[i]['rank'] < cutoff for i in [r['same_concept'], r['different'], *r['backgrounds']])
            assert lookup[r['same_concept']]['concept'] == r['concept'] and r['index'] != r['same_concept']
            assert all(lookup[i]['label'] != r['label'] for i in [r['different'], *r['backgrounds']])
    return dict(categories=archive['categories'], splits=rows)


def register():
    if MANIFEST.exists():
        return read(MANIFEST)
    plans = {ds: partition(ds) for ds in DATASETS}
    for ds, data in plans.items():
        for split, rows in data['splits'].items():
            concepts = {r['concept'] for r in rows}
            exposed = {r['concept'] for r in rows if r['previously_exposed']}
            print('STIMULI', ds, split, len(rows), len(concepts), 'previously exposed', len(exposed), flush=True)
    save(MANIFEST, plans)
    return plans


class Renderer:
    def __init__(self, rows):
        from PIL import Image
        from torchvision.models import VGG16_Weights
        from predify_experiments.native_categories_20261029 import FilteredInput

        self.images = {r['index']: r for r in rows}
        self.tile_cache = {}
        self.normalized = {}
        self.filter = FilteredInput(torch)
        self.transform = VGG16_Weights.IMAGENET1K_V1.transforms()
        self.open_image = Image.open
        self.grey = self.filter.canvas([])

    def tile(self, index, contrast):
        key = index, contrast
        if key not in self.tile_cache:
            if index not in self.normalized:
                row = self.images[index]
                path = Path(row['path'])
                assert digest(path) == row['sha256']
                with self.open_image(path) as im:
                    self.normalized[index] = self.transform(im.convert('RGB'))
            x = self.normalized[index].cuda()
            if contrast is not None:
                mean = x.mean((-2, -1), keepdim=True)
                x = self.filter.separable(mean + contrast * (x - mean), self.filter.gaussian)
            self.tile_cache[key] = self.filter.tile(x)
        return self.tile_cache[key]

    def canvas(self, indices, x_positions, contrast):
        x = self.grey.clone()
        for i, left in zip(indices, x_positions):
            x[:, 80:144, left:left + 64] = self.tile(i, contrast)
        return x

    def __call__(self, row):
        target = self.canvas([row['index']], [80], None)
        first = self.canvas(row['backgrounds'][:2], [5, 155], row['contrast'])
        last = []
        for field in ('index', 'same_concept', 'different'):
            objects = [row['backgrounds'][2]]
            objects.insert(row['side'], row[field])
            last.append(self.canvas(objects, [5, 155], row['contrast']))
        return target, first, torch.stack(last)


if __name__ == '__main__':
    register()
