"""Lock added photos, augmentation draws and held-out visual roles before fitting."""
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import torch

from priming_familiarity_scale_data_20261113 import MANIFEST, ROOT as BASE, Renderer, candidates, earlier, source_manifest
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import DOC as BASE_DOC, read, save


ROOT = BASE.parent / 'expansion_20261118'
DOC = BASE_DOC / ROOT.name
REGISTRY = ROOT / 'stimuli_locked.json'
DATASETS = ('things', 'imagenet')
CONDITIONS = ('A', 'B', 'C', 'D', 'E')
VIEWS = 2


def photos(group, indices, hashes, n, fresh=False):
    accepted = []
    for index, path, expected, historical in group:
        if index in indices or (fresh and historical) or expected in hashes or not path.is_file():
            continue
        actual = digest(path)
        if actual in hashes or (expected is not None and expected != actual):
            continue
        try:
            with Image.open(path) as image:
                image.load()
        except OSError:
            continue
        accepted.append(dict(index=index, path=str(path), sha256=actual))
        indices.add(index)
        hashes.add(actual)
        if len(accepted) == n:
            break
    return accepted


def add_previews(rows, donors, rng):
    by_concept, by_label = defaultdict(list), defaultdict(list)
    for row in donors:
        by_concept[row['concept']].append(row['index'])
        by_label[row['label']].append(row['index'])
    for i, row in enumerate(rows):
        peers = [j for j in by_concept[row['concept']] if j != row['index']]
        labels = [label for label in by_label if label != row['label']]
        order = rng.permutation(labels)
        backgrounds = [int(rng.choice(by_label[label])) for label in order[:3]]
        different = [index for label in labels for index in by_label[label] if index not in backgrounds]
        row.update(same_concept=int(rng.choice(peers)), backgrounds=backgrounds,
                   different=int(rng.choice(different)), side=(i + row['label']) % 2,
                   contrast=(.4, .5, .6)[(i + 2 * row['label']) % 3])


def organize(dataset, plan):
    rng = np.random.default_rng(2026111817)
    splits = plan['splits']
    train = splits['train']
    all_rows = [row for rows in splits.values() for row in rows]
    indices = {r['index'] for r in all_rows}
    hashes = {r['sha256'] for r in all_rows}
    used_concepts = {r['concept'] for r in all_rows}
    category, pool = candidates(dataset)
    exposed, previous_hashes = earlier(dataset, source_manifest(dataset))
    pool = {concept: [group[i] for i in rng.permutation(len(group))] for concept, group in pool.items()}
    added = {condition: [] for condition in ('B', 'C')}
    for label, name in enumerate(plan['categories']):
        novel = rng.permutation(sorted(c for c in pool if category[c] == name and c not in used_concepts))
        ordered = sorted(novel.tolist(), key=lambda c: c in exposed)
        for concept, fresh_only in [(c, strict) for strict in (True, False) for c in ordered]:
            if len(pool[concept]) < 8:
                continue
            new_indices, new_hashes = indices.copy(), hashes.copy()
            excluded = new_hashes | previous_hashes if fresh_only else new_hashes
            selected = photos(pool[concept], new_indices, excluded, 8, fresh=fresh_only)
            if len(selected) == 8:
                indices = new_indices
                hashes.update(r['sha256'] for r in selected)
                added['B'].extend(dict(r, concept=concept, label=label, rank=16) for r in selected)
                break
        else:
            raise ValueError(f'{dataset} {name}: no additional concept with eight unique photos')
        old = {r['concept']: r['rank'] for r in train if r['label'] == label}
        order = rng.permutation(sorted(old)).tolist()
        used = set()
        for concept, fresh_only in [(c, strict) for strict in (True, False) for c in order]:
            if concept in used:
                continue
            excluded = hashes | previous_hashes if fresh_only else hashes
            selected = photos(pool[concept], indices, excluded, 1, fresh=fresh_only)
            added['C'].extend(dict(r, concept=str(concept), label=label, rank=old[concept]) for r in selected)
            if selected:
                used.add(concept)
                hashes.add(selected[0]['sha256'])
            if sum(r['label'] == label for r in added['C']) == 8:
                break
        else:
            raise ValueError(f'{dataset} {name}: fewer than eight ninth photos')
    for condition, rows in added.items():
        assert len(rows) == len(train) // 16
        for i, row in enumerate(rows):
            context = [r for r in train if r['label'] == row['label']][:8][i % 8]
            peers = [r['index'] for r in train + rows if r['concept'] == row['concept'] and r['index'] != row['index']]
            row.update({key: context[key] for key in ('side', 'contrast', 'backgrounds', 'different')})
            row['same_concept'] = (rows[(i // 8) * 8 + (i + 1) % 8]['index'] if condition == 'B'
                                   else int(rng.choice(peers)))
            row['historically_exposed_concept'] = row['concept'] in exposed
            row['historically_exposed_photo'] = row['sha256'] in previous_hashes

    development = {r['concept'] for r in train + splits['validation'] + added['B']}
    fresh = []
    forbidden = hashes | previous_hashes
    for label, name in enumerate(plan['categories']):
        options = rng.permutation(sorted(c for c in pool if category[c] == name and c not in development))
        count = 0
        for concept in sorted(options.tolist(), key=lambda c: (c in used_concepts, c in exposed)):
            new_indices, new_hashes = indices.copy(), forbidden.copy()
            selected = photos(pool[concept], new_indices, new_hashes, 4, fresh=True)
            if len(selected) < 2:
                continue
            indices, forbidden = new_indices, new_hashes
            fresh.extend(dict(r, concept=concept, label=label, rank=count,
                              historically_exposed_concept=concept in exposed,
                              legacy_test_concept=concept in used_concepts) for r in selected)
            count += 1
            if count == 6:
                break
    coverage = {name: len({r['concept'] for r in fresh if r['label'] == label})
                for label, name in enumerate(plan['categories'])}
    if min(coverage.values()) >= 2:
        add_previews(fresh, fresh, rng)

    augmentation = {condition: [] for condition in ('D', 'E')}
    for _ in range(VIEWS):
        position, contrast = [], []
        for row in train:
            lo, hi = (-5, 8) if row['side'] == 0 else (-8, 5)
            dx = int(rng.integers(lo, hi + 1))
            dy = int(rng.integers(-8, 9))
            position.append([dx, dy or int(dx == 0)])
            contrast.append(float(rng.uniform(.4, .6)))
        augmentation['D'].append(position)
        augmentation['E'].append(contrast)
    return dict(added=added, fresh=fresh, fresh_coverage=coverage, augmentation=augmentation)


def register():
    base = read(MANIFEST)
    manifest_hash = digest(MANIFEST)
    if REGISTRY.exists():
        result = read(REGISTRY)
        assert result['baseline_sha256'] == manifest_hash
        return result['datasets']
    plans = {ds: organize(ds, base[ds]) for ds in DATASETS}
    save(REGISTRY, dict(baseline_sha256=manifest_hash, datasets=plans))
    overview = {}
    for ds, plan in plans.items():
        names = base[ds]['categories']
        overview[ds] = dict(
            added={key: dict(Counter(names[r['label']] for r in rows)) for key, rows in plan['added'].items()},
            new_concepts=[r['concept'] for r in plan['added']['B'][::8]],
            added_historical_photos={key: sum(r['historically_exposed_photo'] for r in rows)
                                     for key, rows in plan['added'].items()},
            fresh_concepts=plan['fresh_coverage'],
            fresh_photos=dict(Counter(names[r['label']] for r in plan['fresh'])),
            fresh_historical_concepts=sorted({r['concept'] for r in plan['fresh'] if r['historically_exposed_concept']}),
            fresh_legacy_test_concepts=sorted({r['concept'] for r in plan['fresh'] if r['legacy_test_concept']}))
        print('REGISTERED', ds, overview[ds], flush=True)
    save(DOC / 'stimulus_audit.json', overview)
    return plans


class AugmentedRenderer(Renderer):
    def __init__(self, rows, condition, values):
        super().__init__(rows)
        self.condition = condition
        self.values = {row['index']: value for row, value in zip(rows, values, strict=True)}

    def __call__(self, row):
        target = self.canvas([row['index']], [80], None)
        first = self.canvas(row['backgrounds'][:2], [5, 155], row['contrast'])
        side = row['side']
        left, top, contrast = (5, 155)[side], 80, row['contrast']
        value = self.values[row['index']]
        if self.condition == 'D':
            left += value[0]
            top += value[1]
        else:
            contrast = value
        background = self.canvas([row['backgrounds'][2]], [(155, 5)[side]], row['contrast'])
        last = []
        for key in ('index', 'same_concept', 'different'):
            frame = background.clone()
            frame[:, top:top + 64, left:left + 64] = self.tile(row[key], contrast)
            last.append(frame)
        return target, first, torch.stack(last)


if __name__ == '__main__':
    register()
