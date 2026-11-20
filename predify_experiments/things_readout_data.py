"""THINGS data adapter for the frozen readout/initialization replication."""
from pathlib import Path
from collections import Counter
import json
import random

import numpy as np

from .things_metadata import (load_exact_concept_mapping, load_imagenet_wnids,
    imagefolder_index, load_image_sources, _source_for, sha256_file)
from .paper_stimuli import SUPER_CATEGORIES, imagenet_supercategory
from .supplementary import load
from .readout_init_20261012 import balanced_pick, immutable_json

DATA_PROJECT = Path('C:/Users/liang/Documents/ChatGPT/THINGS数据集测试predify_my')
DATA_ROOT = DATA_PROJECT / 'data/THINGS'
TARGET_CATEGORIES = ('animals', 'clothing', 'food', 'vehicles')


class SubsetReadout:
    """Keep fixed ImageNet readout weights, restricting candidate categories."""
    def __init__(self, head, categories=TARGET_CATEGORIES):
        self.head = head
        self.columns = [SUPER_CATEGORIES.index(c) for c in categories]
        assert list(head.classes_) == list(range(5))
        self.classes_ = np.arange(len(categories))

    def decision_function(self, features):
        return self.head.decision_function(features)[:, self.columns]


class ThingsImages:
    def __init__(self, manifest, transform):
        self.records = manifest['images']
        self.transform = transform

    def __getitem__(self, index):
        from PIL import Image
        row = self.records[str(index)]
        path = DATA_ROOT / 'images' / row['relative_path']
        if sha256_file(path) != row['sha256']:
            raise ValueError(f'Image changed after split freeze: {path}')
        with Image.open(path) as image:
            return self.transform(image.convert('RGB')), row['imagenet_label']


def inventory():
    images_root = DATA_ROOT / 'images'
    meta = DATA_ROOT / 'metadata/things_concepts.tsv'
    wnids_file = DATA_PROJECT / 'predify_experiments/resources/imagenet1k_wnids.txt'
    mappings = load_exact_concept_mapping(meta, load_imagenet_wnids(wnids_file))
    assert len(mappings) == 1854
    assert sum(m.imagenet_index is not None for m in mappings) == 334
    mapping = {m.unique_id: m for m in mappings}
    paths, _ = imagefolder_index(images_root)
    assert len(paths) == 26107
    sources = load_image_sources(DATA_ROOT / 'metadata/01_image-level')
    old_file = DATA_ROOT / 'derived/THINGS_N_Back-1_seed114514.pckl'
    old = load(old_file)
    old_paths = set(old['image_relpaths'])
    for index, relative in zip(old['out_list'], old['image_relpaths']):
        assert paths[index].relative_to(images_root).as_posix() == relative
    old_hashes = {sha256_file(images_root / p) for p in old_paths}
    records = {}; labels = [-1] * len(paths); used = set(); seen_hashes = set()
    counts = {c: Counter() for c in SUPER_CATEGORIES}
    for index, path in enumerate(paths):
        relative = path.relative_to(images_root).as_posix()
        concept = mapping[path.relative_to(images_root).parts[0]]
        label = concept.imagenet_index
        category = imagenet_supercategory(label) if label is not None else None
        if category is None:
            continue
        counts[category]['all_images'] += 1
        source = _source_for(path, images_root, sources)
        if source == 'imagenet':
            counts[category]['excluded_imagenet_source'] += 1
            continue
        digest = sha256_file(path)
        if digest in seen_hashes:
            counts[category]['excluded_duplicate_bytes'] += 1
            continue
        seen_hashes.add(digest)
        labels[index] = label
        records[str(index)] = dict(relative_path=relative, native_concept=concept.unique_id,
            wnid=concept.wnid, imagenet_label=label, category=category, source=source, sha256=digest)
        counts[category]['eligible_images'] += 1
        if source == 'source_unknown': counts[category]['source_unknown'] += 1
        if relative in old_paths or digest in old_hashes:
            used.add(index)
            counts[category]['historical_images'] += 1
        else:
            counts[category]['new_images'] += 1
    metadata_files = [meta, wnids_file, old_file, DATA_PROJECT/'predify_experiments/things.py']
    metadata_files += sorted((DATA_ROOT/'metadata/01_image-level').glob('*.tsv'))
    return records, labels, used, dict(counts={c:dict(v) for c,v in counts.items()},
        metadata_sha256={str(p):sha256_file(p) for p in metadata_files},
        historical_unique_images=len(old_paths), historical_unique_hashes=len(old_hashes),
        exact_concept_mappings=334, all_concepts=1854, all_images=26107)


def select_targets(labels, excluded, per_category, rng):
    # Existing selector visits all five categories; restrict its input for the
    # four-category replication without assigning a new meaning to any label.
    pools = {c: {} for c in TARGET_CATEGORIES}
    for index, label in enumerate(labels):
        category = imagenet_supercategory(label)
        if category in pools and index not in excluded:
            pools[category].setdefault(label, []).append(index)
    picks = {}
    for category, mapping in pools.items():
        concepts = list(mapping); rng.shuffle(concepts)
        for pool in mapping.values(): rng.shuffle(pool)
        chosen = []
        while len(chosen) < per_category:
            previous = len(chosen)
            for concept in concepts:
                if mapping[concept] and len(chosen) < per_category:
                    chosen.append(mapping[concept].pop())
            if len(chosen) == previous:
                raise ValueError(f'Insufficient fresh images: {category}')
        picks[category] = chosen
    return [picks[c][i] for i in range(per_category) for c in TARGET_CATEGORIES]


def prepare_manifest(out, doc, seed, noises, conditions, train_conditions):
    records, labels, historical, audit = inventory()
    rng = random.Random(seed)
    # One unused image per concept is reserved exclusively for previews.
    # This ensures every target has an independent same-concept prime.
    by_label = {}
    for key, rec in records.items(): by_label.setdefault(rec['imagenet_label'], []).append(int(key))
    reserved = set()
    for label, pool in sorted(by_label.items()):
        fresh = [i for i in pool if i not in historical]
        reserved.add(rng.choice(fresh if fresh else pool))
    test = select_targets(labels, historical | reserved, 20, rng)
    train = select_targets(labels, historical | reserved | set(test), 40, rng)
    dev = select_targets(labels, historical | reserved | set(test) | set(train), 20, rng)
    targets = set(train + dev + test)
    assert len(targets) == 320 and not (targets & historical)
    def trials(indices):
        category_pools = {c:[] for c in SUPER_CATEGORIES}; concept_pools = {}
        for key, rec in records.items():
            i = int(key)
            if i in targets: continue
            category_pools[rec['category']].append(i)
            concept_pools.setdefault(rec['imagenet_label'], []).append(i)
        result = []
        for ordinal, index in enumerate(indices):
            rec = records[str(index)]; category = rec['category']; label = rec['imagenet_label']
            trial_rng = random.Random(seed + index)
            previews = [trial_rng.choice(category_pools[c]) for c in SUPER_CATEGORIES if c != category]
            trial_rng.shuffle(previews)
            # With four interleaved categories, ordinal % 2 alone confounds
            # category with preview side. Alternate side within each category.
            slot = 2 + (ordinal // len(TARGET_CATEGORIES) + ordinal % len(TARGET_CATEGORIES)) % 2
            other_concepts = [i for i in category_pools[category] if labels[i] != label]
            result.append(dict(dataset_index=index, imagenet_label=label, native_concept=rec['native_concept'],
                category_index=TARGET_CATEGORIES.index(category), previews=previews, slot=slot,
                primes=dict(exact=index, same_synset=trial_rng.choice(concept_pools[label]),
                    same_category=trial_rng.choice(other_concepts), different_category=previews[slot], neutral=None)))
        return result
    manifest = dict(seed=seed, dataset='THINGS', categories=TARGET_CATEGORIES,
        train=trials(train), dev=trials(dev), test=trials(test), images=records,
        historical_target_exclusions=sorted(historical), reserved_preview_indices=sorted(reserved),
        noise_seeds=noises, steps=15, train_conditions=train_conditions, conditions=conditions,
        primary_readout='pooled', primary_total_backbone_budget=9,
        scope='Four-category balanced targets; exact WordNet mappings only. All five original categories remain available for previews, preserving the original four distractor slots. Known ImageNet-source images excluded everywhere. Targets exclude all old THINGS cohort images and byte duplicates; splits and other-trial previews exclude every new target.',
        legacy_readout='Original five-category ImageNet t9 readout, restricted to animals/clothing/food/vehicles without refitting.',
        no_test_selection=True, data_audit=audit)
    immutable_json(out/'manifest.json',json.loads(json.dumps(manifest)))
    audit['splits'] = {}
    for split in ('train','dev','test'):
        rows=manifest[split]
        audit['splits'][split]=dict(images=len(rows), concepts=len({r['native_concept'] for r in rows}),
            categories=dict(Counter(records[str(r['dataset_index'])]['category'] for r in rows)),
            historical_overlap=len({r['dataset_index'] for r in rows}&historical))
    audit['pairwise_image_overlap']={a+'-'+b:len(set(x)&set(y)) for a,x,b,y in [('train',train,'dev',dev),('train',train,'test',test),('dev',dev,'test',test)]}
    audit['target_categories']=TARGET_CATEGORIES
    immutable_json(doc/'split_audit.json',json.loads(json.dumps(audit)))
    print('SPLITS',audit['splits'],flush=True)
    return manifest
