"""Explore matching to individually recentered objects from the actual preview canvases."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import torch

from priming_sharpen_collect_20261112 import digest, images, vanilla
from priming_sharpen_reserve_20261112 import manifest, renderer
from priming_sharpen_search_20261112 import Candidate, DATASETS, DOC, OUT, build, control_gate, evaluate, rank, read, save, shape, transform
from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_teacher_20261112 import SemanticEvidence, weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


@torch.inference_mode()
def collect(dataset, source, model, buffers):
    folder = OUT / dataset
    path = folder / f'object_memory_{source}.npz'
    note = path.with_suffix('.json')
    if path.exists() and note.exists():
        assert digest(path) == read(note)['feature_sha256']
        return
    chain = 'chain' if source == 'train' else f'chain_{source}'
    current = folder / f'{chain}_manifest.json'
    rows = read(current)['trials']
    render = None if source == 'train' else renderer(manifest(dataset, source))
    bank = {}
    index = []
    tiles = []
    for row in rows:
        frames = images(dataset, row)[1] if render is None else render(row)[1]
        # Keep the pixels exactly as displayed and move each peripheral tile into the center.
        object_ids = []
        for history in frames:
            ids = []
            for frame in history:
                for left in (5, 155):
                    tile = np.ascontiguousarray(frame[:, 80:144, left:left + 64])
                    identity = hashlib.sha256(tile.tobytes()).digest()
                    if identity not in bank:
                        bank[identity] = len(tiles)
                        tiles.append(tile)
                    ids.append(bank[identity])
            object_ids.append(ids)
        index.append(object_ids)
    mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
    sd = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
    encoded = []
    for start in range(0, len(tiles), 24):
        block = tiles[start:start + 24]
        x = ((.5 - mean) / sd).expand(len(block), 3, 224, 224).clone()
        x[:, :, 80:144, 80:144] = torch.from_numpy(np.stack(block)).cuda()
        model(x)
        encoded.append(buffers['fc2'].cpu().numpy())
    p = np.concatenate(encoded, 0)[np.array(index)]
    assert p.shape == (len(rows), 2, 4, 4096)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, preview=p, row=np.array([r['dataset_index'] for r in rows]))
    tmp.replace(path)
    save(note, dict(feature_sha256=digest(path), native_chain_manifest_sha256=digest(current),
                    individual_object_encodings=len(tiles), target_labels_used=False))
    print('OBJECTS', dataset, source, len(rows), len(tiles), flush=True)


def data(dataset, source):
    root = OUT / dataset
    chain = 'chain' if source == 'train' else f'chain_{source}'
    n = root / f'{chain}.npz'
    manifest_path = root / f'{chain}_manifest.json'
    archive = read(manifest_path)
    p = root / f'object_memory_{source}.npz'
    meta = read(p.with_suffix('.json'))
    assert digest(p) == meta['feature_sha256'] and digest(manifest_path) == meta['native_chain_manifest_sha256']
    assert digest(n) == archive['feature_sha256']
    with np.load(n) as raw, np.load(p) as obj:
        np.testing.assert_array_equal(raw['row'], obj['row'])
        return raw['target_fc2'], obj['preview'], raw['y'], archive['trials']


class ObjectEvidence(SemanticEvidence):
    def query(self, x, p):
        result = super().query(x, p)
        similarity = self.bank.similarity(x, p)
        posterior = self.teacher.predict_proba(self.native_scores(p.reshape(-1, p.shape[-1]))).reshape(*p.shape[:2], -1)
        result['preview_posterior'] = (posterior * softmax(20 * similarity, axis=1)[:, :, None]).sum(1)
        return result

    def direction(self, x, q, mechanism, weight):
        if mechanism == 'object_aligned':
            return (1 - weight) * shape(x) + weight * q['aligned']
        if mechanism == 'object_overlap':
            return shape(x * np.maximum(q['aligned'], .1) ** weight)
        p = (1 - weight) * q['teacher'] + weight * q['preview_posterior']
        evidence = (p @ self.bank.prototypes - self.mean) / self.std
        if mechanism == 'object_gain':
            return shape(x * np.exp(.7 * np.tanh(evidence)))
        return shape(np.maximum(shape(x) + (p @ self.bank.prototypes - self.mean), 0))


def fitted(dataset, kernel):
    x, p, y, rows = data(dataset, 'train')
    fold = np.array([r['fold'] for r in rows])
    slot = np.array([r['slot'] for r in rows])
    bank, head, cuts, training = build(x, p, y, fold, slot)
    fit = fold < 3
    semantic = ObjectEvidence(bank, dataset, kernel, x[fit, 2], y[fit])
    return semantic, head, cuts, training, (x, p, y, fold)


def screen():
    kernel = weights()
    grid = [Candidate(mode, q, a, b, w) for mode, q, a, b, w in product(
        ('object_gain', 'object_residual', 'object_aligned', 'object_overlap'),
        (.25, .5, .75, .9), (.15, .3, .45, .6), (.003, .008, .015), (0., .25, .5, .75))
            if w or mode not in ('object_aligned', 'object_overlap')]
    values, match = {}, {}
    for ds in DATASETS:
        semantic, head, cuts, training, (x, p, y, fold) = fitted(ds, kernel)
        test = fold == 3
        external = data(ds, 'reserve')
        for scope, (a, b, c) in (('fold3', (x[test], p[test], y[test])), ('reserve', external[:3])):
            key = f'{ds}_{scope}'
            values[key], info = evaluate(a, b, c, semantic, head, cuts, grid)
            match[key] = dict(training=training, match_auc=info['auc'])
            print('OBJECT MATCH', key, info['auc'], flush=True)
    ordered = [dict(index=i, parameters=asdict(c), **rank([v[i] for v in values.values()])) for i, c in enumerate(grid)]
    ordered.sort(key=lambda r: (r['passed'], r['score']), reverse=True)
    dest = DOC / 'chain/object_memory_exploration.json'
    save(dest, dict(tested=len(grid), matching=match, ranking=ordered,
                    datasets={k: [v[r['index']] for r in ordered[:5]] for k, v in values.items()}))
    print('OBJECT BEST', ordered[:3], flush=True)


def freeze():
    source = DOC / 'chain/object_memory_exploration.json'
    picked = next(r for r in read(source)['ranking'] if r['passed'])
    frozen = dict(choice=picked, screen_sha256=digest(source),
                  replication_manifests={d: digest(OUT / d / 'replication_manifest.json') for d in DATASETS},
                  context='Object-level features and results were unused in the replication before this choice. Earlier primary ImageNet aggregate was known.')
    dest = OUT / 'chain_object_memory_frozen.json'
    if dest.exists():
        assert read(dest) == frozen
    else:
        save(dest, frozen)
    print(frozen, flush=True)


def confirmation_protocol():
    frozen_path = OUT / 'chain_object_memory_frozen.json'
    protocol = dict(frozen_sha256=digest(frozen_path),
                    stimulus_sha256={d: digest(OUT / d / 'confirmation_manifest.json') for d in DATASETS},
                    fit_feature_sha256={d: digest(OUT / d / 'object_memory_train.npz') for d in DATASETS},
                    source_sha256=digest(Path(__file__)),
                    target_step=9, comparisons=['different', 'none'],
                    primary='activity lower; Fisher, true-class margin, and macro AUC higher; BA nondecreasing',
                    data_note='All image hashes new. Some held-out THINGS plant and vehicle concepts recur; fit concepts excluded.')
    path = OUT / 'chain_object_memory_confirmation_protocol.json'
    if path.exists():
        assert read(path) == protocol
    else:
        save(path, protocol)
    print(protocol, flush=True)
    return protocol


def confirm(scope='replication'):
    frozen = read(OUT / 'chain_object_memory_frozen.json')
    assert digest(DOC / 'chain/object_memory_exploration.json') == frozen['screen_sha256']
    cfg = Candidate(**frozen['choice']['parameters'])
    kernel = weights()
    rows = []
    details = {}
    protocol = confirmation_protocol() if scope == 'confirmation' else None
    for ds in DATASETS:
        hashes = protocol['stimulus_sha256'] if protocol else frozen['replication_manifests']
        assert digest(OUT / ds / f'{scope}_manifest.json') == hashes[ds]
        bank, head, cuts, _, _ = fitted(ds, kernel)
        x, p, y, meta = data(ds, scope)
        q = [bank.query(x[:, i], p[:, i]) for i in range(2)]
        gate = [control_gate(r['match'], cuts[cfg.quantile]) for r in q]
        direction = [bank.direction(x[:, i], r, cfg.mechanism, cfg.preview_weight) for i, r in enumerate(q)]
        e, d = [transform(x[:, i], direction[i], gate[i], cfg.mixing, cfg.suppression).astype(x.dtype) for i in range(2)]
        uniform = transform(x[:, 0], direction[0], gate[0], 0., cfg.suppression).astype(x.dtype)
        target_only = transform(x[:, 0], bank.direction(x[:, 0], q[0], cfg.mechanism, 0.), gate[0],
                                cfg.mixing, cfg.suppression).astype(x.dtype)
        assert np.array_equal(e > 0, x[:, 0] > 0) and np.array_equal(d > 0, x[:, 1] > 0)
        np.testing.assert_allclose(e.mean(1), uniform.mean(1), atol=1e-8, rtol=3e-6)
        samples = PairedConcepts(y, np.array([r['concept'] for r in meta]))
        actual = metrics(e, head, samples)
        absolute = {'exact': {k: float(v[0]) for k, v in actual.items()}}
        for name, ref in (('different', d), ('none', x[:, 2]), ('uniform', uniform), ('target_only', target_only)):
            baseline = metrics(ref, head, samples)
            absolute[name] = {k: float(v[0]) for k, v in baseline.items()}
            for k, effect in contrast(actual, baseline).items():
                row = dict(dataset=ds, reference=name, metric=k, **estimate(effect))
                rows.append(row)
                if name in ('different', 'none'):
                    print(row, flush=True)
        details[ds] = dict(targets=len(y), absolute=absolute, exact_gate=float(gate[0].mean()), different_gate=float(gate[1].mean()),
                           match_auc=float(roc_auc_score(np.repeat([1, 0], len(y)), np.r_[q[0]['match'], q[1]['match']])))
    save(DOC / 'chain' / f'object_memory_{scope}_details.json', dict(frozen=frozen, datasets=details))
    with (DOC / 'chain' / f'object_memory_{scope}.csv').open('w', encoding='utf-8', newline='') as f:
        output = csv.DictWriter(f, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--collect', action='store_true')
    action.add_argument('--freeze', action='store_true')
    action.add_argument('--confirm', action='store_true')
    action.add_argument('--confirmation-protocol', action='store_true')
    cohort = p.add_mutually_exclusive_group()
    cohort.add_argument('--replication', action='store_true', help='Collect the replication only, after freezing')
    cohort.add_argument('--confirmation', action='store_true', help='Use the final image-disjoint cohort')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        if args.collect:
            if args.replication and not (OUT / 'chain_object_memory_frozen.json').exists():
                raise RuntimeError('Freeze the object-level choice before encoding the replication')
            if args.confirmation:
                if not (OUT / 'chain_object_memory_confirmation_protocol.json').is_file():
                    raise RuntimeError('Freeze the confirmation protocol before encoding')
                confirmation_protocol()
            model, buffers = vanilla()
            sources = ('confirmation',) if args.confirmation else ('replication',) if args.replication else ('train', 'reserve')
            for dataset in DATASETS:
                for source in sources:
                    collect(dataset, source, model, buffers)
        elif args.confirmation_protocol:
            confirmation_protocol()
        elif args.freeze:
            freeze()
        elif args.confirm:
            confirm('confirmation' if args.confirmation else 'replication')
        else:
            screen()
