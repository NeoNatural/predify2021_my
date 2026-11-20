"""Recheck the frozen object mechanism using the same FP32 render path as Predify."""

from __future__ import annotations

import argparse
import csv
import hashlib

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from hebb_chain_20261111 import old
from priming_sharpen_collect_20261112 import digest, images, vanilla
from priming_sharpen_confirm_20261112 import contrast, metrics
from priming_sharpen_object_memory_20261112 import ObjectEvidence
from priming_sharpen_reserve_20261112 import manifest, renderer
from priming_sharpen_search_20261112 import Candidate, DATASETS, DOC, OUT, build, control_gate, read, save, transform
from priming_sharpen_teacher_20261112 import weights
from report_hebb_post_pool_20261111 import PairedConcepts, estimate


def tiles(previews):
    return [np.ascontiguousarray(frame[:, 80:144, left:left + 64]) for frame in previews for left in (5, 155)]


@torch.inference_mode()
def encode_tiles(model, buffers, patches):
    mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
    sd = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
    result = []
    for i in range(0, len(patches), 24):
        block = patches[i:i + 24]
        x = ((.5 - mean) / sd).expand(24, 3, 224, 224).clone()
        x[:len(block), :, 80:144, 80:144] = torch.from_numpy(np.stack(block)).cuda()
        model(x)
        result.append(buffers['fc2'][:len(block)].cpu().numpy())
    return np.concatenate(result, 0)


def paths(dataset, scope):
    root = OUT / dataset
    chain = 'chain' if scope == 'train' else f'chain_{scope}'
    return root / f'{chain}.npz', root / f'{chain}_manifest.json', root / f'object_memory_fp32_{scope}.npz'


def collect(dataset, scope, model, buffers):
    _, meta, dest = paths(dataset, scope)
    note = dest.with_suffix('.json')
    if dest.exists() and note.exists():
        saved = read(note)
        assert digest(dest) == saved['feature_sha256'] and digest(meta) == saved['chain_manifest_sha256']
        return
    rows = read(meta)['trials']
    render = None if scope == 'train' else renderer(manifest(dataset, scope))
    index, patches, cache = [], [], {}
    for row in rows:
        p = images(dataset, row)[1] if render is None else render(row)[1]
        indices = []
        for history in p:
            group = []
            for tile in tiles(history):
                key = hashlib.sha256(tile.tobytes()).digest()
                if key not in cache:
                    cache[key] = len(patches)
                    patches.append(tile)
                group.append(cache[key])
            indices.append(group)
        index.append(indices)
    p = encode_tiles(model, buffers, patches)[np.array(index)]
    tmp = dest.with_suffix('.tmp')
    with tmp.open('wb') as out:
        np.savez(out, preview=p, row=np.array([r['dataset_index'] for r in rows]))
    tmp.replace(dest)
    save(note, dict(feature_sha256=digest(dest), chain_manifest_sha256=digest(meta),
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                    cudnn_tf32=torch.backends.cudnn.allow_tf32, matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
    print('FP32', dataset, scope, len(patches), flush=True)


def features(dataset, scope):
    n, meta, p = paths(dataset, scope)
    archive = read(meta)
    note = read(p.with_suffix('.json'))
    assert digest(n) == archive['feature_sha256'] and digest(p) == note['feature_sha256']
    assert digest(meta) == note['chain_manifest_sha256']
    assert note['deterministic'] and not note['cudnn_tf32'] and not note['matmul_tf32']
    with np.load(n) as a, np.load(p) as b:
        np.testing.assert_array_equal(a['row'], b['row'])
        return a['target_fc2'], b['preview'], a['y'], archive['trials']


def fitted(dataset, kernel):
    x, p, y, rows = features(dataset, 'train')
    fold = np.array([r['fold'] for r in rows])
    slot = np.array([r['slot'] for r in rows])
    bank, head, cuts, _ = build(x, p, y, fold, slot)
    use = fold < 3
    return ObjectEvidence(bank, dataset, kernel, x[use, 2], y[use]), head, cuts


def adjusted(bank, x, p, cuts, cfg):
    q = bank.query(x, p)
    g = control_gate(q['match'], cuts[cfg.quantile])
    d = bank.direction(x, q, cfg.mechanism, cfg.preview_weight)
    y = transform(x, d, g, cfg.mixing, cfg.suppression).astype(x.dtype)
    assert np.array_equal(y > 0, x > 0) and np.all(y.mean(1) <= x.mean(1) + 1e-7)
    return y


def evaluate(scope, kernel, cfg):
    output = []
    for ds in DATASETS:
        bank, head, cuts = fitted(ds, kernel)
        x, p, y, meta = features(ds, scope)
        e, d = [adjusted(bank, x[:, i], p[:, i], cuts, cfg) for i in range(2)]
        s = PairedConcepts(y, np.array([r['concept'] for r in meta]))
        exact, different, none = [metrics(v, head, s) for v in (e, d, x[:, 2])]
        for ref, base in (('different', different), ('none', none)):
            for name, delta in contrast(exact, base).items():
                result = dict(dataset=ds, reference=ref, metric=name, **estimate(delta))
                output.append(result)
                if name in ('ba', 'auc', 'margin'):
                    print(scope, result, flush=True)
    with (DOC / 'chain' / f'object_memory_{scope}_fp32.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=('reserve', 'replication', 'confirmation', 'validation'), default='replication')
    args = p.parse_args()
    cfg = Candidate(**read(OUT / 'chain_object_memory_frozen.json')['choice']['parameters'])
    old.initialize()
    model, buffers = vanilla()
    for ds in DATASETS:
        collect(ds, 'train', model, buffers)
        collect(ds, args.scope, model, buffers)
    if args.scope != 'validation':
        evaluate(args.scope, weights(), cfg)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
