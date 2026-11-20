"""Encode the original trial pixels for a frozen FC sharpening experiment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from priming_sharpen_search_20261112 import DATASETS, OUT, ROOT, read, save
import numpy as np
import torch


HISTORIES = ('exact', 'different_category')
ARCHIVE = ROOT / 'Hebbian-CNN/Log/native_categories_readouts_20261029'
PREVIOUS = ROOT / 'Hebbian-CNN/Log/hebb_chain_20261111'
CHECKPOINT = Path(r'C:\Users\liang\.cache\torch\hub\checkpoints\vgg16-397923af.pth')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def trials(dataset, scope):
    manifest = read(ARCHIVE / dataset / 'manifest.json')
    if scope == 'train':
        return [r for r in manifest['trials'] if r['fold'] in (1, 2, 3)]
    if scope == 'secondary':
        rows = [r for r in manifest['trials'] if r['fold'] == 4]
    else:
        prior = {r['concept'] for r in read(PREVIOUS / dataset / 'heldout_manifest.json')}
        rows = [r for r in manifest['trials'] if r['fold'] == 0 and r['concept'] not in prior]
    fit = [r for r in manifest['trials'] if r['fold'] in (1, 2, 3)]

    def roles(group, key):
        return {manifest['images'][str(idx)][key] for r in group for idx in
                (r['dataset_index'], *r['previews'], r['primes']['exact'], r['primes']['different_category'])}

    for key in ('concept', 'sha256'):
        assert not roles(rows, key) & roles(fit, key), (dataset, key)
    expected = (32 if dataset == 'imagenet' else 40) * (2 if scope == 'secondary' else 1)
    assert len(rows) == expected
    return rows


def images(dataset, row):
    path = ARCHIVE / dataset / 'formal' / f"stimuli_{row['dataset_index']}.npz"
    with np.load(path) as f:
        x = f['target_0']
        v = np.stack([[f[f'{h}_preview{k}'] for k in range(2)] for h in HISTORIES])
    assert x.shape == (3, 224, 224) and v.shape == (2, 2, 3, 224, 224)
    slot = row['slot'] // 2
    assert np.array_equal(v[0, 1 - slot], v[1, 1 - slot])
    return x, v, digest(path)


def vanilla():
    from torchvision.models import vgg16
    model = vgg16(weights=None)
    model.load_state_dict(torch.load(CHECKPOINT, weights_only=True, map_location='cpu'))
    model.cuda().eval().requires_grad_(False)
    buffers = {}
    for idx, name in ((1, 'fc1'), (4, 'fc2')):
        model.classifier[idx].register_forward_hook(lambda _m, _inp, out, name=name: buffers.update({name: out.detach()}))
    return model, buffers


@torch.inference_mode()
def vanilla_trial(model, buffers, target, preview):
    batch = np.concatenate([target[None], preview.reshape(-1, 3, 224, 224)])
    model(torch.from_numpy(batch).cuda())
    result = {}
    for name, x in buffers.items():
        values = x.cpu().numpy()
        result['target_' + name] = np.repeat(values[None, 0], 3, axis=0)
        result['preview_' + name] = values[1:].reshape(2, 2, -1)
    return result


def collect_vanilla(scope):
    model, buffers = vanilla()
    for dataset in DATASETS:
        rows = trials(dataset, scope)
        folder = OUT / dataset
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f'vanilla_{scope}.npz'
        if destination.exists():
            continue
        bank = {}
        hashes = []
        for i, row in enumerate(rows):
            target, preview, sha = images(dataset, row)
            hashes.append(sha)
            for name, value in vanilla_trial(model, buffers, target, preview).items():
                bank.setdefault(name, []).append(value)
            if (i + 1) % 16 == 0 or i + 1 == len(rows):
                print('ENCODE', scope, dataset, i + 1, '/', len(rows), flush=True)
        temp = destination.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez(f, **{k: np.stack(v) for k, v in bank.items()},
                     y=np.array([r['category_index'] for r in rows]), row=np.array([r['dataset_index'] for r in rows]),
                     concept=np.array([r['concept'] for r in rows]), slot=np.array([r['slot'] // 2 for r in rows]))
        temp.replace(destination)
        save(folder / f'vanilla_{scope}_manifest.json', dict(trials=rows, archived_stimuli_sha256=hashes,
             checkpoint_sha256=digest(CHECKPOINT), feature_sha256=digest(destination)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=('confirm', 'secondary'), default='confirm')
    args = p.parse_args()
    freeze_file = 'vanilla_frozen.json' if args.scope == 'confirm' else 'vanilla_gated_frozen.json'
    if not (OUT / freeze_file).is_file():
        raise RuntimeError('Freeze candidate configurations before encoding confirmation')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    collect_vanilla(args.scope)


if __name__ == '__main__':
    main()
