"""Fit the original familiarity circuit at three nested sample sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch
from torch.nn import functional as F

from priming_familiarity_20261112 import BATCH, DECAY, LR, Familiarity, loss
from priming_familiarity_diagnosis_20261112 import SIDES, regions
from priming_familiarity_report_20261112 import paired_auc
from priming_familiarity_scale_collect_20261113 import SHARD, shard_path
from priming_familiarity_scale_data_20261113 import DATASETS, LEVELS, MANIFEST, ROOT, SEEDS
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import read, save
from report_hebb_post_pool_20261111 import PairedConcepts


ARMS = ('trainable_encoder', 'frozen_encoder')
MAX_EPOCHS = 64
MIN_EPOCHS = 20
PATIENCE = 12
PROTOCOL = ROOT / 'protocol.json'
SELECTION = ROOT / 'selected.json'


def protocol():
    modules = ('priming_familiarity_scale_data_20261113.py', 'priming_familiarity_scale_collect_20261113.py',
               'priming_familiarity_scale_fit_20261113.py', 'priming_familiarity_20261112.py')
    cfg = dict(seeds=list(SEEDS), levels=list(LEVELS), arms=list(ARMS), max_epochs=MAX_EPOCHS, min_epochs=MIN_EPOCHS,
               patience=PATIENCE, batch=BATCH, lr=LR, weight_decay=.001, manifest_sha256=digest(MANIFEST),
               read_query='shared exact t9 across histories', early_stop='minimum validation familiarity loss',
               selection='highest three-seed validation Exact versus Same AUC; Different AUC breaks ties',
               source_sha256={m: digest(Path(__file__).with_name(m)) for m in modules})
    if PROTOCOL.exists():
        assert read(PROTOCOL) == cfg
    else:
        save(PROTOCOL, cfg)


def load_data(dataset, split, rows):
    queries, memories, hashes = [], [], {}
    for start in range(0, len(rows), SHARD):
        path = shard_path(dataset, split, start)
        with np.load(path) as f:
            np.testing.assert_array_equal(f['index'], [r['index'] for r in rows[start:start + SHARD]])
            queries.append(f['query'])
            memories.append(f['memory'])
        hashes[path.name] = digest(path)
    q, p = (torch.from_numpy(np.concatenate(x)).cuda() for x in (queries, memories))
    return q, p, hashes


def weights_for(rows):
    return PairedConcepts(np.array([r['label'] for r in rows]), np.array([r['concept'] for r in rows])).concept_weights


def read_encoded(model, q, p):
    reference = model.reference(q)[:, None]
    h = model.reference(p)
    frames = h.transpose(-1, -2) @ p / p.shape[-2]
    memory = DECAY * frames[:, :, 0] + frames[:, :, 1]
    return model.read_encoded(q[:, None], reference, memory)


@torch.no_grad()
def encode_all(model, x):
    return torch.cat([model.encode(chunk) for chunk in x.split(BATCH)])


@torch.inference_mode()
def prediction(model, q, p, rows=None):
    values, local = [], []
    if rows is not None:
        masks = regions()
        center = torch.as_tensor(masks['center'], device=q.device)
        sides = torch.as_tensor(np.stack([masks[SIDES[r['side']]] for r in rows]), device=q.device)
    for start in range(0, len(q), BATCH):
        idx = slice(start, start + BATCH)
        z = model.encode(q[idx])
        encoded = model.encode(p[idx])
        values.append(read_encoded(model, z, encoded))
        if rows is not None:
            preview = encoded[:, :, 1]
            sample_mask = sides[idx][:, None, :, None].expand(-1, 3, -1, preview.shape[-1])
            pools = F.normalize(preview.gather(2, sample_mask).mean(2), dim=-1)
            target = F.normalize(z[:, center].mean(1), dim=-1)
            local.append((target[:, None] * pools).sum(-1))
    return torch.cat(values), torch.cat(local) if local else None


def aucs(scores, weights):
    return [float(paired_auc(scores[:, 0], scores[:, i], weights)[0]) for i in (1, 2)]


def fit(arm, seed, q, p, validation, n):
    torch.manual_seed(seed)
    model = Familiarity().cuda()
    if arm == 'frozen_encoder':
        model.encoder.requires_grad_(False)
    initial = {name: t.detach().cpu().clone() for name, t in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW([t for t in model.parameters() if t.requires_grad], lr=LR, weight_decay=.001)
    rng = torch.Generator().manual_seed(seed)
    if arm == 'frozen_encoder':
        q, p = encode_all(model, q[:n]), encode_all(model, p[:n])
        validation = tuple(encode_all(model, x) for x in validation)
        encode = lambda x: x
    else:
        encode = model.encode
    best, at, history = float('inf'), 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(n, generator=rng).cuda()
        for idx in order.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            observed = read_encoded(model, encode(q[idx]), encode(p[idx]))
            objective = loss(model, observed)
            assert torch.isfinite(objective)
            objective.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            observed = torch.cat([read_encoded(model, encode(a), encode(b))
                                  for a, b in zip(*(x.split(BATCH) for x in validation))])
            val = float(loss(model, observed))
        history.append(val)
        if val < best - 1e-5:
            best, at = val, epoch
            chosen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch >= MIN_EPOCHS and epoch - at >= PATIENCE:
            break
    model.load_state_dict(chosen)
    if arm == 'frozen_encoder':
        assert all(torch.equal(t.cpu(), initial[k]) for k, t in model.encoder.state_dict().items())
    else:
        assert any(not torch.equal(t.cpu(), initial[k]) for k, t in model.encoder.state_dict().items())
    return model, chosen, dict(best_epoch=at, stopped=epoch, validation_loss=history, best_loss=best,
                               trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_path(dataset, arm, level, seed):
    return ROOT / dataset / 'models' / f'{arm}_{level}_{seed}'


def run_dataset(ds, plan):
    train, val = (plan['splits'][s] for s in ('train', 'validation'))
    q, p, hashes = load_data(ds, 'train', train)
    v, pv, val_hash = load_data(ds, 'validation', val)
    w = weights_for(val)
    records = []
    for level in LEVELS:
        n = sum(r['rank'] < level for r in train)
        assert all(r['rank'] < level for r in train[:n])
        tw = weights_for(train[:n])
        for arm in ARMS:
            for seed in SEEDS:
                path = run_path(ds, arm, level, seed)
                model_path, detail_path, raw_path = (path.with_suffix(s) for s in ('.pth', '.json', '.npz'))
                if detail_path.exists():
                    row = read(detail_path)
                    assert digest(model_path) == row['model_sha256'] and digest(raw_path) == row['score_sha256']
                    assert row['protocol_sha256'] == digest(PROTOCOL) and row['train_cache'] == hashes and row['val_cache'] == val_hash
                else:
                    model, state, progress = fit(arm, seed, q, p, (v, pv), n)
                    tr, _ = prediction(model, q[:n], p[:n])
                    va, local = prediction(model, v, pv, val)
                    tr, va, local = (x.cpu().numpy() for x in (tr, va, local))
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(state, model_path)
                    np.savez(raw_path, train=tr, validation=va, local_validation=local)
                    row = dict(dataset=ds, arm=arm, level=level, seed=seed, n=n,
                               train_auc=aucs(tr, tw), validation_auc=aucs(va, w), local_validation_auc=aucs(local, w),
                               model_sha256=digest(model_path), score_sha256=digest(raw_path),
                               protocol_sha256=digest(PROTOCOL), train_cache=hashes, val_cache=val_hash, **progress)
                    save(detail_path, row)
                records.append(row)
                print('FIT', ds, arm, n, seed, row['best_epoch'], row['stopped'],
                      row['train_auc'], row['validation_auc'], flush=True)
    return records


def train():
    protocol()
    if SELECTION.exists():
        assert read(SELECTION)['protocol_sha256'] == digest(PROTOCOL)
        return
    data = read(MANIFEST)
    runs = {ds: run_dataset(ds, data[ds]) for ds in DATASETS}
    selected = {}
    for ds, records in runs.items():
        cells = []
        for arm in ARMS:
            for level in LEVELS:
                vals = [r['validation_auc'] for r in records if r['level'] == level and r['arm'] == arm]
                cells.append(dict(arm=arm, level=level, validation_auc=np.mean(vals, 0).tolist()))
        selected[ds] = max(cells, key=lambda r: tuple(r['validation_auc']))
    save(SELECTION, dict(protocol_sha256=digest(PROTOCOL), chosen=selected,
                         model_sha256={ds: {f"{r['arm']}_{r['level']}_{r['seed']}": r['model_sha256']
                                             for r in items} for ds, items in runs.items()}))
    print('SELECTED BEFORE TEST', selected, flush=True)


def test():
    protocol()
    selection = read(SELECTION)
    assert selection['protocol_sha256'] == digest(PROTOCOL)
    plans = read(MANIFEST)
    for ds in DATASETS:
        rows = plans[ds]['splits']['test']
        q, p, hashes = load_data(ds, 'test', rows)
        w = weights_for(rows)
        for key, sha in selection['model_sha256'][ds].items():
            path = ROOT / ds / 'models' / key
            model_path = path.with_suffix('.pth')
            assert digest(model_path) == sha
            model = Familiarity().cuda().eval()
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
            scores, local = prediction(model, q, p, rows)
            scores, local = scores.cpu().numpy(), local.cpu().numpy()
            dest = path.with_suffix('.test.npz')
            np.savez(dest, scores=scores, local=local)
            save(path.with_suffix('.test.json'), dict(test_cache=hashes, selection_sha256=digest(SELECTION),
                                                      score_sha256=digest(dest), model_sha256=sha))
            print('TEST', ds, key, aucs(scores, w), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'test'))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.phase == 'register':
            protocol()
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
            (train if args.phase == 'fit' else test)()
