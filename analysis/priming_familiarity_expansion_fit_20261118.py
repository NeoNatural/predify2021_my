"""Fit the augmentation and expansion conditions at the original 8× update budget."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from priming_familiarity_20261112 import BATCH, LR, Familiarity, loss
from priming_familiarity_expansion_collect_20261118 import path_for, view_name
from priming_familiarity_expansion_data_20261118 import (
    CONDITIONS, DATASETS, MANIFEST, REGISTRY, ROOT, VIEWS, register,
)
from priming_familiarity_scale_collect_20261113 import SHARD, shard_path
from priming_familiarity_scale_data_20261113 import SEEDS
from priming_familiarity_scale_fit_20261113 import (
    PROTOCOL as BASE_PROTOCOL, load_data, prediction, read_encoded, run_path,
)
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import read, save


EPOCHS = 20


def baseline_path(dataset, seed):
    return run_path(dataset, 'trainable_encoder', 16, seed)


def model_path(dataset, condition, seed):
    if condition == 'A':
        return baseline_path(dataset, seed).with_suffix('.pth')
    return ROOT / dataset / f'{condition}_{seed}.pth'


def load_features(dataset, condition, rows, *, native=False, legacy=False):
    queries, memories = [], []
    for start in range(0, len(rows), SHARD):
        path = shard_path(dataset, 'test', start) if legacy else path_for(dataset, condition, start)
        with np.load(path) as cache:
            np.testing.assert_array_equal(cache['index'], [r['index'] for r in rows[start:start + SHARD]])
            queries.append(cache['native_query' if native else 'query'])
            memories.append(cache['memory'])
    return tuple(torch.from_numpy(np.concatenate(items)).cuda() for items in (queries, memories))


def evaluate(model, inputs):
    return prediction(model, *inputs)[0].cpu().numpy()


def batches(n, steps, generator):
    # Stream complete random permutations. A batch that spans two permutations
    # can repeat a target; each full pass through the list visits every target once.
    order = torch.empty(0, dtype=torch.long)
    for _ in range(steps):
        if len(order) < BATCH:
            order = torch.cat((order, torch.randperm(n, generator=generator)))
        idx, order = order[:BATCH], order[BATCH:]
        yield idx


def fit_one(train, validation, seed, steps_per_check):
    torch.manual_seed(seed)
    model = Familiarity().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.001)
    order = torch.Generator().manual_seed(seed)
    draw = torch.Generator().manual_seed(seed + 23)
    q, p = train
    n_views, n_train = p.shape[:2]
    best, losses, state = float('inf'), [], None
    for step, idx in enumerate(batches(n_train, steps_per_check * EPOCHS, order), 1):
        view = torch.randint(n_views, (BATCH,), generator=draw).cuda()
        idx = idx.cuda()
        optimizer.zero_grad(set_to_none=True)
        values = read_encoded(model, model.encode(q[view, idx]), model.encode(p[view, idx]))
        objective = loss(model, values)
        assert torch.isfinite(objective)
        objective.backward()
        optimizer.step()
        if step % steps_per_check == 0:
            with torch.no_grad():
                observed = torch.cat([read_encoded(model, model.encode(a), model.encode(b))
                                      for a, b in zip(*(arr.split(BATCH) for arr in validation))])
                val = float(loss(model, observed))
            assert np.isfinite(val)
            losses.append(val)
            if val < best - 1e-5:
                best, selected = val, step
                state = {key: t.detach().cpu().clone() for key, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), state, dict(validation_loss=losses, selected_step=selected, total_updates=step)


def save_arrays(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as file:
        np.savez(file, **arrays)
    tmp.replace(path)


def fit_dataset(dataset, plan, splits, conditions=CONDITIONS):
    base_rows, val_rows = splits['train'], splits['validation']
    baseline, memory, hashes = load_data(dataset, 'train', base_rows)
    val_q, val_mem, val_hashes = load_data(dataset, 'validation', val_rows)
    standard = (baseline, memory)
    validation = (val_q, val_mem)
    steps_per_check = len(base_rows) // BATCH
    assert len(base_rows) % BATCH == 0
    folder = ROOT / dataset
    folder.mkdir(parents=True, exist_ok=True)
    protocol_path = folder / 'protocol.json'
    old_protocol = read(BASE_PROTOCOL)
    assert old_protocol['batch'] == BATCH and old_protocol['lr'] == LR and old_protocol['weight_decay'] == .001
    protocol = dict(registry_sha256=digest(REGISTRY), baseline_protocol_sha256=digest(BASE_PROTOCOL),
                    fitting_sha256=digest(Path(__file__)), seeds=list(SEEDS), conditions=list(CONDITIONS),
                    steps_per_validation=steps_per_check, total_updates=steps_per_check * EPOCHS,
                    learning_rate=LR, weight_decay=.001, batch=BATCH, validation_checks=EPOCHS,
                    selection='minimum validation familiarity loss (improvement > 1e-5)',
                    training_reference='original A, shared exact target t9', augment_views=VIEWS + 1)
    if protocol_path.exists():
        assert read(protocol_path) == protocol
    else:
        save(protocol_path, protocol)

    selected = {}
    for condition in conditions:
        if condition in ('B', 'C'):
            extra = load_features(dataset, condition, plan['added'][condition])
            train = tuple(torch.cat((a, b))[None] for a, b in zip(standard, extra))
            del extra
        if condition in ('D', 'E'):
            augmented = [load_features(dataset, view_name(condition, i), base_rows) for i in range(VIEWS)]
            train = tuple(torch.stack(v) for v in zip(standard, *augmented))
            del augmented
        for seed in SEEDS:
            tag = f'{condition}_{seed}'
            model_file = model_path(dataset, condition, seed)
            outcome = folder / f'{tag}.npz'
            detail = folder / f'{tag}.json'
            if condition == 'A':
                old_path = baseline_path(dataset, seed)
                old = read(old_path.with_suffix('.json'))
                assert old['n'] == len(base_rows) and old['stopped'] == len(old['validation_loss']) == EPOCHS
                assert old['protocol_sha256'] == digest(BASE_PROTOCOL) and old['model_sha256'] == digest(model_file)
                assert old['train_cache'] == hashes and old['val_cache'] == val_hashes
                outcome = old_path.with_suffix('.npz')
                assert old['score_sha256'] == digest(outcome)
                progress = dict(validation_loss=old['validation_loss'], selected_step=old['best_epoch'] * steps_per_check,
                                total_updates=steps_per_check * EPOCHS)
            elif detail.exists():
                progress = read(detail)
                assert progress['protocol_sha256'] == digest(protocol_path)
                assert progress['model_sha256'] == digest(model_file) and progress['scores_sha256'] == digest(outcome)
            else:
                model, state, progress = fit_one(train, validation, seed, steps_per_check)
                tmp = model_file.with_suffix('.tmp')
                torch.save(state, tmp)
                tmp.replace(model_file)
                save_arrays(outcome, train=evaluate(model, standard), validation=evaluate(model, validation))
                del model
            progress.update(model_sha256=digest(model_file), scores_sha256=digest(outcome),
                            protocol_sha256=digest(protocol_path))
            if not detail.exists():
                save(detail, progress)
            selected[tag] = progress['model_sha256']
            print('FIT', dataset, condition, seed, progress['selected_step'], '/', progress['total_updates'], flush=True)
        if condition != 'A':
            del train
    if tuple(conditions) == CONDITIONS:
        save(folder / 'selection.json', dict(protocol_sha256=digest(protocol_path), models=selected))


@torch.inference_mode()
def score_dataset(dataset, plan, splits):
    folder = ROOT / dataset
    selection_path = folder / 'selection.json'
    selection = read(selection_path)
    assert selection['protocol_sha256'] == digest(folder / 'protocol.json')
    for split in ('legacy', 'fresh'):
        if split == 'fresh' and min(plan['fresh_coverage'].values()) < 2:
            continue
        test = splits['test'] if split == 'legacy' else plan['fresh']
        q, p = load_features(dataset, split, test, native=True, legacy=split == 'legacy')
        for condition in CONDITIONS:
            for seed in SEEDS:
                tag = f'{condition}_{seed}'
                path = model_path(dataset, condition, seed)
                assert selection['models'][tag] == digest(path)
                model = Familiarity().cuda().eval()
                model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
                scores = torch.cat([model(a, b) for a, b in zip(q.split(BATCH), p.split(BATCH))]).cpu().numpy()
                assert scores.shape == (len(test), 3) and np.isfinite(scores).all()
                save_arrays(folder / f'{tag}.{split}.npz', scores=scores, index=[r['index'] for r in test],
                            selection_sha256=digest(selection_path))
        del q, p
        print('SCORED', dataset, split, len(test), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('phase', choices=('fit', 'test'))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        inputs = args.dataset, register()[args.dataset], read(MANIFEST)[args.dataset]['splits']
        if args.phase == 'fit':
            fit_dataset(*inputs, conditions=args.conditions)
        else:
            score_dataset(*inputs)
