"""Check whether the large-sample frozen-encoder control was limited by its budget."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import torch

import priming_familiarity_scale_fit_20261113 as trainer
from priming_familiarity_20261112 import BATCH, Familiarity
from priming_familiarity_report_20261112 import paired_auc
from priming_familiarity_scale_collect_20261113 import SHARD, shard_path
from priming_familiarity_scale_data_20261113 import DATASETS, LEVELS, MANIFEST, ROOT, SEEDS
from priming_familiarity_scale_report_20261113 import row_summary, write
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_search_20261112 import read, save


FOLDER = ROOT / 'frozen_budget_check'
PLAN = FOLDER / 'protocol.json'
FROZEN = FOLDER / 'models.json'
MAX = 192
MIN = 64
PATIENCE = 24
LEVEL = LEVELS[-1]


def register():
    trainer.protocol()
    record = dict(level=LEVEL, seeds=list(SEEDS), max_epochs=MAX, min_epochs=MIN, patience=PATIENCE,
                  selection='validation loss only; original learning-curve selection unchanged',
                  decision='the original frozen controls reached epoch 64 with falling validation loss',
                  primary_selection_sha256=digest(trainer.SELECTION), source_sha256=digest(Path(__file__)))
    if PLAN.exists():
        assert read(PLAN) == record
    else:
        save(PLAN, record)


def fit(plans):
    result = {}
    budget = (trainer.MAX_EPOCHS, trainer.MIN_EPOCHS, trainer.PATIENCE)
    trainer.MAX_EPOCHS, trainer.MIN_EPOCHS, trainer.PATIENCE = MAX, MIN, PATIENCE
    try:
        for ds in DATASETS:
            p = plans[ds]['splits']
            q, m, _ = trainer.load_data(ds, 'train', p['train'])
            v, vm, _ = trainer.load_data(ds, 'validation', p['validation'])
            vw = trainer.weights_for(p['validation'])
            result[ds] = {}
            for seed in SEEDS:
                model, state, info = trainer.fit('frozen_encoder', seed, q, m, (v, vm), len(q))
                observed, _ = trainer.prediction(model, v, vm)
                path = FOLDER / f'{ds}_{seed}.pth'
                torch.save(state, path)
                result[ds][str(seed)] = dict(model_sha256=digest(path),
                                              validation_auc=trainer.aucs(observed.cpu().numpy(), vw), **info)
                print('EXTENDED FROZEN', ds, seed, info['best_epoch'], info['stopped'],
                      result[ds][str(seed)]['validation_auc'], flush=True)
    finally:
        trainer.MAX_EPOCHS, trainer.MIN_EPOCHS, trainer.PATIENCE = budget
    save(FROZEN, dict(protocol_sha256=digest(PLAN), datasets=result))


@torch.inference_mode()
def test(plans):
    models = read(FROZEN)
    assert models['protocol_sha256'] == digest(PLAN)
    table = []
    for ds in DATASETS:
        rows = plans[ds]['splits']['test']
        native, memory = [], []
        for start in range(0, len(rows), SHARD):
            with np.load(shard_path(ds, 'test', start)) as data:
                np.testing.assert_array_equal(data['index'], [r['index'] for r in rows[start:start + SHARD]])
                native.append(data['native_query'])
                memory.append(data['memory'])
        q, m = (torch.from_numpy(np.concatenate(v)).cuda() for v in (native, memory))
        w = trainer.weights_for(rows)
        cells = {k: [] for k in ('extended_frozen', 'original_frozen', 'trainable_encoder')}
        for seed in SEEDS:
            path = FOLDER / f'{ds}_{seed}.pth'
            assert digest(path) == models['datasets'][ds][str(seed)]['model_sha256']
            model = Familiarity().cuda().eval()
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            scores = torch.cat([model(a, b) for a, b in zip(q.split(BATCH), m.split(BATCH))]).cpu().numpy()
            cells['extended_frozen'].append(scores)
            for key, arm in (('original_frozen', 'frozen_encoder'), ('trainable_encoder', 'trainable_encoder')):
                src = trainer.run_path(ds, arm, LEVEL, seed).with_suffix('.native_test.npz')
                with np.load(src) as f:
                    cells[key].append(f['scores'])
        for i, name in enumerate(('same_concept', 'different'), 1):
            values = {k: np.stack([paired_auc(x[:, 0], x[:, i], w) for x in v]) for k, v in cells.items()}
            table.extend(dict(dataset=ds, negative=name, arm=k, reference='', **row_summary(v)) for k, v in values.items())
            for reference in ('original_frozen', 'trainable_encoder'):
                table.append(dict(dataset=ds, negative=name, arm='extended_frozen', reference=reference,
                                  **row_summary(values['extended_frozen'] - values[reference])))
    write('frozen_budget_test', table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'fit', 'test'))
    args = parser.parse_args()
    register()
    if args.phase == 'register':
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    (fit if args.phase == 'fit' else test)(read(MANIFEST))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
