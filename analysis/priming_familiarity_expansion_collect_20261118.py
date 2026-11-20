"""Collect new four-layer trajectories with the frozen 8× circuit and controls."""
import argparse

import numpy as np
import torch

from hebb_chain_20261111 import Controls, Experiment, old
from priming_familiarity_expansion_data_20261118 import (
    AugmentedRenderer, DATASETS, MANIFEST, ROOT, Renderer, register,
)
from priming_familiarity_scale_collect_20261113 import BATCH, SHARD, BatchExperiment, pack, shard_path
from priming_familiarity_scale_native_20261113 import native_batch
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_search_20261112 import read


def path_for(dataset, condition, start):
    return ROOT / dataset / 'features' / condition / f'{start:04d}.npz'


def view_name(condition, view):
    return f'{condition}{view + 1}'


def collect_view(dataset, condition, batch, rows, render, *, native=False):
    collector = native_batch if native else lambda run, *frames: run.collect(*frames)
    for start in range(0, len(rows), SHARD):
        segment = rows[start:start + SHARD]
        ids = [r['index'] for r in segment]
        path = path_for(dataset, condition, start)
        if path.exists():
            with np.load(path) as cached:
                np.testing.assert_array_equal(cached['index'], ids)
                assert ('native_query' in cached) == native
            continue
        results = [collector(batch, *pack(render, segment[j:j+BATCH])) for j in range(0, len(segment), BATCH)]
        arrays = [np.concatenate(group) for group in zip(*results)]
        data = dict(index=ids, query=arrays[0], memory=arrays[1])
        if native:
            data.update(native_query=arrays[3], native_fc2=arrays[4])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as file:
            np.savez(file, **data)
        tmp.replace(path)
        print('CACHE', dataset, condition, start + len(segment), '/', len(rows), flush=True)


def audit(dataset, batch, rows):
    baseline = Renderer(rows)
    with np.load(shard_path(dataset, 'train', 0)) as cache:
        q, p, _ = batch.collect(*pack(baseline, rows[:2]))
        np.testing.assert_allclose(q, cache['query'][:2], atol=.004, rtol=.003)
        np.testing.assert_allclose(p, cache['memory'][:2], atol=.004, rtol=.003)
    # Zero displacement or the original contrast must exactly reproduce the old renderer.
    for condition, values in [('D', [[0, 0]] * len(rows)), ('E', [r['contrast'] for r in rows])]:
        renderer = AugmentedRenderer(rows, condition, values)
        assert all(torch.equal(a, b) for a, b in zip(renderer(rows[0]), baseline(rows[0])))
    print('BASELINE REPLAY VERIFIED', dataset, flush=True)


def collect(dataset, phase):
    plans = register()
    plan = plans[dataset]
    splits = read(MANIFEST)[dataset]['splits']
    train = splits['train']
    if phase == 'test':
        selection = ROOT / dataset / 'selection.json'
        assert selection.is_file(), 'Freeze all checkpoints before collecting held-out features'
        if min(plan['fresh_coverage'].values()) < 2:
            print('FRESH TEST UNAVAILABLE', dataset, plan['fresh_coverage'], flush=True)
            return
    controls = Controls(tuple(frozen()['hebbian_lambdas']))
    run = Experiment(dataset, old.initialize())
    run.fit()
    batch = BatchExperiment(run, controls)
    try:
        if phase == 'test':
            collect_view(dataset, 'fresh', batch, plan['fresh'], Renderer(plan['fresh']), native=True)
        else:
            audit(dataset, batch, train)
            for condition, rows in plan['added'].items():
                collect_view(dataset, condition, batch, rows, Renderer(train + rows))
            for condition, draws in plan['augmentation'].items():
                for i, values in enumerate(draws):
                    collect_view(dataset, view_name(condition, i), batch, train, AugmentedRenderer(train, condition, values))
    finally:
        run.detach()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('phase', choices=('fit', 'test'))
    args = parser.parse_args()
    collect(args.dataset, args.phase)
