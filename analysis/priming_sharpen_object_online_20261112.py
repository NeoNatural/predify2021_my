"""Verify the frozen object-memory update in the real terminal FC2 hook."""

from __future__ import annotations

import argparse

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from hebb_chain_20261111 import Controls, Experiment, old
from hebb_chain_20261111 import OUT as CHAIN
from priming_sharpen_chain_20261112 import simulate
from priming_sharpen_collect_20261112 import vanilla
from priming_sharpen_confirm_20261112 import read_features
from priming_sharpen_object_memory_20261112 import data
from priming_sharpen_object_precision_20261112 import adjusted, encode_tiles, features, fitted, tiles
from priming_sharpen_reserve_20261112 import manifest, renderer
from priming_sharpen_search_20261112 import Candidate, DATASETS, DOC, OUT, read, save
from priming_sharpen_teacher_20261112 import weights


def check(dataset, parts, model, buffers, kernel, scope, cfg, recurrent):
    bank, _, cuts = fitted(dataset, kernel)
    plan = manifest(dataset, scope)
    render = renderer(plan)
    archived = read_features(dataset, 'chain', scope)
    _, memory, _, _ = features(dataset, scope)
    original_archive = None if scope == 'validation' else data(dataset, scope)[1]

    def apply(x, p):
        memory = np.broadcast_to(p, (len(x), *p.shape))
        if scope == 'validation':
            from priming_sharpen_corroboration_20261112 import update
            return update(bank, x, bank.query(x, memory), cuts, cfg)
        return adjusted(bank, x, memory, cuts, cfg)

    for i, row in enumerate(plan['targets']):
        fc2 = archived['trajectory_fc2'][i, 0]
        expected = apply(fc2, memory[i, 0])
        if np.max(np.abs(expected - fc2)) > 1e-4:
            break
    else:
        raise AssertionError('No active object-memory trial')
    image, preview = render(row)
    object_vectors = encode_tiles(model, buffers, tiles(preview[0]))
    np.testing.assert_allclose(object_vectors, memory[i, 0], atol=3e-5, rtol=3e-5)

    def edit(x, _):
        return apply(x, object_vectors)

    run = Experiment(dataset, parts)
    run.fit()
    img = torch.from_numpy(image).cuda()
    target, _ = simulate(run, img, torch.from_numpy(preview[0]).cuda(), recurrent, active=True, edit=edit)
    np.testing.assert_array_equal(target['fc1'], archived['trajectory_fc1'][i, 0])
    np.testing.assert_array_equal(target['fc2'], fc2)
    np.testing.assert_allclose(target['terminal'], expected, rtol=1e-5, atol=1e-5)
    none, _ = simulate(run, img, [run.c.grey, run.c.grey], recurrent, active=False, edit=edit)
    np.testing.assert_array_equal(none['fc2'], archived['trajectory_fc2'][i, 2])
    run.detach()
    results = dict(target=row['index'], steps=15, scope=scope, original_fc1_identical=True, no_preview_identical=True,
                object_encoding_max_abs_error=float(np.max(np.abs(object_vectors - memory[i, 0]))),
                terminal_max_abs_error=float(np.max(np.abs(expected - target['terminal']))),
                terminal_max_change=float(np.max(np.abs(fc2 - target['terminal']))))
    if original_archive is not None:
        results['fp32_vs_initial_archive_max_abs_difference'] = float(np.max(np.abs(object_vectors - original_archive[i, 0])))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('replication', 'confirmation', 'validation'), default='replication')
    args = parser.parse_args()
    if args.scope == 'validation':
        cfg = read(OUT / 'chain_object_corroboration_frozen.json')['config']
    else:
        cfg = Candidate(**read(OUT / 'chain_object_memory_frozen.json')['choice']['parameters'])
    recurrent = Controls(tuple(read(CHAIN / 'selection.json')['suppression']['lam']))
    kernel = weights()
    parts = old.initialize()
    model, buffers = vanilla()
    checks = {d: check(d, parts, model, buffers, kernel, args.scope, cfg, recurrent) for d in DATASETS}
    print(checks, flush=True)
    save(DOC / 'chain' / f'online_object_{args.scope}.json', checks)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
