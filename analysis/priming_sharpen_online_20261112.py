"""Check that frozen FC2 replay is equivalent to a real intervention before the final linear layer."""

from __future__ import annotations

import argparse

from hebb_chain_20261111 import Experiment, Controls, old
from hebb_chain_20261111 import OUT as CHAIN
from priming_sharpen_chain_20261112 import simulate
from priming_sharpen_confirm_20261112 import read_features
from priming_sharpen_reserve_20261112 import manifest, renderer
from priming_sharpen_search_20261112 import DATASETS, DOC, OUT, control_gate, read, save, shape, transform
from priming_sharpen_semantic_confirm_20261112 import frozen_profile
from priming_sharpen_teacher_20261112 import models, weights
import numpy as np
from threadpoolctl import threadpool_limits
import torch


def audit(dataset, parts, cfg, kernel, recurrent, scope='reserve', crosslayer=False):
    run = Experiment(dataset, parts)
    run.fit()
    if crosslayer:
        from priming_sharpen_crosslayer_20261112 import directions, load_models
        bank, _, cuts, auxiliary, _ = load_models(dataset, kernel)

        def update(x, memory, fc1):
            q = bank.query(x, memory)
            p = auxiliary.predict_proba(shape(fc1))
            d = directions(bank, x, q['teacher'], p, cfg['mode'], cfg['weight'])
            g = control_gate(q['match'], cuts[cfg['quantile']])
            return transform(x, d, g, cfg['mixing'], cfg['suppression']).astype(np.float32)
    else:
        bank, _, cuts, _ = models(dataset, 'chain', kernel)

        def update(x, memory, _):
            q = bank.query(x, memory)
            g = control_gate(q['match'], cuts[cfg.quantile])
            d = bank.direction(x, q, cfg.mechanism, cfg.preview_weight)
            return transform(x, d, g, cfg.mixing, cfg.suppression).astype(np.float32)

    def edit(x, memory):
        return update(x, memory, run.layers[2].record['signed'].cpu().numpy())

    plan = manifest(dataset, scope)
    render = renderer(plan)
    archived = read_features(dataset, 'chain', scope)
    for i, row in enumerate(plan['targets']):
        fc2 = archived['trajectory_fc2'][i, 0]
        v = archived['preview_fc2'][i, 0][None]
        expected = update(fc2, np.broadcast_to(v, (15, *v.shape[1:])), archived['trajectory_fc1'][i, 0])
        if not np.allclose(expected, fc2, atol=1e-6, rtol=1e-6):
            break
    else:
        raise AssertionError('No reserve trial activated the frozen transform')
    image, frames = render(row)
    image = torch.from_numpy(image).cuda()
    previews = torch.from_numpy(frames[0]).cuda()
    target, preview = simulate(run, image, previews, recurrent, active=True, edit=edit)
    np.testing.assert_array_equal(target['fc1'], archived['trajectory_fc1'][i, 0])
    np.testing.assert_array_equal(target['fc2'], fc2)
    np.testing.assert_array_equal(preview['fc2'], v[0])
    np.testing.assert_allclose(target['terminal'], expected, rtol=5e-6, atol=5e-6)
    none, _ = simulate(run, image, [run.c.grey, run.c.grey], recurrent, active=False, edit=edit)
    np.testing.assert_array_equal(none['fc2'], archived['trajectory_fc2'][i, 2])
    run.detach()
    return dict(target=row['index'], steps=15, original_fc1_identical=True, no_preview_identical=True,
                terminal_max_abs_vs_offline=float(np.abs(expected - target['terminal']).max()),
                terminal_max_change=float(np.abs(fc2 - target['terminal']).max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=('reserve', 'replication'), default='reserve')
    p.add_argument('--crosslayer', action='store_true')
    args = p.parse_args()
    if args.crosslayer:
        if args.scope != 'replication':
            p.error('--crosslayer requires --scope replication')
        cfg = read(OUT / 'chain_crosslayer_frozen.json')['choice']['config']
    else:
        _, _, cfg = frozen_profile(args.scope)
    recurrent = Controls(tuple(read(CHAIN / 'selection.json')['suppression']['lam']))
    kernel = weights()
    parts = old.initialize()
    checks = {d: audit(d, parts, cfg, kernel, recurrent, args.scope, args.crosslayer) for d in DATASETS}
    print(checks, flush=True)
    name = 'online_audit.json' if args.scope == 'reserve' else 'online_replication_audit.json'
    if args.crosslayer:
        name = 'online_crosslayer_audit.json'
    save(DOC / 'chain' / name, checks)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
