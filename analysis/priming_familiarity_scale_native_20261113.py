"""Check learning-curve test results with each history's native target state."""

from __future__ import annotations

import argparse

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from hebb_chain_20261111 import Controls, Experiment, old
from priming_familiarity_20261112 import BATCH as FIT_BATCH, Familiarity
from priming_familiarity_report_20261112 import paired_auc
from priming_familiarity_scale_collect_20261113 import BATCH, SHARD, BatchExperiment, audit, pack, shard_path
from priming_familiarity_scale_data_20261113 import DATASETS, LEVELS, MANIFEST, ROOT, SEEDS, Renderer
from priming_familiarity_scale_fit_20261113 import ARMS, PROTOCOL, SELECTION, protocol, weights_for
from priming_familiarity_scale_report_20261113 import OUT as DOC, row_summary, write
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_search_20261112 import read, save


def native_batch(batch, x, first, last):
    exact, memory, f = batch.collect(x, first, last)
    targets, fc2 = [exact], [f]
    for history, order in ((1, [1, 0, 2]), (2, [2, 0, 1])):
        q, observed, values = batch.collect(x, first, last[:, order])
        np.testing.assert_array_equal(observed[:, 0], memory[:, history])
        targets.append(q)
        fc2.append(values)
    return exact, memory, f, np.stack(targets, 1), np.stack(fc2, 1)


def collect(plans, selection):
    parts = old.initialize()
    controls = Controls(tuple(frozen()['hebbian_lambdas']))
    for ds in DATASETS:
        run = Experiment(ds, parts)
        run.fit()
        rows = plans[ds]['splits']['test']
        render = Renderer(rows)
        sample = pack(render, rows[:2])
        serial = audit(run, controls, sample)
        original_methods = [getattr(run.model, f'pcoder{k}').forward for k in range(1, 6)]
        batch = BatchExperiment(run, controls)
        try:
            check = native_batch(batch, *sample)
            for a, b in zip(check[:3], serial):
                np.testing.assert_allclose(a, b, atol=.004, rtol=.003)
            for start in range(0, len(rows), SHARD):
                segment = rows[start:start + SHARD]
                path = shard_path(ds, 'test', start)
                if path.exists():
                    with np.load(path) as f:
                        np.testing.assert_array_equal(f['index'], [r['index'] for r in segment])
                        assert 'native_query' in f.files
                    continue
                items = [native_batch(batch, *pack(render, segment[j:j+BATCH])) for j in range(0, len(segment), BATCH)]
                q, p, f, native_q, native_f = [np.concatenate(a) for a in zip(*items)]
                tmp = path.with_suffix('.tmp')
                with tmp.open('wb') as file:
                    np.savez(file, query=q, memory=p, fc2=f, native_query=native_q, native_fc2=native_f,
                             index=[r['index'] for r in segment])
                tmp.replace(path)
                print('TEST CACHE', ds, start + len(segment), '/', len(rows), flush=True)
            save(ROOT / ds / 'native_test_audit.json',
                 dict(selection_sha256=digest(SELECTION), model_count=len(selection['model_sha256'][ds]),
                      shards={str(start): digest(shard_path(ds, 'test', start)) for start in range(0, len(rows), SHARD)}))
        finally:
            run.detach()
            for k, forward in enumerate(original_methods, 1):
                getattr(run.model, f'pcoder{k}').forward = forward


@torch.inference_mode()
def test(plans, selection):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    table, growth, stable = [], [], {}
    for ds in DATASETS:
        rows = plans[ds]['splits']['test']
        audit = read(ROOT / ds / 'native_test_audit.json')
        assert audit['selection_sha256'] == digest(SELECTION)
        q, p = [], []
        for start in range(0, len(rows), SHARD):
            path = shard_path(ds, 'test', start)
            assert digest(path) == audit['shards'][str(start)]
            with np.load(path) as f:
                q.append(f['native_query'])
                p.append(f['memory'])
        q, p = (torch.from_numpy(np.concatenate(values)).cuda() for values in (q, p))
        w = weights_for(rows)
        views = {}
        for arm in ARMS:
            for level in LEVELS:
                values = []
                for seed in SEEDS:
                    key = f'{arm}_{level}_{seed}'
                    path = ROOT / ds / 'models' / f'{key}.pth'
                    assert digest(path) == selection['model_sha256'][ds][key]
                    model = Familiarity().cuda().eval()
                    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
                    score = torch.cat([model(a, b) for a, b in zip(q.split(FIT_BATCH), p.split(FIT_BATCH))]).cpu().numpy()
                    values.append([paired_auc(score[:, 0], score[:, i], w) for i in (1, 2)])
                    np.savez(path.with_suffix('.native_test.npz'), scores=score)
                for i, name in enumerate(('same_concept', 'different')):
                    v = np.stack(values)[:, i]
                    views[arm, level, name] = v
                    table.append(dict(dataset=ds, arm=arm, level=level, negative=name, **row_summary(v)))
                if level > LEVELS[0]:
                    for negative in ('same_concept', 'different'):
                        effect = views[arm, level, negative] - views[arm, LEVELS[0], negative]
                        growth.append(dict(dataset=ds, arm=arm, level=level, reference_level=LEVELS[0],
                                           negative=negative, **row_summary(effect)))
        choice = selection['chosen'][ds]
        arm, level = choice['arm'], choice['level']
        score = views[arm, level, 'same_concept']
        delta = score - views[arm, LEVELS[0], 'same_concept']
        stable[ds] = dict(arm=arm, level=level, test_auc=row_summary(score), improvement_over_small=row_summary(delta),
                          evidence_of_growth=bool(level > LEVELS[0] and row_summary(score)['ci_low'] > .5
                                                  and row_summary(delta)['ci_low'] > 0))
    write('native_test', table)
    write('native_growth', growth)
    save(DOC / 'native_heldout_summary.json', stable)
    print('NATIVE TEST', stable, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('collect', 'score'))
    args = parser.parse_args()
    protocol()
    selection = read(SELECTION)
    assert selection['protocol_sha256'] == digest(PROTOCOL)
    (collect if args.phase == 'collect' else test)(read(MANIFEST), selection)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
