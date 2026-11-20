"""Cache exact-history t9 queries and paired preview endpoints at larger scale."""

from __future__ import annotations

import argparse
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from hebb_chain_20261111 import Controls, Experiment, PREVIEW, old
from priming_familiarity_data_20261112 import trial
from priming_familiarity_scale_data_20261113 import DATASETS, MANIFEST, ROOT, SPLITS, Renderer
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_same_concept_data_20261112 import frozen
from priming_sharpen_search_20261112 import read, save
from predify_experiments.conv_preview_memory_20261106 import DEFAULTS, OFFSETS
from predify_experiments.fc_sae_gated_20261106 import TrialMemory
from predify_experiments.hebb_chain import Layer
from predify_experiments.sequence import reset_hebbian_state, run_recurrent_sequence
from predify_experiments.tripartite_20261027 import restore_state, state_snapshot


BATCH = 6
SHARD = 24


def pcoder_batch(self, ff, fb, target, build_graph=False, ffm=None, fbm=None, erm=None):
    assert self.rep is not None and not build_graph
    if self.has_feedback:
        self.rep = ffm * ff + fbm * fb + (1 - ffm - fbm) * self.rep - erm * self.grd
    else:
        self.rep = ffm * ff + (1 - ffm) * self.rep - erm * self.grd
    with torch.enable_grad():
        self.rep.requires_grad_(True)
        self.prd = self.pmodule(self.rep)
        self.prediction_error = (self.prd - target).square().flatten(1).mean(1).sum()
        self.grd = torch.autograd.grad(self.prediction_error, self.rep)[0].detach()
    self.rep = self.rep.detach()
    self.prd = self.prd.detach()
    self.prediction_error = self.prediction_error.detach()
    return self.rep, self.prd


def coactivation(x):
    z = x.clamp_min(0)
    z = z / z.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    _, _, h, w = z.shape
    flat = z.flatten(2)
    intra = (flat @ flat.transpose(1, 2)) / (h * w)
    intra = (intra + intra.transpose(1, 2)) / 2
    intra.diagonal(dim1=1, dim2=2).zero_()
    inter = []
    for dy, dx in OFFSETS:
        dst = z[:, :, max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)].flatten(2)
        src = z[:, :, max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)].flatten(2)
        inter.append((dst @ src.transpose(1, 2) / dst.shape[2]) * torch.exp(z.new_tensor(-.5 * (dy * dy + dx * dx))))
    inter = torch.stack(inter, dim=-1)
    intra /= intra.sum(2).amax(1)[:, None, None].clamp_min(1e-8)
    inter /= inter.sum((2, 3)).amax(1)[:, None, None, None].clamp_min(1e-8)
    return intra, inter


class SpatialMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.writes = 0
        self.frozen = False
        self.intra = self.trace = self.inter = None

    def write(self, x):
        assert not self.frozen and self.writes < 2
        intra, inter = coactivation(x)
        decay, eta = DEFAULTS['decay'], DEFAULTS['eta']
        self.intra = eta * intra if self.writes == 0 else self.intra * decay + eta * intra
        self.trace = eta * inter if self.writes == 0 else self.trace * decay + eta * inter
        b, c, _, _ = self.trace.shape
        flat = self.trace.reshape(b, c, -1)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(2, flat.topk(DEFAULTS['inter_k'], dim=2).indices, True)
        mask = mask.reshape_as(self.trace) & (self.trace > 0)
        mask = mask & mask.transpose(1, 2).flip((-1,))
        sparse = self.trace * mask
        self.inter = sparse.new_zeros((b, c, c, 3, 3))
        for i, (y, x) in enumerate(OFFSETS):
            self.inter[..., y + 1, x + 1] = sparse[..., i]
        self.writes += 1

    def freeze(self):
        assert self.writes == 2 and not self.frozen
        self.frozen = True


class BatchedLayer(Layer):
    def __init__(self, source):
        super().__init__(source.index, source.cal, source.proxy)
        if self.spatial:
            self.memory = SpatialMemory()
        self.collect = True

    def prepare(self, batch):
        if self.spatial:
            self.memory.reset()
        else:
            self.memory = TrialMemory(batch, len(self.proxy.A), self.device)
        self.record = {}

    def read(self, r, gain=False):
        assert not gain
        if self.spatial:
            b, c, h, w = r.shape
            x = r.reshape(1, b * c, h, w)
            u = F.conv2d(x, self.memory.intra.reshape(b * c, c, 1, 1), groups=b)
            v = F.conv2d(x, self.memory.inter.reshape(b * c, c, 3, 3), padding=1, groups=b)
            z = r
            response = ((2 / 3) * u + (1 / 3) * v).reshape_as(r)
        else:
            z = self.proxy.encode(r)
            response = self.memory.read(z) if self.memory.phase == 'target' else (
                self.memory.W @ z.unsqueeze(-1)).squeeze(-1)
        return (z * response).sum(self.dim, keepdim=True) / (z.square().sum(self.dim, keepdim=True) + 1e-12)


class BatchExperiment:
    def __init__(self, run, controls):
        self.run = run
        self.controls = controls
        run.detach()
        run.layers = [BatchedLayer(layer) for layer in run.layers]
        run.p4.hebb, run.p5.hebb = run.layers[:2]
        for i, module in zip((2, 3), (run.pipe[1], run.pipe[4])):
            run.hooks.append(module.register_forward_hook(lambda _m, _inp, out, i=i: run.layers[i](out)))
        for k in range(1, 6):
            pc = getattr(run.model, f'pcoder{k}')
            pc.forward = MethodType(pcoder_batch, pc)

    def step(self, inputs, steps):
        return run_recurrent_sequence(self.run.model, inputs, steps, hebbian_layers=self.run.c.layers,
                                      commit_update=False, preserve_model_state=True)

    @torch.no_grad()
    def collect(self, target, first, last):
        run = self.run
        batch = len(target)
        run.c.reset()
        reset_hebbian_state(run.c.layers)
        for k in range(1, 6):
            pc = getattr(run.model, f'pcoder{k}')
            for name in ('rep', 'prd', 'grd'):
                a = getattr(pc, name)
                setattr(pc, name, a.expand(batch, *a.shape[1:]).contiguous())
        for layer in run.layers:
            layer.prepare(batch)
        run.controls(PREVIEW)
        self.step(first, 2)
        conv5 = run.layers[1]
        p1 = conv5.record['signed'].to(torch.float16)
        for layer in run.layers:
            layer.write(layer.record['signed'])
        prefix = state_snapshot(run.model)
        p2 = [None] * 3
        for history in (1, 2, 0):
            restore_state(run.model, prefix)
            self.step(last[:, history], 2)
            p2[history] = conv5.record['signed'].to(torch.float16)
        for layer in run.layers:
            layer.write(layer.record['signed'])
            layer.memory.freeze()
        run.controls(self.controls)
        self.step(target, 10)
        query = conv5.record['signed'].to(torch.float16)
        memory = torch.stack((p1[:, None].expand(-1, 3, -1, -1, -1), torch.stack(p2, 1)), 2)
        return query.cpu().numpy(), memory.cpu().numpy(), run.layers[3].record['signed'].cpu().numpy()


def shard_path(dataset, split, start):
    dest = ROOT / dataset / split / f'{start:04d}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def pack(render, rows):
    return tuple(torch.stack(group) for group in zip(*(render(row) for row in rows)))


def audit(run, controls, args):
    target, first, last = args
    examples = []
    for i in range(len(target)):
        old = [trial(run, target[i], (first[i], last[i, h]), controls, steps=10) for h in range(3)]
        examples.append((old[0][0][0], np.stack([r[1] for r in old]), old[0][2][0]))
    return [np.stack(v) for v in zip(*examples)]


def collect_dataset(dataset, plan, parts, controls, splits):
    run = Experiment(dataset, parts)
    run.fit()
    # Check independent batched trajectories before collecting the learning curve.
    sample_rows = plan['splits']['train'][:2]
    sample = pack(Renderer(plan['splits']['train']), sample_rows)
    reference = audit(run, controls, sample)
    original_methods = [getattr(run.model, f'pcoder{k}').forward for k in range(1, 6)]
    batch = BatchExperiment(run, controls)
    try:
        verified = batch.collect(*sample)
        errors = []
        for actual, expected in zip(verified, reference):
            np.testing.assert_allclose(actual, expected, atol=.004, rtol=.003)
            diff = np.abs(actual.astype(np.float32) - expected)
            errors.append(float(diff.max()))
        save(ROOT / dataset / 'batch_audit.json', dict(max_abs_errors=errors, targets=[r['index'] for r in sample_rows]))
        print('BATCH AUDIT', dataset, errors, flush=True)
        for split in splits:
            rows = plan['splits'][split]
            render = Renderer(rows)
            for start in range(0, len(rows), SHARD):
                segment = rows[start:start + SHARD]
                dest = shard_path(dataset, split, start)
                if dest.exists():
                    with np.load(dest) as z:
                        np.testing.assert_array_equal(z['index'], [r['index'] for r in segment])
                    continue
                items = [batch.collect(*pack(render, segment[j:j+BATCH])) for j in range(0, len(segment), BATCH)]
                q, p, f = [np.concatenate(v) for v in zip(*items)]
                tmp = dest.with_suffix('.tmp')
                with tmp.open('wb') as file:
                    np.savez(file, query=q, memory=p, fc2=f, index=[r['index'] for r in segment])
                tmp.replace(dest)
                print('CACHE', dataset, split, start + len(segment), '/', len(rows), flush=True)
    finally:
        run.detach()
        for k, forward in enumerate(original_methods, 1):
            getattr(run.model, f'pcoder{k}').forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fit', 'test'))
    args = parser.parse_args()
    if args.phase == 'test':
        assert (ROOT / 'selected.json').exists()
    plans = read(MANIFEST)
    cfg = frozen()
    parts = old.initialize()
    controls = Controls(tuple(cfg['hebbian_lambdas']))
    for ds in DATASETS:
        collect_dataset(ds, plans[ds], parts, controls, SPLITS[:2] if args.phase == 'fit' else ('test',))


if __name__ == '__main__':
    main()
