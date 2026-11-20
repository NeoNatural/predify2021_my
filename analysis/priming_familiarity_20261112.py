"""Fit and freeze a full-frame Conv5 familiarity circuit on training folds only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from priming_familiarity_data_20261112 import OUT, paths
from priming_sharpen_collect_20261112 import digest
from priming_sharpen_object_precision_20261112 import fitted
from priming_sharpen_same_concept_data_20261112 import PLAN
from priming_sharpen_search_20261112 import DATASETS, control_gate, read, save
from priming_sharpen_teacher_20261112 import weights


WIDTH = 64
ACTIVE = 12
DECAY = .5
SHUFFLES = tuple(range(2026111231, 2026111239))
FIT_SEEDS = (2026111221, 2026111222)
EPOCHS = 48
PATIENCE = 9
BATCH = 8
LR = .002


class Familiarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, WIDTH)
        self.excitation = nn.Parameter(torch.eye(WIDTH) * 4 - 4)
        self.bias = nn.Parameter(torch.full((WIDTH,), -4.))
        self.population = nn.Parameter(torch.zeros(WIDTH))
        self.inhibition = nn.Parameter(torch.tensor(16.))
        self.center = nn.Parameter(torch.tensor(-1.))

    def encode(self, image):
        x = image.float().flatten(-2).transpose(-1, -2)
        x = F.normalize(x, dim=-1) * 512**.5
        z = self.encoder(x).relu()
        indices = z.topk(ACTIVE, dim=-1).indices
        selected = torch.zeros_like(z, dtype=torch.bool).scatter_(-1, indices, True)
        return F.normalize(z * selected, dim=-1)

    def reference(self, z):
        weights = F.normalize(F.softplus(self.excitation), p=1, dim=1)
        return (F.linear(z, weights) - F.softplus(self.bias)).relu()

    def coactivation(self, frame):
        z = self.encode(frame)
        return self.reference(z).transpose(-1, -2) @ z / z.shape[-2]

    def write(self, memory, frame):
        return DECAY * memory + self.coactivation(frame)

    def memory(self, preview):
        before, after = preview.unbind(-4)
        a = self.coactivation(before)
        return self.write(a, after)

    def read_encoded(self, z, h0, memory):
        inhibition = F.softplus(self.inhibition) * (z @ memory.transpose(-1, -2))
        residual = (h0 - inhibition).relu()
        weights = F.softplus(self.population)
        weights = weights / weights.mean()
        lost = ((h0 - residual) * weights).sum(-1).mean(-1)
        return lost / (.01 + h0.sum(-1).mean(-1))

    def forward(self, target, preview):
        z = self.encode(target)
        return self.read_encoded(z, self.reference(z), self.memory(preview))


class PreviewMemory:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.value = self.model.encoder.weight.new_zeros(WIDTH, WIDTH)
        self.writes = 0
        self.frozen = False

    def write(self, frame):
        if self.frozen or self.writes == 2:
            raise RuntimeError('Only two preview writes are permitted')
        self.value = self.model.write(self.value, frame)
        self.writes += 1

    def freeze(self):
        assert self.writes in (0, 2)
        self.frozen = True

    def read(self, target):
        assert self.frozen
        z = self.model.encode(target)
        return self.model.read_encoded(z, self.model.reference(z), self.value)


def shuffled(memory, seed):
    rng = np.random.default_rng(seed)
    rows = rng.permutation(WIDTH)
    cols = rng.permutation(WIDTH)
    return memory[..., rows, :][..., :, cols]


def loss(model, values):
    relative = F.softplus(24 * (values[:, 1:] - values[:, :1])).mean()
    logits = 16 * (values - model.center.sigmoid())
    binary = (F.softplus(-logits[:, 0]).mean() + F.softplus(logits[:, 1:]).mean()) / 2
    return relative + .5 * binary


@torch.no_grad()
def scores(model, data, memory, select):
    return torch.cat([model(data[idx], memory[idx]) for idx in select.split(BATCH)])


def fit(data, memory, folds, seed):
    torch.manual_seed(seed)
    model = Familiarity().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.001)
    rng = torch.Generator().manual_seed(seed)
    train = torch.tensor(np.flatnonzero(np.isin(folds, (1, 2))), device='cuda')
    dev = torch.tensor(np.flatnonzero(folds == 3), device='cuda')
    best, best_epoch = float('inf'), 0
    for epoch in range(1, EPOCHS + 1):
        shuffle = train[torch.randperm(len(train), generator=rng).cuda()]
        for indices in shuffle.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            objective = loss(model, model(data[indices], memory[indices]))
            objective.backward()
            optimizer.step()
        val = float(loss(model, scores(model, data, memory, dev)).detach())
        if val < best - 1e-5:
            best, best_epoch = val, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(state)
    val = scores(model, data, memory, dev).cpu().numpy()
    return model, val, dict(seed=seed, epoch=best_epoch, stopped=epoch, validation_loss=best,
                           auc_same=auc(val, 1), auc_different=auc(val, 2))


def auc(values, negative):
    return float(roc_auc_score(np.r_[np.ones(len(values)), np.zeros(len(values))],
                               np.r_[values[:, 0], values[:, negative]]))


def gate(values, threshold, slope):
    return np.clip(slope * (values - threshold), 0, 1)


def intensity(values):
    weights = np.array([.5, .25, .25])
    return np.array([values.mean(0) @ weights, (values[:, 1:] > 0).mean(),
                     (values >= 1).mean(0) @ weights])


def calibrate(values, old_gate):
    """Match mean strength, pooled negative trigger rate and saturation on fold 3."""
    target = intensity(old_gate)
    candidates = []
    thresholds = np.unique(np.r_[0., np.quantile(values[:, 1:], np.linspace(0, 1, 41))])
    for threshold in thresholds:
        for slope in np.geomspace(1, 512, 81):
            calibrated = gate(values, threshold, slope)
            stats = intensity(calibrated)
            error = np.abs(stats - target) @ np.array([2., .75, .35])
            candidates.append((float(error), float(threshold), float(slope)))
    error, threshold, slope = min(candidates)
    calibrated = gate(values, threshold, slope)
    assert threshold >= 0 and gate(0, threshold, slope) == 0
    return dict(threshold=threshold, slope=slope, error=error,
                new_intensity=intensity(calibrated).tolist(), old_intensity=target.tolist(),
                new_history_mean=calibrated.mean(0).tolist(), old_history_mean=old_gate.mean(0).tolist())


def fit_dataset(dataset, kernel):
    path, meta = paths(dataset, 'train')
    assert digest(path) == read(meta)['feature_sha256']
    with np.load(path) as archive:
        data = torch.from_numpy(archive['conv']).cuda()
        memory = torch.from_numpy(archive['memory']).cuda()
        folds = archive['folds']
        f, p = archive['fc2'], archive['old_preview']
    assert set(folds) == {1, 2, 3}
    bank, _, cuts = fitted(dataset, kernel)
    f, p = f[folds == 3], p[folds == 3]
    old = np.column_stack([control_gate(bank.query(f[:, j], p[:, j])['match'], cuts[.5]) for j in range(3)])
    results = []
    best_loss = float('inf')
    for seed in FIT_SEEDS:
        model, val, stats = fit(data, memory, folds, seed)
        results.append(stats)
        print('FIT', dataset, stats, flush=True)
        if stats['validation_loss'] < best_loss:
            best_loss = stats['validation_loss']
            selected, values, best = model, val, stats
    calibration = calibrate(values, old)
    model_path = OUT / dataset / 'circuit.pth'
    tmp = model_path.with_suffix('.tmp')
    torch.save({k: v.detach().cpu() for k, v in selected.state_dict().items()}, tmp)
    tmp.replace(model_path)
    print('CALIBRATION', dataset, calibration, flush=True)
    return dict(train_sha256=digest(path), train_audit_sha256=digest(meta),
                model_sha256=digest(model_path), fresh_stimuli_sha256=digest(OUT / dataset / 'fresh_stimuli.json'),
                chosen=best, starts=results, calibration=calibration,
                folds={str(j): int(np.sum(folds == j)) for j in (1, 2, 3)})


def freeze():
    dest = OUT / 'frozen.json'
    if dest.exists():
        checked()
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    kernel = weights()
    datasets = {d: fit_dataset(d, kernel) for d in DATASETS}
    sources = ('priming_familiarity_20261112.py', 'priming_familiarity_data_20261112.py',
               'priming_familiarity_verify_20261112.py', 'priming_familiarity_report_20261112.py')
    result = dict(datasets=datasets, width=WIDTH, max_active=ACTIVE, memory_decay=DECAY,
                  shuffle_seeds=SHUFFLES, fit_seeds=FIT_SEEDS, epochs=EPOCHS, patience=PATIENCE, lr=LR, batch=BATCH,
                  plan_sha256=digest(PLAN), source_sha256={p: digest(Path(__file__).with_name(p)) for p in sources})
    save(dest, result)
    print('FROZEN', digest(dest), flush=True)


def checked():
    result = read(OUT / 'frozen.json')
    assert result['plan_sha256'] == digest(PLAN)
    assert tuple(result['shuffle_seeds']) == SHUFFLES
    for name, sha in result['source_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == sha
    for ds, spec in result['datasets'].items():
        assert digest(OUT / ds / 'circuit.pth') == spec['model_sha256']
        assert digest(OUT / ds / 'fresh_stimuli.json') == spec['fresh_stimuli_sha256']
    return result


def load_circuit(dataset):
    model = Familiarity().cuda().eval()
    model.load_state_dict(torch.load(OUT / dataset / 'circuit.pth', weights_only=True, map_location='cpu'))
    return model.requires_grad_(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with threadpool_limits(limits=4):
        freeze()
