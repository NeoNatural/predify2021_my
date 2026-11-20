"""Trial-local convolutional Hebbian traces. No training-set connection matrices."""
from __future__ import annotations

import hashlib
import torch
import torch.nn.functional as F

OFFSETS = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0))
DEFAULTS = dict(decay=.5, eta=.5, gain=.25, rho=1/3, lam=.5,
                alpha=.5, steps=3, a=.25, b=.1, inter_k=32)


def digest(x):
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()


def shift(x, dy, dx):
    h, w = x.shape[-2:]
    return F.pad(x, (1, 1, 1, 1))[..., 1+dy:1+dy+h, 1+dx:1+dx+w]


def inhibition(x):
    col = x.mean(1, keepdim=True)
    neighbors = sum(shift(col, *o) for o in OFFSETS)
    counts = sum(shift(torch.ones_like(col), *o) for o in OFFSETS)
    return .5*col + .5*neighbors/counts.clamp_min(1)


def coactivation(activity):
    """Shared channel/offset outer products, averaged over valid spatial pairs.

    Each preview endpoint is normalized by its own nonnegative RMS. Separate
    intra/inter global row-sum normalization bounds each Hebbian increment.
    No across-image centering, target activity, or labels enter this function.
    """
    if activity.ndim != 4 or activity.shape[0] != 1:
        raise ValueError('One trial per memory is required')
    if not torch.isfinite(activity).all():
        raise ValueError('Non-finite preview')
    z = activity.detach().clamp_min(0)
    rms = z.square().mean().sqrt().clamp_min(1e-8)
    z = z/rms
    c, h, w = z.shape[1:]
    if min(h, w) < 2:
        raise ValueError('At least two spatial positions per axis are required')
    flat = z[0].reshape(c, -1)
    intra = flat @ flat.T / (h*w)
    intra = (intra+intra.T)/2
    intra.fill_diagonal_(0)
    inter = torch.zeros(c, c, 3, 3, device=z.device, dtype=z.dtype)
    for dy, dx in OFFSETS:
        # Destination at p, source at p+offset, consistent with conv2d.
        dst = z[:, :, max(0, -dy):min(h, h-dy), max(0, -dx):min(w, w-dx)]
        src = z[:, :, max(0, dy):min(h, h+dy), max(0, dx):min(w, w+dx)]
        dst, src = dst[0].reshape(c, -1), src[0].reshape(c, -1)
        inter[:, :, dy+1, dx+1] = (dst @ src.T / dst.shape[1]) * torch.exp(
            z.new_tensor(-.5*(dy*dy+dx*dx)))
    # The same scalar for every edge preserves reciprocal symmetry.
    intra /= intra.sum(1).max().clamp_min(1e-8)
    inter /= inter.sum((1, 2, 3)).max().clamp_min(1e-8)
    return intra, inter, float(rms)


def mutual_sparse(inter, k):
    c = inter.shape[0]
    scores = torch.stack([inter[:, :, y+1, x+1] for y, x in OFFSETS], -1)
    flat = scores.reshape(c, -1)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, flat.topk(min(k, flat.shape[1]), dim=1).indices, True)
    mask = mask.reshape_as(scores) & (scores > 0)
    result = torch.zeros_like(inter)
    for j, (dy, dx) in enumerate(OFFSETS):
        rev = OFFSETS.index((-dy, -dx))
        result[:, :, dy+1, dx+1] = scores[:, :, j]*mask[:, :, j]*mask[:, :, rev].T
    return result


class PreviewMemory:
    def __init__(self, channels=512, device='cpu', config=None):
        self.config = dict(DEFAULTS if config is None else config)
        self.intra = torch.zeros(channels, channels, device=device)
        self.trace_inter = torch.zeros(channels, channels, 3, 3, device=device)
        self.inter = torch.zeros_like(self.trace_inter)
        self.reset()

    def reset(self):
        self.intra.zero_(); self.trace_inter.zero_(); self.inter.zero_()
        self.writes = 0
        self.phase = 'preview'

    def write(self, activity):
        if self.phase != 'preview' or self.writes >= 2:
            raise RuntimeError('Only two preview endpoint writes are allowed')
        h0, hl, rms = coactivation(activity)
        q = self.config
        self.intra.mul_(q['decay']).add_(h0, alpha=q['eta'])
        self.trace_inter.mul_(q['decay']).add_(hl, alpha=q['eta'])
        # Sparsify the accumulated trace, not each segment independently.
        self.inter.copy_(mutual_sparse(self.trace_inter, q['inter_k']))
        self.writes += 1
        return dict(source_rms=rms, **self.stats())

    def freeze(self):
        if self.phase != 'preview':
            raise RuntimeError('Already frozen')
        self.phase = 'target'

    def fingerprint(self):
        return hashlib.sha256(''.join(digest(x) for x in
            (self.intra, self.trace_inter, self.inter)).encode()).hexdigest()

    def stats(self):
        return dict(writes=self.writes, weight_sha256=self.fingerprint(),
                    intra_edges=int((self.intra>0).sum()),
                    inter_edges=int((self.inter>0).sum()),
                    inter_degree_max=int((self.inter>0).sum((1, 2, 3)).max()),
                    intra_max_rowsum=float(self.intra.sum(1).max()),
                    inter_max_rowsum=float(self.inter.sum((1, 2, 3)).max()))


class PreviewOperator(torch.nn.Module):
    def __init__(self, memory, scale, family, source, config=None):
        super().__init__()
        if family not in ('clip', 'gate', 'multiply') or source not in ('none', 'rep4', 'prd5'):
            raise ValueError((family, source))
        self.memory, self.scale, self.family, self.source = memory, float(scale), family, source
        self.config = dict(DEFAULTS if config is None else config)
        self.calls = 0

    def forward(self, a):
        self.calls += 1
        with torch.no_grad():
            x = a.detach().clamp_min(0)/self.scale
            r = x
            if self.family != 'clip':
                q, m = self.config, self.memory
                for _ in range(q['steps']):
                    if self.source == 'none' or m.writes == 0:
                        u = torch.zeros_like(r)
                    else:
                        u = (1-q['rho'])*F.conv2d(r, m.intra[:, :, None, None])
                        u = u + q['rho']*F.conv2d(r, m.inter, padding=1)
                    d = x*(1+q['gain']*u/(1+u)) if self.family == 'multiply' else x+q['gain']*u
                    out = (1-q['alpha'])*r + q['alpha']*d/(1+q['lam']*inhibition(r))
                    if self.family == 'gate':
                        mu = inhibition(d)
                        sd = (inhibition(d.square())-mu.square()).clamp_min(0).sqrt()
                        out = out*(d > q['a']*mu+q['b']*sd)
                    r = out
            return (r*self.scale).requires_grad_(True)
