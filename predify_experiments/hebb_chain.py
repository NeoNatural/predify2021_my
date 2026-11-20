"""Trial-local Hebbian operators for the two PCoders and two physical FC layers."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .conv_preview_memory_20261106 import PreviewMemory, inhibition
from .fc_sae_gated_20261106 import TrialMemory


LAYERS = ("conv4", "conv5", "fc1", "fc2")
STEPS = 5


@dataclass(frozen=True)
class Controls:
    lam: tuple[float, ...] = (0.,) * 4
    epsilon: tuple[float, ...] = (0.,) * 4


def centered_gain(p, q, c, dim, epsilon):
    direction = torch.tanh(q / c)
    total = p.sum(dim, keepdim=True)
    center = (p * direction).sum(dim, keepdim=True) / torch.where(total > 0, total, 1)
    return p * (1 + epsilon * (direction - center))


class Layer(nn.Module):
    def __init__(self, index, calibration, proxy=None, device="cuda"):
        super().__init__()
        self.index, self.cal, self.proxy = index, calibration, proxy
        self.spatial = index < 2
        self.dim = 1 if self.spatial else -1
        self.memory = PreviewMemory(512, device) if self.spatial else None
        self.device = device
        self.lam = self.epsilon = 0.
        self.collect = False
        self.diagnose = False
        self.collect_scale = False
        self.local_only = False
        self.rms = [0., 0]
        self.record = {}
        self.reset()

    def reset(self):
        if self.spatial:
            self.memory.reset()
        elif self.proxy is not None:
            self.memory = TrialMemory(1, len(self.proxy.A), self.device)
        self.record = {}

    @torch.no_grad()
    def write(self, x):
        if self.spatial:
            self.memory.write(x.clamp_min(0) / self.cal["scale"])
            return
        if self.memory.writes >= 2:
            raise RuntimeError("Only two preview endpoint writes are allowed")
        self.memory.write(self.proxy.encode(x))

    def weight_tensors(self):
        m = self.memory
        return (m.intra, m.trace_inter, m.inter) if self.spatial else (m.W,)

    def fingerprint(self):
        return self.memory.fingerprint() if self.spatial else self.memory.fingerprints()[0]

    def read(self, r, gain=False):
        if self.spatial:
            z = r
            h = (2 / 3) * F.conv2d(z, self.memory.intra[:, :, None, None])
            h = h + (1 / 3) * F.conv2d(z, self.memory.inter, padding=1)
        else:
            z = self.proxy.encode(r)
            # Leading rows are timesteps of this one trial and share its W.
            h = F.linear(z, self.memory.W[0])
        kappa = (z * h).sum(self.dim, keepdim=True) / (z.square().sum(self.dim, keepdim=True) + 1e-12)
        if not gain:
            return kappa
        gated = h * (z > 0)
        return gated if self.spatial else self.proxy.gain_signal(gated)

    def population(self, r):
        if self.spatial:
            mu = inhibition(r)
            return mu, (inhibition(r.square()) - mu.square()).clamp_min(0).sqrt()
        return r.mean(-1, keepdim=True), r.std(dim=-1, keepdim=True, correction=0)

    @torch.no_grad()
    def update(self, x, lam, epsilon=0.):
        scale = self.cal.get("scale", 1.)
        drive = x / scale if self.spatial else x
        r = drive
        trace = []
        for _ in range(STEPS):
            kappa = self.read(r)
            mu, sigma = self.population(r)
            threshold = lam * (mu + sigma) * kappa / (kappa + self.cal["kappa0"])
            r = (drive - threshold).clamp_min(0)
            if self.diagnose:
                trace.append(torch.stack([v.flatten(1).mean(1) for v in (kappa, mu + sigma, threshold, r)], -1))
        if self.diagnose:
            self.record["trace"] = torch.stack(trace, 1)
        p = r * scale if self.spatial else r
        if epsilon or self.collect_scale:
            q = self.read(r, gain=True)
        if self.collect_scale:
            support = p > 0
            self.rms[0] += float(q[support].double().square().sum())
            self.rms[1] += int(support.sum())
        if epsilon:
            out = centered_gain(p, q, self.cal["c"], self.dim, epsilon)
            a = p.double().sum(self.dim)
            b = out.double().sum(self.dim)
            relative = float(((a - b).abs() / torch.where(a > 0, a, 1)).max())
            if relative > 8 * torch.finfo(torch.float32).eps or not torch.equal(p > 0, out > 0):
                raise AssertionError("Gain changed the local activity budget or support")
            self.record["local_pool"] = p
            self.record["conservation"] = relative
            return out
        return p

    def forward(self, raw):
        # The PCoder's predictor differentiates its error w.r.t. the returned rep.
        # Build a new leaf only where the experiment changes that representation.
        with torch.no_grad():
            positive = raw.clamp_min(0) if self.spatial else raw
            active = self.lam > 0 and self.memory is not None and self.memory.writes > 0
            out = self.update(positive, self.lam, self.epsilon) if active else positive
            if active and self.epsilon and self.local_only:
                self.record["local_gain"] = out
                out = self.record["local_pool"]
            rep = out
            if self.index == 1:
                rep = out + raw.clamp_max(0) if active else raw
            if self.collect:
                self.record.update(input=positive.detach(), output=out.detach(), signed=rep.detach())
        if not self.spatial or rep is raw:
            return rep
        return rep.requires_grad_(True)
