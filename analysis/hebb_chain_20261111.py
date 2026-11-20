"""Four-layer trial-local Hebbian experiment with training-only parameter selection."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from analysis import conv_fc_transfer_20261106 as old
from predify_experiments import conv_fc_transfer_20261106 as old_rule
from predify_experiments.fc_sae_gain_20261105 import FIT, GainProxy, Projection, model_hash
from predify_experiments import fc_sae_gated_20261106 as fc_rule
from predify_experiments.hebb_chain import Controls, LAYERS, Layer
from predify_experiments.sequence import reset_hebbian_state, run_recurrent_sequence
from predify_experiments.supplementary import load
from predify_experiments.tripartite_20261027 import restore_state, state_snapshot


OUT = ROOT / "Hebbian-CNN/Log/hebb_chain_20261111"
DOC = ROOT / "docs/hebb_chain_20261111"
PLAN = ROOT / "docs/pcoder4_pcoder5_fc1_fc2_hebbian_plan_20261111.md"
DATASETS = ("imagenet", "things")
HISTORIES = ("exact", "different_category")
WINDOWS = (slice(0, 5), slice(5, 10), slice(10, 15))
PREVIEW = Controls((.03, .015, 0., 0.))
REFERENCE = Controls((.03, .015, .03, .03))
GRID = {f"a{p}{f}": Controls((lam, lam / 2, fc, fc))
        for (p, lam), (f, fc) in itertools.product(enumerate((.015, .03, .045)), repeat=2)}
GAIN = {"gain4": (.005, 0, 0, 0), "gain5": (0, .005, 0, 0), "gainfc": (0, 0, .005, .005)}


def log(*parts):
    print(*parts, flush=True)


def save_json(path, value):
    old.writejson(path, value)


def save_arrays(path, **arrays):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez(f, **arrays)
    temp.replace(path)


def load_arrays(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), PLAN, ROOT / "predify_experiments/hebb_chain.py",
             ROOT / "predify_experiments/conv_preview_memory_20261106.py",
             ROOT / "predify_experiments/fc_sae_gain_20261105.py"]
    old.immutable(OUT / "protocol.json", dict(
        grid={k: list(v.lam) for k, v in GRID.items()}, gain={k: list(v) for k, v in GAIN.items()},
        preview_lambda=list(PREVIEW.lam), target_steps=15, side_steps=5, folds=dict(train=[1, 2, 3], dev=4, heldout=0),
        sources={str(f.relative_to(ROOT)): old.sha(f) for f in files},
    ))


class Experiment:
    def __init__(self, dataset, model_parts):
        self.dataset = dataset
        self.c = old.old.c
        self.model = self.c.model
        self.p4, self.p5, self.transform, self.filter = model_parts
        self.folder = OUT / dataset
        self.folder.mkdir(exist_ok=True)
        self.spec = old.read(old.DOC / f"{dataset}_manifest.json")
        manifest, train, dev = old.old.split(dataset)
        self.train = old.training_subset(train)
        self.dev = dev
        assert train == self.spec["training"] and dev == self.spec["development"]
        assert self.train == self.spec["calibration_training"]
        assert all(row["fold"] in (1, 2, 3) for row in self.train) and all(row["fold"] == 4 for row in dev)
        self.manifest = manifest
        source = old.OUT / dataset
        head_info = old.read(source / "fixed_heads.json")
        assert head_info["training"] == [r["dataset_index"] for r in train]
        assert old.sha(source / "fixed_heads.pckl") == head_info["sha256"]
        self.heads = load(source / "fixed_heads.pckl")
        self.pipe = self.model.backbone.classifier.pipe
        self.bridge = (self.pipe[3].weight, self.pipe[3].bias)
        self.inputs = {}
        self.layers = []
        self.hooks = []
        old.immutable(self.folder / "manifest.json", dict(dataset=dataset, train=self.train, development=dev,
                      source_manifest=old.sha(self.c.native.OUT / dataset / "manifest.json"),
                      fixed_heads=head_info["sha256"], protocol=old.sha(OUT / "protocol.json")))

    def attach(self, calibration, proxies=(None, None)):
        self.detach()
        self.layers = [Layer(i, calibration[i], proxies[i - 2] if i >= 2 else None) for i in range(4)]
        self.p4.hebb, self.p5.hebb = self.layers[:2]
        for i, module in zip((2, 3), (self.pipe[1], self.pipe[4])):
            self.hooks.append(module.register_forward_hook(lambda _m, _inp, out, i=i: self.layers[i](out)))

    def detach(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def stimulus(self, row, history):
        key = row["dataset_index"]
        if key not in self.inputs:
            target, previews = self.c.test_inputs(self.dataset, row, self.transform, self.filter, torch)
            self.inputs[key] = target[0].cpu(), {h: [p.cpu() for p in previews[h]] for h in HISTORIES}
        target, previews = self.inputs[key]
        return target.cuda(), [self.c.grey] * 2 if history == "none" else [p.cuda() for p in previews[history]]

    def controls(self, config):
        for layer, lam, epsilon in zip(self.layers, config.lam, config.epsilon, strict=True):
            layer.lam = lam
            layer.epsilon = epsilon

    def step(self, image, steps, callback=None):
        run_recurrent_sequence(self.model, image[None], steps, hebbian_layers=self.c.layers, commit_update=False,
                               preserve_model_state=True, after_step=callback)

    @torch.no_grad()
    def reset(self):
        self.c.reset()
        reset_hebbian_state(self.c.layers)
        for layer in self.layers:
            layer.reset()
            layer.collect = True
            layer.collect_scale = layer.local_only = layer.diagnose = False
            assert layer.memory is None or layer.memory.writes == 0

    @torch.no_grad()
    def prepare_trial(self, previews, preview="shared", fingerprints=False):
        self.reset()
        active = preview != "off"
        boundaries = []
        for j, image in enumerate(previews):
            cfg = PREVIEW if active else Controls()
            if preview == "fc_read":
                cfg = Controls((*PREVIEW.lam[:2], .03, .03))
            self.controls(cfg)
            self.step(image, 2)
            if active:
                for layer in self.layers:
                    layer.write(layer.record["signed"])
                assert all(layer.memory.writes == j + 1 for layer in self.layers)
            if fingerprints:
                boundaries.append([layer.fingerprint() for layer in self.layers])
        for layer in self.layers:
            layer.memory.freeze()
        assert all(layer.memory.writes == (2 if active else 0) for layer in self.layers)
        return boundaries

    def versions(self):
        return tuple(w._version for layer in self.layers for w in layer.weight_tensors())

    def shuffle(self, seed):
        g = torch.Generator().manual_seed(seed)
        for layer in self.layers:
            if layer.spatial:
                order = torch.randperm(512, generator=g).cuda()
                old_rule.shuffle_memory(layer.memory, order)
            else:
                order = torch.randperm(len(layer.proxy.A), generator=g).cuda()
                weight = layer.memory.W[0]
                weight.copy_(weight[order][:, order])

    @torch.no_grad()
    def target(self, image, config, *, audit=False, local=False, trace=False, calibrate=False):
        self.controls(config)
        for layer in self.layers:
            layer.local_only = local
            layer.diagnose = trace
            layer.collect_scale = calibrate and layer.lam > 0
        before = self.versions()
        values = {}
        stats = []

        def capture(t, _):
            assert before == self.versions()
            for layer, name in zip(self.layers, LAYERS):
                for field, label in (("output", name), ("input", name + "_input")):
                    values.setdefault(label, []).append(layer.record[field].flatten())
                if local:
                    values.setdefault(name + "_local", []).append(layer.record["local_gain"].flatten())
                if trace and layer.lam > 0:
                    values.setdefault(name + "_trace", []).append(layer.record["trace"][0])
            values.setdefault("conv5_signed", []).append(self.layers[1].record["signed"].flatten())
            pc = self.model.pcoder5
            stats.append(torch.stack((pc.prediction_error, pc.prd.square().mean().sqrt())))

        self.step(image, 15, capture)
        result = {k: torch.stack(v) for k, v in values.items()}
        result["pc5_diagnostics"] = torch.stack(stats)
        assert all(bool(torch.isfinite(x).all()) for x in result.values())
        assert all(bool((result[name].sum(-1) > 0).all()) for name in LAYERS)
        if audit:
            expected = F.linear(result["fc1"], *self.bridge).relu()
            torch.testing.assert_close(expected, result["fc2_input"], rtol=5e-5, atol=2e-5)
        return result

    @torch.no_grad()
    def replay_fc(self, fc1, config, *, calibrate=False, trace=False):
        incoming = fc1
        values = {}
        for i, name in enumerate(LAYERS[2:], 2):
            layer = self.layers[i]
            layer.collect_scale = calibrate
            layer.diagnose = trace
            values[name + "_input"] = incoming
            lam = config.lam[i]
            out = layer.update(incoming, lam, config.epsilon[i]) if lam else incoming
            values[name] = out
            if trace and lam:
                values[name + "_trace"] = layer.record["trace"]
            if i == 2:
                incoming = F.linear(out, *self.bridge).relu()
        return values

    @torch.no_grad()
    def fit_samples(self):
        self.attach([{"scale": 1.}, {"scale": 1.}, {}, {}])
        samples = {name: [] for name in LAYERS}
        for row in self.train:
            for history in HISTORIES:
                target, previews = self.stimulus(row, history)
                self.reset()
                self.controls(Controls())
                bank = [[] for _ in LAYERS]

                def capture():
                    for layer, frames in zip(self.layers, bank):
                        frames.append(layer.record["output"][0].cpu())

                for preview in previews:
                    self.step(preview, 2)
                    capture()
                self.step(target, 10, lambda t, _: capture() if t in (0, 9) else None)
                for name, frames in zip(LAYERS, bank):
                    samples[name].append(torch.stack(frames))
        return samples

    def fit(self):
        done = self.folder / "initial_calibration.json"
        if done.exists():
            cal = old.read(done)
            proxies = [load_proxy(self.folder / f"{name}.pth", cal[i]) for i, name in enumerate(LAYERS[2:], 2)]
            self.attach(cal, proxies)
            return cal
        samples = self.fit_samples()
        cal = []
        proxies = []
        for i, name in enumerate(LAYERS):
            x = torch.stack(samples.pop(name))
            spec = {}
            proxy = None
            if i < 2:
                spec["scale"] = float(x[:, 2:].square().mean().sqrt())
            else:
                seed = 20261111 + 100 * (self.dataset == "things") + i
                proxy = fit_proxy(x, seed)
                spec.update(seed=seed, proxy_sha256=model_hash(proxy))
                torch.save({k: t.cpu() for k, t in proxy.state_dict().items()}, self.folder / f"{name}.pth")
                proxies.append(proxy)
            layer = Layer(i, spec, proxy)
            total = count = 0
            with torch.no_grad():
                for trial in x:
                    layer.reset()
                    for frame in trial[:2]:
                        layer.write(frame[None].cuda())
                    layer.memory.freeze()
                    for frame in trial[2:]:
                        k = layer.read(frame[None].cuda() / spec.get("scale", 1.))
                        positives = k[k > 0]
                        total += float(positives.double().sum())
                        count += positives.numel()
            assert count and total > 0
            spec.update(kappa0=total / count, positive_kappa_count=count)
            cal.append(spec)
            log("CAL", self.dataset, name, spec)
        save_json(done, cal)
        self.attach(cal, proxies)
        return cal

    @torch.no_grad()
    def audit(self):
        path = self.folder / "audit.json"
        if path.exists():
            return
        target, previews = self.stimulus(self.train[0], "exact")
        layers = self.layers
        legacy5 = self.c.layers[2]
        reference = old_rule.TransferOperator(old_rule.new_memory(512, "cuda"), old_rule.SpatialCodec(),
                    old.read(old.DOC / f"{self.dataset}_calibration.json")["direct_rep4"],
                    self.spec["scale"], "rep4", "pool")
        self.p4.hebb, self.p5.hebb = reference, legacy5
        prior, _ = old.run_trial(target, previews, reference)
        self.p4.hebb, self.p5.hebb = layers[:2]
        saved_cal = layers[0].cal
        layers[0].cal = dict(reference.calibration, scale=self.spec["scale"])
        self.reset()
        self.controls(Controls((.03, 0., 0., 0.)))
        for preview in previews:
            self.step(preview, 2)
            layers[0].write(layers[0].record["output"])
        for layer in layers:
            layer.memory.freeze()
        new = self.target(target, Controls((.03, 0., 0., 0.)))
        legacy_error = {k: float(np.max(np.abs(prior[k] - new[k].cpu().numpy()))) for k in LAYERS}
        for k in LAYERS:
            np.testing.assert_allclose(prior[k], new[k].cpu().numpy(), rtol=1e-5, atol=1e-5)
        layers[0].cal = saved_cal

        boundaries = self.prepare_trial(previews, fingerprints=True)
        frozen = [layer.fingerprint() for layer in layers]
        snapshot = state_snapshot(self.model)
        configs = {"off": Controls(), "p4": Controls((.03, 0., 0., 0.)), "p5": Controls((0., .015, 0., 0.)),
                   "fc": Controls((0., 0., .03, .03)), "all": REFERENCE}
        results = {}
        errors = {}
        for name, cfg in configs.items():
            restore_state(self.model, snapshot)
            results[name] = self.target(target, cfg, audit=True)
        for name in ("conv4", "conv5", "conv5_signed", "pc5_diagnostics"):
            assert torch.equal(results["off"][name], results["fc"][name])
        for i, name in enumerate(LAYERS[2:], 2):
            a = results["all"][name + "_input"]
            layer = layers[i]
            reference_cal = dict(layer.cal, q_rms=1.)  # Epsilon is zero in this independent formula check.
            b = torch.cat([fc_rule.side(x[None], layer.proxy, layer.memory, reference_cal, 0, .03)[0] for x in a])
            torch.testing.assert_close(b, results["all"][name], rtol=2e-5, atol=1e-6)
            errors[name] = float((b - results["all"][name]).abs().max())
        assert not torch.equal(results["off"]["conv5"], results["p5"]["conv5"])
        assert [layer.fingerprint() for layer in layers] == frozen
        for layer in layers:
            try:
                layer.write(layer.record["signed"])
            except RuntimeError:
                pass
            else:
                raise AssertionError("A frozen memory accepted a write")
        self.prepare_trial(previews, preview="off")
        zero_snapshot = state_snapshot(self.model)
        zero = self.target(target, Controls())
        restore_state(self.model, zero_snapshot)
        bypass = self.target(target, REFERENCE)
        assert all(torch.equal(zero[k], bypass[k]) for k in LAYERS)
        assert self.prepare_trial(previews, fingerprints=True) == boundaries
        replay = self.target(target, REFERENCE)
        assert all(torch.equal(results["all"][k], replay[k]) for k in LAYERS)
        save_json(path, dict(passed=True, legacy_max_abs=legacy_error, fc_formula_max_abs=errors,
                             preview_weight_hashes=boundaries, target_weight_hashes=frozen,
                             zero_identity=True, replay_exact=True, fc_to_pc_identity=True,
                             pcoder5_diagnostics=replay["pc5_diagnostics"].cpu().tolist()))
        log("AUDIT PASSED", self.dataset, legacy_error, errors)


def fit_proxy(samples, seed):
    projection = Projection(seed=seed + 1000).cuda()
    with torch.no_grad():
        projected = projection(samples.flatten(0, 1).cuda())
        scale = float(projected.square().mean().sqrt())
        center = (projected.mean(0) / scale).cpu()
    model = GainProxy(projection, center, scale, seed=seed + 2000).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=FIT["learning_rate"])
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FIT["steps"], eta_min=FIT["minimum_learning_rate"])
    for _ in range(FIT["steps"]):
        opt.zero_grad(set_to_none=True)
        z = model.encode_projected(projected)
        reconstruction = model.decode_projected(z)
        mse = .5 * ((reconstruction - projected) / model.scale).square().sum(-1)
        loss = (mse + FIT["l1"] * z.sum(-1)).mean()
        loss.backward()
        model.project_gradient()
        opt.step()
        model.constrain()
        decay.step()
    assert torch.isfinite(loss) and sum(p.numel() for p in model.parameters()) == 2112
    return model.eval().requires_grad_(False)


def load_proxy(path, spec):
    seed = spec["seed"]
    model = GainProxy(Projection(seed=seed + 1000), torch.zeros(32), 1., seed=seed + 2000)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert model_hash(model) == spec["proxy_sha256"]
    return model.cuda().eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, nargs="+")
    args = parser.parse_args()
    protocol()
    model_parts = old.initialize()
    for dataset in args.dataset or DATASETS:
        run = Experiment(dataset, model_parts)
        run.fit()
        run.audit()
        run.detach()


if __name__ == "__main__":
    main()
