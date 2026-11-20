"""Frozen two-point experiment on the post-suppression Hebbian gain."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import fc_sae_gated_search_20261106 as fc
from analysis import conv_fc_transfer_20261106 as conv
from analysis import fc_sae_gain_pilot_20261105 as sae
from predify_experiments import conv_fc_transfer_20261106 as conv_rule
from predify_experiments import fc_sae_gated_20261106 as fc_rule
from predify_experiments.supplementary import load
import numpy as np
import torch
import torch.nn.functional as F


OUT = ROOT / "Hebbian-CNN/Log/hebb_post_pool_20261111"
DOC = ROOT / "docs/hebb_post_pool_20261111"
DATASETS = ("imagenet", "things")
HISTORIES = ("exact", "different")
EPSILON = 0.005
FLOAT_TOL = 8 * torch.finfo(torch.float32).eps
DOUBLE_TOL = 16 * torch.finfo(torch.float64).eps
RS_TOL = 400 * FLOAT_TOL
sha = fc.exp.sha
read = fc.exp.read


def freeze_file(path, value):
    if path.exists():
        assert read(path) == value, f"Frozen file changed: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fc.exp.write_json(path, value)


def post_gain(p, q, c, dim, epsilon):
    if epsilon == 0:
        return p
    u = torch.tanh(q / c)
    total = p.sum(dim=dim, keepdim=True)
    center = (p * u).sum(dim=dim, keepdim=True) / torch.where(total > 0, total, 1)
    return p * (1 + epsilon * (u - center))


class LocalAudit:
    def __init__(self):
        self.checks = {}
        self.activities = {}

    def check(self, p, q, c, dim, history, layer):
        y = post_gain(p, q, c, dim, EPSILON)
        assert torch.equal(post_gain(p, q, c, dim, 0), p)
        assert (p >= 0).all() and torch.isfinite(q).all() and torch.isfinite(y).all()
        support = p > 0
        assert torch.equal(y > 0, support) and (y >= 0).all()
        ratios = (y[support].double() / p[support].double())
        lo, hi = (float(ratios.min()), float(ratios.max())) if ratios.numel() else (1., 1.)
        assert lo >= .99 - FLOAT_TOL and hi <= 1.01 + FLOAT_TOL

        # Independent FP64 evaluation using normalized activity as weights.
        x = p.double()
        total = x.sum(dim=dim, keepdim=True)
        weights = x / torch.where(total > 0, total, 1)
        u = torch.tanh(q.double() / c)
        reference = x + EPSILON * x * (u - (weights * u).sum(dim=dim, keepdim=True))
        error = (y.double() - reference).abs()
        relative = lambda diff, base: float((diff / torch.where(base > 0, base, 1)).max())
        element_rel = relative(error, x)
        exact_diff = (reference.sum(dim=dim, keepdim=True) - total).abs()
        group_diff = (y.double().sum(dim=dim, keepdim=True) - total).abs()
        image_p = x.flatten(1).sum(1)
        image_y = y.double().flatten(1).sum(1)
        image_diff = (image_y - image_p).abs()
        assert relative(exact_diff, total) <= DOUBLE_TOL
        assert max(element_rel, relative(group_diff, total), relative(image_diff, image_p)) <= FLOAT_TOL

        key = f"{history}/{layer}"
        row = self.checks.setdefault(key, {})
        for name, value in dict(
            max_fp64_sum_abs=float(exact_diff.max()), max_fp64_sum_rel=relative(exact_diff, total),
            max_element_abs_vs_fp64=float(error.max()), max_element_rel_vs_fp64=element_rel,
            max_group_sum_abs=float(group_diff.max()), max_group_sum_rel=relative(group_diff, total),
            max_image_sum_abs=float(image_diff.max()), max_image_sum_rel=relative(image_diff, image_p),
            gain_max=hi,
        ).items():
            row[name] = max(row.get(name, 0), value)
        row["gain_min"] = min(row.get("gain_min", 1), lo)
        row["groups"] = row.get("groups", 0) + total.numel()
        row["zero_groups"] = row.get("zero_groups", 0) + int((total == 0).sum())
        self.activities.setdefault((history, layer), []).append(torch.stack([image_p, image_y], 1).cpu().numpy())
        return y

    def finish(self):
        repetition = {}
        for layer in sorted({layer for _, layer in self.activities}):
            exact, different = [np.concatenate(self.activities[h, layer]) for h in HISTORIES]
            assert (different > 0).all()
            ed = 100 * (1 - exact / different)
            diff = np.abs(ed[:, 1] - ed[:, 0])
            assert diff.max() <= RS_TOL
            repetition[layer] = dict(pairs=len(diff), max_abs_pp=float(diff.max()))
        return dict(passed=True, relative_fp32_tolerance=FLOAT_TOL,
                    relative_fp64_tolerance=DOUBLE_TOL, suppression_tolerance_pp=RS_TOL,
                    local=self.checks, repetition=repetition)


class Rms:
    def __init__(self):
        self.squares = 0.
        self.count = 0

    def add(self, p, q):
        signal = q[p > 0].double()
        self.squares += float(signal.square().sum())
        self.count += signal.numel()

    def result(self):
        c = (self.squares / self.count) ** .5
        assert np.isfinite(c) and c > 0
        return dict(c=c, active_units=self.count)


def compact(bank, *, keep=()):
    result = {}
    for layer, values in bank.items():
        if layer in keep:
            result[layer] = values
        x = values.astype(np.float64)
        result[layer + "_mean"] = x.mean(-1)
        gram = []
        for step in x.transpose(1, 0, 2):
            u = step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-12)
            gram.append(u @ u.T)
        result[layer + "_gram"] = np.stack(gram)
    return result


def fc_bridge(x, w, b):
    return F.linear(x, w, b).relu()


@torch.inference_mode()
def run_fc(data, models, bridge, cal, cfg, *, epsilon=0, rms=None, audit=None):
    base = fc.gpu(data["target_fc1"])
    bank, local = {}, {}
    for idx, history in enumerate((*HISTORIES, "none")):
        incoming = base
        for k, (layer, proxy) in enumerate(zip(fc.LAYERS, models)):
            preview = None if history == "none" else fc.gpu(data["preview_" + layer][:, idx])
            memory = fc_rule.memory_from_preview(proxy, preview, len(base), base.device, base.dtype)
            before = memory.fingerprints()
            p, _ = fc_rule.side(incoming, proxy, memory, cal[k], 0, cfg["lambda"], cfg["steps"])
            out = p
            if history != "none":
                _, _, q, _ = fc_rule.query(proxy, p, memory)
                if rms is not None:
                    rms[k].add(p, q)
                if audit is not None:
                    local[history, layer] = audit.check(p, q, cal[k]["c"], -1, history, layer).cpu().numpy()
                if epsilon:
                    out = post_gain(p, q, cal[k]["c"], -1, epsilon)
            else:
                assert torch.equal(out, incoming)
            assert before == memory.fingerprints() and memory.writes == (0 if preview is None else 2)
            bank[history, layer] = out.cpu().numpy()
            if k == 0:
                incoming = fc_bridge(out, *bridge)
                bank[history, "fc2_input"] = incoming.cpu().numpy()
    return bank, local


def fc_group(bank, history, heads, shared):
    values = {l: bank[history, l][:, None] for l in ("fc1", "fc2_input", "fc2")}
    result = compact(values)
    for layer in fc.LAYERS:
        for kind, head in (("fixed", heads[layer]), ("shared", shared[layer])):
            result[f"{layer}_{kind}_scores"] = head.decision_function(bank[history, layer])[:, None]
    return result


def fc_dataset(dataset, bridge, protocol, config, old, stage):
    folder = OUT / "fc" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    pilot = fc.exp.OUT / dataset
    prior = fc.OUT / dataset
    proxy_dir = sae.OUT / dataset / "proxy"
    fc.refit.verify(fc.refit.OUT / dataset / "inputs")
    fc.first.verify(proxy_dir)
    train = fc.data(dataset, "training")
    rows = protocol["datasets"][dataset]["training_rows"]
    assert train["row"].tolist() == rows == read(pilot / "nuisance.json")["training_rows"]
    proxy_provenance = read(proxy_dir / "data_provenance.json")
    assert proxy_provenance["train_target_rows"] == rows
    trained = old["datasets"][dataset]
    assert trained["training_inputs"] == sha(fc.refit.OUT / dataset / "inputs/complete.json")
    assert trained["proxies"] == sha(proxy_dir / "complete.json")
    assert trained["fixed_svm"] == sha(pilot / "reference_svm.pckl") == read(pilot / "nuisance.json")["svm_sha256"]
    models = fc.load_models(dataset)
    model_state = [fc.model_hash(model) for model in models]
    cal = read(fc.OUT / "calibration.json")[dataset]
    assert all(c["targets"] == len(rows) and c["histories"] == list(HISTORIES) for c in cal)
    frozen = folder / "calibration.json"
    if not frozen.exists():
        rms = [Rms() for _ in models]
        pool, _ = run_fc(train, models, bridge, cal, config, rms=rms)
        with np.load(prior / "pool_only_training.npz") as previous:
            assert np.array_equal(previous["row"], train["row"])
            for (h, l), values in pool.items():
                np.testing.assert_array_equal(values, previous[f"{h}__{l}"])
        result = [dict(**r.result(), **cal[k]) for k, r in enumerate(rms)]
        freeze_file(frozen, dict(layers=result, training_rows=rows, fixed_svm=trained["fixed_svm"],
                                 proxy_state=model_state, old_calibration_sha256=sha(fc.OUT / "calibration.json")))
    assert read(frozen)["proxy_state"] == model_state
    if stage == "calibrate":
        return
    frozen_hash = sha(frozen)
    cal = read(frozen)["layers"]
    dev = fc.exp.features(dataset)
    assert trained["pilot_inputs"] == sha(pilot / "inputs/complete.json")
    assert set(train["concept"]).isdisjoint(dev["concept"]) and set(rows).isdisjoint(dev["row"])
    if stage == "local":
        audit = LocalAudit()
        pool, local = run_fc(dev, models, bridge, cal, config, audit=audit)
        with np.load(prior / "representations.npz") as previous:
            for (h, l), values in pool.items():
                np.testing.assert_array_equal(values, previous[f"pool_only__{h}__{l}"])
        np.savez_compressed(folder / "local.npz",
                            **{f"{h}__{l}": v for (h, l), v in pool.items()},
                            **{f"{h}__{l}_gain": v for (h, l), v in local.items()},
                            **{k: dev[k] for k in ("y", "concept", "row")})
        freeze_file(folder / "local.json", dict(**audit.finish(), calibration_sha256=frozen_hash,
                                                old_pool_bitwise=True, local_sha256=sha(folder / "local.npz")))
    if stage == "network":
        audit = read(folder / "local.json")
        assert audit["passed"] and audit["local_sha256"] == sha(folder / "local.npz")
        with np.load(folder / "local.npz") as saved:
            pool = {(h, l): saved[f"{h}__{l}"] for h in (*HISTORIES, "none")
                    for l in ("fc1", "fc2_input", "fc2")}
            local = {(h, l): saved[f"{h}__{l}_gain"] for h in HISTORIES for l in fc.LAYERS}
        t0, _ = run_fc(train, models, bridge, cal, config)
        t1, _ = run_fc(train, models, bridge, cal, config, epsilon=EPSILON)
        with (pilot / "reference_svm.pckl").open("rb") as f:
            fixed = pickle.load(f)
        shared, fit = {}, {}
        head_dir = folder / "heads"
        head_dir.mkdir(exist_ok=True)
        for l in fc.LAYERS:
            tx, yy, ww = fc.refit.pool_histories([t[h, l] for t in (t0, t1) for h in (*HISTORIES, "none")],
                                                train["y"], train["concept"])
            shared[l], fit[l] = fc.fit_head(head_dir, fc.refit.signature(tx, yy, ww), tx, yy, ww)
        freeze_file(folder / "shared_heads.json", dict(heads=fit, training_rows=rows, calibration_sha256=frozen_hash))
        gain, _ = run_fc(dev, models, bridge, cal, config, epsilon=EPSILON)
        for h in HISTORIES:
            np.testing.assert_array_equal(local[h, "fc1"], gain[h, "fc1"])
        for l in ("fc1", "fc2_input", "fc2"):
            np.testing.assert_array_equal(pool["none", l], gain["none", l])
        for case, bank in (("eps0", pool), ("eps005", gain), ("local", pool | local)):
            for h in HISTORIES:
                np.savez_compressed(folder / f"{case}_{h}.npz", **fc_group(bank, h, fixed, shared))
        np.savez_compressed(folder / "clip.npz", **fc_group(pool, "none", fixed, shared))
    assert sha(frozen) == frozen_hash and model_state == [fc.model_hash(model) for model in models]


class PostPoolConv(conv_rule.TransferOperator):
    def __init__(self, config, cal, scale, *, epsilon=0, rms=None, audit=None):
        super().__init__(conv_rule.new_memory(512, "cuda"), conv_rule.SpatialCodec(), cal, scale,
                         "rep4", "pool", config)
        self.epsilon, self.rms, self.audit = epsilon, rms, audit

    def reset(self, profile=None):
        super().reset(profile)
        self.local_gain = []

    def forward(self, a):
        with torch.no_grad():
            native = a.detach().clamp_min(0)
            self.input_activity = native
            out = native
            # run_trial freezes the graph briefly between previews; calls identifies target.
            if self.memory.writes:
                cfg = self.config
                p, _ = conv_rule.side(native / self.scale, self.memory, self.codec, self.calibration,
                                      0, cfg["lam"], cfg["steps"])
                out = p * self.scale
                if self.calls >= 4:
                    q, _, _ = conv_rule.query(p, self.memory, self.codec, cfg["rho"])
                    if self.rms is not None:
                        self.rms.add(out, q)
                    if self.audit is not None:
                        y = self.audit.check(out, q, self.calibration["c"], 1, self.history, "conv4")
                        self.local_gain.append(y.flatten().cpu().numpy())
                    if self.epsilon:
                        out = post_gain(out, q, self.calibration["c"], 1, self.epsilon)
            self.calls += 1
            self.output_means.append(float(out.mean()))
            return out.requires_grad_(True)


def conv_spec(dataset):
    src = conv.OUT / dataset
    spec = read(src / "manifest.json")
    _, train, dev = conv.old.split(dataset)
    assert train == spec["training"] and dev == spec["development"]
    assert conv.training_subset(train) == spec["calibration_training"]
    assert spec == read(conv.DOC / f"{dataset}_manifest.json")
    previous = conv.previous.OUT / dataset / "manifest.json"
    assert sha(previous) == spec["previous_manifest_sha256"]
    scale_info = Path(read(previous)["scale_source"])
    assert sha(scale_info) == read(previous)["scale_source_sha256"]
    assert read(scale_info)["scale"] == spec["scale"]
    assert read(conv.OUT / "protocol.json") == read(conv.DOC / "protocol.json")
    frozen = read(src / "freeze.json")
    assert frozen["protocol"] == sha(conv.OUT / "protocol.json") == spec["protocol_sha256"]
    assert frozen["config"] == conv.CONFIG
    assert frozen["calibration"] == sha(src / "calibration.json")
    assert read(src / "calibration.json") == read(conv.DOC / f"{dataset}_calibration.json")
    head_info = read(src / "fixed_heads.json")
    assert head_info["training"] == [t["dataset_index"] for t in train]
    assert head_info["input_sha256"] == sha(src / "fixed_head_training.npz")
    assert head_info["sha256"] == frozen["heads"] == sha(src / "fixed_heads.pckl")
    assert set(t["concept"] for t in train).isdisjoint(t["concept"] for t in dev)
    return src, spec, frozen


def conv_scores(values, heads):
    for l in fc.LAYERS:
        n, t, d = values[l].shape
        values[f"{l}_fixed_scores"] = heads[l].decision_function(values.pop(l).reshape(n * t, d)).reshape(n, t, -1)
    return values


def read_conv_group(src, case, h, expected):
    path = src / "development" / case / f"{h}.npz"
    note = read(path.with_suffix(".json"))
    assert sha(path) == note["sha256"]
    with np.load(path) as z:
        np.testing.assert_array_equal(z["indices"], expected["row"])
        np.testing.assert_array_equal(z["labels"], expected["y"])
        return {k: z[k] for k in z.files if k.endswith(("_mean", "_gram")) or k in fc.LAYERS}, note


def conv_run_group(dataset, trials, history, p4, transform, filt, op):
    values, memories = [], []
    op.history = history
    p4.hebb = op
    for j, trial in enumerate(trials):
        targets, previews = conv.old.c.test_inputs(dataset, trial, transform, filt, torch)
        features, audit = conv.run_trial(targets[0], previews[history], op)
        assert audit["reset_zero"] and audit["writes"] == 2 and audit["before_target"] == audit["after_target"]
        memories.append(audit["after_target"])
        if op.audit is not None:
            assert len(op.local_gain) == 15
            features["local_gain"] = np.stack(op.local_gain)
        if op.rms is None:
            values.append(features)
        if j == 0 or (j + 1) % 8 == 0 or j + 1 == len(trials):
            print("CONV", dataset, history, j + 1, "/", len(trials), flush=True)
    return values, memories


def conv_dataset(dataset, p4, transform, filt, stage):
    src, spec, previous = conv_spec(dataset)
    folder = OUT / "conv" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    cal = read(src / "calibration.json")["direct_rep4"]
    frozen = folder / "calibration.json"
    if not frozen.exists():
        rms = Rms()
        op = PostPoolConv(conv.CONFIG, cal, spec["scale"], rms=rms)
        for history in ("exact", "different_category"):
            conv_run_group(dataset, spec["calibration_training"], history, p4, transform, filt, op)
        freeze_file(frozen, dict(**rms.result(), **cal, scale=spec["scale"], old_frozen=previous,
                                 training_rows=[t["dataset_index"] for t in spec["calibration_training"]],
                                 target_steps=list(range(15)), manifest_sha256=sha(src / "manifest.json")))
    assert read(frozen)["old_frozen"] == previous
    if stage == "calibrate":
        return
    frozen_hash = sha(frozen)
    cal = read(frozen)
    trials = spec["development"]
    data = dict(row=np.array([t["dataset_index"] for t in trials]), y=np.array([t["category_index"] for t in trials]),
                concept=np.array([t["concept"] for t in trials]))
    heads = load(src / "fixed_heads.pckl")
    if stage == "local":
        audit, memories = LocalAudit(), {}
        op = PostPoolConv(conv.CONFIG, cal, spec["scale"], audit=audit)
        largest = 0.
        for h, original in zip(HISTORIES, ("exact", "different_category")):
            # Canonical history labels for the independent local acceptance report.
            values, memories[h] = conv_run_group(dataset, trials, original, p4, transform, filt, op)
            if original != h:
                audit.activities[h, "conv4"] = audit.activities.pop((original, "conv4"))
                audit.checks[h + "/conv4"] = audit.checks.pop(original + "/conv4")
            raw = {l: np.stack([v[l] for v in values]) for l in values[0]}
            del values
            bank = compact(raw, keep=fc.LAYERS)
            old_pool, note = read_conv_group(src, "direct_rep4_pool", original, data)
            assert [a["after_target"] for a in note["trials"]] == memories[h]
            for k, v in old_pool.items():
                np.testing.assert_allclose(bank[k], v, rtol=1e-6, atol=1e-7, err_msg=k)
                largest = max(largest, float(np.max(np.abs(bank[k] - v))))
            local = {f"conv4_{stat}": bank.pop(f"local_gain_{stat}") for stat in ("mean", "gram")}
            np.savez_compressed(folder / f"local_{h}.npz", **local)
            np.savez_compressed(folder / f"eps0_{h}.npz", **conv_scores(bank, heads))
            clip, _ = read_conv_group(src, "clip", original, data)
            np.savez_compressed(folder / f"clip_{h}.npz", **conv_scores(clip, heads))
        np.savez_compressed(folder / "labels.npz", **data)
        freeze_file(folder / "local.json", dict(**audit.finish(), memory_hashes=memories,
                                                old_pool_max_abs=largest, calibration_sha256=frozen_hash))
    if stage == "network":
        audit = read(folder / "local.json")
        assert audit["passed"] and audit["calibration_sha256"] == frozen_hash
        op = PostPoolConv(conv.CONFIG, cal, spec["scale"], epsilon=EPSILON)
        for h, original in zip(HISTORIES, ("exact", "different_category")):
            values, memories = conv_run_group(dataset, trials, original, p4, transform, filt, op)
            assert memories == audit["memory_hashes"][h], "Preview memory differs across epsilon"
            raw = {l: np.stack([v[l] for v in values]) for l in values[0]}
            del values
            bank = compact(raw, keep=fc.LAYERS)
            with np.load(folder / f"local_{h}.npz") as local:
                np.testing.assert_array_equal(local["conv4_mean"][:, 0], bank["conv4_mean"][:, 0])
            np.savez_compressed(folder / f"eps005_{h}.npz", **conv_scores(bank, heads))
    assert sha(frozen) == frozen_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=("fc", "conv"))
    parser.add_argument("stage", choices=("calibrate", "local", "network"))
    args = parser.parse_args()
    freeze_file(OUT / "protocol.json", dict(
        epsilons=[0, EPSILON], steps=5, lam=.03,
        gain="p*[1+epsilon*(tanh(q/c)-sum(p*tanh(q/c))/sum(p))], zero group gives zero",
        gain_phase="target only, after five original suppression-only steps",
        gain_scale="active-unit RMS of q on training pool endpoints, both histories, every target network step for Conv4",
        preview="same old suppression-only path, two writes, zero at reset, frozen target",
        fc="original FC1 -> linear+bias+ReLU -> FC2, existing frozen SAE and kappa calibration",
        conv="direct native-channel rep4 graph, existing rep4 scale and kappa calibration",
        conv_windows={"early": [0, 5], "middle": [5, 10], "late": [10, 15]},
        shared_svm="one model per dataset/layer, equal training history and epsilon weights, no preview included",
        bootstrap="4000 paired concept-block draws, stratified by category; percentile 95% interval",
        local_tolerances=dict(relative_fp32=FLOAT_TOL, relative_fp64=DOUBLE_TOL, repetition_pp=RS_TOL),
        source_sha256=sha(__file__),
    ))
    if args.stage != "calibrate":
        for branch in ("fc", "conv"):
            for dataset in DATASETS:
                folder = OUT / branch / dataset
                assert (folder / "calibration.json").exists()
                if args.stage == "network":
                    gate = read(folder / "local.json")
                    assert gate["passed"] and gate["calibration_sha256"] == sha(folder / "calibration.json")
    fc.exp.setup()
    if args.branch == "fc":
        protocol = read(fc.exp.OUT / "protocol.json")
        fc.exp.validate(protocol)
        old = read(fc.OUT / "protocol.json")
        for name, digest in old["dependencies"].items():
            assert sha(ROOT / name) == digest, name
        selection = read(fc.OUT / "selection.json")
        assert selection["training_only"] and not selection["heldout_used"]
        assert selection["protocol_sha256"] == sha(fc.OUT / "protocol.json")
        bridge = fc.exp.load_bridge()
        for dataset in DATASETS:
            fc_dataset(dataset, bridge, protocol, selection["config"], old, args.stage)
            print("DONE", args.branch, args.stage, dataset, flush=True)
    else:
        p4, _, transform, filt = conv.initialize()
        for dataset in DATASETS:
            conv_dataset(dataset, p4, transform, filt, args.stage)
            print("DONE", args.branch, args.stage, dataset, flush=True)


if __name__ == "__main__":
    main()
