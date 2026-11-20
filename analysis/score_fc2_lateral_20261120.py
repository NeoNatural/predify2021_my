"""Replay FC2 lateral directions on the original carry trajectories and score matched contrasts."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from bootstrap_pcoder_carry_20261119 import N_BOOT, population_from_features, summarize
from collect_fc2_unmodulated_20261120 import DEST, MANIFEST, RAW, read
from fc2_lateral_20261120 import CACHE, candidate, prepare
from priming_familiarity_timecourse_20261119 import (
    OUT as CAL, SELECTION, SEEDS, SHARD, apply_gate, digest, fitted, frozen, gate,
    run_path, shard_path, target_direction, weights,
)
from priming_sharpen_search_20261112 import save, shape
from score_familiarity_timecourse_20261119 import (
    DOC, FIELDS, HISTORIES, PairedConcepts, check_archived_carry, concept_folds,
)

DESTINATION = DOC / "fc2_ablation" / "lateral"
ARMS = ("on", "lateral", "scale", "shuffled")
METRICS = (*FIELDS, "mvpa_auc", "mvpa_ba", "active_pct", "weak_pct")
RELATIVE = {"mean", "within", "between", "fisher"}
COMPARISONS = (("lateral", "scale"), ("lateral", "shuffled"), ("lateral", "on"),
               ("on", "scale"))
TIMES = 15


def cache_inputs(ds, rows):
    bare, full, original_t9 = [], [], []
    fingerprints = {}
    for start in range(0, len(rows), SHARD):
        name = f"{start:04d}.npz"
        index = [r["index"] for r in rows[start : start + SHARD]]
        paths = dict(raw=RAW / ds / name, complete=DEST / ds / name, native=shard_path(ds, "test", start))
        fingerprints[name] = {key: digest(path) for key, path in paths.items()}
        with np.load(paths["raw"]) as a, np.load(paths["complete"]) as b, np.load(paths["native"]) as c:
            for pack in (a, b, c):
                np.testing.assert_array_equal(pack["index"], index)
            bare.append(a["features"][:, 0])
            full.append(b["features"])
            original_t9.append(c["native_fc2"])
    x, y, native = (np.concatenate(v) for v in (bare, full, original_t9))
    assert y.shape == (len(rows), len(SEEDS), len(HISTORIES), TIMES, 4096)
    assert x.shape == (len(rows), len(HISTORIES), TIMES, 4096)
    np.testing.assert_allclose(x[:, :, 9], native, atol=1e-5, rtol=1e-5)
    return x, y, fingerprints, float(np.max(np.abs(x[:, :, 9] - native)))


def recover_gate(ds, bare, full, bank):
    config = frozen()["configs"]["target_only"]
    assert config["mixing"] == .45 and config["suppression"] == .003
    cal_file = CAL / f"{ds}_fc2_calibration.json"
    cal = read(cal_file)
    assert cal["selection_sha256"] == digest(SELECTION)
    selection = read(SELECTION)
    assert selection["chosen"][ds]["level"] == 16
    assert selection["chosen"][ds]["arm"] == "trainable_encoder"

    # The cache stores activations, whose row mean determines g up to rounding.
    mass = bare.mean(-1, dtype=np.float64)
    output_mass = full.mean(-1, dtype=np.float64)
    assert mass.min() > 0
    inverse = (1 - output_mass / mass[:, None]) / .003
    assert inverse.min() > -5e-5 and inverse.max() < 1 + 5e-5
    values = np.clip(inverse, 0, 1)
    same_vector = np.all(full == bare[:, None], axis=-1)
    values[same_vector] = 0
    files = {}
    for j, seed in enumerate(SEEDS):
        path = run_path(ds, "trainable_encoder", 16, seed)
        model_path = path.with_suffix(".pth")
        model_sha = digest(model_path)
        assert model_sha == selection["model_sha256"][ds][path.name]
        scores_path = path.with_suffix(".native_test.npz")
        with np.load(scores_path) as archive:
            c = cal["calibrations"][f"selected_{seed}"]
            expected = gate(archive["scores"], c["threshold"], c["slope"])
        np.testing.assert_allclose(values[:, j, :, 9], expected, atol=5e-5, rtol=0)
        files[str(seed)] = dict(model_sha256=model_sha, native_scores_sha256=digest(scores_path),
                                t9_max_abs_gate_error=float(np.max(np.abs(values[:, j, :, 9] - expected))))
        values[:, j, :, 9] = expected

    flat = bare.reshape(-1, 4096)
    q = target_direction(bank, flat, config)
    errors = []
    for j in range(len(SEEDS)):
        replay = apply_gate(flat, q, values[:, j].astype(np.float32).reshape(-1), config).reshape(bare.shape)
        np.testing.assert_allclose(replay, full[:, j], atol=5e-5, rtol=5e-5)
        errors.append(float(np.max(np.abs(replay - full[:, j]))))
    target_mean = mass[:, None] * (1 - .003 * values)
    np.testing.assert_allclose(target_mean, output_mass, atol=1e-7, rtol=3e-7)
    return values, target_mean, dict(calibration_sha256=digest(cal_file), per_seed=files,
                                     explicit_replay_max_abs_error=errors)


def match_mean(bare, direction, gates, target_mean):
    output = np.empty((len(bare), len(SEEDS), *bare.shape[1:]), dtype=np.float32)
    q = shape(direction) if direction is not None else None
    baseline = shape(bare)
    for j in range(len(SEEDS)):
        mix = (.45 * gates[:, j, ..., None]).astype(np.float32)
        profile = baseline if q is None else (1 - mix) * baseline + mix * q
        output[:, j] = target_mean[:, j, ..., None] * profile
        output[:, j][gates[:, j] == 0] = bare[gates[:, j] == 0]
    return output


class FoldBootstrap:
    """Reweight fixed out-of-fold predictions using the original concept draws."""
    def __init__(self, labels, concepts, counts):
        self.labels = labels
        self.classes = np.unique(labels)
        self.folds = list(concept_folds(labels, concepts))
        self.members = []
        memberships = []
        proportions = []
        for _, test in self.folds:
            labels_fold = labels[test]
            groups = [np.flatnonzero(labels_fold == c) for c in self.classes]
            self.members.append(groups)
            memberships.append([np.flatnonzero(test)[group[0]] for group in groups])
            size = np.array([len(group) for group in groups])
            mass = size[None, :] / (len(labels_fold) - size[:, None])
            np.fill_diagonal(mass, 0)
            proportions.append(mass)
        block = counts[:, np.asarray(memberships)]
        self.recall_weights = block / len(self.folds) / len(self.classes)
        self.pair_weights = block[..., :, None] * block[..., None, :] * np.stack(proportions)
        self.denominator = self.pair_weights.sum(axis=(1, 3))
        assert (self.denominator > 0).all()

    def measure(self, x):
        recalls, aucs = [], []
        for (train, test), groups in zip(self.folds, self.members, strict=True):
            head = make_pipeline(
                StandardScaler(), LinearSVC(C=1, dual="auto", max_iter=50000, random_state=20261119)
            )
            head.fit(x[train], self.labels[train])
            np.testing.assert_array_equal(head.classes_, self.classes)
            scores = head.decision_function(x[test])
            predicted = scores.argmax(1)
            recalls.append([(predicted[idx] == c).mean() for c, idx in enumerate(groups)])
            matrix = np.zeros((len(groups), len(groups)))
            for c, idx in enumerate(groups):
                positive = scores[idx, c, None]
                for other, j in enumerate(groups):
                    if c != other:
                        negative = scores[None, j, c]
                        matrix[c, other] = np.mean((positive > negative) + .5 * (positive == negative))
            aucs.append(matrix)
        ba = np.einsum("bfc,fc->b", self.recall_weights, np.stack(recalls))
        numerator = np.einsum("bfck,fck->bc", self.pair_weights, np.stack(aucs))
        auc = (numerator / self.denominator).mean(axis=1)
        return {"mvpa_auc": 100 * auc, "mvpa_ba": 100 * ba}


def measure(ds, arm, x, head, sampler, mvpa):
    support = x > 0
    active = 100 * support.mean(-1)
    weak = 100 * (support & (x < .01 * x.max(-1, keepdims=True))).mean(-1)
    values = population_from_features(x, sampler, dict(active_pct=active, weak_pct=weak))
    shape = (len(sampler.counts), len(SEEDS), len(HISTORIES), TIMES)
    for key in ("margin", "auc", "ba", "mvpa_auc", "mvpa_ba"):
        values[key] = np.empty(shape)
    for j, seed in enumerate(SEEDS):
        for h in range(len(HISTORIES)):
            for t in range(TIMES):
                a = np.ascontiguousarray(x[:, j, h, t])
                fixed = sampler.readout(head.decision_function(a)[:, None], fc=True)
                for key in ("margin", "auc", "ba"):
                    values[key][:, j, h, t] = fixed[key] * (1 if key == "margin" else 100)
                for key, series in mvpa.measure(a).items():
                    values[key][:, j, h, t] = series
        print("SCORED", ds, arm, seed, flush=True)
    return values


def save_results(ds, measured):
    rows, recur, inter, absolute = [], [], [], []
    for arm, data in measured.items():
        for j, seed in enumerate(SEEDS):
            for h, hist in enumerate(HISTORIES):
                for t in range(TIMES):
                    rows.append(dict(dataset=ds, arm=arm, seed=seed, history=hist, step=t,
                                     **{m: data[m][0, j, h, t] for m in METRICS}))
    for m in METRICS:
        effects = {}
        for h, hist in enumerate(HISTORIES[1:], 1):
            for arm, data in measured.items():
                a, ref = data[m][:, :, 0], data[m][:, :, h]
                contrast = 100 * (a / ref - 1) if m in RELATIVE else a - ref
                effects[arm] = contrast.mean(axis=1)
            all_effects = effects | {f"{a}_minus_{b}": effects[a] - effects[b] for a, b in COMPARISONS}
            for arm, values in all_effects.items():
                for t in range(TIMES):
                    recur.append(dict(dataset=ds, reference=hist, metric=m, comparison=arm,
                                      step=t, **summarize(values[:, t])))
                for start in (0, 5, 10):
                    inter.append(dict(dataset=ds, reference=hist, metric=m, comparison=arm,
                                      start=start, **summarize(values[:, start : start + 5].mean(axis=1))))
        for a, b in COMPARISONS:
            for h, hist in enumerate(HISTORIES):
                delta = (measured[a][m][:, :, h] - measured[b][m][:, :, h]).mean(axis=1)
                for start in (0, 5, 10):
                    absolute.append(dict(dataset=ds, history=hist, metric=m, comparison=f"{a}_minus_{b}",
                                         start=start, **summarize(delta[:, start : start + 5].mean(axis=1))))
    frame = pd.DataFrame(rows)
    orig = frame[frame.arm == "on"]
    check_archived_carry(ds, orig)
    archived = pd.read_csv(DOC / f"{ds}_by_seed.csv").sort_values(["seed", "history", "step"])
    fields = list(FIELDS) + ["mvpa_auc", "mvpa_ba"]
    np.testing.assert_allclose(orig.sort_values(["seed", "history", "step"])[fields], archived[fields], atol=2e-5)
    for file, records in (("by_seed", rows), ("pointwise_intervals", recur), ("intervals", inter),
                          ("absolute_contrasts", absolute)):
        path = DESTINATION / f"{ds}_{file}.csv"
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(records).to_csv(tmp, index=False)
        tmp.replace(path)


def score_lateral(ds):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    prepare(ds)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    rows = read(MANIFEST)[ds]["splits"]["test"]
    assert len(rows) == {"things": 108, "imagenet": 96}[ds]
    bare, full, source, bare_t9 = cache_inputs(ds, rows)
    bank, head, _ = fitted(ds, weights())
    gates, target_mean, calibration = recover_gate(ds, bare, full, bank)
    print("BASELINE AND GATE VERIFIED", ds, calibration, flush=True)
    labels = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    sampler = PairedConcepts(labels, concepts, N_BOOT)
    mvpa = FoldBootstrap(labels, concepts, sampler.counts)
    graph_path = CACHE / ds / "graph.npz"
    audit = dict(dataset=ds, test_count=len(rows), test_concepts=len(set(concepts)),
                 image_order=[r["index"] for r in rows], seeds=list(SEEDS), histories=list(HISTORIES),
                 timesteps=list(range(TIMES)), source_sha256=digest(Path(__file__)), shards=source,
                 nonmodulated_t9_max_abs_error=bare_t9, calibration=calibration, mean_budget={},
                 graph=read(graph_path.with_suffix(".json")), resamples=N_BOOT,
                 protocol="one frozen hard-k32 direction; same gate and per-trial activity budget per arm")
    measured = {}
    for arm in ARMS:
        if arm == "on":
            activation = full
        else:
            q = candidate(bare, graph_path, arm == "shuffled") if arm != "scale" else None
            activation = match_mean(bare, q, gates, target_mean)
            del q
        assert np.isfinite(activation).all() and activation.min() >= 0
        output = activation.mean(-1, dtype=np.float64)
        np.testing.assert_allclose(output, target_mean, atol=1e-7, rtol=3e-7)
        audit["mean_budget"][arm] = float(np.max(np.abs(output / target_mean - 1)))
        measured[arm] = measure(ds, arm, activation, head, sampler, mvpa)
        del activation
    save_results(ds, measured)
    save(DESTINATION / f"{ds}_audit.json", audit)
    with (CACHE / ds / "gate_reconstruction.npz").open("wb") as stream:
        np.savez(stream, index=audit["image_order"], seed=SEEDS, histories=HISTORIES, g=gates,
                 target_mean=target_mean)
    print("COMPLETE", ds, flush=True)
