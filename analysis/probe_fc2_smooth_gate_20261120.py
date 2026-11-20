"""Probe a smooth, strength-matched familiarity gate at the archived native t9."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from bootstrap_pcoder_carry_20261119 import N_BOOT, population_from_features, summarize
from fc2_lateral_20261120 import CACHE, candidate
from priming_familiarity_timecourse_20261119 import OUT as CAL, SHARD, gate, run_path, shard_path
from priming_sharpen_search_20261112 import save
from probe_fc2_lateral_strengthening_20261120 import OUT, reweight
from score_fc2_lateral_20261120 import (
    DESTINATION, HISTORIES, MANIFEST, RELATIVE, SEEDS,
    FoldBootstrap, PairedConcepts, digest, fitted, read, weights,
)


def remap(scores, threshold, tau):
    distance = np.maximum(scores.astype(np.float64) - threshold, 0)
    return -np.expm1(-distance / tau)


def fit_tau(validation, threshold, slope):
    mass = gate(validation[:, 0], threshold, slope).mean()
    distance = np.maximum(validation[:, 0] - threshold, 0)
    assert 0 < mass < (distance > 0).mean()
    high = float(distance.max()) * 1e7
    tau = brentq(lambda t: remap(validation[:, 0], threshold, t).mean() - mass, 1e-12, high)
    np.testing.assert_allclose(remap(validation[:, 0], threshold, tau).mean(), mass, atol=1e-8, rtol=0)
    return tau, float(mass)


def readouts(x, head, sampler, cv):
    values = population_from_features(x, sampler)
    for metric in ("auc", "ba", "margin", "mvpa_auc", "mvpa_ba"):
        values[metric] = np.empty((len(sampler.counts), len(SEEDS), 3, 1))
    for j in range(len(SEEDS)):
        for h in range(3):
            a = np.ascontiguousarray(x[:, j, h, 0])
            fixed = sampler.readout(head.decision_function(a)[:, None], fc=True)
            for key in ("auc", "ba", "margin"):
                values[key][:, j, h, 0] = fixed[key] * (1 if key == "margin" else 100)
            for key, v in cv.measure(a).items():
                values[key][:, j, h, 0] = v
    return values


def probe(ds):
    rows = read(MANIFEST)[ds]["splits"]["test"]
    indices = [r["index"] for r in rows]
    pieces = []
    for i in range(0, len(rows), SHARD):
        with np.load(shard_path(ds, "test", i)) as cache:
            np.testing.assert_array_equal(cache["index"], indices[i:i + SHARD])
            pieces.append(cache["native_fc2"])
    bare = np.concatenate(pieces)[:, :, None]
    cfg_path = CAL / f"{ds}_fc2_calibration.json"
    calibrations = read(cfg_path)["calibrations"]
    old, smooth, fitted_tau = [], [], {}
    for seed in SEEDS:
        path = run_path(ds, "trainable_encoder", 16, seed)
        cal = calibrations[f"selected_{seed}"]
        threshold, slope = cal["threshold"], cal["slope"]
        with np.load(path.with_suffix(".npz")) as f:
            tau, mass = fit_tau(f["validation"], threshold, slope)
        with np.load(path.with_suffix(".native_test.npz")) as f:
            old.append(gate(f["scores"], threshold, slope))
            smooth.append(remap(f["scores"], threshold, tau))
        fitted_tau[str(seed)] = dict(tau=tau, threshold=threshold, exact_validation_mean=mass)
    old, smooth = (np.stack(value, axis=1)[..., None] for value in (old, smooth))
    with np.load(CACHE / ds / "gate_reconstruction.npz") as f:
        np.testing.assert_array_equal(f["index"], indices)
        np.testing.assert_array_equal(f["g"][..., 9:10], old)
        target_mean = f["target_mean"][..., 9:10]
    y = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    sampler = PairedConcepts(y, concepts, N_BOOT)
    mvpa = FoldBootstrap(y, concepts, sampler.counts)
    _, head, _ = fitted(ds, weights())
    graph_path = CACHE / ds / "graph.npz"
    q = candidate(bare, graph_path, shuffled=False)
    measurements = {}
    for key, g in (("original", old), ("smooth", smooth)):
        x = reweight(bare, q, .45, g, target_mean)
        np.testing.assert_allclose(x.mean(-1, dtype=np.float64) / target_mean, 1, atol=3e-7, rtol=0)
        measurements[key] = readouts(x, head, sampler, mvpa)
        del x

    archived = pd.read_csv(DESTINATION / f"{ds}_by_seed.csv")
    archived = archived[(archived.arm == "lateral") & (archived.step == 9)].set_index(["seed", "history"])
    records = []
    for metric, original in measurements["original"].items():
        new = measurements["smooth"][metric]
        for j, seed in enumerate(SEEDS):
            for h, label in enumerate(HISTORIES):
                np.testing.assert_allclose(original[0, j, h, 0], archived.loc[(seed, label), metric], atol=2e-7)
        for h, label in enumerate(HISTORIES[1:], 1):
            def effect(a):
                v, ref = a[:, :, 0, 0], a[:, :, h, 0]
                return (100 * (v / ref - 1) if metric in RELATIVE else v - ref).mean(axis=1)
            a, b = map(effect, (original, new))
            for comparison, values in (("original", a), ("smooth", b), ("smooth_minus_original", b-a)):
                records.append(dict(dataset=ds, reference=label, metric=metric, comparison=comparison,
                                    **summarize(values)))
    for key, g in (("original", old), ("smooth", smooth)):
        means = sampler.concept_weights[0] @ g.mean(1)[..., 0]
        print(ds, key, "mean gates", means, "exact-same", means[0] - means[1], flush=True)
    pd.DataFrame(records).to_csv(OUT / f"{ds}_smooth_t9.csv", index=False)
    save(OUT / f"{ds}_smooth_t9_audit.json", dict(
        source_sha256=digest(Path(__file__)), calibration_sha256=digest(cfg_path), parameters=fitted_tau,
        original_baseline_reproduced=True, graph_sha256=digest(graph_path),
        gate="1 - exp(-max(score-threshold,0)/tau); tau matches original exact mean using validation only",
        note="Only a t9 sensitivity check on previously inspected native test images; original suppression retained.",
    ))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        for dataset in ("things", "imagenet"):
            probe(dataset)
