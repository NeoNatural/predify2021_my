"""Fit early Conv5-to-FC2 gates on validation and evaluate frozen on new photos."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

from bootstrap_pcoder_carry_20261119 import N_BOOT, population_from_features, summarize
from early_familiarity_gate_data_20261120 import (
    CACHE, DATASETS, LATERAL, LOCK, MANIFEST, ORIGINAL_CAL, OUT, SEEDS, SHARD, STEPS, digest, gate, read,
)
from fc2_lateral_20261120 import candidate
from priming_familiarity_scale_data_20261113 import HISTORIES
from priming_sharpen_search_20261112 import save
from probe_fc2_lateral_strengthening_20261120 import reweight
from score_fc2_lateral_20261120 import FoldBootstrap, PairedConcepts, RELATIVE, fitted, weights
from score_familiarity_timecourse_20261119 import concept_folds

FACTORS = (1., .25, .5, 2., 4., 8.)
MIXING = .45
SUPPRESSION = .003
REFERENCE = "different"
ARMS = ("original", "calibrated")


def verify_calibration_source(expected):
    current = digest(Path(__file__))
    if current != expected:
        archive = CACHE / "calibration_frozen_source.py"
        assert archive.is_file() and digest(archive) == expected


def load(ds, split, rows):
    scores, features, sha = [], [], {}
    folder = CACHE / ds / split
    for start in range(0, len(rows), SHARD):
        path = folder / f"{start:04d}.npz"
        with np.load(path) as f:
            np.testing.assert_array_equal(f["index"], [r["index"] for r in rows[start:start + SHARD]])
            scores.append(f["scores"])
            features.append(f["fc2"])
        sha[path.name] = digest(path)
    return np.concatenate(scores), np.concatenate(features), sha


def original_gate(ds, scores):
    path = ORIGINAL_CAL / f"{ds}_fc2_calibration.json"
    cals = read(path)["calibrations"]
    old = [cals[f"selected_{seed}"] for seed in SEEDS]
    g = np.stack([gate(scores[:, j], c["threshold"], c["slope"])
                  for j, c in enumerate(old)], 1)
    return g, old, digest(path)


def ramp(s, threshold, slope):
    return np.clip((s - threshold) * slope, 0, 1)


def threshold_for(scores, target, slope):
    low = float(scores.min() - 1 / slope)
    high = float(scores.max())
    if target <= 0:
        return high
    if target >= 1:
        return low
    return brentq(lambda u: ramp(scores, u, slope).mean() - target, low, high, xtol=1e-13)


def estimate_map(scores, original, calibration, factor):
    c = np.empty((len(SEEDS), len(STEPS)))
    s = np.array([v["slope"] * factor for v in calibration])
    for j, (cal, slope) in enumerate(zip(calibration, s, strict=True)):
        for t in STEPS:
            c[j, t] = (cal["threshold"] if factor == 1 else
                       threshold_for(scores[:, j, 0, t], original[:, j, 0, t].mean(), slope))
    result = dict(thresholds=c.tolist(), slopes=s.tolist())
    predicted = apply(scores, result)
    np.testing.assert_allclose(predicted[:, :, 0].mean(0), original[:, :, 0].mean(0), atol=2e-7, rtol=0)
    return result


def apply(scores, params):
    threshold = np.asarray(params["thresholds"])[None, :, None, :]
    slope = np.asarray(params["slopes"])[None, :, None, None]
    return ramp(scores, threshold, slope)


def calibration_stats(dataset, values, concept_weights, *, stage):
    table = []
    for arm, g in values.items():
        mean = (concept_weights @ g.reshape(len(g), -1)).reshape(g.shape[1:])
        for j, seed in enumerate(SEEDS):
            for h, hist in enumerate(HISTORIES):
                for t in STEPS:
                    table.append(dict(dataset=dataset, stage=stage, arm=arm, seed=seed, history=hist,
                                      step=t, mean_mixing=100 * MIXING * mean[j, h, t]))
    return table


def fit(ds):
    data = read(MANIFEST)[ds]["splits"]["validation"]
    scores, _, sha = load(ds, "validation", data)
    original, cals, cal_hash = original_gate(ds, scores)
    y = np.array([r["label"] for r in data])
    concepts = np.array([r["concept"] for r in data])
    x = scores.astype(np.float64)
    protocols = []
    out_of_fold = {}
    ref = HISTORIES.index(REFERENCE)
    for factor in FACTORS:
        cv = np.empty_like(x)
        for train, held in concept_folds(y, concepts):
            params = estimate_map(x[train], original[train], cals, factor)
            cv[held] = apply(x[held], params)
        delta = cv - original
        objective = float((delta[:, :, 0] - delta[:, :, ref]).mean())
        protocols.append(dict(factor=factor, out_of_fold_gap_gain=objective,
                              exact_mean_change=float(delta[:, :, 0].mean()),
                              different_mean_change=float(delta[:, :, ref].mean())))
        out_of_fold[factor] = cv
    selection = max(protocols, key=lambda r: r["out_of_fold_gap_gain"])["factor"]
    params = estimate_map(x, original, cals, selection)
    calibrated = apply(x, params)
    delta = calibrated - original
    np.testing.assert_allclose(delta[:, :, 0].mean(0), 0, atol=2e-7, rtol=0)
    summary = dict(dataset=ds, validation_images=len(y), validation_concepts=len(set(concepts)),
                   reference=REFERENCE, steps=list(STEPS), alpha_max=MIXING, suppression=SUPPRESSION,
                   objective="maximize held-concept (Exact - Different) mean gate gain, matching Exact each seed/step when fitting",
                   scale_factors=protocols, selected=selection, parameters=params,
                   validation_exact_mean_max_error=float(np.max(np.abs(delta[:, :, 0].mean(0)))),
                   native_validation_shards=sha, original_calibration_sha256=cal_hash, lock_sha256=digest(LOCK),
                   source_sha256=digest(Path(__file__)))
    path = OUT / f"{ds}_calibration.json"
    if path.exists():
        prior = read(path)
        verify_calibration_source(prior["source_sha256"])
        assert prior | {"source_sha256": summary["source_sha256"]} == summary
    else:
        save(path, summary)
    w = PairedConcepts(y, concepts, 0).concept_weights[0]
    pd.DataFrame(calibration_stats(ds, dict(original=original, calibrated=calibrated), w, stage="fit")
                 + calibration_stats(ds, dict(original=original, calibrated=out_of_fold[selection]),
                                     w, stage="cross_validation")).to_csv(
                                         OUT / f"{ds}_validation.csv", index=False)
    print("FROZEN", ds, selection, protocols, flush=True)


def measure(x, alpha, head, sampler, mvpa, subset, dataset_title):
    values = population_from_features(x, sampler, dict(mixing=100 * alpha, active_pct=100 * (x > 0).mean(-1)))
    dim = (len(sampler.counts), len(SEEDS), len(HISTORIES), len(STEPS))
    for metric in ("auc", "ba", "margin", "mvpa_auc", "mvpa_ba"):
        values[metric] = np.empty(dim)
    for j in range(len(SEEDS)):
        for h in range(len(HISTORIES)):
            for t in STEPS:
                a = np.ascontiguousarray(x[:, j, h, t])
                fixed = sampler.readout(head.decision_function(a)[:, None], fc=True)
                for key in ("auc", "ba", "margin"):
                    values[key][:, j, h, t] = fixed[key] * (1 if key == "margin" else 100)
                for key, v in mvpa.measure(a[subset]).items():
                    values[key][:, j, h, t] = v
    print("SCORED", dataset_title, flush=True)
    return values


def tables(dataset, measurements):
    scores, effects, absolute_delta = [], [], []
    first = measurements[ARMS[0]]
    for metric in first:
        data = {arm: results[metric] for arm, results in measurements.items()}
        for j, seed in enumerate(SEEDS):
            for h, hist in enumerate(HISTORIES):
                for t in STEPS:
                    for arm, values in data.items():
                        scores.append(dict(dataset=dataset, seed=seed, history=hist, step=t, arm=arm,
                                           metric=metric, estimate=values[0, j, h, t]))
        for h, hist in enumerate(HISTORIES):
            gain = data["calibrated"][:, :, h] - data["original"][:, :, h]
            for t in STEPS:
                absolute_delta.append(dict(dataset=dataset, history=hist, step=t, metric=metric,
                                           **summarize(gain[:, :, t].mean(1))))
            absolute_delta.append(dict(dataset=dataset, history=hist, step="0–4", metric=metric,
                                       **summarize(gain.mean((1, 2)))))
        for h, ref in enumerate(HISTORIES[1:], 1):
            contrast = {}
            for arm, values in data.items():
                a, b = values[:, :, 0], values[:, :, h]
                contrast[arm] = (100 * (a / b - 1) if metric in RELATIVE else a - b).mean(1)
            contrast["change"] = contrast["calibrated"] - contrast["original"]
            for name, values in contrast.items():
                for t in STEPS:
                    effects.append(dict(dataset=dataset, reference=ref, step=t, metric=metric, comparison=name,
                                        **summarize(values[:, t])))
                effects.append(dict(dataset=dataset, reference=ref, step="0–4", metric=metric, comparison=name,
                                    **summarize(values.mean(1))))
    return scores, effects, absolute_delta


def score(ds):
    calibration_path = OUT / f"{ds}_calibration.json"
    calibration = read(calibration_path)
    assert calibration["lock_sha256"] == digest(LOCK)
    verify_calibration_source(calibration["source_sha256"])
    seal = read(CACHE / ds / "fresh" / "protocol.json")
    assert seal["calibration_sha256"] == digest(calibration_path)
    rows = read(LOCK)["datasets"][ds]["rows"]
    scores, bare, sha = load(ds, "fresh", rows)
    original, _, cal_hash = original_gate(ds, scores)
    assert calibration["original_calibration_sha256"] == cal_hash
    calibrated = apply(scores, calibration["parameters"])
    target = bare.mean(-1, dtype=np.float64)[:, None] * (1 - SUPPRESSION * original.astype(np.float64))
    graph = LATERAL / ds / "graph.npz"
    assert read(graph.with_suffix(".json"))["graph_sha256"] == digest(graph)
    direction = candidate(bare, graph, shuffled=False)
    y = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    sampler = PairedConcepts(y, concepts, N_BOOT)
    valid = [c for c in np.unique(y) if len(set(concepts[y == c])) == 6]
    subset = np.isin(y, valid)
    mvpa_sampler = PairedConcepts(y[subset], concepts[subset], N_BOOT)
    mvpa = FoldBootstrap(y[subset], concepts[subset], mvpa_sampler.counts)
    _, head, _ = fitted(ds, weights())
    measurements, errors = {}, {}
    for arm, g in (("original", original), ("calibrated", calibrated)):
        active = reweight(bare, direction, MIXING, g, target)
        # The mixing gate can be off while the original suppression gate is on.
        # In these trials the output is scaled uniformly according to the old budget.
        for j in range(len(SEEDS)):
            uniform_only = (g[:, j] == 0) & (original[:, j] > 0)
            scale = 1 - SUPPRESSION * original[:, j][uniform_only].astype(np.float64)
            active[:, j][uniform_only] = bare[uniform_only] * scale[:, None]
        ratio = active.mean(-1, dtype=np.float64) / target
        np.testing.assert_allclose(ratio, 1, atol=3e-7, rtol=0)
        errors[arm] = float(np.max(np.abs(ratio - 1)))
        measurements[arm] = measure(active, MIXING * g, head, sampler, mvpa, subset, f"{ds} {arm}")
    points, effects, absolute = tables(ds, measurements)
    for suffix, data in (("by_seed", points), ("effects", effects), ("absolute_change", absolute)):
        pd.DataFrame(data).to_csv(OUT / f"{ds}_{suffix}.csv", index=False)
    audit = dict(dataset=ds, calibration_sha256=digest(calibration_path), fresh_shards=sha,
                 evaluation_source_sha256=digest(Path(__file__)),
                 mean_budget_max_relative_errors=errors, images=len(rows), concepts=len(set(concepts)),
                 mvpa_images=int(subset.sum()), mvpa_categories=[int(c) for c in valid],
                 mvpa_draw_scope="paired concept blocks conditional on original six-fold OOF predictions",
                 native_replay=read(CACHE / ds / "native_audit.json"), graph_sha256=digest(graph))
    save(OUT / f"{ds}_audit.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("phase", choices=("fit", "score"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        (fit if args.phase == "fit" else score)(args.dataset)
