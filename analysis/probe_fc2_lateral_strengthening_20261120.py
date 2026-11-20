"""Diagnose FC2 mixing strength, support and gate saturation on fixed carry trajectories."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from bootstrap_pcoder_carry_20261119 import N_BOOT, summarize
from fc2_lateral_20261120 import CACHE, candidate
from priming_familiarity_timecourse_20261119 import run_path
from priming_sharpen_search_20261112 import save, shape
from score_fc2_lateral_20261120 import (
    DESTINATION, HISTORIES, MANIFEST, METRICS, RELATIVE, SEEDS,
    FoldBootstrap, PairedConcepts, cache_inputs, digest, fitted, measure, read, weights,
)

OUT = DESTINATION / "strengthening"
ARMS = {
    "baseline": (.45, False, False),
    "higher_mixing": (.75, False, False),
    "higher_mixing_shuffled": (.75, True, False),
    "higher_mixing_original_support": (.75, False, True),
}
COMPARISONS = (
    ("higher_mixing", "baseline"),
    ("higher_mixing", "higher_mixing_shuffled"),
    ("higher_mixing_original_support", "higher_mixing"),
    ("higher_mixing_original_support", "baseline"),
)
WINDOWS = (5, 10)


def reweight(bare, direction, mixing, gate, target_mean):
    basis, lateral = shape(bare), shape(direction)
    out = np.empty((len(bare), len(SEEDS), *bare.shape[1:]), dtype=np.float32)
    for j in range(len(SEEDS)):
        amount = (mixing * gate[:, j, ..., None]).astype(np.float32)
        out[:, j] = target_mean[:, j, ..., None] * ((1 - amount) * basis + amount * lateral)
        out[:, j][gate[:, j] == 0] = bare[gate[:, j] == 0]
    return out


def summarize_gate(g, score, sampler):
    data = []
    for start in WINDOWS:
        gate = g[..., start : start + 5]
        saturated = gate >= .9998
        x, y = gate[:, :, 0], gate[:, :, 1]
        terms = dict(
            exact_mean=x, same_mean=y, mean_gap=x - y,
            exact_ceiling=saturated[:, :, 0], same_ceiling=saturated[:, :, 1],
            both_ceiling=saturated[:, :, 0] & saturated[:, :, 1],
            paired_accuracy=((x - y) > 1e-4) + .5 * (np.abs(x - y) <= 1e-4),
        )
        for name, values in terms.items():
            data.append(dict(start=start, measure=name, **summarize(sampler.concept_weights @ values.mean((1, 2)))))
    at9 = g[:, :, :, 9]
    ties = (at9[:, :, :2] >= .9998).all(axis=2)
    full_acc = (score[:, :, 0] > score[:, :, 1]) + .5 * (score[:, :, 0] == score[:, :, 1])
    tie_weight = sampler.concept_weights @ ties.mean(axis=1)
    raw_on_ties = sampler.concept_weights @ (ties * full_acc).mean(axis=1)
    for name, v in (("raw_pair_accuracy", sampler.concept_weights @ full_acc.mean(axis=1)),
                    ("gated_pair_accuracy", sampler.concept_weights @ (
                        ((at9[:, :, 0] - at9[:, :, 1]) > 1e-4)
                        + .5 * (np.abs(at9[:, :, 0] - at9[:, :, 1]) <= 1e-4)
                    ).mean(axis=1)),
                    ("both_ceiling", tie_weight),
                    ("raw_pair_accuracy_among_both_ceiling", raw_on_ties / tie_weight)):
        assert np.isfinite(v).all()
        data.append(dict(start=9, measure=name, **summarize(v)))
    return data


def result_tables(measures):
    relative, absolute = [], []
    for start in WINDOWS:
        for metric in METRICS:
            data = {arm: values[metric][..., start : start + 5] for arm, values in measures.items()}
            for h, history in enumerate(HISTORIES[1:], 1):
                effects = {}
                for arm, values in data.items():
                    exact, ref = values[:, :, 0], values[:, :, h]
                    effect = 100 * (exact / ref - 1) if metric in RELATIVE else exact - ref
                    effects[arm] = effect.mean(axis=(1, 2))
                comparisons = effects | {f"{a}_minus_{b}": effects[a] - effects[b] for a, b in COMPARISONS}
                relative.extend(dict(start=start, reference=history, metric=metric, comparison=key, **summarize(v))
                                for key, v in comparisons.items())
            for h, history in enumerate(HISTORIES):
                for a, b in COMPARISONS:
                    delta = (data[a][:, :, h] - data[b][:, :, h]).mean(axis=(1, 2))
                    absolute.append(dict(start=start, history=history, metric=metric, comparison=f"{a}_minus_{b}",
                                         **summarize(delta)))
    return relative, absolute


def run(ds):
    OUT.mkdir(exist_ok=True, parents=True)
    rows = read(MANIFEST)[ds]["splits"]["test"]
    index = [r["index"] for r in rows]
    graph_path = CACHE / ds / "graph.npz"
    gate_path = CACHE / ds / "gate_reconstruction.npz"
    frozen = dict(dataset=ds, graph_sha256=digest(graph_path), gate_sha256=digest(gate_path),
                  source_sha256=digest(Path(__file__)), windows=list(WINDOWS),
                  arms={k: list(v) for k, v in ARMS.items()}, comparisons=[list(x) for x in COMPARISONS],
                  note="exploratory replay on previously inspected test batches; mean suppression unchanged; no parameter search")
    protocol = OUT / f"{ds}_protocol.json"
    if protocol.exists():
        assert read(protocol) == frozen
    else:
        save(protocol, frozen)

    bare, archived, _, _ = cache_inputs(ds, rows)
    del archived
    with np.load(gate_path) as f:
        np.testing.assert_array_equal(f["index"], index)
        np.testing.assert_array_equal(f["seed"], SEEDS)
        np.testing.assert_array_equal(f["histories"], HISTORIES)
        g, target = f["g"], f["target_mean"]
    np.testing.assert_allclose(target, bare.mean(-1, dtype=np.float64)[:, None] * (1 - .003 * g))
    labels = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    sampler = PairedConcepts(labels, concepts, N_BOOT)
    mvpa = FoldBootstrap(labels, concepts, sampler.counts)
    scores = []
    for seed in SEEDS:
        base = run_path(ds, "trainable_encoder", 16, seed)
        with np.load(base.with_suffix(".native_test.npz")) as f:
            scores.append(f["scores"])
    diagnostics = summarize_gate(g, np.stack(scores, axis=1), sampler)
    pd.DataFrame(diagnostics).assign(dataset=ds).to_csv(OUT / f"{ds}_gate.csv", index=False)

    _, head, _ = fitted(ds, weights())
    true = candidate(bare, graph_path, shuffled=False)
    measures, mean_errors = {}, {}
    for arm, (mixing, shuffled, preserve_support) in ARMS.items():
        direction = true
        if shuffled:
            direction = candidate(bare, graph_path, shuffled=True)
        if preserve_support:
            direction = np.where(bare > 0, true, 0)
            assert direction.mean(-1).min() > 0
        activation = reweight(bare, direction, mixing, g, target)
        del direction
        budget_ratio = activation.mean(-1, dtype=np.float64) / target
        mean_errors[arm] = float(np.max(np.abs(budget_ratio - 1)))
        np.testing.assert_allclose(budget_ratio, 1, atol=3e-7, rtol=0)
        if preserve_support:
            np.testing.assert_array_equal(activation > 0, np.broadcast_to(bare[:, None] > 0, activation.shape))
        measures[arm] = measure(ds, arm, activation, head, sampler, mvpa)
        del activation

    effects, absolute = result_tables(measures)
    table = pd.DataFrame(effects).assign(dataset=ds)
    old = pd.read_csv(DESTINATION / f"{ds}_intervals.csv")
    keys = ["start", "reference", "metric"]
    new = table[table.comparison == "baseline"].set_index(keys).sort_index()
    check = old[(old.comparison == "lateral") & old.start.isin(WINDOWS)].set_index(keys).sort_index()
    np.testing.assert_array_equal(new.index, check.index)
    np.testing.assert_allclose(new[["estimate", "low", "high"]], check[["estimate", "low", "high"]], atol=2e-11, rtol=0)
    table.to_csv(OUT / f"{ds}_effects.csv", index=False)
    pd.DataFrame(absolute).assign(dataset=ds).to_csv(OUT / f"{ds}_absolute_changes.csv", index=False)
    save(OUT / f"{ds}_audit.json", dict(protocol_sha256=digest(protocol), n_test=len(rows),
                                      reference_reproduced=True, max_relative_mean_error=mean_errors))
    print("COMPLETE", ds, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("things", "imagenet"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        run(parser.parse_args().dataset)
