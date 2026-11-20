"""Score frozen real, scaled and shuffled FC2 responses on the fresh cohort."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from bootstrap_pcoder_carry_20261119 import N_BOOT, summarize
from collect_fc2_fresh_replication_20261120 import (
    CACHE, DATASETS, LATERAL, LOCK, MIXING, OUT, SEEDS, SHARD, STEPS, SUPPRESSION, digest, read, save,
)
from early_familiarity_gate_fit_20261120 import OUT as EARLY_DOC, original_gate
from fc2_lateral_20261120 import candidate
from score_fc2_lateral_20261120 import (
    HISTORIES, METRICS, FoldBootstrap, PairedConcepts, RELATIVE, fitted, match_mean, measure, weights,
)

ARMS = ("lateral", "scale", "shuffled")
CONTRASTS = {"lateral_minus_scale": ("lateral", "scale"),
             "lateral_minus_shuffled": ("lateral", "shuffled")}
WINDOWS = (5, 10)


class MaskedMVPA:
    """Keep the original six-fold test on categories with six held-out concepts."""
    def __init__(self, labels, concepts):
        self.categories = [x for x in np.unique(labels) if len(set(concepts[labels == x])) == 6]
        self.keep = np.isin(labels, self.categories)
        y, c = labels[self.keep], concepts[self.keep]
        self.mvpa = FoldBootstrap(y, c, PairedConcepts(y, c, N_BOOT).counts)

    def measure(self, x):
        return self.mvpa.measure(x[self.keep])


def inputs(ds, rows):
    folder = CACHE / ds
    cfg_path = folder / "protocol.json"
    cfg = read(cfg_path)
    proof = read(folder / "collection.json")
    assert cfg["frozen_stimuli_sha256"] == digest(LOCK)
    assert cfg["mixing"] == MIXING and cfg["suppression"] == SUPPRESSION
    assert cfg["arms"] == list(ARMS)
    assert proof["protocol_sha256"] == digest(cfg_path)
    s, x = [], []
    for start in range(0, len(rows), SHARD):
        path = folder / f"{start:04d}.npz"
        assert proof["shards"][path.name]["sha256"] == digest(path)
        with np.load(path) as f:
            np.testing.assert_array_equal(f["index"], [r["index"] for r in rows[start:start+SHARD]])
            s.append(f["scores"])
            x.append(f["fc2"])
    return np.concatenate(s), np.concatenate(x), cfg, proof


def check_early(ds, pop):
    published = pd.read_csv(EARLY_DOC / f"{ds}_by_seed.csv")
    published = published.loc[published.arm == "original"]
    stable = ("mean", "within", "between", "fisher")
    largest = {}
    for m in stable:
        expect = published.loc[published.metric == m].set_index(["seed", "history", "step"]).estimate
        actual = pop[m][0, ..., :5]
        check = expect.reindex(pd.MultiIndex.from_product([SEEDS, HISTORIES, range(5)])).to_numpy().reshape(actual.shape)
        np.testing.assert_allclose(actual, check, atol=3e-7, rtol=3e-6)
        largest[m] = float(np.max(np.abs(actual - check)))
    return largest


def summarize_arms(ds, arms, contrasts=CONTRASTS, window_starts=WINDOWS):
    seeds, pointwise, windows, absolutes = [], [], [], []
    for arm, table in arms.items():
        for j, seed in enumerate(SEEDS):
            for h, history in enumerate(HISTORIES):
                for t in STEPS:
                    seeds.append(dict(dataset=ds, arm=arm, seed=seed, history=history, step=t,
                                      **{m: table[m][0, j, h, t] for m in METRICS}))

    for metric in METRICS:
        data = {arm: values[metric] for arm, values in arms.items()}
        for h, reference in enumerate(HISTORIES[1:], 1):
            effects = {}
            for arm, values in data.items():
                exact, other = values[:, :, 0], values[:, :, h]
                difference = 100 * (exact / other - 1) if metric in RELATIVE else exact - other
                effects[arm] = difference.mean(axis=1)
            for name, (a, b) in contrasts.items():
                effects[name] = effects[a] - effects[b]
            for name, a in effects.items():
                row = dict(dataset=ds, reference=reference, comparison=name, metric=metric)
                for step in STEPS:
                    pointwise.append(dict(**row, step=step, **summarize(a[:, step])))
                for step in window_starts:
                    windows.append(dict(**row, start=step, **summarize(a[:, step:step + 5].mean(axis=1))))
        for h, history in enumerate(HISTORIES):
            for name, (a, b) in contrasts.items():
                delta = (data[a][:, :, h] - data[b][:, :, h]).mean(axis=1)
                for start in window_starts:
                    absolutes.append(dict(dataset=ds, history=history, comparison=name, metric=metric,
                                          start=start, **summarize(delta[:, start:start+5].mean(axis=1))))
    return seeds, pointwise, windows, absolutes


def score(ds):
    rows = read(LOCK)["datasets"][ds]["rows"]
    scores, bare, cfg, proof = inputs(ds, rows)
    g, _, calibration = original_gate(ds, scores)
    assert cfg["original_gate_calibration_sha256"] == calibration
    target_mean = bare.mean(-1, dtype=np.float64)[:, None] * (1 - SUPPRESSION * g.astype(np.float64))
    y = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    sampler = PairedConcepts(y, concepts, N_BOOT)
    mvpa = MaskedMVPA(y, concepts)
    _, head, _ = fitted(ds, weights())
    graph = LATERAL / ds / "graph.npz"
    assert cfg["graph_sha256"] == digest(graph)
    arms, budget, early = {}, {}, {}
    for arm in ARMS:
        direction = candidate(bare, graph, shuffled=arm == "shuffled") if arm != "scale" else None
        features = match_mean(bare, direction, g, target_mean)
        del direction
        raw_mean = features.mean(axis=-1, dtype=np.float64)
        np.testing.assert_allclose(raw_mean / target_mean, 1, atol=3e-7, rtol=0)
        budget[arm] = float(np.max(np.abs(raw_mean / target_mean - 1)))
        arms[arm] = measure(ds, arm, features, head, sampler, mvpa)
        del features
        if arm == "lateral":
            early = check_early(ds, arms[arm])
    tags = ("by_seed", "pointwise_intervals", "windows", "absolute_contrasts")
    OUT.mkdir(exist_ok=True, parents=True)
    for tag, table in zip(tags, summarize_arms(ds, arms), strict=True):
        path = OUT / f"{ds}_{tag}.csv"
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(table).to_csv(tmp, index=False)
        tmp.replace(path)
    save(OUT / f"{ds}_audit.json",
         dict(dataset=ds, targets=len(rows), concepts=len(set(concepts)),
              mvpa_targets=int(mvpa.keep.sum()), mvpa_categories=[int(c) for c in mvpa.categories],
              graph_sha256=digest(graph), original_calibration_sha256=calibration,
              source_sha256=digest(Path(__file__)), collection=proof,
              mean_budget_max_relative_error=budget, early_results_max_error=early))
    print("COMPLETE", ds, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        score(parser.parse_args().dataset)
