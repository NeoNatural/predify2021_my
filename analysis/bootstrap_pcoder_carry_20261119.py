"""Paired concept-block intervals for window effects and their carry dependence."""

import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from priming_familiarity_timecourse_20261119 import DEST, NO_CARRY, MANIFEST, read
from score_familiarity_timecourse_20261119 import DOC, PairedConcepts
from priming_sharpen_search_20261112 import unit

OUT = DOC / "carry"
N_BOOT = 4000


def population(folder, ds, rows, sampler):
    pieces = []
    indices = []
    for path in sorted((folder / ds).glob("*.npz")):
        with np.load(path) as z:
            pieces.append(z["features"])
            indices.extend(z["index"])
    np.testing.assert_array_equal(indices, [r["index"] for r in rows])
    x = np.concatenate(pieces)
    del pieces
    return population_from_features(x, sampler)


def population_from_features(x, sampler, diagnostics=None):
    """Use the same geometry bootstrap for cached and in-memory FC2 arms."""
    n, seeds, histories, steps, _ = x.shape
    gram = np.empty((seeds, histories, steps, n, n))
    activation = np.empty((n, seeds, histories, steps))
    for k in range(seeds):
        for h in range(histories):
            for t in range(steps):
                v = np.ascontiguousarray(x[:, k, h, t])
                u = unit(v.astype(np.float64))
                gram[k, h, t] = u @ u.T
                activation[:, k, h, t] = v.mean(axis=1)
    del x
    draws = len(sampler.counts)
    gram = gram.reshape(-1, n, n)
    w = sampler.class_weights.mean(axis=1)
    same = sampler.labels[:, None] == sampler.labels
    within_kernel = (gram * same * len(sampler.classes)).reshape(-1, n * n).T.copy()
    global_kernel = gram.reshape(-1, n * n).T.copy()
    norm = w @ np.diagonal(gram, axis1=1, axis2=2).T
    centroid = np.empty_like(norm)
    grand = np.empty_like(norm)
    for j in range(0, draws, 160):
        chunk = w[j : j + 160]
        pairs = (chunk[:, :, None] * chunk[:, None, :]).reshape(len(chunk), -1)
        centroid[j : j + len(chunk)] = pairs @ within_kernel
        grand[j : j + len(chunk)] = pairs @ global_kernel
    shape = (draws, seeds, histories, steps)
    within = (norm - centroid).reshape(shape)
    between = (centroid - grand).reshape(shape)
    mean = (sampler.concept_weights @ activation.reshape(n, -1)).reshape(shape)
    result = dict(mean=mean, within=within, between=between, fisher=between / within)
    if diagnostics:
        for name, values in diagnostics.items():
            result[name] = (sampler.concept_weights @ values.reshape(n, -1)).reshape(shape)
    return result


def summarize(vector):
    lo, hi = np.quantile(vector[1:], [0.025, 0.975])
    return dict(estimate=vector[0], low=lo, high=hi)


def main(ds):
    rows = read(MANIFEST)[ds]["splits"]["test"]
    sampler = PairedConcepts(
        np.array([r["label"] for r in rows]), np.array([r["concept"] for r in rows]), N_BOOT
    )
    runs = {
        key: population(folder, ds, rows, sampler)
        for key, folder in [("carry", DEST), ("reset", NO_CARRY)]
    }
    records = []
    curves = []
    for h, ref in [(1, "same_concept"), (2, "different")]:
        for m in runs["carry"]:
            effects = {}
            for arm, values in runs.items():
                a = values[m]
                delta = 100 * (a[:, :, 0] / a[:, :, h] - 1)
                effects[arm] = delta.mean(axis=1)
                for t in range(15):
                    curves.append(
                        dict(
                            dataset=ds,
                            reference=ref,
                            metric=m,
                            arm=arm,
                            step=t,
                            **summarize(effects[arm][:, t]),
                        )
                    )
            for start in (0, 5, 10):
                on, off = (effects[arm][:, start : start + 5].mean(axis=1) for arm in ("carry", "reset"))
                estimates = {"carry": on, "reset": off, "change": off - on}
                # Ratios have an interpretable scale only with a well-separated denominator.
                if np.prod(np.quantile(on[1:], [0.025, 0.975])) > 0:
                    estimates["ratio_pct"] = 100 * off / on
                for stat, v in estimates.items():
                    records.append(
                        dict(dataset=ds, reference=ref, metric=m, start=start, stat=stat, **summarize(v))
                    )
    OUT.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(records).to_csv(OUT / f"{ds}_window_intervals.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / f"{ds}_pointwise_intervals.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["things", "imagenet"])
    with threadpool_limits(limits=3):
        main(parser.parse_args().dataset)
