"""Matched FC2 contrasts with carry and frozen familiarity, using the original six MVPA folds."""

import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from collect_fc2_unmodulated_20261120 import RAW, DEST, MANIFEST, read
from bootstrap_pcoder_carry_20261119 import population, summarize, N_BOOT
from score_familiarity_timecourse_20261119 import (
    DOC,
    SEEDS,
    HISTORIES,
    fitted,
    weights,
    concept_folds,
    decode,
    PairedConcepts,
    metrics,
    FIELDS,
)
from report_familiarity_timecourse_20261119 import RELATIVE

OUT = DOC / "fc2_ablation"


def score(ds):
    rows = read(MANIFEST)[ds]["splits"]["test"]
    y = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    folds = list(concept_folds(y, concepts))
    s = PairedConcepts(y, concepts, 0)
    bs = PairedConcepts(y, concepts, N_BOOT)
    _, head, _ = fitted(ds, weights())
    on, raw = [], []
    ids = []
    for path in sorted((RAW / ds).glob("*.npz")):
        with np.load(path) as r, np.load(DEST / ds / path.name) as f:
            np.testing.assert_array_equal(r["index"], f["index"])
            ids.extend(r["index"])
            on.append(f["features"])
            raw.append(r["features"])
    np.testing.assert_array_equal(ids, [r["index"] for r in rows])
    full = np.concatenate(on)
    bare = np.concatenate(raw)
    del raw, on
    fixed = pd.read_csv(DOC / f"{ds}_by_seed.csv").assign(arm="on")
    records = []
    pop_on = population(DEST, ds, rows, bs)
    pop_off = population(RAW, ds, rows, bs)
    pop = {"on": pop_on, "off": pop_off, "scale": dict(pop_off, mean=pop_on["mean"])}
    for arm, n_seeds in [("on", 3), ("off", 1), ("scale", 3)]:
        for m in ["auc", "ba", "margin"]:
            pop[arm][m] = np.empty((N_BOOT + 1, n_seeds, 3, 15))
        for k in range(n_seeds):
            for h, history in enumerate(HISTORIES):
                for t in range(15):
                    original = np.ascontiguousarray(bare[:, 0, h, t])
                    if arm == "on":
                        x = full[:, k, h, t]
                    elif arm == "off":
                        x = original
                    else:
                        target_mean = full[:, k, h, t].mean(1)
                        x = original * (target_mean / original.mean(1))[:, None]
                        np.testing.assert_allclose(x.mean(1), target_mean, rtol=5e-7)
                    x = np.ascontiguousarray(x)
                    scores = head.decision_function(x)
                    measured = bs.readout(scores[:, None], fc=True)
                    for m in ["auc", "ba", "margin"]:
                        pop[arm][m][:, k, h, t] = measured[m] * (1 if m == "margin" else 100)
                    if arm == "on":
                        continue
                    vals = metrics(x, head, s)
                    row = {m: vals[m][0] * (100 if m in ("auc", "ba") else 1) for m in FIELDS}
                    for m in FIELDS:
                        np.testing.assert_allclose(
                            row[m],
                            pop[arm][m][0, k if pop[arm][m].shape[1] > 1 else 0, h, t],
                            rtol=2e-6,
                            atol=2e-6,
                        )
                    records.append(
                        dict(
                            dataset=ds,
                            seed=SEEDS[k] if arm == "scale" else 0,
                            history=history,
                            step=t,
                            arm=arm,
                            **row,
                            **decode(x, y, folds),
                        )
                    )
            print(ds, arm, k, flush=True)
    frame = pd.concat([fixed, pd.DataFrame(records)], ignore_index=True)
    for row in fixed.itertuples():
        for m in FIELDS:
            np.testing.assert_allclose(
                getattr(row, m),
                pop_on[m][0, SEEDS.index(row.seed), HISTORIES.index(row.history), row.step],
                atol=2e-5,
            )
    inter = []
    for h in [1, 2]:
        for start in [0, 5, 10]:
            for m in FIELDS:
                estimates = {}
                for arm in pop:
                    values = pop[arm][m][:, :, :, start : start + 5]
                    exact, ref = values[:, :, 0], values[:, :, h]
                    effect = 100 * (exact / ref - 1) if m in RELATIVE else exact - ref
                    estimates[arm] = effect.mean(axis=(1, 2))
                for name, values in dict(
                    estimates,
                    category_increment=estimates["on"] - estimates["scale"],
                    total_increment=estimates["on"] - estimates["off"],
                ).items():
                    inter.append(
                        dict(
                            dataset=ds,
                            reference=HISTORIES[h],
                            start=start,
                            metric=m,
                            comparison=name,
                            **summarize(values),
                        )
                    )
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{ds}_absolute.csv", index=False)
    pd.DataFrame(inter).to_csv(OUT / f"{ds}_intervals.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["things", "imagenet"])
    parser.add_argument("--lateral", action="store_true", help="Compare the four frozen direction controls")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.lateral:
            from score_fc2_lateral_20261120 import score_lateral

            score_lateral(args.dataset)
        else:
            score(args.dataset)
