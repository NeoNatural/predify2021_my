"""Measure the archived fixed decoder and concept-held-out, time-resolved MVPA."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from priming_familiarity_timecourse_20261119 import DEST, NO_CARRY, MANIFEST, SEEDS, fitted, read, weights
from priming_familiarity_scale_data_20261113 import HISTORIES
from priming_sharpen_confirm_20261112 import metrics
from report_hebb_post_pool_20261111 import PairedConcepts

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/hebbian_week_20261112_14/timecourse_20261119"
ARCHIVE = ROOT / "docs/priming_sharpen_20261112/learning_curve_20261113/fc2_encoder_control.csv"
FIELDS = ("mean", "within", "between", "fisher", "margin", "auc", "ba")


def concept_folds(labels, concepts):
    """One novel concept from every category in each of the six test folds."""
    rng = np.random.default_rng(20261119)
    fold_id = np.full(len(labels), -1)
    for category in np.unique(labels):
        mask = labels == category
        groups = rng.permutation(np.unique(concepts[mask]))
        assert len(groups) == 6
        for i, concept in enumerate(groups):
            fold_id[mask & (concepts == concept)] = i
    assert np.all(fold_id >= 0)
    for i in range(6):
        held = fold_id == i
        assert set(concepts[held]).isdisjoint(concepts[~held])
        yield ~held, held


def decode(x, labels, folds):
    rates, areas = [], []
    for train, test in folds:
        svm = make_pipeline(
            StandardScaler(), LinearSVC(C=1, dual="auto", max_iter=50000, random_state=20261119)
        )
        svm.fit(x[train], labels[train])
        scores = svm.decision_function(x[test])
        rates.append(balanced_accuracy_score(labels[test], svm.classes_[scores.argmax(1)]))
        areas.append(
            np.mean([roc_auc_score(labels[test] == c, scores[:, k]) for k, c in enumerate(svm.classes_)])
        )
    return dict(mvpa_ba=100 * np.mean(rates), mvpa_auc=100 * np.mean(areas))


def measure(ds, no_carry=False):
    rows = read(MANIFEST)[ds]["splits"]["test"]
    labels = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    folds = list(concept_folds(labels, concepts))
    sampler = PairedConcepts(labels, concepts, draws=0)
    _, head, _ = fitted(ds, weights())
    pieces, ids = [], []
    folder = NO_CARRY if no_carry else DEST
    for path in sorted((folder / ds).glob("*.npz")):
        with np.load(path) as data:
            pieces.append(data["features"])
            ids.extend(data["index"])
    np.testing.assert_array_equal(ids, [r["index"] for r in rows])
    features = np.concatenate(pieces)
    del pieces
    records = []
    for k, seed in enumerate(SEEDS):
        for h, history in enumerate(HISTORIES):
            for t in range(15):
                x = np.ascontiguousarray(features[:, k, h, t])
                fixed = metrics(x, head, sampler)
                values = {m: fixed[m][0] * (100 if m in ("auc", "ba") else 1) for m in FIELDS}
                records.append(
                    dict(
                        dataset=ds,
                        seed=seed,
                        history=history,
                        step=t,
                        **values,
                        **decode(x, labels, folds),
                    )
                )
            print(ds, seed, history, flush=True)
    frame = pd.DataFrame(records)
    output = DOC / "carry" if no_carry else DOC
    if not no_carry:
        check_archived_carry(ds, frame)
    assert np.isfinite(frame[list(FIELDS) + ["mvpa_ba", "mvpa_auc"]]).all().all()
    output.mkdir(exist_ok=True, parents=True)
    frame.to_csv(output / f"{ds}_by_seed.csv", index=False)


def check_archived_carry(ds, frame):
    original = pd.read_csv(ARCHIVE)
    check = original.loc[
        (original.dataset == ds)
        & (original.comparison == "absolute")
        & (original.condition == "trainable")
        & (original.seed != "mean")
    ]
    assert len(check) == len(SEEDS) * len(HISTORIES) * len(FIELDS)
    indexed = frame.set_index(["seed", "history", "step"])
    for r in check.itertuples():
        np.testing.assert_allclose(
            indexed.loc[(int(r.seed), r.history, 9), r.metric], r.estimate, atol=2e-5, rtol=2e-5
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["things", "imagenet"])
    parser.add_argument("--no-carry", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        measure(args.dataset, args.no_carry)
