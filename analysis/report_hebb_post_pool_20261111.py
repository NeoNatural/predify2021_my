"""Report raw metrics and paired concept-bootstrap contrasts for the two-point run."""

from __future__ import annotations

import csv
import shutil
from collections import Counter

from hebb_post_pool_20261111 import DOC, OUT, DATASETS, HISTORIES, freeze_file, read, sha
import numpy as np


class PairedConcepts:
    def __init__(self, labels, concepts, draws=4000):
        self.labels = labels
        self.classes = np.unique(labels)
        self.members = self.classes[:, None] == labels
        concept_sizes = Counter(concepts)
        self.concept_sizes = np.array([concept_sizes[g] for g in concepts])
        rng = np.random.default_rng(20261111)
        self.counts = np.ones((draws + 1, len(labels)), dtype=np.float64)
        for members in self.members:
            groups, inverse = np.unique(concepts[members], return_inverse=True)
            counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
            self.counts[1:, members] = counts[:, inverse]
        self.weights = self.counts / self.counts.sum(axis=1, keepdims=True)
        self.class_weights = self.normalized_classes(self.counts)
        self.concept_weights = self.normalized_classes(self.counts / self.concept_sizes).mean(axis=1)

    def normalized_classes(self, counts):
        weights = counts[:, None] * self.members
        return weights / weights.sum(axis=-1, keepdims=True)

    def geometry(self, gram, equal_class):
        weights = self.class_weights
        if equal_class:
            class_mass = np.full((len(weights), len(self.classes)), 1 / len(self.classes))
        else:
            class_mass = self.weights @ self.members.T
        all_weights = np.einsum("bc,bci->bi", class_mass, weights)
        center_norm = np.einsum("bci,tij,bcj->bct", weights, gram, weights, optimize=True)
        mean_center_norm = np.einsum("bc,bct->bt", class_mass, center_norm)
        global_norm = np.einsum("bi,tij,bj->bt", all_weights, gram, all_weights, optimize=True)
        image_norm = all_weights @ np.diagonal(gram, axis1=1, axis2=2).T
        within = np.maximum(image_norm - mean_center_norm, 0)
        between = np.maximum(mean_center_norm - global_norm, 0)
        return dict(within=within, between=between, fisher=between / np.maximum(within, 1e-20))

    def auc(self, scores):
        auc = []
        for label in self.classes:
            pos = self.labels == label
            a = scores[pos, label, None]
            b = scores[~pos, label][None]
            compare = (a > b) + .5 * (a == b)
            wpos, wneg = self.counts[:, pos], self.counts[:, ~pos]
            auc.append(((wpos @ compare) * wneg).sum(1) / (wpos.sum(1) * wneg.sum(1)))
        return np.mean(auc, axis=0)

    def readout(self, scores, fc):
        truth = self.labels[:, None]
        correct = scores.argmax(-1) == truth
        actual = np.take_along_axis(scores, truth[:, :, None], axis=2).squeeze(2)
        others = np.where(np.arange(scores.shape[-1])[None, None, :] == truth[:, :, None], -np.inf, scores)
        margin = actual - others.max(2)
        avg_scores = scores.mean(1)
        return dict(accuracy=self.weights @ correct.mean(1),
                    ba=self.class_weights.mean(1) @ correct.mean(1),
                    window_mean_score_ba=self.class_weights.mean(1) @ (avg_scores.argmax(1) == self.labels),
                    auc=self.auc(avg_scores),
                    margin=(self.concept_weights if fc else self.weights) @ margin.mean(1))


def archive_data(path):
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


def read_groups(branch, dataset):
    folder = OUT / branch / dataset
    for name in ("calibration.json", "local.json"):
        shutil.copyfile(folder / name, DOC / f"{branch}_{dataset}_{name}")
    assert read(folder / "local.json")["passed"]
    labels = archive_data(folder / ("local.npz" if branch == "fc" else "labels.npz"))
    groups = {}
    for h in HISTORIES:
        for condition in ("eps0", "eps005", "local", "clip"):
            source = "clip" if condition == "clip" and branch == "fc" else f"{condition}_{h}"
            groups[condition, h] = archive_data(folder / f"{source}.npz")
    if branch == "fc":
        shutil.copyfile(folder / "shared_heads.json", DOC / f"fc_{dataset}_shared_heads.json")
    return groups, {k: labels[k] for k in ("y", "concept", "row")}


def estimate(values):
    lo, hi = np.quantile(values[1:], [.025, .975])
    return dict(estimate=float(values[0]), ci_low=float(lo), ci_high=float(hi))


def difference(x, ref, scale=1):
    return {k: scale * (x[k] - ref[k]) for k in x.keys() & ref.keys()}


def contrasts(absolute, metadata, contrasts_out, *, relative=(), pp=()):
    for test, ref, comparison in (
        (("eps005", "exact"), ("eps0", "exact"), "network_gain"),
        (("eps005", "different"), ("eps0", "different"), "network_gain"),
        (("local", "exact"), ("eps0", "exact"), "local_gain"),
        (("local", "different"), ("eps0", "different"), "local_gain"),
        *[((c, "exact"), (c, "different"), "exact_vs_different")
          for c in ("eps0", "eps005", "local", "clip")],
        *[((c, h), ("clip", h), "vs_no_hebb") for h in HISTORIES for c in ("eps0", "eps005")],
    ):
        if test not in absolute or ref not in absolute:
            continue
        a, b = absolute[test], absolute[ref]
        base = dict(**metadata, comparison=comparison, condition=test[0], history=test[1],
                    reference_condition=ref[0], reference_history=ref[1])
        for metric, delta in sorted(difference(a, b).items()):
            contrasts_out.append(dict(**base, metric=metric, unit="pp" if metric in pp else "absolute",
                                      **estimate(delta * (100 if metric in pp else 1))))
            if metric in relative:
                change = 100 * (a[metric] / b[metric] - 1)
                contrasts_out.append(dict(**base, metric=metric, unit="percent", **estimate(change)))


def summarize(branch, dataset, groups, labels, output):
    fc = branch == "fc"
    samples = PairedConcepts(labels["y"], labels["concept"])
    weights = samples.concept_weights if fc else samples.weights
    windows = {"feedforward": slice(0, 1)} if fc else {
        "early": slice(0, 5), "middle": slice(5, 10), "late": slice(10, 15),
    }
    layers = ("fc1", "fc2_input", "fc2") if fc else ("conv4_input", "conv4", "conv5", "fc1", "fc2")
    for layer in layers:
        step_geometry = {
            condition: samples.geometry(values[layer + "_gram"], equal_class=fc)
            for condition, values in groups.items() if layer + "_gram" in values
        }
        for (condition, history), metrics in step_geometry.items():
            means = groups[condition, history][layer + "_mean"]
            for time, mean in enumerate(weights[0] @ means):
                output["step_metrics"].append(dict(branch=branch, dataset=dataset, layer=layer, condition=condition,
                    history=history, t=time, mean=float(mean), **{k: float(v[0, time]) for k, v in metrics.items()}))
        for window, sl in windows.items():
            meta = dict(branch=branch, dataset=dataset, layer=layer, window=window)
            means = {case: groups[case][layer + "_mean"][:, sl].mean(1) for case in step_geometry}
            absolute = {}
            for (condition, history), metrics in step_geometry.items():
                absolute[condition, history] = dict(mean=weights @ means[condition, history],
                    **{k: v[:, sl].mean(1) for k, v in metrics.items()})
                for metric, value in absolute[condition, history].items():
                    output["layer_metrics"].append(dict(**meta, condition=condition, history=history,
                                                        metric=metric, **estimate(value)))
            contrasts(absolute, meta, output["geometry_contrasts"], relative=("mean", "within", "between", "fisher"))

            suppression = {}
            for condition in ("eps0", "eps005", "local", "clip"):
                if (condition, "exact") in means:
                    exact, different = [means[condition, h] for h in HISTORIES]
                    suppression[condition] = 100 * (1 - exact / different)
            for condition, values in suppression.items():
                output["repetition"].append(dict(**meta, condition=condition, reference="", unit="percent",
                                                 **estimate(weights @ values)))
                for j, row in enumerate(labels["row"]):
                    output["per_target"].append(dict(**meta, condition=condition, target=int(row),
                        concept=str(labels["concept"][j]), category=int(labels["y"][j]),
                        exact=float(means[condition, "exact"][j]), different=float(means[condition, "different"][j]),
                        suppression=float(values[j])))
                if condition != "eps0":
                    output["repetition"].append(dict(**meta, condition=condition, reference="eps0", unit="pp",
                        **estimate(weights @ (values - suppression["eps0"]))))

            if layer not in ("fc1", "fc2"):
                continue
            for kind in (("fixed", "shared") if fc else ("fixed",)):
                scores = f"{layer}_{kind}_scores"
                values = {(condition, h): samples.readout(group[scores][:, sl], fc)
                          for (condition, h), group in groups.items()
                          if scores in group and condition != "local"}
                if fc:
                    values["clip", "none"] = values.pop(("clip", "exact"))
                    values.pop(("clip", "different"))
                readout_meta = dict(**meta, head=kind)
                for (condition, h), metrics in values.items():
                    for metric, value in metrics.items():
                        output["readouts"].append(dict(**readout_meta, condition=condition, history=h,
                                                       metric=metric, **estimate(value)))
                if fc:
                    # The FC feedforward no-preview baseline is shared across histories.
                    values.update({("clip", h): values["clip", "none"] for h in HISTORIES})
                contrasts(values, readout_meta, output["readout_contrasts"],
                          pp=("accuracy", "ba", "window_mean_score_ba", "auc"))


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    output = {name: [] for name in ("step_metrics", "layer_metrics", "geometry_contrasts", "repetition",
                                   "per_target", "readouts", "readout_contrasts")}
    for branch in ("fc", "conv"):
        for dataset in DATASETS:
            groups, labels = read_groups(branch, dataset)
            summarize(branch, dataset, groups, labels, output)
    files = []
    for name, rows in output.items():
        path = DOC / f"{name}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files.append(path)
        print(name, len(rows), flush=True)
    shutil.copyfile(OUT / "protocol.json", DOC / "protocol.json")
    freeze_file(DOC / "complete.json", dict(complete=True, results={f.name: sha(f) for f in files},
                                           reporter_sha256=sha(__file__), protocol_sha256=sha(DOC / "protocol.json")))


if __name__ == "__main__":
    main()
