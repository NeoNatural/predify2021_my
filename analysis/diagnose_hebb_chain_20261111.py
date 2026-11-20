"""Activity-matched and bridge-specific FC diagnostics on the frozen chain."""

from __future__ import annotations

from hebb_chain_20261111 import DATASETS, DOC, HISTORIES, OUT, WINDOWS, Controls, Experiment, load_arrays, log, old, protocol, save_arrays
from run_hebb_chain_20261111 import calibrated, estimate, samples, write_csv
import numpy as np
import torch
from torch.nn import functional as F


def match_mean(raw, reference):
    source = raw.sum(-1, keepdim=True)
    target = reference.sum(-1, keepdim=True)
    if not (source > 0).all():
        raise ValueError("FC mean matching requires positive input activity")
    scaled = raw * (target / source)
    torch.testing.assert_close(scaled.sum(-1), target[:, 0], rtol=1e-6, atol=1e-7)
    return scaled


@torch.no_grad()
def fc_paths(run, image, config):
    base = run.target(image, Controls((*config.lam[:2], 0, 0)))
    l1, l2 = run.layers[2:]
    x = base["fc1"]
    p = l1.update(x, config.lam[2])
    fc1 = {"raw": x, "pool": p, "scale": match_mean(x, p)}
    fc2 = {}
    for name, x in fc1.items():
        y = F.linear(x, *run.bridge).relu()
        z = l2.update(y, config.lam[3])
        fc2[f"{name}_raw"] = y
        fc2[f"{name}_pool"] = z
    fc2["scale_scale"] = match_mean(fc2["scale_raw"], fc2["pool_pool"])
    return dict(fc1=fc1, fc2=fc2)


def collect(run, config, history):
    path = run.folder / f"dev_bridge__{history}.npz"
    if path.exists():
        return load_arrays(path)
    bank = {}
    for i, row in enumerate(run.dev):
        image, previews = run.stimulus(row, history)
        run.prepare_trial(previews)
        versions = run.versions()
        groups = fc_paths(run, image, config)
        assert versions == run.versions()
        for layer, variants in groups.items():
            for name, tensor in variants.items():
                bank.setdefault(f"{layer}__{name}", []).append(tensor.cpu().numpy())
        if (i + 1) % 8 == 0 or i + 1 == len(run.dev):
            log("BRIDGE", run.dataset, history, i + 1, "/", len(run.dev))
    result = {}
    for key, values in bank.items():
        layer = key.split("__", 1)[0]
        a = np.stack(values).astype(np.float64)
        norm = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        result[key + "_gram"] = np.einsum("itd,jtd->tij", norm, norm)
        result[key + "_mean"] = a.mean(-1)
        result[key + "_score"] = run.heads[layer].decision_function(a.reshape(-1, a.shape[-1])).reshape(*a.shape[:2], -1)
    save_arrays(path, **result)
    return result


def summarize(run, history, arrays, output):
    contrasts = {"fc1": (("pool", "raw"), ("scale", "raw"), ("pool", "scale")),
                 "fc2": (("pool_raw", "raw_raw"), ("pool_pool", "pool_raw"), ("raw_pool", "raw_raw"),
                         ("pool_pool", "raw_pool"), ("pool_pool", "raw_raw"),
                         ("scale_scale", "raw_raw"), ("pool_pool", "scale_scale"))}
    s = samples(run, "dev", draws=4000)
    for layer, comparisons in contrasts.items():
        for case, reference in comparisons:
            a, b = (f"{layer}__{key}" for key in (case, reference))
            geoms = [s.geometry(arrays[key + "_gram"], equal_class=True) for key in (a, b)]
            for w, sl in enumerate(WINDOWS):
                for metric in ("within", "between", "fisher"):
                    x, y = [geo[metric][:, sl].mean(1) for geo in geoms]
                    output.append(dict(dataset=run.dataset, history=history, layer=layer, case=case,
                                       reference=reference, window=w, metric=metric + "_percent", **estimate(100 * (x / y - 1))))
                x, y = [s.concept_weights @ arrays[key + "_mean"][:, sl].mean(1) for key in (a, b)]
                output.append(dict(dataset=run.dataset, history=history, layer=layer, case=case,
                                   reference=reference, window=w, metric="activity_percent", **estimate(100 * (x / y - 1))))
                reads = [s.readout(arrays[key + "_score"][:, sl], fc=True) for key in (a, b)]
                for metric in ("ba", "auc", "margin"):
                    delta = (reads[0][metric] - reads[1][metric]) * (1 if metric == "margin" else 100)
                    output.append(dict(dataset=run.dataset, history=history, layer=layer, case=case,
                                       reference=reference, window=w, metric=metric, **estimate(delta)))


def main():
    protocol()
    selection = old.read(OUT / "selection.json")
    cfg = Controls(tuple(selection["suppression"]["lam"]))
    model_parts = old.initialize()
    output = []
    for dataset in DATASETS:
        run = Experiment(dataset, model_parts)
        assert old.sha(run.folder / "calibration.json") == selection["calibration"][dataset]
        run.fit()
        calibrated(run)
        for history in HISTORIES:
            arrays = collect(run, cfg, history)
            summarize(run, history, arrays, output)
        run.detach()
    DOC.mkdir(exist_ok=True)
    write_csv(DOC / "fc_bridge_diagnostics.csv", output)


if __name__ == "__main__":
    main()
