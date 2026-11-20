"""Run the frozen four-layer search, confirm it on clean development trials, and test mechanisms."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from hebb_chain_20261111 import (DATASETS, DOC, GAIN, GRID, HISTORIES, LAYERS, OUT, REFERENCE,
                                WINDOWS, Controls, Experiment, load_arrays, log, old, protocol,
                                save_arrays, save_json)
import numpy as np
import torch
from report_hebb_post_pool_20261111 import PairedConcepts


class Features:
    """Keep only one group's transient full-map features; archive exact geometry."""

    def __init__(self, root, size):
        self.root, self.size = root, size
        self.data = {}

    def add(self, index, values):
        for key, tensor in values.items():
            arr = tensor.detach().cpu().numpy()
            if key not in self.data:
                self.data[key] = np.lib.format.open_memmap(self.root / f"{key}.npy", mode="w+", dtype=arr.dtype,
                                                          shape=(self.size, *arr.shape))
            self.data[key][index] = arr

    def compact(self, heads):
        result = {}
        for key, arr in self.data.items():
            if key.endswith(("_trace", "_diagnostics")):
                result[key] = np.array(arr)
                continue
            means, density, negative, gram = [], [], [], []
            for t in range(arr.shape[1]):
                x = torch.from_numpy(np.ascontiguousarray(arr[:, t])).to("cuda", torch.float64)
                means.append(x.mean(1).cpu().numpy())
                density.append((x != 0).double().mean(1).cpu().numpy())
                negative.append((x < 0).double().mean(1).cpu().numpy())
                x /= x.norm(dim=1, keepdim=True).clamp_min(1e-12)
                gram.append((x @ x.T).cpu().numpy())
            result[key + "_mean"] = np.stack(means, 1)
            result[key + "_density"] = np.stack(density, 1)
            result[key + "_negative"] = np.stack(negative, 1)
            result[key + "_gram"] = np.stack(gram)
            name = key.removesuffix("_local")
            if name in heads and key in (name, name + "_local"):
                result[key + "_scores"] = heads[name].decision_function(arr.reshape(-1, arr.shape[-1])).reshape(
                    len(arr), arr.shape[1], -1)
        return result

    def close(self):
        for arr in self.data.values():
            arr.flush()
            arr._mmap.close()
        self.data.clear()


def path_for(run, scope, condition, history):
    return run.folder / f"{scope}__{condition}__{history}.npz"


def seed_for(dataset, row):
    key = f"20261111/{dataset}/{row['dataset_index']}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def run_group(run, scope, configs, *, replay=False, preview="shared", histories=HISTORIES, local=False,
              shuffle=False, calibrate=False, trace=False, check_replay=False):
    rows = trial_rows(run, scope)
    for history in histories:
        destinations = {key: path_for(run, scope, key, history) for key in configs}
        rms_path = run.folder / f"gain_rms__{history}.json"
        if all(p.exists() for p in destinations.values()) and (not calibrate or rms_path.exists()):
            continue
        log("RUN", run.dataset, scope, history, tuple(configs), len(rows))
        start = time.monotonic()
        for layer in run.layers:
            layer.rms = [0., 0]
        signatures = []
        with TemporaryDirectory(dir=run.folder, prefix="features_") as scratch:
            root = Path(scratch)
            buffers = {key: Features(root / key, len(rows)) for key in (*configs, "shared")}
            for buffer in buffers.values():
                buffer.root.mkdir()
            try:
                for i, row in enumerate(rows):
                    image, previews = run.stimulus(row, history)
                    run.prepare_trial(previews, preview=preview)
                    fingerprints = [layer.fingerprint() for layer in run.layers]
                    signatures.append(fingerprints)
                    if shuffle:
                        run.shuffle(seed_for(run.dataset, row))
                    if replay:
                        config = next(iter(configs.values()))
                        # All replay branches must share the same physical PCoder path.
                        assert all((c.lam[:2], c.epsilon[:2]) == (config.lam[:2], config.epsilon[:2])
                                   for c in configs.values())
                        only_pc = Controls((*config.lam[:2], 0, 0), (*config.epsilon[:2], 0, 0))
                        snapshot = None
                        if check_replay and i == 0:
                            from predify_experiments.tripartite_20261027 import state_snapshot
                            snapshot = state_snapshot(run.model)
                        raw = run.target(image, only_pc, trace=trace, calibrate=calibrate)
                        buffers["shared"].add(i, {k: v for k, v in raw.items() if not k.startswith("fc")})
                        for key, cfg in configs.items():
                            v = run.replay_fc(raw["fc1"], cfg, calibrate=calibrate and cfg == REFERENCE, trace=trace)
                            buffers[key].add(i, v)
                            if snapshot is not None:
                                from predify_experiments.tripartite_20261027 import restore_state
                                restore_state(run.model, snapshot)
                                full = run.target(image, cfg, audit=True)
                                for name in LAYERS:
                                    torch.testing.assert_close(full[name], (v if name.startswith("fc") else raw)[name],
                                                               rtol=5e-5, atol=2e-5)
                    else:
                        assert len(configs) == 1
                        key, cfg = next(iter(configs.items()))
                        buffers[key].add(i, run.target(image, cfg, local=local, trace=trace, audit=i == 0))
                    if (i + 1) % 8 == 0 or i + 1 == len(rows):
                        log("PROGRESS", run.dataset, scope, history, tuple(configs), i + 1, "/", len(rows),
                            f"{time.monotonic() - start:.1f}s")
                if preview == "shared":
                    old.immutable(run.folder / f"{scope}__preview_hashes__{history}.json", signatures)
                if calibrate:
                    save_json(rms_path, [layer.rms for layer in run.layers])
                shared = buffers["shared"].compact(run.heads)
                for key, path in destinations.items():
                    save_arrays(path, **shared, **buffers[key].compact(run.heads))
            finally:
                for buffer in buffers.values():
                    buffer.close()


def calibrated(run):
    result = []
    by_history = [old.read(run.folder / f"gain_rms__{history}.json") for history in HISTORIES]
    for i, layer in enumerate(run.layers):
        squares, count = np.sum([a[i] for a in by_history], 0)
        assert squares > 0 and count > 0
        layer.cal.update(c=float(np.sqrt(squares / count)), gain_active_units=int(count))
        result.append(layer.cal)
    old.immutable(run.folder / "calibration.json", result)


def stage_a(run):
    assert old.read(run.folder / "audit.json")["passed"]
    run_group(run, "train", {"shared_off": Controls()})
    for p in range(3):
        run_group(run, "train", {k: v for k, v in GRID.items() if k[1] == str(p)}, replay=True, calibrate=p == 1)
    calibrated(run)


def views(run, scope, cases):
    return {case: [load_arrays(path_for(run, scope, case, h)) for h in HISTORIES] for case in cases}


def samples(run, scope, draws=0):
    rows = trial_rows(run, scope)
    return PairedConcepts(np.array([r["category_index"] for r in rows]), np.array([r["concept"] for r in rows]), draws)


def trial_rows(run, scope):
    if scope == "train":
        return run.train
    if scope == "dev":
        return run.dev
    assert scope == "heldout"
    rows = []
    for category in range(len(run.manifest["categories"])):
        pool = [r for r in run.manifest["trials"] if r["fold"] == 0 and r["category_index"] == category]
        for concept in sorted({r["concept"] for r in pool})[:2]:
            rows.extend(sorted((r for r in pool if r["concept"] == concept), key=lambda r: r["dataset_index"])[:2])
    assert len(rows) == len(run.dev)
    roles = lambda trials: {i for t in trials for i in [t["dataset_index"], *t["previews"], *t["primes"].values()]}
    used = [t for t in run.manifest["trials"] if t["fold"] in (1, 2, 3)] + run.dev
    for key in ("concept", "sha256"):
        meta = run.manifest["images"]
        assert not {meta[str(i)][key] for i in roles(rows)} & {meta[str(i)][key] for i in roles(used)}
    old.immutable(run.folder / "heldout_manifest.json", rows)
    return rows


def repetition(samples, pair, layer, window):
    exact, diff = [g[layer + "_mean"][:, window].mean(1) for g in pair]
    assert (diff > 0).all()
    return samples.concept_weights @ (100 * (1 - exact / diff))


def fisher(samples, data, layer, window):
    return samples.geometry(data[layer + "_gram"][window], equal_class=True)["fisher"].mean(1)


def margin_by_target(scores, labels):
    actual = np.take_along_axis(scores, labels[:, None, None], axis=-1).squeeze(-1)
    other = np.where(np.arange(scores.shape[-1])[None, None, :] == labels[:, None, None], -np.inf, scores)
    return actual - other.max(-1)


def screen(run, keys, baseline):
    s = samples(run, "train")
    data = views(run, "train", (baseline, *keys))
    reference = data[baseline]
    rs_window = slice(5, 15)
    result = {}
    for key in keys:
        pair = data[key]
        rs = {name: float((repetition(s, pair, name, rs_window) - repetition(s, reference, name, rs_window))[0])
              for name in LAYERS}
        geom = {name: [float(100 * (fisher(s, a, name, rs_window)[0] / fisher(s, b, name, rs_window)[0] - 1))
                       for a, b in zip(pair, reference)] for name in LAYERS}
        checks = []
        for h, a, b in zip(HISTORIES, pair, reference):
            for window in (WINDOWS[0], rs_window):
                ax, bx = a["fc2_scores"][:, window], b["fc2_scores"][:, window]
                ma, mb = (margin_by_target(x, s.labels) for x in (ax, bx))
                ba = s.class_weights[0].mean(0) @ ((ax.argmax(-1) == s.labels[:, None]).mean(1) -
                                                   (bx.argmax(-1) == s.labels[:, None]).mean(1))
                margin = s.concept_weights[0] @ (ma - mb).mean(1)
                sd = float(mb.std())
                checks.append(dict(history=h, window=[window.start, window.stop], ba_pp=float(100 * ba),
                                   margin=float(margin), margin_sd=sd,
                                   passed=bool(ba >= -.01000001 and margin >= -.1 * sd)))
        result[key] = dict(repetition_pp=rs, fisher_percent=geom, readout=checks,
                           feasible=all(c["passed"] for c in checks))
    return result


def select_a(runs, configs):
    keys = tuple(configs)
    records = {r.dataset: screen(r, keys, "shared_off") for r in runs}
    summary = {}
    for key, cfg in configs.items():
        rows = [a[key] for a in records.values()]
        rs = min(r["repetition_pp"][layer] for r in rows for layer in ("conv4", "fc1", "fc2"))
        geom = min(v for row in rows for pair in row["fisher_percent"].values() for v in pair)
        margin = min(c["margin"] / max(c["margin_sd"], 1e-8) for row in rows for c in row["readout"])
        summary[key] = dict(worst_rs=rs, worst_fisher=geom, worst_margin_sd=margin,
                            feasible=all(row["feasible"] for row in rows), sum_lambda=sum(cfg.lam))
    pool = [key for key, row in summary.items() if row["feasible"]]
    consistent = [key for key in pool if summary[key]["worst_rs"] > 0]
    pool = consistent or pool
    frontier = [key for key in pool if not any(
        (summary[other]["worst_rs"] >= summary[key]["worst_rs"] and
         summary[other]["worst_fisher"] >= summary[key]["worst_fisher"] and
         (summary[other]["worst_rs"], summary[other]["worst_fisher"]) !=
         (summary[key]["worst_rs"], summary[key]["worst_fisher"])) for other in pool)]
    chosen = max(frontier, key=lambda k: (summary[k]["worst_fisher"], summary[k]["worst_margin_sd"],
                                         -summary[k]["sum_lambda"])) if frontier else "a11"
    return chosen, dict(records=records, summary=summary, frontier=frontier, selected=chosen,
                        feasible=bool(frontier), consistent_rs=bool(consistent))


def stage_b(run, config, key):
    # On the suppress-only trajectory compute each local gain without propagating it.
    run_group(run, "train", {"local": Controls(config.lam, (.005,) * 4)}, local=True)
    for name, epsilon in GAIN.items():
        cfg = Controls(config.lam, epsilon)
        if name == "gainfc":
            run_group(run, "train", {name: cfg}, replay=True)
        else:
            run_group(run, "train", {name: cfg})
    base = views(run, "train", (key, "local"))
    for direct, replayed in zip(base["local"], base[key]):
        for name in LAYERS:
            np.testing.assert_allclose(direct[name + "_gram"], replayed[name + "_gram"], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(direct["fc2_scores"], replayed["fc2_scores"], rtol=5e-5, atol=5e-5)
    s = samples(run, "train")
    rows = screen(run, tuple(GAIN), key)
    for name, eps in GAIN.items():
        row = rows[name]
        delta = {}
        for layer, enabled in zip(LAYERS, eps):
            if enabled:
                a = fisher(s, base["local"][0], layer + "_local", slice(5, 15))[0]
                b = fisher(s, base["local"][0], layer, slice(5, 15))[0]
                delta[layer] = float(100 * (a / b - 1))
        row["local_fisher_exact_percent"] = delta
        row["eligible"] = (row["feasible"] and min(row["repetition_pp"].values()) >= -.02 and
                           all(value > 0 and row["fisher_percent"][layer][0] > 0 for layer, value in delta.items()))
    return rows


def configure(run):
    run.fit()
    run.audit()


def check_extra(selection):
    rows = [r[selection["selected"]]["repetition_pp"] for r in selection["records"].values()]
    return selection["feasible"] and all(r["conv5"] <= 0 and min(r[k] for k in ("conv4", "fc1", "fc2")) > 0
                                         for r in rows)


def fit_and_select(model_parts):
    for dataset in DATASETS:
        run = Experiment(dataset, model_parts)
        configure(run)
        stage_a(run)
        run.detach()
    runs = [Experiment(dataset, model_parts) for dataset in DATASETS]
    grid = dict(GRID)
    chosen, selection = select_a(runs, grid)
    if check_extra(selection):
        lam = grid[chosen].lam
        extra = {name: Controls((lam[0], lam[1] * factor, *lam[2:]))
                 for name, factor in (("p5_low", .5), ("p5_high", 1.5))}
        for run in runs:
            configure(run)
            calibrated(run)
            for name, config in extra.items():
                run_group(run, "train", {name: config}, replay=True)
            run.detach()
        grid.update(extra)
        chosen, selection = select_a(runs, grid)
    cfg = grid[chosen]
    save_json(OUT / "training_a.json", selection)
    log("A SELECTED", chosen, cfg, selection["summary"][chosen])
    gain_results = {}
    for run in runs:
        configure(run)
        calibrated(run)
        gain_results[run.dataset] = stage_b(run, cfg, chosen)
        run.detach()
    gains = [name for name in GAIN if selection["feasible"] and all(r[name]["eligible"] for r in gain_results.values())]
    if gains:
        gains.sort(key=lambda name: min(v for d in gain_results.values()
                                        for v in d[name]["local_fisher_exact_percent"].values()), reverse=True)
    selected_gain = gains[0] if gains else None
    save_json(OUT / "training_b.json", gain_results)
    frozen = dict(suppression=dict(id=chosen, **asdict(cfg)), gain=selected_gain,
                  gain_epsilon=list(GAIN[selected_gain]) if selected_gain else [0] * 4,
                  feasible=selection["feasible"], consistent_training_rs=selection["consistent_rs"],
                  calibration={r.dataset: old.sha(r.folder / "calibration.json") for r in runs})
    save_json(OUT / "selection.json", frozen)
    log("FROZEN", frozen)
    return old.read(OUT / "selection.json")


@torch.no_grad()
def probe_matching(run, config):
    path = run.folder / "dev_fixed_input_match.npz"
    if path.exists():
        return
    rows = []
    indices = [0, 5, 9, 14]
    for row in run.dev:
        image, exact = run.stimulus(row, HISTORIES[0])
        _, different = run.stimulus(row, HISTORIES[1])
        run.prepare_trial(exact)
        actual = run.target(image, config)
        shapes = [layer.record["input"].shape for layer in run.layers]

        def fixed_inputs():
            records = []
            for layer, name, shape in zip(run.layers, LAYERS, shapes):
                layer.diagnose = True
                steps = []
                for t in indices:
                    x = actual[name + "_input"][t].reshape(shape)
                    layer.update(x, layer.lam)
                    steps.append(layer.record["trace"][0])
                records.append(torch.stack(steps))
            return torch.stack(records)

        a = fixed_inputs()
        run.prepare_trial(different)
        run.controls(config)
        b = fixed_inputs()
        rows.append(torch.stack((a, b)).cpu().numpy())
    save_arrays(path, trace=np.stack(rows), times=np.array(indices),
                columns=np.array(["kappa", "mu_plus_sigma", "threshold", "activity_normalized"]),
                layers=np.array(LAYERS), histories=np.array(HISTORIES))
    log("MATCH PROBE", run.dataset, len(rows))


def confirm(run, selection, scope):
    assert old.sha(run.folder / "calibration.json") == selection["calibration"][run.dataset]
    configure(run)
    calibrated(run)
    lam = tuple(selection["suppression"]["lam"])
    cfg = Controls(lam)
    cases = ["shared_off", "suppress", "no_hebb"]
    run_group(run, scope, {"suppress": cfg}, trace=scope == "dev")
    run_group(run, scope, {"no_hebb": Controls()}, preview="off")
    run_group(run, scope, {"none": Controls()}, preview="off", histories=("none",))
    if selection["gain"]:
        run_group(run, scope, {"gain": Controls(lam, tuple(selection["gain_epsilon"]))})
        cases.append("gain")
    if scope == "heldout":
        run_group(run, scope, {"shared_off": Controls()})
        return cases
    run_group(run, scope, {"shared_off": Controls(), "fc_only": Controls((0, 0, *lam[2:]))}, replay=True)
    run_group(run, scope, {"pc_only": Controls((*lam[:2], 0, 0))}, trace=True)
    run_group(run, scope, {"pc4_only": Controls((lam[0], 0, *lam[2:]))}, trace=True)
    run_group(run, scope, {"shuffle": cfg}, shuffle=True, trace=True)
    run_group(run, scope, {"preview_fc": cfg}, preview="fc_read", trace=True)
    run_group(run, scope, {"local": Controls(lam, (.005,) * 4)}, local=True)
    probe_matching(run, cfg)
    return [*cases, "fc_only", "pc_only", "pc4_only", "shuffle", "preview_fc", "local"]


def estimate(x):
    lo, hi = np.quantile(x[1:], [.025, .975])
    return dict(estimate=float(x[0]), ci_low=float(lo), ci_high=float(hi))


def summarize(run, scope, cases, output):
    s = samples(run, scope, 4000)
    data = views(run, scope, cases)
    data["none"] = [load_arrays(path_for(run, scope, "none", "none"))]
    geom = {}
    readouts = {}
    for condition, pair in data.items():
        histories = ("none",) if condition == "none" else HISTORIES
        for h, z in zip(histories, pair):
            for key in z:
                if not key.endswith("_gram"):
                    continue
                name = key.removesuffix("_gram")
                geo = s.geometry(z[key], equal_class=True)
                stats = {k: s.concept_weights @ z[f"{name}_{k}"] for k in ("mean", "density", "negative")}
                stats.update(geo)
                for t in range(15):
                    output["steps"].append(dict(dataset=run.dataset, scope=scope, condition=condition, history=h,
                                                layer=name, t=t, **{k: float(v[0, t]) for k, v in stats.items()}))
                for w, sl in enumerate(WINDOWS):
                    item = {k: v[:, sl].mean(1) for k, v in stats.items()}
                    geom[condition, h, name, w] = item
                    for metric, values in item.items():
                        output["geometry"].append(dict(dataset=run.dataset, scope=scope, condition=condition, history=h,
                                                       layer=name, window=w, metric=metric, **estimate(values)))
            for w, sl in enumerate(WINDOWS):
                scores = z["fc2_scores"][:, sl]
                readouts[condition, h, w] = s.readout(scores, fc=True)
                for metric, values in readouts[condition, h, w].items():
                    output["readout"].append(dict(dataset=run.dataset, scope=scope, condition=condition, history=h,
                                                  window=w, metric=metric, **estimate(values)))
    for (case, h, layer, w), item in geom.items():
        sl = WINDOWS[w]
        refs = [("no_hebb", h, layer, w), ("shared_off", h, layer, w)] if h != "none" else []
        if case in ("gain", "local", "shuffle", "preview_fc", "pc4_only"):
            refs.append(("suppress", h, layer.removesuffix("_local"), w))
        if layer in LAYERS:
            refs.append((case, h, layer + "_input", w))
        for reference in refs:
            if reference not in geom or reference == (case, h, layer, w):
                continue
            for metric in ("mean", "within", "between", "fisher"):
                reference_values = geom[reference][metric]
                pct = 100 * (item[metric] / np.maximum(reference_values, 1e-15) - 1)
                output["contrasts"].append(dict(dataset=run.dataset, scope=scope, condition=case, history=h,
                     layer=layer, window=w, metric=metric, reference=reference[0], reference_layer=reference[2],
                     **estimate(pct)))
        if h == HISTORIES[0] and layer in LAYERS:
            value = repetition(s, data[case], layer, sl)
            for reference in ("absolute", "shared_off", "no_hebb", "suppress"):
                if reference != case:
                    delta = value if reference == "absolute" else value - repetition(s, data[reference], layer, sl)
                    output["repetition"].append(dict(dataset=run.dataset, scope=scope, condition=case,
                         layer=layer, window=w, reference=reference, **estimate(delta)))
    if scope == "dev":
        def interaction(values):
            both, pc, fc, neither = (values[k] for k in ("suppress", "pc_only", "fc_only", "shared_off"))
            return both - pc - fc + neither

        for w, sl in enumerate(WINDOWS):
            for layer in LAYERS:
                values = {}
                for case in ("suppress", "pc_only", "fc_only", "shared_off"):
                    e, d = [a[layer + "_mean"][:, sl].mean(1) for a in data[case]]
                    values[case] = s.concept_weights @ (100 * np.log(d / e))
                output["interactions"].append(dict(dataset=run.dataset, layer=layer, window=w,
                    metric="log_repeat_x100", **estimate(interaction(values))))
                for h in HISTORIES:
                    values = {case: 100 * np.log(geom[case, h, layer, w]["fisher"])
                              for case in ("suppress", "pc_only", "fc_only", "shared_off")}
                    output["interactions"].append(dict(dataset=run.dataset, layer=layer, window=w,
                        metric=f"log_fisher_x100_{h}", **estimate(interaction(values))))
            for h in HISTORIES:
                values = {case: readouts[case, h, w]["margin"] for case in ("suppress", "pc_only", "fc_only", "shared_off")}
                output["interactions"].append(dict(dataset=run.dataset, layer="fc2", window=w,
                    metric=f"margin_{h}", **estimate(interaction(values))))


def write_csv(path, values):
    if not values:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "train", "dev", "heldout"), default="all")
    args = parser.parse_args()
    protocol()
    old.immutable(OUT / "search_protocol.json", dict(source_sha256=old.sha(Path(__file__)),
                  primary_protocol=old.sha(OUT / "protocol.json"), ba_guardrail_pp=1., margin_guardrail_sd=.1,
                  gain_repeat_guardrail_pp=.02, paired_concept_draws=4000))
    model_parts = old.initialize()
    selection_path = OUT / "selection.json"
    selection = old.read(selection_path) if selection_path.exists() else fit_and_select(model_parts)
    if args.stage == "train":
        return
    output = {key: [] for key in ("steps", "geometry", "repetition", "readout", "contrasts", "interactions")}
    scopes = ("dev", "heldout") if args.stage == "all" else (args.stage,)
    DOC.mkdir(exist_ok=True)
    for scope in scopes:
        for dataset in DATASETS:
            run = Experiment(dataset, model_parts)
            cases = confirm(run, selection, scope)
            summarize(run, scope, cases, output)
            run.detach()
        for key, values in output.items():
            write_csv(DOC / f"{args.stage}_{key}.csv", values)
        log("REPORT", scope, DOC)
    for name in ("selection", "training_a", "training_b", "protocol", "search_protocol"):
        save_json(DOC / f"{name}.json", old.read(OUT / f"{name}.json"))
    for dataset in DATASETS:
        for name in ("audit", "calibration"):
            save_json(DOC / f"{dataset}_{name}.json", old.read(OUT / dataset / f"{name}.json"))
    save_json(DOC / f"{args.stage}_complete.json", dict(scopes=list(scopes), selection_sha256=old.sha(selection_path)))


if __name__ == "__main__":
    main()
