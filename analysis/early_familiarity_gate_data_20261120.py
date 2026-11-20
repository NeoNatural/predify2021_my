"""Lock new photos and collect native Conv5 scores for the early gate experiment."""
import argparse
from collections import Counter
from copy import copy
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fc2_lateral_20261120 import CACHE as LATERAL
from hebb_chain_20261111 import PREVIEW
from priming_familiarity_expansion_data_20261118 import (
    REGISTRY as PREVIOUS, add_previews, candidates, earlier, photos, source_manifest,
)
from priming_familiarity_timecourse_20261119 import (
    BATCH, MANIFEST, OUT as ORIGINAL_CAL, SEEDS, SELECTION, SHARD,
    Controls, Experiment, Familiarity, Renderer, digest, frozen, gate, old,
    pack, read, run_path, run_recurrent_sequence, restore_state, shard_path, state_snapshot,
)
from priming_familiarity_scale_collect_20261113 import BatchExperiment, reset_hebbian_state
from priming_sharpen_search_20261112 import save
from score_fc2_lateral_20261120 import DESTINATION

DATASETS = ("things", "imagenet")
CACHE = MANIFEST.parent / "early_fc2_gate_20261120"
OUT = DESTINATION / "early_gate"
LOCK = CACHE / "fresh_stimuli.json"
STEPS = tuple(range(5))


def register():
    fixed = dict(original_sha256=digest(MANIFEST), expansion_sha256=digest(PREVIOUS))
    if LOCK.exists():
        doc = read(LOCK)
        assert all(doc[k] == value for k, value in fixed.items())
        return doc["datasets"]
    base = read(MANIFEST)
    expansion = read(PREVIOUS)["datasets"]
    plans = {}
    for ds in DATASETS:
        rng = np.random.default_rng(2026112031)
        sets = base[ds]["splits"]
        previous = expansion[ds]["fresh"] + expansion[ds]["added"]["B"] + expansion[ds]["added"]["C"]
        used = previous + [r for rows in sets.values() for r in rows]
        idx = {r["index"] for r in used}
        exposed, forbidden = earlier(ds, source_manifest(ds))
        forbidden = forbidden | {r["sha256"] for r in used}
        exposed = exposed | {r["concept"] for r in used}
        dev = {r["concept"] for r in sets["train"] + sets["validation"]}
        categories, pool = candidates(ds)
        rows = []
        for n, label in enumerate(base[ds]["categories"]):
            options = sorted(concept for concept in pool if categories[concept] == label and concept not in dev)
            order = sorted(rng.permutation(options).tolist(), key=lambda c: c in exposed)
            n_chosen = 0
            for concept in order:
                indices, hashes = idx.copy(), forbidden.copy()
                group = pool[concept]
                ordered = [group[i] for i in rng.permutation(len(group))]
                new = photos(ordered, indices, hashes, 4, fresh=True)
                if len(new) < 2:
                    continue
                rows.extend(dict(r, concept=concept, label=n, rank=n_chosen,
                                 historically_exposed_concept=concept in exposed) for r in new)
                idx, forbidden = indices, hashes
                n_chosen += 1
                if n_chosen == 6:
                    break
            assert n_chosen >= 1
        add_previews(rows, rows, rng)
        assert not dev & {r["concept"] for r in rows}
        assert len({r["sha256"] for r in rows}) == len(rows)
        concepts = Counter(base[ds]["categories"][j] for j, _ in {(r["label"], r["concept"]) for r in rows})
        counts = Counter(base[ds]["categories"][r["label"]] for r in rows)
        plans[ds] = dict(rows=rows, concepts=dict(concepts), photos=dict(counts),
                         historically_seen_concepts=sorted({r["concept"] for r in rows
                                                           if r["historically_exposed_concept"]}))
        print("LOCKED", ds, len(rows), dict(concepts), dict(counts), flush=True)
    save(LOCK, dict(**fixed, datasets=plans))
    OUT.mkdir(exist_ok=True, parents=True)
    save(OUT / "fresh_inventory.json", {ds: {k: v for k, v in plan.items() if k != "rows"}
                                        for ds, plan in plans.items()})
    return plans


def original_models(ds):
    selected, calibrations = read(SELECTION), read(ORIGINAL_CAL / f"{ds}_fc2_calibration.json")
    assert calibrations["selection_sha256"] == digest(SELECTION)
    models = []
    for seed in SEEDS:
        path = run_path(ds, "trainable_encoder", 16, seed)
        checkpoint = path.with_suffix(".pth")
        assert digest(checkpoint) == selected["model_sha256"][ds][path.name]
        model = Familiarity().cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        models.append(model)
    return models, [calibrations["calibrations"][f"selected_{s}"] for s in SEEDS]


class NativeEarly(BatchExperiment):
    """Fork each native history from the same first-preview state and one-write memory."""
    @torch.no_grad()
    def native(self, target, first, last, models, n_steps):
        run = self.run
        run.c.reset()
        reset_hebbian_state(run.c.layers)
        for k in range(1, 6):
            pc = getattr(run.model, f"pcoder{k}")
            for name in ("rep", "prd", "grd"):
                tensor = getattr(pc, name)
                setattr(pc, name, tensor.expand(len(target), *tensor.shape[1:]).contiguous())
        for layer in run.layers:
            layer.prepare(len(target))

        run.controls(PREVIEW)
        self.step(first, 2)
        first_map = run.layers[1].record["signed"].to(torch.float16)
        for layer in run.layers:
            layer.write(layer.record["signed"])
        prefix = state_snapshot(run.model)
        memories = [copy(layer.memory) for layer in run.layers]
        observations, features = [], []
        for h in range(3):
            restore_state(run.model, prefix)
            for layer, memory in zip(run.layers, memories, strict=True):
                layer.memory = copy(memory)
                if not layer.spatial:
                    layer.memory.W = memory.W.clone()
            run.controls(PREVIEW)
            self.step(last[:, h], 2)
            second = run.layers[1].record["signed"].to(torch.float16)
            for layer in run.layers:
                layer.write(layer.record["signed"])
                layer.memory.freeze()
            run.controls(self.controls)
            conv, fc = [], []

            def capture(t, _):
                conv.append(run.layers[1].record["signed"].to(torch.float16))
                fc.append(run.layers[3].record["signed"].clone())

            run_recurrent_sequence(run.model, target, n_steps, hebbian_layers=run.c.layers,
                                   commit_update=False, preserve_model_state=True, after_step=capture)
            states = torch.stack(conv, 1)
            preview = torch.stack((first_map, second), 1)
            values = []
            for model in models:
                encoded = model.encode(states)
                values.append(model.read_encoded(encoded, model.reference(encoded), model.memory(preview)[:, None]))
            observations.append(torch.stack(values, 1))
            features.append(torch.stack(fc, 1))
        return torch.stack(observations, 2).cpu().numpy(), torch.stack(features, 1).cpu().numpy()


def check_dataset(ds, base, run, models, calibration, controls):
    rows = base["splits"]["test"]
    size = BATCH
    scores, replay = run.native(*pack(Renderer(rows), rows[:size]), models, n_steps=10)
    archive = MANIFEST.parent / "timecourse_unmodulated_20261120" / ds / "0000.npz"
    with np.load(archive) as f:
        np.testing.assert_array_equal(f["index"][:size], [r["index"] for r in rows[:size]])
        x = f["features"][:size, 0, :, :10]
    np.testing.assert_allclose(replay, x, atol=1e-5, rtol=1e-5)
    errors = []
    for j, seed in enumerate(SEEDS):
        path = run_path(ds, "trainable_encoder", 16, seed)
        with np.load(path.with_suffix(".native_test.npz")) as f:
            score = f["scores"][:size]
        np.testing.assert_allclose(scores[:, j, :, 9], score, atol=2e-5, rtol=2e-5)
        errors.append(float(np.max(np.abs(scores[:, j, :, 9] - score))))
    with np.load(LATERAL / ds / "gate_reconstruction.npz") as f:
        archived_gate = f["g"][:size, ..., :len(STEPS)]
    reconstructed = np.stack([gate(scores[:, j, :, :len(STEPS)], cal["threshold"], cal["slope"])
                              for j, cal in enumerate(calibration)], 1)
    np.testing.assert_allclose(reconstructed, archived_gate, atol=2e-4, rtol=0)
    print("NATIVE AUDIT", ds, errors, flush=True)
    save(CACHE / ds / "native_audit.json", dict(fc2_max_error=float(np.max(np.abs(replay - x))),
             t9_score_max_errors=errors, early_gate_max_error=float(np.max(np.abs(reconstructed - archived_gate)))))


def collect(ds, split):
    plans = register()
    base = read(MANIFEST)[ds]
    seal = dict(split=split, lock_sha256=digest(LOCK), source_sha256=digest(Path(__file__)))
    if split == "fresh":
        freeze = OUT / f"{ds}_calibration.json"
        assert freeze.is_file(), "Calibrate and freeze on validation before opening the fresh trial set"
        seal["calibration_sha256"] = digest(freeze)
    rows = base["splits"]["validation"] if split == "validation" else plans[ds]["rows"]
    target = CACHE / ds / split
    target.mkdir(exist_ok=True, parents=True)
    record = target / "protocol.json"
    if record.is_file():
        assert read(record) == seal
    else:
        save(record, seal)
    controls = Controls(tuple(frozen()["hebbian_lambdas"]))
    run = Experiment(ds, old.initialize())
    run.fit()
    models, calibration = original_models(ds)
    batch = NativeEarly(run, controls)
    try:
        if not (CACHE / ds / "native_audit.json").is_file():
            check_dataset(ds, base, batch, models, calibration, controls)
        renderer = Renderer(rows)
        for start in range(0, len(rows), SHARD):
            chunk = rows[start : start + SHARD]
            indices = [r["index"] for r in chunk]
            path = target / f"{start:04d}.npz"
            if path.exists():
                with np.load(path) as f:
                    np.testing.assert_array_equal(f["index"], indices)
                continue
            parts = [batch.native(*pack(renderer, chunk[i:i + BATCH]), models, n_steps=len(STEPS))
                     for i in range(0, len(chunk), BATCH)]
            scores, fc2 = (np.concatenate(arrays) for arrays in zip(*parts))
            assert scores.shape == (len(indices), len(SEEDS), 3, len(STEPS))
            assert np.isfinite(scores).all() and np.isfinite(fc2).all()
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.savez(f, index=indices, scores=scores, fc2=fc2)
            tmp.replace(path)
            print("CACHE", ds, split, start + len(chunk), "/", len(rows), flush=True)
    finally:
        run.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("split", choices=("validation", "fresh"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        collect(args.dataset, args.split)
