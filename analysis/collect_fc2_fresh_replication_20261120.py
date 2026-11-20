"""Extend the locked fresh native trajectories through target step 14."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from early_familiarity_gate_data_20261120 import (
    BATCH, CACHE as EARLY_CACHE, Controls, DATASETS, Experiment, LATERAL, LOCK, MANIFEST,
    NativeEarly, ORIGINAL_CAL, Renderer, SEEDS, SELECTION, SHARD, digest, frozen, gate, old,
    original_models, pack, read, save,
)
from score_fc2_lateral_20261120 import DESTINATION

CACHE = EARLY_CACHE.parent / "fresh_lateral_20261120"
OUT = DESTINATION / "fresh_replication"
STEPS = tuple(range(15))
MIXING = .45
SUPPRESSION = .003


def audit_legacy(ds, runner, models, calibrations):
    test = read(MANIFEST)[ds]["splits"]["test"]
    rows = test[:BATCH]
    scores, x = runner.native(*pack(Renderer(test), rows), models, n_steps=len(STEPS))
    archive = MANIFEST.parent / "timecourse_unmodulated_20261120" / ds / "0000.npz"
    index = [r["index"] for r in rows]
    with np.load(archive) as f:
        np.testing.assert_array_equal(f["index"][:BATCH], index)
        original = f["features"][:BATCH, 0]
    np.testing.assert_allclose(x, original, atol=1e-5, rtol=1e-5)
    with np.load(LATERAL / ds / "gate_reconstruction.npz") as f:
        np.testing.assert_array_equal(f["index"][:BATCH], index)
        g = f["g"][:BATCH]
    replay = np.stack([gate(scores[:, j], c["threshold"], c["slope"]) for j, c in enumerate(calibrations)], 1)
    np.testing.assert_allclose(replay, g, atol=2e-4, rtol=0)
    return dict(index=index, fc2_error=float(np.max(np.abs(x - original))),
                gate_error=float(np.max(np.abs(replay - g))), raw_sha256=digest(archive))


def compare_early(ds, name, index, scores, fc2):
    early = EARLY_CACHE / ds / "fresh" / name
    with np.load(early) as ref:
        np.testing.assert_array_equal(ref["index"], index)
        original_scores, original_fc2 = ref["scores"], ref["fc2"]
    five = original_scores.shape[-1]
    assert five == 5
    np.testing.assert_allclose(scores[..., :five], original_scores, atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(fc2[:, :, :five], original_fc2)
    return dict(early_sha256=digest(early), max_score_error=float(np.max(np.abs(scores[..., :five] - original_scores))))


def collect(ds):
    rows = read(LOCK)["datasets"][ds]["rows"]
    assert len(rows) == {"things": 97, "imagenet": 96}[ds]
    target = CACHE / ds
    target.mkdir(exist_ok=True, parents=True)
    graph = LATERAL / ds / "graph.npz"
    graph_hash = digest(graph)
    assert read(graph.with_suffix(".json"))["graph_sha256"] == graph_hash
    controls = frozen()
    assert controls["configs"]["target_only"]["mixing"] == MIXING
    assert controls["configs"]["target_only"]["suppression"] == SUPPRESSION
    plan = dict(dataset=ds, n_targets=len(rows), frozen_stimuli_sha256=digest(LOCK),
                selection_sha256=digest(SELECTION), source_sha256=digest(Path(__file__)),
                original_gate_calibration_sha256=digest(ORIGINAL_CAL / f"{ds}_fc2_calibration.json"),
                graph_sha256=graph_hash, carry=True, seeds=list(SEEDS), steps=list(STEPS),
                mixing=MIXING, suppression=SUPPRESSION,
                arms=["lateral", "scale", "shuffled"],
                primary=dict(history="exact", reference="same_concept", start=10, end=14, metric="fisher",
                             controls=["scale", "shuffled"]))
    protocol = target / "protocol.json"
    if protocol.exists() and (any(target.glob("*.npz")) or (target / "collection.json").exists()):
        assert read(protocol) == plan
    else:
        save(protocol, plan)
    run = Experiment(ds, old.initialize())
    run.fit()
    models, calibrations = original_models(ds)
    runner = NativeEarly(run, Controls(tuple(controls["hebbian_lambdas"])))
    try:
        refs = audit_legacy(ds, runner, models, calibrations)
        renderer = Renderer(rows)
        shards = {}
        for start in range(0, len(rows), SHARD):
            segment = rows[start:start + SHARD]
            index = [r["index"] for r in segment]
            name = f"{start:04d}.npz"
            path = target / name
            if path.is_file():
                with np.load(path) as f:
                    np.testing.assert_array_equal(f["index"], index)
                    scores, features = f["scores"], f["fc2"]
            else:
                pieces = [runner.native(*pack(renderer, segment[i:i+BATCH]), models, n_steps=len(STEPS))
                          for i in range(0, len(segment), BATCH)]
                scores, features = (np.concatenate(group) for group in zip(*pieces))
            assert scores.shape == (len(index), len(SEEDS), 3, len(STEPS))
            assert features.shape == (len(index), 3, len(STEPS), 4096)
            assert np.isfinite(scores).all() and np.isfinite(features).all()
            check = compare_early(ds, name, index, scores, features)
            if not path.exists():
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as stream:
                    np.savez(stream, index=index, scores=scores, fc2=features)
                tmp.replace(path)
            shards[name] = dict(**check, sha256=digest(path))
            print("NATIVE 0–14", ds, start + len(segment), "/", len(rows), flush=True)
        save(target / "collection.json", dict(protocol_sha256=digest(protocol), legacy_check=refs, shards=shards))
    finally:
        run.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    with threadpool_limits(limits=4):
        collect(parser.parse_args().dataset)
