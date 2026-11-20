"""Recover full trajectories for the frozen November 13 selection on its paired test trials."""

import os
import argparse

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from priming_familiarity_scale_fc2_20261113 import (
    frozen,
    Controls,
    Experiment,
    old,
    Familiarity,
    read,
    SELECTION,
    SEEDS,
    MANIFEST,
    run_path,
    digest,
    OUT,
    gate,
    Renderer,
    fitted,
    weights,
    target_direction,
    apply_gate,
)
from priming_familiarity_scale_collect_20261113 import BatchExperiment, BATCH, SHARD, pack, shard_path
from predify_experiments.sequence import run_recurrent_sequence
from predify_experiments.tripartite_20261027 import restore_state, state_snapshot

DEST = MANIFEST.parent / "timecourse_20261119"
NO_CARRY = DEST.parent / "timecourse_no_carry_20261119"


class Timeline(BatchExperiment):
    def step(self, inputs, steps):
        if steps != 10:
            return super().step(inputs, steps)

        def capture(t, _):
            self.conv.append(self.run.layers[1].record["signed"].to(torch.float16))
            self.fc.append(self.run.layers[3].record["signed"].clone())

        self.conv, self.fc = [], []
        return run_recurrent_sequence(
            self.run.model,
            inputs,
            15,
            hebbian_layers=self.run.c.layers,
            commit_update=False,
            preserve_model_state=True,
            after_step=capture,
        )


class ResetBeforeTarget(Timeline):
    """Save the expanded trial-start grey state, then restore only PCoder tensors."""

    gray_state = None

    def step(self, inputs, steps):
        if self.gray_state is None:
            assert steps == 2
            self.gray_state = state_snapshot(self.run.model)
        if steps == 10:

            def memory_signature():
                signature = []
                for layer in self.run.layers:
                    memory = layer.memory
                    assert memory.writes == 2
                    assert memory.frozen if layer.spatial else memory.phase == "target"
                    names = ("intra", "trace", "inter") if layer.spatial else ("W",)
                    signature.extend(
                        (getattr(memory, n).data_ptr(), getattr(memory, n)._version) for n in names
                    )
                return signature

            before = memory_signature()
            restore_state(self.run.model, self.gray_state)
            for k, saved in enumerate(self.gray_state, 1):
                pc = getattr(self.run.model, f"pcoder{k}")
                assert all(torch.equal(getattr(pc, name), value) for name, value in saved.items())
            assert before == memory_signature()
            self.gray_state = None
        return super().step(inputs, steps)


def collect(no_carry=False):
    parts = old.initialize()
    cfg = frozen()
    chosen = read(SELECTION)
    for ds in ["things", "imagenet"]:
        rows = read(MANIFEST)[ds]["splits"]["test"]
        run = Experiment(ds, parts)
        run.fit()
        original = [getattr(run.model, f"pcoder{k}").forward for k in range(1, 6)]
        timeline = ResetBeforeTarget if no_carry else Timeline
        batch = timeline(run, Controls(tuple(cfg["hebbian_lambdas"])))
        bank, _, _ = fitted(ds, weights())
        render = Renderer(rows)
        models = []
        ref = []
        cal = read(OUT / f"{ds}_fc2_calibration.json")["calibrations"]
        for seed in SEEDS:
            path = run_path(ds, "trainable_encoder", 16, seed)
            assert digest(path.with_suffix(".pth")) == chosen["model_sha256"][ds][path.name]
            model = Familiarity().cuda().eval()
            model.load_state_dict(torch.load(path.with_suffix(".pth"), weights_only=True))
            models.append(model)
            with np.load(path.with_suffix(".native_test.npz")) as f:
                ref.append(f["scores"])
        try:
            folder = (NO_CARRY if no_carry else DEST) / ds
            folder.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(rows), SHARD):
                dest = folder / f"{start:04}.npz"
                if dest.exists():
                    continue
                seg = rows[start : start + SHARD]
                features = []
                with np.load(shard_path(ds, "test", start)) as f:
                    ids = f["index"]
                    np.testing.assert_array_equal(ids, [r["index"] for r in seg])
                    cache = f["memory"] if no_carry else f["native_fc2"]
                for j in range(0, len(seg), BATCH):
                    x, first, last = pack(render, seg[j : j + BATCH])
                    size = len(x)
                    histories = []
                    for h, order in enumerate(([0, 1, 2], [1, 0, 2], [2, 0, 1])):
                        _, mem, _ = batch.collect(x, first, last[:, order])
                        raw = torch.stack(batch.fc, 1).cpu().numpy()
                        if no_carry:
                            np.testing.assert_array_equal(mem[:, 0], cache[j : j + size, h])
                        else:
                            np.testing.assert_allclose(
                                raw[:, 9], cache[j : j + size, h], atol=1e-5, rtol=1e-5
                            )
                        flat = raw.reshape(-1, raw.shape[-1])
                        direction = target_direction(bank, flat, cfg["configs"]["target_only"])
                        with torch.no_grad():
                            conv = torch.stack(batch.conv, 1)
                            p = torch.from_numpy(mem[:, 0]).cuda()
                            vals = []
                            for k, (seed, model) in enumerate(zip(SEEDS, models)):
                                z = model.encode(conv)
                                sc = (
                                    model.read_encoded(z, model.reference(z), model.memory(p)[:, None])
                                    .cpu()
                                    .numpy()
                                )
                                if not no_carry:
                                    np.testing.assert_allclose(
                                        sc[:, 9],
                                        ref[k][start + j : start + j + size, h],
                                        atol=2e-5,
                                        rtol=2e-5,
                                    )
                                c = cal[f"selected_{seed}"]
                                g = gate(sc, c["threshold"], c["slope"]).ravel()
                                vals.append(
                                    apply_gate(flat, direction, g, cfg["configs"]["target_only"]).reshape(
                                        raw.shape
                                    )
                                )
                        histories.append(np.stack(vals, 1))
                    features.append(np.stack(histories, 2))
                tmp = dest.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.savez(f, index=ids, features=np.concatenate(features))
                tmp.replace(dest)
                print(ds, start + len(seg), "/", len(rows), flush=True)
        finally:
            run.detach()
            for k, fn in enumerate(original, 1):
                getattr(run.model, f"pcoder{k}").forward = fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-carry", action="store_true")
    with threadpool_limits(limits=4):
        collect(parser.parse_args().no_carry)
