"""Replay the archived carry trajectory, saving FC2 before terminal modulation."""

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from priming_familiarity_timecourse_20261119 import (
    DEST,
    MANIFEST,
    old,
    Controls,
    Experiment,
    frozen,
    Timeline,
    Renderer,
    read,
    pack,
    BATCH,
    SHARD,
    shard_path,
)

RAW = DEST.parent / "timecourse_unmodulated_20261120"


def collect():
    parts = old.initialize()
    cfg = frozen()
    for ds in ("things", "imagenet"):
        rows = read(MANIFEST)[ds]["splits"]["test"]
        run = Experiment(ds, parts)
        run.fit()
        original = [getattr(run.model, f"pcoder{k}").forward for k in range(1, 6)]
        batch = Timeline(run, Controls(tuple(cfg["hebbian_lambdas"])))
        renderer = Renderer(rows)
        folder = RAW / ds
        folder.mkdir(exist_ok=True, parents=True)
        try:
            for start in range(0, len(rows), SHARD):
                path = folder / f"{start:04d}.npz"
                if path.exists():
                    continue
                segment = rows[start : start + SHARD]
                with np.load(shard_path(ds, "test", start)) as z:
                    ids, t9 = z["index"], z["native_fc2"]
                np.testing.assert_array_equal(ids, [row["index"] for row in segment])
                pieces = []
                for j in range(0, len(segment), BATCH):
                    x, first, last = pack(renderer, segment[j : j + BATCH])
                    histories = []
                    for h, order in enumerate(([0, 1, 2], [1, 0, 2], [2, 0, 1])):
                        batch.collect(x, first, last[:, order])
                        data = torch.stack(batch.fc, 1).cpu().numpy()
                        np.testing.assert_allclose(
                            data[:, 9], t9[j : j + len(x), h], atol=1e-5, rtol=1e-5
                        )
                        histories.append(data)
                    pieces.append(np.stack(histories, 1))
                features = np.concatenate(pieces)[:, None]
                with np.load(DEST / ds / path.name) as z:
                    np.testing.assert_array_equal(ids, z["index"])
                    scale = z["features"].mean(-1) / features.mean(-1)
                assert (
                    scale.min() >= 1 - cfg["configs"]["target_only"]["suppression"] - 2e-6
                    and scale.max() <= 1 + 2e-6
                )
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as stream:
                    np.savez(stream, index=ids, features=features)
                tmp.replace(path)
                print(ds, start + len(segment), "/", len(rows), flush=True)
        finally:
            run.detach()
            for k, forward in enumerate(original, 1):
                getattr(run.model, f"pcoder{k}").forward = forward


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        collect()
