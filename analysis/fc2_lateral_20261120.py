"""Freeze the historical FC2 corr/k32 rule on clean images from the current training split."""
import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from priming_familiarity_scale_data_20261113 import MANIFEST, Renderer
from priming_sharpen_collect_20261112 import CHECKPOINT, digest, vanilla
from priming_sharpen_search_20261112 import read, save
from predify_experiments.fc_lateral_study_20261028 import connection_model
from predify_experiments.fc_sparse_lateral_20261028 import mutual_topk
from predify_experiments.fc_finite_steps_20261028 import trajectories

CACHE = MANIFEST.parent / "fc2_lateral_20261120"
REPO = Path(__file__).resolve().parents[1]
ORI = REPO / "Hebbian-CNN/Log/hebb_fc_native_pilot_20261102"
RULE = dict(rule="corr", graph="k32", mode="hard", gain=.75, budget=1.)
N_UPDATES = 5
PERMUTATION_SEED = 20261120


@torch.inference_mode()
def prepare(ds):
    dest = CACHE / ds
    dest.mkdir(parents=True, exist_ok=True)
    graph_path, audit_path = dest / "graph.npz", dest / "graph.json"
    if audit_path.exists():
        audit = read(audit_path)
        assert audit["manifest_sha256"] == digest(MANIFEST)
        assert audit["graph_sha256"] == digest(graph_path)
        assert audit["source_sha256"] == digest(Path(__file__))
        return
    splits = read(MANIFEST)[ds]["splits"]
    train = splits["train"]
    reserved = splits["validation"] + splits["test"]
    for key in ("index", "sha256", "concept"):
        assert not ({r[key] for r in train} & {r[key] for r in reserved})
    previous = read(ORI / ds / "pilot_manifest.json")["training"]
    previous_train = set(previous["concepts"])
    overlaps = {k: sorted(previous_train & {r["concept"] for r in rows}) for k, rows in splits.items()}
    assert overlaps["test"] or overlaps["validation"], "Re-audit the archive for a compatible graph"

    model, buffers = vanilla()
    render = Renderer(train)
    pieces = []
    for start in range(0, len(train), 12):
        canvases = [render.canvas([row["index"]], [80], None) for row in train[start : start + 12]]
        model(torch.stack(canvases))
        pieces.append(buffers["fc2"].clone())
    x = torch.cat(pieces)
    assert x.shape == (len(train), 4096) and torch.isfinite(x).all() and x.min() >= 0
    model = connection_model(x)
    matrix, stats = mutual_topk(model["matrices"]["corr"], 32)
    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(matrix.shape[0])
    assert torch.equal(matrix, matrix.T) and not matrix.diagonal().any() and matrix.min() >= 0
    assert torch.isclose(matrix.sum(1).max(), matrix.new_tensor(1.))
    assert stats["degree_max"] <= 32
    tmp = graph_path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, graph=matrix.cpu().numpy(), scale=float(model["scale"]), permutation=permutation,
                 train_index=[r["index"] for r in train])
    tmp.replace(graph_path)
    stats.update(variance_floor=model["correlation_variance_floor"], rms=float(model["scale"]),
                 edges=int((matrix > 0).sum()) // 2)
    audit = dict(dataset=ds, n_train=len(train), concepts=len({r["concept"] for r in train}),
                 train_source="clean original central canvases, no preview, standalone frozen VGG16 FC2 ReLU",
                 previous_graph_concept_overlap=overlaps, rule=RULE, updates=N_UPDATES, division=2,
                 relaxation=.5, shuffle_seed=PERMUTATION_SEED, statistics=stats,
                 manifest_sha256=digest(MANIFEST), vgg_sha256=digest(CHECKPOINT), graph_sha256=digest(graph_path),
                 source_sha256=digest(Path(__file__)))
    save(audit_path, audit)
    print("FROZEN GRAPH", ds, stats, flush=True)


@torch.inference_mode()
def candidate(source, graph_path, shuffled):
    with np.load(graph_path) as archive:
        graph = torch.from_numpy(archive["graph"]).cuda()
        rms = float(archive["scale"])
        if shuffled:
            order = torch.from_numpy(archive["permutation"]).cuda()
            graph = graph[order][:, order].contiguous()
    model = dict(scale=rms, matrices={"corr_k32": graph})
    results = []
    base = torch.from_numpy(np.ascontiguousarray(source.reshape(-1, 4096))).cuda()
    for chunk in base.split(512):
        _, response, _ = next(trajectories(chunk, model, RULE, steps=(N_UPDATES,)))
        # Reject zero-drive cases before normalizing the candidate direction.
        assert torch.all(response.mean(1) > 0)
        results.append(response.cpu().numpy())
    return np.concatenate(results).reshape(source.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("things", "imagenet"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=4):
        prepare(args.dataset)
