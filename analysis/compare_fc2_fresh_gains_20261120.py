"""Replay the two frozen FC2 gain operators with identical fresh-cohort inputs."""
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from score_fc2_fresh_replication_20261120 import (
    DATASETS, LATERAL, LOCK, METRICS, MIXING, N_BOOT, OUT, SEEDS, SUPPRESSION,
    MaskedMVPA, PairedConcepts, candidate, digest, fitted, inputs, match_mean,
    measure, original_gate, read, summarize_arms, weights,
)
from priming_familiarity_timecourse_20261119 import apply_gate, frozen, target_direction

CONTRAST = {"lateral_minus_explicit": ("lateral", "explicit")}


def explicit_features(bare, bank, gates):
    cfg = frozen()["configs"]["target_only"]
    assert (cfg["mixing"], cfg["suppression"]) == (MIXING, SUPPRESSION)
    flat = bare.reshape(-1, bare.shape[-1])
    direction = target_direction(bank, flat, cfg)
    output = np.empty((len(bare), len(SEEDS), *bare.shape[1:]), dtype=np.float32)
    for i in range(len(SEEDS)):
        g = gates[:, i].astype(np.float32).reshape(-1)
        output[:, i] = apply_gate(flat, direction, g, cfg).reshape(bare.shape)
    return output


def compare(ds):
    rows = read(LOCK)["datasets"][ds]["rows"]
    scores, bare, cfg, _ = inputs(ds, rows)
    gates, _, calibration = original_gate(ds, scores)
    graph = LATERAL / ds / "graph.npz"
    assert calibration == cfg["original_gate_calibration_sha256"]
    assert digest(graph) == cfg["graph_sha256"]
    means = bare.mean(-1, dtype=np.float64)[:, None] * (1 - SUPPRESSION * gates.astype(np.float64))
    bank, head, _ = fitted(ds, weights())
    labels = np.array([r["label"] for r in rows])
    concepts = np.array([r["concept"] for r in rows])
    sampler, mvpa = PairedConcepts(labels, concepts, N_BOOT), MaskedMVPA(labels, concepts)
    measured = {}
    for arm in ("explicit", "lateral"):
        if arm == "explicit":
            features = explicit_features(bare, bank, gates)
        else:
            features = match_mean(bare, candidate(bare, graph, shuffled=False), gates, means)
        error = abs(features.mean(-1, dtype=np.float64) / means - 1).max()
        print(ds, arm, "max relative mean error", error, flush=True)
        assert error < 1e-6
        measured[arm] = measure(ds, arm, features, head, sampler, mvpa)
        del features
    tables = [pd.DataFrame(rows) for rows in summarize_arms(ds, measured, CONTRAST, (0, 5, 10))]
    keys = ["seed", "history", "step"]
    original = pd.read_csv(OUT / f"{ds}_by_seed.csv")
    def lateral_scores(df):
        return df.loc[df.arm == "lateral"].set_index(keys).sort_index()[list(METRICS)]
    pd.testing.assert_frame_equal(lateral_scores(tables[0]), lateral_scores(original),
                                  check_exact=False, atol=1e-7, rtol=2e-6)
    return tables


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        results = [compare(ds) for ds in DATASETS]
    for name, tables in zip(("by_seed", "points", "windows", "absolute_contrasts"),
                            zip(*results, strict=True), strict=True):
        pd.concat(tables, ignore_index=True).to_csv(OUT / f"gain_{name}.csv", index=False)
