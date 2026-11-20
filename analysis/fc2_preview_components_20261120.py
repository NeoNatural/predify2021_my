"""Training-only FC2 category axes and held-out within-concept residual distances."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from bootstrap_pcoder_carry_20261119 import N_BOOT, summarize
from collect_fc2_fresh_replication_20261120 import CACHE as FRESH, LOCK
from priming_familiarity_timecourse_20261119 import DEST, NO_CARRY, MANIFEST, SEEDS, digest, read
from score_familiarity_timecourse_20261119 import DOC, HISTORIES, PairedConcepts

OUT = DOC / "preview_components"
WINDOWS = {"t0": [0], "early": range(5), "middle": range(5, 10), "late": range(10, 15)}
PAIRS = {"exact-different": (0, 2), "same_concept-different": (1, 2), "same_concept-exact": (1, 0)}


def shards(folder, key, rows):
    offset = 0
    for path in sorted(folder.glob("[0-9][0-9][0-9][0-9].npz")):
        with np.load(path) as f:
            n = len(f["index"])
            assert path.stem == f"{offset:04d}"
            np.testing.assert_array_equal(f["index"], [r["index"] for r in rows[offset:offset+n]])
            yield f[key]
            offset += n
    assert offset == len(rows)


def basis(ds, rows):
    x = np.concatenate(list(shards(MANIFEST.parent / ds / "train", "fc2", rows))).astype(float)
    labels, concepts = (np.array([r[k] for r in rows]) for k in ("label", "concept"))
    classes = np.unique(labels)
    np.testing.assert_array_equal(classes, np.arange(len(classes)))
    centers = []
    for k in classes:
        groups = np.unique(concepts[labels == k])
        centers.append(np.mean([x[(labels == k) & (concepts == g)].mean(0) for g in groups], axis=0))
    centers = np.stack(centers)
    origin = centers.mean(0)
    centered = centers - origin
    u = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    assert s[-2] > 1e-8 * s[0] and s[-1] < 1e-10 * s[0]
    q = vt[:-1]
    np.testing.assert_allclose(q @ q.T, np.eye(len(q)), atol=1e-12)
    np.testing.assert_allclose((centered @ q.T) @ q, centered, atol=1e-12)
    return origin, u, q


def residual_distance(residual, concepts):
    # Each image contributes its average squared distance to different photos of its concept.
    # One half of the mean pairwise squared distance equals the unbiased within-concept variance.
    out = np.empty(residual.shape[:-1])
    for c in np.unique(concepts):
        idx = np.flatnonzero(concepts == c)
        assert len(idx) >= 2
        r = residual[idx]
        centered = r - r.mean(axis=0)
        sq = (centered ** 2).mean(axis=-1)
        out[idx] = len(idx) / (2 * (len(idx)-1)) * (sq + sq.mean(axis=0))
    return out


def components(x, labels, concepts, axes):
    # Work one history and seed at a time. Keep actual amplitudes for the primary analysis.
    origin, target_axes, q = axes
    z = x.astype(np.float64) - origin
    n_units = x.shape[-1]
    scores = z @ q.T
    residual = z - scores @ q
    total = np.mean(z*z, axis=-1)
    outside = np.mean(residual*residual, axis=-1)
    np.testing.assert_allclose(total, (scores*scores).sum(-1)/n_units + outside, atol=1e-12)
    rms = np.sqrt(total)
    assert rms.min() > 0
    shared = np.einsum("ntd,nd->nt", z, target_axes[labels]) / np.sqrt(n_units)
    return dict(shared=shared, residual_rms=np.sqrt(outside), photo_var=residual_distance(residual, concepts),
                shared_alignment=shared/rms, photo_var_unit=residual_distance(residual/rms[..., None], concepts))


def draw_metrics(folder, key, rows, axes, expected_seeds):
    labels, concepts = (np.array([r[k] for r in rows]) for k in ("label", "concept"))
    x = np.concatenate(list(shards(folder, key, rows)))
    if x.ndim == 4:
        x = x[:, None]
    assert x.shape == (len(rows), expected_seeds, 3, 15, 4096) and np.isfinite(x).all()
    w = PairedConcepts(labels, concepts, N_BOOT).concept_weights
    result = {}
    for h in range(3):
        channels = {}
        for k in range(expected_seeds):
            for name, per_image in components(x[:, k, h], labels, concepts, axes).items():
                if name not in channels:
                    channels[name] = per_image.copy()
                else:
                    channels[name] += per_image
        for name, values in channels.items():
            if name not in result:
                result[name] = np.empty((len(w), 3, 15))
            result[name][:, h] = w @ (values / expected_seeds)
    return result


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read(MANIFEST)
    fresh = read(LOCK)
    assert fresh["original_sha256"] == digest(MANIFEST)
    curves, effects, stats, audits = [], [], [], {}
    for ds in ("things", "imagenet"):
        splits = manifest[ds]["splits"]
        train, test, recent = splits["train"], splits["test"], fresh["datasets"][ds]["rows"]
        for held in (splits["validation"], test, recent):
            for k in ("index", "sha256", "concept"):
                assert {r[k] for r in train}.isdisjoint(r[k] for r in held)
        assert {r["sha256"] for r in recent}.isdisjoint(r["sha256"] for r in test)
        axes = basis(ds, train)
        np.savez(OUT / f"{ds}_training_axes.npz", origin=axes[0], target_axes=axes[1], basis=axes[2],
                 categories=manifest[ds]["categories"], train_index=[r["index"] for r in train])
        audits[ds] = dict(train=len(train), train_concepts=len({r["concept"] for r in train}),
                          categories=manifest[ds]["categories"], rank=len(axes[2]), paired=len(test),
                          fresh=len(recent), paired_concepts=len({r["concept"] for r in test}),
                          fresh_concepts=len({r["concept"] for r in recent}),
                          training_sha256={p.name:digest(p) for p in sorted((MANIFEST.parent/ds/"train").glob("*.npz"))})
        draws = {}
        for arm, folder, key, rows, seeds in (
            ("fresh_carry", FRESH/ds, "fc2", recent, 1),
            ("carry", DEST/ds, "features", test, len(SEEDS)),
            ("reset", NO_CARRY/ds, "features", test, len(SEEDS)),
        ):
            print(ds, arm, flush=True)
            draws[arm] = draw_metrics(folder, key, rows, axes, seeds)
            for m, vals in draws[arm].items():
                for h, name in enumerate(HISTORIES):
                    for t in range(15):
                        curves.append(dict(dataset=ds, arm=arm, history=name, step=t, metric=m,
                                           **summarize(vals[:, h, t])))
        for arm in (*draws, "carry-reset"):
            for m in draws["carry"]:
                v = (draws["carry"][m]-draws["reset"][m]) if arm == "carry-reset" else draws[arm][m]
                for contrast, (h, r) in PAIRS.items():
                    delta = v[:, h] - v[:, r]
                    tags = dict(dataset=ds, arm=arm, comparison=contrast, metric=m)
                    for t in range(15):
                        effects.append(dict(**tags, step=t, **summarize(delta[:, t])))
                    for name, steps in WINDOWS.items():
                        stats.append(dict(**tags, window=name, **summarize(delta[:, steps].mean(axis=1))))
    for key, values in (("absolute", curves), ("contrasts", effects), ("windows", stats)):
        result = pd.DataFrame(values)
        assert np.isfinite(result.select_dtypes("number")).all().all()
        result.to_csv(OUT/f"{key}.csv", index=False)
    (OUT/"audit.json").write_text(json.dumps(dict(datasets=audits, manifest=digest(MANIFEST), fresh=digest(LOCK),
            script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), bootstrap=N_BOOT),
            ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=3):
        collect()
