"""Audit the independent vanilla cache and saved out-of-fold readout."""
from pathlib import Path
import json
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predify_experiments.fc_lateral_vanilla_20261028 import OUT, DOC, sha, CATEGORIES, write_json


def main():
    import torch
    from threadpoolctl import threadpool_limits

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    w3, b3 = (weights[f"classifier.3.{k}"].to(device) for k in ("weight", "bias"))
    w6, b6 = (weights[f"classifier.6.{k}"].to(device) for k in ("weight", "bias"))
    report = dict(passed=True, datasets={},
        scope="source and cache integrity, full layer equations, concept folds, frozen readout replay; no candidate lateral experiment")
    source_hashes = {sha(p) for p in (OUT / "source_snapshot").glob("*.py")}
    source_hashes.add(sha(ROOT / "predify_experiments/fc_lateral_vanilla_20261028.py"))
    all_hashes = []
    for name, expected in (("imagenet", 1750), ("things", 1000)):
        base = OUT / name
        m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        e = json.loads((base / "extraction.json").read_text(encoding="utf-8"))
        baseline = json.loads((base / "baseline_readout.json").read_text(encoding="utf-8"))
        assert sha(base / "manifest.json") == e["manifest_sha256"]
        assert sha(base / "activations.npz") == e["cache_sha256"]
        assert sha(checkpoint) == e["checkpoint_sha256"]
        assert e["script_sha256"] in source_hashes and m["selection_script_sha256"] in source_hashes
        assert baseline["source_sha256"] in source_hashes
        assert baseline["models_sha256"] == sha(base / "baseline_models.pckl")
        with np.load(base / "activations.npz") as z:
            a = {k:z[k] for k in z.files}
        rows = m["rows"]
        assert len(rows) == expected
        for key, field in (("dataset_index", "dataset_index"), ("category", "category_index"), ("concept", "concept"), ("fold", "fold")):
            assert np.array_equal(a[key], [r[field] for r in rows])
        assert a["fc1"].shape == a["fc2"].shape == (expected, 4096)
        assert a["logits"].shape == (expected, 1000)
        for key in ("fc1", "fc2", "logits"):
            assert a[key].dtype == np.float32 and np.isfinite(a[key]).all()
        for key in ("fc1", "fc2"):
            assert (a[key] >= 0).all() and (np.linalg.norm(a[key], axis=1) > 0).all()
        error2 = error6 = 0.0
        with torch.inference_mode():
            for first in range(0, expected, 128):
                x1 = torch.from_numpy(a["fc1"][first:first+128]).to(device)
                x2 = torch.from_numpy(a["fc2"][first:first+128]).to(device)
                y = torch.from_numpy(a["logits"][first:first+128]).to(device)
                error2 = max(error2, float((torch.relu(x1 @ w3.T+b3)-x2).abs().max()))
                error6 = max(error6, float((x2 @ w6.T+b6-y).abs().max()))
        assert error2 < 1e-4 and error6 < 1e-4
        folds = []
        for f in range(5):
            train = [r for r in rows if r["fold"] != f]
            test = [r for r in rows if r["fold"] == f]
            assert not {r["concept"] for r in train} & {r["concept"] for r in test}
            assert not {r["sha256"] for r in train} & {r["sha256"] for r in test}
            folds.append(dict(fold=f, images=len(test),
                categories={c:sum(r["category"] == c for r in test) for c in CATEGORIES}))
        with (base / "baseline_models.pckl").open("rb") as f:
            fitted = pickle.load(f)
        assert fitted["cache_sha256"] == e["cache_sha256"]
        assert fitted["manifest_sha256"] == e["manifest_sha256"]
        replay_error = 0.0
        with np.load(base / "baseline_oof.npz") as z, threadpool_limits(limits=4):
            for layer in ("fc1", "fc2"):
                for f in range(5):
                    te = a["fold"] == f
                    predicted = fitted["models"][layer][f].decision_function(a[layer][te])
                    replay_error = max(replay_error, float(np.max(np.abs(predicted-z[layer+"_scores"][te]))))
        assert replay_error < 1e-8
        label_status = None
        if name == "things":
            assert all(r["source"] not in ("imagenet", "source_unknown") and r["imagenet_label"] == -1 for r in rows)
            concepts = {r["concept"]:r for r in rows}
            label_status = dict(agree=sum(r["human_agreement"] for r in concepts.values()),
                missing=sum(not r["native_human"].strip() for r in concepts.values()),
                different_or_nonexclusive=sum(not r["human_agreement"] and bool(r["native_human"].strip()) for r in concepts.values()))
        hashes = {r["sha256"] for r in rows}
        assert len(hashes) == expected
        all_hashes.append(hashes)
        report["datasets"][name] = dict(images=expected, cache_bytes=(base / "activations.npz").stat().st_size,
            fc1_to_fc2_max_error=error2, fc2_to_logits_max_error=error6,
            frozen_probe_replay_max_error=replay_error, folds=folds, native_human_label_status_concepts=label_status)
    assert not all_hashes[0] & all_hashes[1]
    report["cross_dataset_byte_duplicates"] = 0
    write_json(DOC / "cache_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
