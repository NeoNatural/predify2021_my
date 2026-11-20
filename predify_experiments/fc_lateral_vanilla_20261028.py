"""Independent, clean vanilla VGG16 FC cache; never imports the Predify model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Hebbian-CNN/Log/fc_lateral_vanilla_20261028"
DOC = ROOT / "docs/fc_lateral_vanilla_20261028"
IMAGENET = Path("C:/Users/liang/Documents/ImageNet")
THINGS = Path("C:/Users/liang/Documents/ChatGPT/THINGS数据集测试predify_my/data/THINGS")
INVENTORY = ROOT / "docs/things_native_inventory_20261028"
CATEGORIES = ("animals", "clothing", "food", "plants", "vehicles")
NATIVE = dict(zip(("animal", "clothing", "food", "plant", "vehicle"), CATEGORIES))
SEED = 20261028
BROAD = {"plant", "tree", "flower", "animal", "bird", "insect", "vehicle", "food", "clothing", "car", "boat"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def assign_folds(rows, seed=SEED):
    """Concept-disjoint, category-stratified, with greedy image-count balance."""
    rng = random.Random(seed)
    for category in CATEGORIES:
        groups = defaultdict(list)
        for r in rows:
            if r["category"] == category:
                groups[r["concept"]].append(r)
        if len(groups) < 5:
            raise ValueError(f"Fewer than five concepts: {category}")
        names = sorted(groups)
        rng.shuffle(names)
        names.sort(key=lambda c: -len(groups[c]))
        sizes = [0] * 5
        for concept in names:
            candidates = [f for f in range(5) if sizes[f] == min(sizes)]
            fold = rng.choice(candidates)
            for r in groups[concept]:
                r["fold"] = fold
            sizes[fold] += len(groups[concept])
    for i, r in enumerate(rows):
        r["row"] = i
        r["category_index"] = CATEGORIES.index(r["category"])
    for fold in range(5):
        train = {r["concept"] for r in rows if r["fold"] != fold}
        test = {r["concept"] for r in rows if r["fold"] == fold}
        assert not train & test
        assert {r["category"] for r in rows if r["fold"] == fold} == set(CATEGORIES)
    return rows


def prepare_imagenet():
    from torchvision.datasets import ImageNet
    from .paper_stimuli import imagenet_supercategory

    dataset = ImageNet(str(IMAGENET), split="val")
    pools = defaultdict(lambda: defaultdict(list))
    for index, (path, label) in enumerate(dataset.samples):
        category = imagenet_supercategory(label)
        if category in CATEGORIES:
            pools[category][label].append((index, path))
    rng = random.Random(SEED)
    rows = []
    for category in CATEGORIES:
        concepts = sorted(pools[category])
        rng.shuffle(concepts)
        selected = sorted(concepts if category == "plants" else concepts[:35])
        assert len(selected) == (7 if category == "plants" else 35)
        per_concept = 50 if category == "plants" else 10
        for label in selected:
            images = sorted(pools[category][label])
            rng.shuffle(images)
            assert len(images) >= per_concept
            for index, path in images[:per_concept]:
                rows.append(dict(dataset_index=index, path=str(Path(path).resolve()),
                    relative_path=Path(path).relative_to(IMAGENET / "val").as_posix(),
                    concept=dataset.wnids[label], imagenet_label=int(label), category=category,
                    category_policy="existing audited ImageNet supercategory subset", source="imagenet_val"))
    assert len(rows) == 1750
    return assign_folds(rows)


def prepare_things():
    concepts = read_csv(INVENTORY / "concepts.csv")
    images = read_csv(INVENTORY / "images.csv")
    pools = defaultdict(list)
    for r in images:
        if r["eligible"] == "True" and r["historical_image"] == "False" and r["source"] != "source_unknown":
            pools[r["concept"]].append(r)
    rng = random.Random(SEED + 1)
    rows = []
    for native, category in NATIVE.items():
        candidates = sorted(r["concept"] for r in concepts
            if r["manual_exclusive"] == native and r["historical_concept"] == "False"
            and r["concept"] not in BROAD and len(pools[r["concept"]]) >= 10)
        rng.shuffle(candidates)
        if len(candidates) < 20:
            raise ValueError(f"Insufficient native THINGS concepts: {category}: {len(candidates)}")
        lookup = {r["concept"]: r for r in concepts}
        for concept in sorted(candidates[:20]):
            meta = lookup[concept]
            selected = sorted(pools[concept], key=lambda r: r["relative_path"])
            rng.shuffle(selected)
            for r in selected[:10]:
                rows.append(dict(dataset_index=int(r["index"]), path=str(THINGS / "images" / r["relative_path"]),
                    relative_path=r["relative_path"], concept=concept, imagenet_label=-1, category=category,
                    category_policy="native manual exclusive across five categories; no ImageNet label matching",
                    source=r["source"], inventory_sha256=r["sha256"],
                    native_manual=meta["manual_original"], native_human=meta["human_original"],
                    native_wordnet=meta["wordnet_original"],
                    human_agreement=meta["human_exclusive"] == native,
                    wordnet_agreement=meta["wordnet_exclusive"] == native))
    assert len(rows) == 1000
    return assign_folds(rows, SEED + 1)


def prepare():
    manifests = {}
    for name, builder in (("imagenet", prepare_imagenet), ("things", prepare_things)):
        path = OUT / name / "manifest.json"
        if path.exists():
            manifests[name] = json.loads(path.read_text(encoding="utf-8"))
            continue
        rows = builder()
        for r in rows:
            digest = sha(r["path"])
            if "inventory_sha256" in r:
                assert digest == r.pop("inventory_sha256")
            r["sha256"] = digest
        assert len({r["sha256"] for r in rows}) == len(rows)
        manifest = dict(schema_version=1, dataset=name, seed=SEED, categories=CATEGORIES, rows=rows,
            extraction="torchvision vanilla VGG16 IMAGENET1K_V1; eval; one forward; clean standard 224 crop",
            features="post-ReLU fc1=classifier[1], fc2=classifier[4], full 4096 coordinates, FP32",
            exploratory=True, split="five concept-disjoint folds; inner splits must refit every learned transform",
            selection_script_sha256=sha(__file__),
            limitations=["ImageNet plants have only seven fine concepts; categories are not matched between datasets.",
                "THINGS uses native manual labels; other native labels retained for sensitivity; no visual stimulus review claimed.",
                "ImageNet images may overlap earlier exploratory work; fresh extraction is not independent confirmation.",
                "No claim of independence from pretrained ImageNet training semantics or perceptual near duplicates."])
        if name == "things":
            manifest["inventory_sha256"] = {p.name: sha(p) for p in
                (INVENTORY / "concepts.csv", INVENTORY / "images.csv", THINGS / "metadata/concepts-metadata_things.tsv")}
        write_json(path, manifest)
        manifests[name] = manifest
        print("PREPARED", name, len(rows), dict(Counter(r["category"] for r in rows)), flush=True)
    hashes = [{r["sha256"] for r in manifests[d]["rows"]} for d in ("imagenet", "things")]
    assert not hashes[0] & hashes[1], "Cross-dataset byte duplicates"
    return manifests


def extract(manifests, batch_size=32):
    import torch
    import torchvision
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import VGG16_Weights, vgg16

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
    if not checkpoint.exists():
        raise FileNotFoundError("Require the existing cached VGG16 checkpoint; no download")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg16(weights=None)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().to(device)
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()
    captured = {}
    handles = [model.classifier[i].register_forward_hook(
        lambda module, inputs, output, name=name: captured.__setitem__(name, output.detach().clone()))
        for name, i in (("fc1", 1), ("fc2", 4))]

    class Images(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            if sha(r["path"]) != r["sha256"]:
                raise ValueError(f"Source changed: {r['path']}")
            with Image.open(r["path"]) as im:
                return transform(im.convert("RGB")), i

    try:
        for name, manifest in manifests.items():
            dest = OUT / name
            if (dest / "extraction.json").exists():
                assert sha(dest / "activations.npz") == json.loads((dest / "extraction.json").read_text())["cache_sha256"]
                print("VERIFIED EXISTING", name, flush=True)
                continue
            rows = manifest["rows"]
            arrays = {"fc1": np.empty((len(rows), 4096), np.float32),
                      "fc2": np.empty((len(rows), 4096), np.float32),
                      "logits": np.empty((len(rows), 1000), np.float32)}
            started = time.monotonic()
            loader = DataLoader(Images(rows), batch_size=batch_size, shuffle=False, num_workers=0)
            first_error = 0.0
            with torch.inference_mode():
                for images, indices in loader:
                    x = images.to(device)
                    logits = model(x)
                    if int(indices[0]) == 0:
                        a = captured["fc1"].clone()
                        b = captured["fc2"].clone()
                        manual = model.features(x)
                        manual = torch.flatten(model.avgpool(manual), 1)
                        manual_fc1 = model.classifier[:3](manual)
                        manual_fc2 = model.classifier[3:6](manual_fc1)
                        manual_logits = model.classifier[6](manual_fc2)
                        first_error = max(float((a-manual_fc1).abs().max()),
                            float((b-manual_fc2).abs().max()), float((logits-manual_logits).abs().max()))
                        assert first_error < 1e-5
                    ix = indices.numpy()
                    for layer in ("fc1", "fc2"):
                        arrays[layer][ix] = captured[layer].cpu().numpy()
                    arrays["logits"][ix] = logits.cpu().numpy()
                    if int(indices[-1]) // 256 != int(indices[0]) // 256 or int(indices[-1]) == len(rows)-1:
                        print("EXTRACT", name, int(indices[-1])+1, "/", len(rows),
                              "seconds", round(time.monotonic()-started, 1), flush=True)
            arrays.update(category=np.array([r["category_index"] for r in rows], np.int64),
                fold=np.array([r["fold"] for r in rows], np.int64),
                concept=np.array([r["concept"] for r in rows]),
                dataset_index=np.array([r["dataset_index"] for r in rows], np.int64))
            assert all(np.isfinite(arrays[k]).all() for k in ("fc1", "fc2", "logits"))
            assert all((arrays[k] >= 0).all() for k in ("fc1", "fc2"))
            cache = dest / "activations.npz"
            with cache.with_suffix(".tmp").open("wb") as f:
                np.savez_compressed(f, **arrays)
            cache.with_suffix(".tmp").replace(cache)
            metadata = dict(dataset=name, rows=len(rows), seconds=time.monotonic()-started,
                device=str(device), torch=torch.__version__, torchvision=torchvision.__version__,
                checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                manifest_sha256=sha(dest / "manifest.json"), script_sha256=sha(__file__),
                preprocess=repr(transform), cache_sha256=sha(cache), manual_forward_max_error=first_error,
                model="unmodified torchvision vgg16", mode="eval; inference_mode; dropout disabled; no autocast; TF32 disabled",
                statistics={k:dict(mean=float(arrays[k].mean()), nonzero_fraction=float((arrays[k]>0).mean()),
                    zero_vectors=int((np.linalg.norm(arrays[k], axis=1)==0).sum()),
                    dead_units=int((arrays[k].max(0)==0).sum())) for k in ("fc1", "fc2")})
            write_json(dest / "extraction.json", metadata)
            print("SAVED", name, metadata["statistics"], flush=True)
    finally:
        for handle in handles: handle.remove()


def baseline():
    """Fixed-C out-of-fold readout to validate cache usability, not rule selection."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from threadpoolctl import threadpool_limits

    summary = {}
    for dataset in ("imagenet", "things"):
        dest = OUT / dataset
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        rows = manifest["rows"]
        with np.load(dest / "activations.npz") as z:
            a = {key: z[key] for key in z.files}
        assert len(rows) == len(a["category"])
        y, fold, concept = a["category"], a["fold"], a["concept"]
        assert all(rows[i]["row"] == i and rows[i]["concept"] == concept[i]
            and rows[i]["category_index"] == y[i] and rows[i]["fold"] == fold[i] for i in range(len(rows)))
        current = dict(images=len(rows), concepts=len(set(concept)),
            category_images=dict(Counter(r["category"] for r in rows)),
            category_concepts={c:len({r["concept"] for r in rows if r["category"]==c}) for c in CATEGORIES},
            readout="StandardScaler + LinearSVC fixed C=1; train-only scaling; class/concept balanced training weights; five concept-disjoint folds",
            purpose="baseline cache validation; no hyperparameter search or lateral processing", layers={})
        scores_saved = {}
        fitted = {}
        for layer in ("fc1", "fc2"):
            x = a[layer]
            scores = np.full((len(y), 5), np.nan)
            fit_info = []
            fitted[layer] = {}
            with threadpool_limits(limits=4):
                for f in range(5):
                    tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                    assert not set(concept[tr]) & set(concept[te])
                    weights = np.zeros(len(tr))
                    for c in range(5):
                        local = y[tr] == c
                        groups, counts = np.unique(concept[tr][local], return_counts=True)
                        by_group = dict(zip(groups, counts))
                        weights[local] = [1/(len(groups)*by_group[g]) for g in concept[tr][local]]
                    weights *= len(tr)/weights.sum()
                    model = make_pipeline(StandardScaler(), LinearSVC(C=1, dual="auto", max_iter=30000, random_state=SEED))
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(x[tr], y[tr], standardscaler__sample_weight=weights, linearsvc__sample_weight=weights)
                    scores[te] = model.decision_function(x[te])
                    fitted[layer][f] = model
                    fit_info.append(dict(fold=f, train=len(tr), test=len(te),
                        test_per_class=np.bincount(y[te], minlength=5).tolist(), n_iter=int(model[-1].n_iter_),
                        macro_ovr_auc=float(np.mean([roc_auc_score(y[te]==c, scores[te,c]) for c in range(5)]))))
                    print("READOUT", dataset, layer, "fold", f, flush=True)
            assert np.isfinite(scores).all()
            pred = scores.argmax(1)
            # Scale-free angular geometry, reported separately from activity mean.
            raw_norm = np.linalg.norm(x, axis=1)
            assert (raw_norm > 0).all()
            u = x/raw_norm[:, None]
            centers = np.stack([u[y==c].mean(0) for c in range(5)])
            within = float(np.mean([np.square(u[y==c]-centers[c]).sum(1).mean() for c in range(5)]))
            between = float(np.square(centers-centers.mean(0)).sum(1).mean())
            metrics = dict(balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro")),
                macro_ovr_auc=float(np.mean([r["macro_ovr_auc"] for r in fit_info])),
                auc_aggregation="equal mean of within-fold macro OVR AUC; no ranking of margins across different fitted models",
                pooled_margin_macro_ovr_auc_descriptive=float(np.mean([roc_auc_score(y==c, scores[:, c]) for c in range(5)])),
                confusion_matrix=confusion_matrix(y, pred, labels=np.arange(5)).tolist(), fits=fit_info,
                mean_activation=float(x.mean()), normalized_within_scatter=within,
                normalized_between_scatter=between, normalized_fisher_ratio=between/within,
                geometry_scope="descriptive whole-cache baseline; no learned transform and no significance claim")
            current["layers"][layer] = metrics
            scores_saved[layer+"_scores"] = scores
            print("BASELINE", dataset, layer, {k:v for k,v in metrics.items() if isinstance(v, float)}, flush=True)
        np.savez_compressed(dest / "baseline_oof.npz", y=y, fold=fold, concept=concept, **scores_saved)
        models_path = dest / "baseline_models.pckl"
        with models_path.open("wb") as f:
            pickle.dump(dict(models=fitted, manifest_sha256=sha(dest / "manifest.json"),
                cache_sha256=sha(dest / "activations.npz"), source_sha256=sha(__file__),
                purpose="Frozen original-input readout for each outer fold; preprocessing must stay frozen too"), f,
                protocol=pickle.HIGHEST_PROTOCOL)
        current["source_sha256"] = sha(__file__)
        current["models_sha256"] = sha(models_path)
        write_json(dest / "baseline_readout.json", current)
        summary[dataset] = current
    write_json(DOC / "baseline_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "extract", "baseline", "all"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.action in ("prepare", "extract", "all"):
        manifests = prepare()
        if args.action in ("extract", "all"): extract(manifests, args.batch_size)
    if args.action in ("baseline", "all"): baseline()


if __name__ == "__main__": main()
