"""Nested concept-held-out lateral FC experiments on immutable vanilla caches."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from .fc_lateral_vanilla_20261028 import ROOT, OUT as DATA, CATEGORIES, sha, write_json

OUT = ROOT / "Hebbian-CNN/Log/fc_lateral_study_20261028"
DOC = ROOT / "docs/fc_lateral_study_20261028"
SEED = 20261028
RULES = ("raw", "cov", "corr")
INHIBITIONS = ("none", "subtract", "divide", "local")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAINS = (.25, .75)
STRENGTHS = (.5, 1., 2.)
ALPHA = .5
TOL = 1e-5
MAX_STEPS = 100


def configurations():
    configs = [dict(family="baseline", rule="zero", inhibition="none", gain=0., strength=0.),
               dict(family="scale_half", rule="zero", inhibition="scale", gain=0., strength=.5)]
    for inhibition in ("subtract", "divide", "threshold"):
        for strength in STRENGTHS:
            configs.append(dict(family="only_"+inhibition, rule="zero", inhibition=inhibition, gain=0., strength=strength))
    for rule in RULES:
        for inhibition in INHIBITIONS:
            for gain in GAINS:
                for strength in ((0.,) if inhibition == "none" else STRENGTHS):
                    configs.append(dict(family=rule+"_"+inhibition, rule=rule, inhibition=inhibition, gain=gain, strength=strength))
    for i, c in enumerate(configs): c["id"] = i
    return configs


def balanced_weights(y, groups):
    """Labels are used by evaluation/readouts, never connection estimation."""
    weights = np.zeros(len(y), np.float64)
    for c in range(5):
        ix = np.flatnonzero(y == c)
        unique, counts = np.unique(groups[ix], return_counts=True)
        if not len(unique): raise ValueError("Missing readout class")
        sizes = dict(zip(unique, counts))
        weights[ix] = [1/(len(unique)*sizes[g]) for g in groups[ix]]
    return weights * len(y)/weights.sum()


def inner_folds(y, groups, seed):
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1, int)
    for c in range(5):
        unique = np.unique(groups[y == c]); rng.shuffle(unique)
        if len(unique) < 2: raise ValueError("Need at least two training concepts per category")
        loads = [0, 0]
        for group in unique:
            ix = np.flatnonzero(groups == group)
            f = int(np.argmin(loads))
            fold[ix] = f; loads[f] += len(ix)
    assert (fold >= 0).all()
    for f in range(2):
        assert not set(groups[fold == f]) & set(groups[fold != f])
        assert set(y[fold == f]) == set(range(5))
    return fold


def connection_model(train):
    """Only an unlabeled training activation matrix is accepted."""
    scale = train.square().mean().sqrt().clamp_min(1e-8)
    x = train/scale
    mu = x.mean(0)
    second = x.T @ x/len(x)
    covariance = second-mu[:, None]*mu[None, :]
    covariance = (covariance+covariance.T)*.5
    var = covariance.diagonal().clamp_min(0)
    positive_var = var[var > 0]
    floor = (.1*positive_var.median()).clamp_min(1e-6)
    std = (var+floor).sqrt()
    matrices = {"raw": second, "cov": covariance.clamp_min(0),
                "corr": (covariance/(std[:, None]*std[None, :])).clamp_min(0)}
    norms = {}
    for rule in RULES:
        e = matrices[rule].clone()
        e.fill_diagonal_(0)
        # One global gain constraint preserves symmetry and relative input strengths.
        norm = e.sum(1).max().clamp_min(1e-8)
        matrices[rule] = e/norm
        norms[rule] = float(norm)
    return dict(scale=scale, mu=mu, matrices=matrices, normalization=norms,
                correlation_variance_floor=float(floor), input_mean=float(x.mean()))


def dynamics(original, model, config, matrix=None):
    x = original/model["scale"]
    inhibition = config["inhibition"]
    if inhibition == "scale":
        return original*config["strength"], dict(steps=0, residual=0., converged=True)
    if inhibition == "threshold":
        r = torch.relu(x-config["strength"]*model["input_mean"])
        return r*model["scale"], dict(steps=0, residual=0., converged=True)
    gain, strength = config["gain"], config["strength"]
    if gain == 0 and strength == 0:
        return original.clone(), dict(steps=0, residual=0., converged=True)
    e = matrix if matrix is not None else model["matrices"].get(config["rule"])
    bias = gain*(e @ model["mu"]) if inhibition == "local" else None
    r = x.clone()
    residual = math.inf
    for step in range(1, MAX_STEPS+1):
        excitation = gain*(r @ e.T) if gain else 0.
        drive = x+excitation
        population = r.mean(1, keepdim=True)
        if inhibition == "local": drive = drive-bias-strength*population
        elif inhibition == "subtract": drive = drive-strength*population
        target = torch.relu(drive)
        if inhibition == "divide": target = target/(1+strength*population)
        updated = (1-ALPHA)*r+ALPHA*target
        # Convergence is checked against the fixed point, not just a finite last step.
        if step % 5 == 0:
            residual = float(((updated-r).norm(dim=1)/(x.norm(dim=1)+1e-8)).max())
            if not math.isfinite(residual) or float(updated.abs().max()) > 1e5: break
            if residual < TOL:
                r = updated
                break
        r = updated
    return r*model["scale"], dict(steps=step, residual=residual,
        converged=bool(math.isfinite(residual) and residual < TOL))


def geometry(r, y, detailed=False):
    norm = r.norm(dim=1)
    zeros = float((norm <= 1e-8).float().mean())
    u = r/norm.clamp_min(1e-8)[:, None]
    centers = torch.stack([u[y == c].mean(0) for c in range(5)])
    within = torch.stack([(u[y == c]-centers[c]).square().sum(1).mean() for c in range(5)]).mean()
    between = (centers-centers.mean(0)).square().sum(1).mean()
    activation = torch.stack([r[y == c].mean() for c in range(5)]).mean()
    result = dict(activation=float(activation), raw_mean=float(r.mean()), within=float(within),
        between=float(between), fisher=float(between/within.clamp_min(1e-12)), zero_fraction=zeros,
        nonzero=float((r > 1e-7).float().mean()))
    if detailed:
        per_unit = r.mean(0)
        mean_square = r.square().mean(1)
        centered = u-u.mean(0)
        gram = centered @ centered.T
        effective_dim = gram.trace().square()/gram.square().sum().clamp_min(1e-12)
        result.update(effective_dimension=float(effective_dim),
            population_activity_fraction=float((r.mean(1).square()/mean_square.clamp_min(1e-12)).mean()),
            largest_unit_activity_share=float(per_unit.max()/per_unit.sum().clamp_min(1e-12)),
            per_class_activation=[float(r[y == c].mean()) for c in range(5)],
            per_class_within=[float((u[y == c]-centers[c]).square().sum(1).mean()) for c in range(5)])
    return result


def classification(scores, y):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    scores = np.asarray(scores)
    pred = scores.argmax(1)
    return dict(accuracy=float(balanced_accuracy_score(y, pred)),
        auc=float(np.mean([roc_auc_score(y == c, scores[:, c]) for c in range(5)])),
        macro_f1=float(f1_score(y, pred, average="macro")),
        recalls=[float((pred[y == c] == c).mean()) for c in range(5)],
        confusion_matrix=confusion_matrix(y, pred, labels=np.arange(5)).tolist())


def ridge_scores(train, test, y, groups):
    """Fixed-regularization linear ridge readout; all preprocessing train-only."""
    weights = torch.tensor(balanced_weights(y, groups), dtype=train.dtype, device=train.device)
    mu = (weights[:, None]*train).sum(0)/weights.sum()
    var = ((train-mu).square()*weights[:, None]).sum(0)/weights.sum()
    std = var.sqrt()
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    z = (train-mu)/std
    q = (test-mu)/std
    root = weights.sqrt()
    z = z*root[:, None]
    labels = torch.nn.functional.one_hot(torch.as_tensor(y, dtype=torch.long, device=train.device), 5).to(train.dtype)
    prior = (weights[:, None]*labels).sum(0)/weights.sum()
    target = (labels-prior)*root[:, None]
    kernel = (z @ z.T)/train.shape[1]
    kernel.diagonal().add_(1.)
    dual = torch.linalg.solve(kernel, target)
    scores = ((q @ z.T)/train.shape[1]) @ dual+prior
    return scores.cpu().numpy()


def effects(metrics, baseline):
    return dict(activation_pct=100*(metrics["activation"]/baseline["activation"]-1),
        within_pct=100*(metrics["within"]/baseline["within"]-1),
        between_pct=100*(metrics["between"]/baseline["between"]-1),
        fisher_pct=100*(metrics["fisher"]/baseline["fisher"]-1),
        accuracy_pp=100*(metrics["accuracy"]-baseline["accuracy"]),
        auc_delta=metrics["auc"]-baseline["auc"])


def joint_score(metrics, baseline):
    # Maximize the weakest of the three requested sharpening changes.
    if metrics["zero_fraction"] > 0 or not metrics.get("converged", True): return -1e6
    terms = [math.log(baseline["activation"]/max(metrics["activation"], 1e-20)),
        math.log(baseline["within"]/max(metrics["within"], 1e-20)),
        math.log(max(metrics["fisher"], 1e-20)/baseline["fisher"])]
    return min(terms)


def choose(candidate_rows, configs):
    choices = {}
    for family in sorted({c["family"] for c in configs}):
        options = []
        for config in (c for c in configs if c["family"] == family):
            rows = [r for r in candidate_rows if r["config"]["id"] == config["id"]]
            assert len(rows) == 2
            stable = all(r["metrics"]["converged"] and r["metrics"]["zero_fraction"] == 0 for r in rows)
            scores = [joint_score(r["metrics"], r["baseline"]) for r in rows]
            deltas = [effects(r["metrics"], r["baseline"]) for r in rows]
            guard = stable and np.mean([r["accuracy_pp"] for r in deltas]) >= -2. and np.mean([r["auc_delta"] for r in deltas]) >= -.005
            score = float(np.mean(scores)) if stable else -1e6
            options.append(dict(config=config, score=score, readout_guard=bool(guard), stable=bool(stable),
                inner_joint=bool(stable and score > 1e-6 and guard),
                inner_effects={k:float(np.mean([r[k] for r in deltas])) for k in deltas[0]}))
        # First jointly valid, then readout-preserving, then stable; never inspect outer labels.
        choices[family] = max(options, key=lambda o:(o["inner_joint"], o["readout_guard"], o["stable"], o["score"], -o["config"]["gain"], -o["config"]["strength"]))
    learned = [v for k,v in choices.items() if k.split("_")[0] in RULES]
    choices["selected"] = copy.deepcopy(max(learned, key=lambda o:(o["inner_joint"], o["readout_guard"], o["stable"], o["score"])))
    return choices


def matched_controls(original, selected):
    scaled = original*(selected.mean(1)/original.mean(1).clamp_min(1e-8))[:, None]
    counts = (selected > 1e-7).sum(1)
    available = (original > 0).sum(1)
    counts = torch.minimum(counts, available).clamp_min(1)
    ordered = original.sort(dim=1, descending=True).values
    hi = ordered.gather(1, (counts-1)[:, None]).squeeze(1)
    lo = ordered.gather(1, counts.clamp_max(original.shape[1]-1)[:, None]).squeeze(1)
    cutoff = (hi+lo)*.5
    threshold = torch.relu(original-cutoff[:, None])
    # Ties can prevent exact support matching; retain the mismatch diagnostic.
    mismatch = float(((threshold > 1e-7).sum(1)-(selected > 1e-7).sum(1)).abs().float().mean())
    return scaled, threshold, mismatch


def svm_fit(train, test, y, groups):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    x, q = train.cpu().numpy(), test.cpu().numpy()
    weights = balanced_weights(y, groups)
    model = make_pipeline(StandardScaler(), LinearSVC(C=1, dual="auto", max_iter=30000, random_state=SEED))
    with threadpool_limits(limits=4), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y, standardscaler__sample_weight=weights, linearsvc__sample_weight=weights)
        scores = model.decision_function(q)
    return scores, model


def freeze():
    protocol = dict(version=1, exploratory=True, seed=SEED, configs=configurations(),
        datasets=["imagenet", "things"], layers=["fc1", "fc2"], outer_folds=5, inner_folds=2,
        connection_learning="unlabeled image-uniform training moments only; symmetric nonnegative zero diagonal; max row-sum normalization",
        dynamics=dict(alpha=ALPHA, max_steps=MAX_STEPS, tolerance=TOL, persistent_input=True),
        selection="within each family maximize mean inner-fold minimum log improvement of activation/W/Fisher; joint-first then readout guard then stability; selected chooses among learned families using inner data only",
        readout_guard="inner BA drop <=2pp and macro OVR AUC drop <=0.005; selection guard, not a statistical noninferiority claim",
        ridge="train-only weighted StandardScaler; class/concept balanced; K=ZZ'/4096; fixed dual regularization 1; intercept from weighted labels",
        svm="fixed C=1 confirmation for inner-selected Hebbian, selected inhibition-only subtract, and density-matched threshold; frozen original SVM for all outer conditions",
        inference="within outer fold geometry and AUC; five folds are descriptive robustness units, not independent subjects; no significance claim",
        original_model_sources={p:sha(ROOT/p) for p in ["predify2021/model_factory/get_model.py", "predify2021/model_factory/pvgg16_shared.py", "Hebbian-CNN/Hebbian_VGG_Lib.py", "Hebbian-CNN/Hebb_Priming_Utils.py"]},
        data={d:{p:sha(DATA/d/p) for p in ["manifest.json", "activations.npz", "baseline_models.pckl"]} for d in ["imagenet", "things"]},
        source_sha256=sha(__file__))
    path = OUT / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        assert previous == json.loads(json.dumps(protocol)), "Frozen protocol/source changed; use a new version directory"
    else:
        write_json(path, protocol)
        source = OUT / "source_snapshot.py"
        source.write_bytes(Path(__file__).read_bytes())
    return protocol


def run(dataset, layer):
    protocol = freeze()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(DATA/dataset/"activations.npz") as z:
        x = torch.as_tensor(z[layer], device=DEVICE)
        y, groups, folds = z["category"], z["concept"], z["fold"]
    yt = torch.as_tensor(y, device=DEVICE)
    with (DATA/dataset/"baseline_models.pckl").open("rb") as f:
        frozen_heads = pickle.load(f)["models"][layer]
    configs = configurations()
    started = time.monotonic()
    for fold in range(5):
        dest = OUT/dataset/layer/f"fold{fold}"
        if (dest/"complete.json").exists():
            print("EXISTS", dataset, layer, fold, flush=True); continue
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(groups[train]) & set(groups[test])
        inner = inner_folds(y[train], groups[train], SEED+fold)
        candidate_rows = []
        for inner_id in range(2):
            tr, va = train[inner != inner_id], train[inner == inner_id]
            assert not set(groups[tr]) & set(groups[va])
            model = connection_model(x[tr])
            baseline_metrics = None
            for config in configs:
                r, diagnostic = dynamics(x[train], model, config)
                # The contiguous order is outer-train, with inner subsets indexed locally.
                tr_local, va_local = np.flatnonzero(inner != inner_id), np.flatnonzero(inner == inner_id)
                met = geometry(r[va_local], yt[va])
                scores = ridge_scores(r[tr_local], r[va_local], y[tr], groups[tr])
                met.update(classification(scores, y[va]), **diagnostic)
                if config["family"] == "baseline": baseline_metrics = met
                candidate_rows.append(dict(inner=inner_id, config=config, metrics=met, baseline=baseline_metrics))
            print("INNER", dataset, layer, fold, inner_id, "candidates", len(configs),
                "elapsed", round(time.monotonic()-started, 1), flush=True)
        choices = choose(candidate_rows, configs)
        write_json(dest/"selection.json", dict(train_rows=train.tolist(), test_rows=test.tolist(),
            inner_fold=inner.tolist(), choices=choices, candidates=candidate_rows))
        model = connection_model(x[train])
        ordered = np.concatenate([train, test])
        transformed, diagnostics = {}, {}
        for family, choice in choices.items():
            if family == "selected": continue
            transformed[family], diagnostics[family] = dynamics(x[ordered], model, choice["config"])
        chosen_family = choices["selected"]["config"]["family"]
        transformed["selected"] = transformed[chosen_family]
        diagnostics["selected"] = diagnostics[chosen_family]
        rng = np.random.default_rng(SEED+fold)
        perm = torch.as_tensor(rng.permutation(x.shape[1]), device=DEVICE)
        chosen_e = model["matrices"][choices["selected"]["config"]["rule"]]
        shuffled = chosen_e[perm][:, perm]
        transformed["selected_shuffle"], diagnostics["selected_shuffle"] = dynamics(x[ordered], model, choices["selected"]["config"], shuffled)
        scaled, threshold, mismatch = matched_controls(x[ordered], transformed["selected"])
        transformed["selected_matched_scale"], transformed["selected_matched_threshold"] = scaled, threshold
        diagnostics["selected_matched_scale"] = dict(steps=0, residual=0., converged=True)
        diagnostics["selected_matched_threshold"] = dict(steps=0, residual=0., converged=True, support_mismatch_mean=mismatch)
        results, saved_scores, frozen_scores, arrays, models = {}, {}, {}, {}, {}
        confirm = {"selected", "only_subtract", "selected_matched_threshold"}
        for family, r in transformed.items():
            a, b = r[:len(train)], r[len(train):]
            met = geometry(b, yt[test], detailed=True)
            scores = ridge_scores(a, b, y[train], groups[train])
            met.update(classification(scores, y[test]), **diagnostics[family])
            fixed = frozen_heads[fold].decision_function(b.cpu().numpy())
            met["frozen_svm"] = classification(fixed, y[test])
            if family in confirm:
                svm_scores, head = svm_fit(a, b, y[train], groups[train])
                met["refit_svm"] = classification(svm_scores, y[test])
                models[family] = head
                saved_scores[family+"_svm"] = svm_scores
            if family == "baseline":
                met["refit_svm"] = met["frozen_svm"]
            results[family] = met
            saved_scores[family] = scores
            frozen_scores[family] = fixed
            arrays[family] = b.cpu().numpy()
        baseline_metrics = results["baseline"]
        for family, met in results.items():
            met["effects"] = effects(met, baseline_metrics)
            met["descriptive_joint"] = bool(joint_score(met, baseline_metrics) > 1e-6)
        # Pure rescaling must preserve angular geometry and refitted standardized readout.
        for family in ("scale_half", "selected_matched_scale"):
            assert abs(results[family]["within"]/baseline_metrics["within"]-1) < 2e-5
            assert abs(results[family]["fisher"]/baseline_metrics["fisher"]-1) < 2e-5
        assert np.allclose(saved_scores["scale_half"], saved_scores["baseline"], atol=2e-4)
        write_json(dest/"results.json", dict(dataset=dataset, layer=layer, fold=fold,
            train_size=len(train), test_size=len(test), choices=choices, results=results,
            connection_normalization=model["normalization"], correlation_variance_floor=model["correlation_variance_floor"]))
        np.savez_compressed(dest/"test_representations.npz", row=test, y=y[test], concept=groups[test], **arrays)
        np.savez_compressed(dest/"readout_scores.npz", row=test, y=y[test], **saved_scores,
            **{k+"_frozen":v for k,v in frozen_scores.items()})
        with (dest/"confirmation_svm_models.pckl").open("wb") as f:
            pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(dest/"complete.json", dict(seconds=time.monotonic()-started,
            files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}, protocol_sha256=sha(OUT/"protocol.json")))
        print("OUTER COMPLETE", dataset, layer, fold, "selected", choices["selected"]["config"],
            "effects", results["selected"]["effects"], "elapsed", round(time.monotonic()-started, 1), flush=True)
        del transformed, model, shuffled, chosen_e
    for path, digest in protocol["original_model_sources"].items(): assert sha(ROOT/path) == digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--dataset", choices=("imagenet", "things"))
    parser.add_argument("--layer", choices=("fc1", "fc2"))
    args = parser.parse_args()
    if args.action == "freeze":
        p = freeze(); print("FROZEN", len(p["configs"]), "configurations", flush=True)
    else:
        with torch.inference_mode():
            for dataset in ([args.dataset] if args.dataset else ["imagenet", "things"]):
                for layer in ([args.layer] if args.layer else ["fc1", "fc2"]): run(dataset, layer)


if __name__ == "__main__": main()
