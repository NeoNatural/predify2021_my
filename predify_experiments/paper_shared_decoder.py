from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .paper_analysis import paired_interval


def _key(result: dict[str, Any]) -> tuple[str, str, str]:
    metadata = result["run_metadata"]
    return (
        str(metadata["degradation_mode"]),
        str(metadata["memory_carrier"]),
        str(metadata["prime_identity"]),
    )


def _load(directory: Path) -> dict[tuple[str, str, str], tuple[Path, dict[str, Any]]]:
    results = {}
    for path in directory.glob("*.pckl"):
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if result.get("paper_schema_version") != 1:
            continue
        results[_key(result)] = (path.resolve(), result)
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_shared_decoder(
    train_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    bootstrap_samples: int,
    max_iter: int = 50_000,
) -> dict[str, Path]:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from scipy.stats import binomtest

    training = _load(train_dir)
    evaluation = _load(evaluation_dir)
    training_by_mode = {
        mode: (path, result)
        for (mode, carrier, identity), (path, result) in training.items()
        if carrier == "none" and identity == "none"
    }
    if not training_by_mode:
        raise ValueError("shared decoder training directory has no target-only results")
    if not evaluation:
        raise ValueError("shared decoder evaluation directory has no paper results")

    all_evaluation_indices = set()
    for _path, result in evaluation.values():
        all_evaluation_indices.update(int(value) for value in result["dataset_indices"])
    for mode, (_path, result) in training_by_mode.items():
        overlap = set(int(value) for value in result["dataset_indices"]) & all_evaluation_indices
        if overlap:
            raise ValueError(
                f"shared decoder train/evaluation image overlap for {mode}: {len(overlap)}"
            )

    result_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    classifiers: dict[tuple[str, str, int], Any] = {}
    predictions: dict[tuple[str, str, str, str, int], np.ndarray] = {}
    correctness: dict[tuple[str, str, str, str, int], np.ndarray] = {}

    layers = ("conv4", "conv5", "fc1", "fc2")
    for mode, (training_path, train_result) in training_by_mode.items():
        mode_evaluations = {
            key: value for key, value in evaluation.items() if key[0] == mode
        }
        if not mode_evaluations:
            continue
        train_categories = np.asarray(train_result["category_labels"], dtype=np.int64)
        first_eval = next(iter(mode_evaluations.values()))[1]
        timesteps = np.asarray(first_eval["target_logits"]).shape[1]
        for layer in layers:
            train_timeline = np.asarray(
                train_result["target_representations"][layer], dtype=np.float32
            )
            if train_timeline.shape[1] != timesteps:
                raise ValueError("training and evaluation timestep counts differ")
            for timestep in range(timesteps):
                classifier = make_pipeline(
                    StandardScaler(),
                    LinearSVC(
                        C=1.0,
                        dual="auto",
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                )
                classifier.fit(train_timeline[:, timestep], train_categories)
                classifiers[(mode, layer, timestep)] = classifier
                fitted_svc = classifier.named_steps["linearsvc"]
                iterations = int(fitted_svc.n_iter_)
                fit_rows.append(
                    {
                        "degradation_mode": mode,
                        "layer": layer,
                        "timestep": timestep,
                        "n_iter": iterations,
                        "max_iter": max_iter,
                        "converged": iterations < max_iter,
                    }
                )
                for (eval_mode, carrier, identity), (path, result) in mode_evaluations.items():
                    features = np.asarray(
                        result["target_representations"][layer][:, timestep],
                        dtype=np.float32,
                    )
                    categories = np.asarray(result["category_labels"], dtype=np.int64)
                    predicted = classifier.predict(features)
                    correct = predicted == categories
                    prediction_key = (eval_mode, carrier, identity, layer, timestep)
                    predictions[prediction_key] = predicted
                    correctness[prediction_key] = correct
                    result_rows.append(
                        {
                            "degradation_mode": eval_mode,
                            "memory_carrier": carrier,
                            "prime_identity": identity,
                            "layer": layer,
                            "timestep": timestep,
                            "accuracy": float(correct.mean()),
                            "n_train": len(train_categories),
                            "n_test": len(categories),
                            "train_source_file": str(training_path),
                            "test_source_file": str(path),
                        }
                    )

    for (mode, carrier, identity), (_path, result) in evaluation.items():
        if identity != "same" or carrier == "none":
            continue
        different_key = (mode, carrier, "different")
        if different_key not in evaluation:
            continue
        different = evaluation[different_key][1]
        if not np.array_equal(result["dataset_indices"], different["dataset_indices"]):
            raise ValueError("shared decoder Same/Different files contain different targets")
        for layer in layers:
            timesteps = np.asarray(result["target_logits"]).shape[1]
            for timestep in range(timesteps):
                same_correct = correctness[(mode, carrier, "same", layer, timestep)]
                different_correct = correctness[
                    (mode, carrier, "different", layer, timestep)
                ]
                mean, low, high = paired_interval(
                    same_correct.astype(float) - different_correct.astype(float),
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                same_only = int(np.sum(same_correct & ~different_correct))
                different_only = int(np.sum(~same_correct & different_correct))
                discordant = same_only + different_only
                if discordant:
                    two_sided_p = float(
                        binomtest(same_only, discordant, 0.5).pvalue
                    )
                    one_sided_p = float(
                        binomtest(
                            same_only,
                            discordant,
                            0.5,
                            alternative="greater",
                        ).pvalue
                    )
                else:
                    two_sided_p = 1.0
                    one_sided_p = 1.0
                effect_rows.append(
                    {
                        "degradation_mode": mode,
                        "memory_carrier": carrier,
                        "contrast": "same_minus_different",
                        "layer": layer,
                        "timestep": timestep,
                        "accuracy_delta_pp": mean,
                        "ci95_low_pp": low,
                        "ci95_high_pp": high,
                        "significantly_positive": low > 0.0,
                        "same_only_correct": same_only,
                        "different_only_correct": different_only,
                        "exact_mcnemar_two_sided_p": two_sided_p,
                        "exact_mcnemar_one_sided_p": one_sided_p,
                    }
                )

    output_paths = {
        "accuracies": output_dir / "shared_decoder_accuracies.csv",
        "effects": output_dir / "shared_decoder_same_vs_different.csv",
        "fit_diagnostics": output_dir / "shared_decoder_fit_diagnostics.csv",
        "classifiers": output_dir / "shared_decoders.pckl",
        "predictions": output_dir / "shared_decoder_predictions.pckl",
    }
    _write_csv(output_paths["accuracies"], result_rows)
    _write_csv(output_paths["effects"], effect_rows)
    _write_csv(output_paths["fit_diagnostics"], fit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_paths["classifiers"].open("wb") as handle:
        pickle.dump(
            {
                "schema_version": 1,
                "seed": seed,
                "linear_svc_max_iter": max_iter,
                "training_directory": str(train_dir.resolve()),
                "evaluation_directory": str(evaluation_dir.resolve()),
                "classifiers": classifiers,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with output_paths["predictions"].open("wb") as handle:
        pickle.dump(
            {
                "schema_version": 1,
                "seed": seed,
                "predictions": predictions,
                "correctness": correctness,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and test a frozen paper-category readout")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=50_000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = analyze_shared_decoder(
        args.train_dir.resolve(),
        args.evaluation_dir.resolve(),
        args.output_dir.resolve(),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        max_iter=args.max_iter,
    )
    for name, path in paths.items():
        print(f"[shared-decoder] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
