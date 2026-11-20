from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def identity_grouped_folds(
    categories: np.ndarray,
    identities: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    dummy = np.zeros((len(categories), 1), dtype=np.float32)
    return list(splitter.split(dummy, categories, groups=identities))


def cross_validated_linear_svm(
    features: np.ndarray,
    categories: np.ndarray,
    identities: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return identity-grouped out-of-fold predictions and correctness."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    features = np.asarray(features, dtype=np.float32)
    categories = np.asarray(categories, dtype=np.int64)
    identities = np.asarray(identities, dtype=np.int64)
    predictions = np.full(len(categories), -1, dtype=np.int64)
    split_indices = identity_grouped_folds(
        categories, identities, folds=folds, seed=seed
    )
    for train, test in split_indices:
        classifier = make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=seed),
        )
        classifier.fit(features[train], categories[train])
        predictions[test] = classifier.predict(features[test])
    if np.any(predictions < 0):
        raise RuntimeError("linear SVM did not produce an out-of-fold prediction")
    correctness = predictions == categories
    return predictions, correctness, float(correctness.mean())


@lru_cache(maxsize=16)
def _bootstrap_weights(length: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, length, size=(samples, length))
    counts = np.zeros((samples, length), dtype=np.float64)
    row_indices = np.repeat(np.arange(samples), length)
    np.add.at(counts, (row_indices, draws.ravel()), 1.0)
    counts /= float(length)
    counts.setflags(write=False)
    return counts


def paired_intervals(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap paired means for one or many value columns at once."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not len(values):
        raise ValueError("paired_intervals requires a non-empty [samples, metrics] array")
    distribution = _bootstrap_weights(len(values), samples, seed) @ values
    distribution *= scale
    low, high = np.quantile(distribution, (0.025, 0.975), axis=0)
    mean = values.mean(axis=0) * scale
    # Resampling a constant column has zero uncertainty. Avoid BLAS summation
    # roundoff creating a spurious interval around that constant.
    constant = np.all(values == values[0], axis=0)
    mean[constant] = values[0, constant] * scale
    low[constant] = mean[constant]
    high[constant] = mean[constant]
    return mean, low, high


def paired_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    scale: float = 1.0,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired_interval requires a non-empty vector")
    mean, low, high = paired_intervals(
        values,
        samples=samples,
        seed=seed,
        scale=scale,
    )
    return float(mean[0]), float(low[0]), float(high[0])


def decode_result(
    result: dict[str, Any],
    *,
    folds: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], np.ndarray]]:
    categories = np.asarray(result["category_labels"])
    identities = np.asarray(result["object_identity_labels"])
    rows: list[dict[str, Any]] = []
    correctness: dict[tuple[str, int], np.ndarray] = {}
    metadata = result["run_metadata"]
    for layer, timeline in result["target_representations"].items():
        timeline = np.asarray(timeline)
        for timestep in range(timeline.shape[1]):
            _, correct, accuracy = cross_validated_linear_svm(
                timeline[:, timestep],
                categories,
                identities,
                folds=folds,
                seed=seed,
            )
            correctness[(layer, timestep)] = correct
            rows.append(
                {
                    "condition": metadata["condition"],
                    "prime_identity": metadata["prime_identity"],
                    "memory_carrier": metadata["memory_carrier"],
                    "degradation_mode": metadata["degradation_mode"],
                    "degradation_level": metadata["degradation_level"],
                    "layer": layer,
                    "timestep": timestep,
                    "decoding_accuracy": accuracy,
                    "chance_accuracy": 1.0 / len(np.unique(categories)),
                }
            )
    return rows, correctness


def _result_key(result: dict[str, Any]) -> tuple[str, float, str, str, str]:
    metadata = result["run_metadata"]
    return (
        str(metadata["degradation_mode"]),
        float(metadata["degradation_level"]),
        str(metadata["memory_carrier"]),
        str(metadata["condition"]),
        str(metadata["prime_identity"]),
    )


def _load_results(paths: Iterable[Path]) -> dict[tuple[str, float, str, str, str], tuple[Path, dict[str, Any]]]:
    indexed = {}
    for path in paths:
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if result.get("paper_schema_version") != 1:
            continue
        indexed[_result_key(result)] = (path.resolve(), result)
    return indexed


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aligned_reference_indices(
    result: dict[str, Any], reference: dict[str, Any]
) -> np.ndarray:
    reference_lookup = {
        int(dataset_index): offset
        for offset, dataset_index in enumerate(reference["dataset_indices"])
    }
    try:
        return np.asarray(
            [reference_lookup[int(value)] for value in result["dataset_indices"]],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"feedforward reference is missing dataset index {error.args[0]}"
        ) from error


def analyze_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    folds: int,
    seed: int,
    bootstrap_samples: int,
    feedforward_dir: Path | None = None,
) -> dict[str, Path]:
    indexed = _load_results(input_dir.glob("*.pckl"))
    decoding_rows: list[dict[str, Any]] = []
    decoded: dict[tuple[str, float, str, str, str], dict[tuple[str, int], np.ndarray]] = {}
    for key, (path, result) in indexed.items():
        rows, correctness = decode_result(result, folds=folds, seed=seed)
        for row in rows:
            row["source_file"] = str(path)
        decoding_rows.extend(rows)
        decoded[key] = correctness

    effect_rows: list[dict[str, Any]] = []
    for key, (_path, result) in indexed.items():
        mode, level, carrier, condition, identity = key
        if condition not in {"one_back", "two_back"} or identity != "same":
            continue
        control_key = (mode, level, carrier, "no_prime", "none")
        if control_key not in indexed:
            continue
        control = indexed[control_key][1]
        for layer, timeline in result["target_representations"].items():
            control_timeline = np.asarray(control["target_representations"][layer])
            timeline = np.asarray(timeline)
            for timestep in range(timeline.shape[1]):
                decoding_delta = (
                    decoded[key][(layer, timestep)].astype(float)
                    - decoded[control_key][(layer, timestep)].astype(float)
                )
                decode_mean, decode_low, decode_high = paired_interval(
                    decoding_delta,
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                activation = np.mean(np.abs(timeline[:, timestep]), axis=1)
                control_activation = np.mean(
                    np.abs(control_timeline[:, timestep]), axis=1
                )
                activation_delta_pct = (
                    (activation - control_activation)
                    / np.maximum(np.abs(control_activation), 1e-12)
                    * 100.0
                )
                act_mean, act_low, act_high = paired_interval(
                    activation_delta_pct,
                    samples=bootstrap_samples,
                    seed=seed,
                )
                top1_delta = (
                    np.asarray(result["target_top1_correct"])[:, timestep].astype(float)
                    - np.asarray(control["target_top1_correct"])[:, timestep].astype(float)
                )
                top1_mean, top1_low, top1_high = paired_interval(
                    top1_delta,
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                top5_delta = (
                    np.asarray(result["target_top5_correct"])[:, timestep].astype(float)
                    - np.asarray(control["target_top5_correct"])[:, timestep].astype(float)
                )
                top5_mean, top5_low, top5_high = paired_interval(
                    top5_delta,
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                effect_rows.append(
                    {
                        "degradation_mode": mode,
                        "degradation_level": level,
                        "memory_carrier": carrier,
                        "condition_vs_no_prime": condition,
                        "layer": layer,
                        "timestep": timestep,
                        "decoding_delta_pp": decode_mean,
                        "decoding_ci95_low_pp": decode_low,
                        "decoding_ci95_high_pp": decode_high,
                        "activation_delta_pct": act_mean,
                        "activation_ci95_low_pct": act_low,
                        "activation_ci95_high_pct": act_high,
                        "top1_delta_pp": top1_mean,
                        "top1_ci95_low_pp": top1_low,
                        "top1_ci95_high_pp": top1_high,
                        "top5_delta_pp": top5_mean,
                        "top5_ci95_low_pp": top5_low,
                        "top5_ci95_high_pp": top5_high,
                        "complete_paper_phenotype": (
                            decode_low > 0.0 and act_high < 0.0
                        ),
                    }
                )

    causal_rows: list[dict[str, Any]] = []
    causal_recovery_rows: list[dict[str, Any]] = []
    for key, (_path, result) in indexed.items():
        mode, level, carrier, condition, identity = key
        if condition not in {"one_back", "two_back"} or identity != "same":
            continue
        different_key = (mode, level, carrier, condition, "different")
        if different_key not in indexed:
            continue
        different = indexed[different_key][1]
        for layer, timeline in result["target_representations"].items():
            timeline = np.asarray(timeline)
            different_timeline = np.asarray(
                different["target_representations"][layer]
            )
            for timestep in range(timeline.shape[1]):
                decoding_delta = (
                    decoded[key][(layer, timestep)].astype(float)
                    - decoded[different_key][(layer, timestep)].astype(float)
                )
                decode_mean, decode_low, decode_high = paired_interval(
                    decoding_delta,
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                activation = np.mean(np.abs(timeline[:, timestep]), axis=1)
                different_activation = np.mean(
                    np.abs(different_timeline[:, timestep]), axis=1
                )
                activation_delta_pct = (
                    (activation - different_activation)
                    / np.maximum(np.abs(different_activation), 1e-12)
                    * 100.0
                )
                act_mean, act_low, act_high = paired_interval(
                    activation_delta_pct,
                    samples=bootstrap_samples,
                    seed=seed,
                )
                top1_delta = (
                    np.asarray(result["target_top1_correct"])[:, timestep].astype(float)
                    - np.asarray(different["target_top1_correct"])[:, timestep].astype(float)
                )
                top1_mean, top1_low, top1_high = paired_interval(
                    top1_delta,
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                top5_delta = (
                    np.asarray(result["target_top5_correct"])[:, timestep].astype(float)
                    - np.asarray(different["target_top5_correct"])[:, timestep].astype(float)
                )
                top5_mean, top5_low, top5_high = paired_interval(
                    top5_delta,
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                causal_rows.append(
                    {
                        "degradation_mode": mode,
                        "degradation_level": level,
                        "memory_carrier": carrier,
                        "condition": condition,
                        "contrast": "same_minus_different",
                        "layer": layer,
                        "timestep": timestep,
                        "decoding_delta_pp": decode_mean,
                        "decoding_ci95_low_pp": decode_low,
                        "decoding_ci95_high_pp": decode_high,
                        "activation_delta_pct": act_mean,
                        "activation_ci95_low_pct": act_low,
                        "activation_ci95_high_pct": act_high,
                        "top1_delta_pp": top1_mean,
                        "top1_ci95_low_pp": top1_low,
                        "top1_ci95_high_pp": top1_high,
                        "top5_delta_pp": top5_mean,
                        "top5_ci95_low_pp": top5_low,
                        "top5_ci95_high_pp": top5_high,
                    }
                )
        if (
            result.get("representation_recovery") is not None
            and different.get("representation_recovery") is not None
        ):
            if not np.array_equal(result["dataset_indices"], different["dataset_indices"]):
                raise ValueError("Same/Different recovery files contain different targets")
            for recovery_kind in ("representation_recovery", "prediction_recovery"):
                same_values = np.asarray(result[recovery_kind])
                different_values = np.asarray(different[recovery_kind])
                raw_deltas = same_values - different_values
                metric_signs = np.asarray(
                    [
                        1.0 if name == "cosine" else -1.0
                        for name in result["recovery_metric_names"]
                    ]
                )
                oriented_deltas = raw_deltas * metric_signs[None, None, None, :]
                flat_shape = raw_deltas.shape[1:]
                raw_mean, raw_low, raw_high = paired_intervals(
                    raw_deltas.reshape(len(raw_deltas), -1),
                    samples=bootstrap_samples,
                    seed=seed,
                )
                improve_mean, improve_low, improve_high = paired_intervals(
                    oriented_deltas.reshape(len(oriented_deltas), -1),
                    samples=bootstrap_samples,
                    seed=seed,
                )
                for timestep in range(same_values.shape[1]):
                    for pcoder_offset, pcoder_index in enumerate(result["pcoder_indices"]):
                        for metric_offset, metric_name in enumerate(
                            result["recovery_metric_names"]
                        ):
                            flat_index = np.ravel_multi_index(
                                (timestep, pcoder_offset, metric_offset), flat_shape
                            )
                            causal_recovery_rows.append(
                                {
                                    "degradation_mode": mode,
                                    "degradation_level": level,
                                    "memory_carrier": carrier,
                                    "condition": condition,
                                    "contrast": "same_minus_different",
                                    "recovery_kind": recovery_kind,
                                    "pcoder": pcoder_index,
                                    "timestep": timestep,
                                    "metric": metric_name,
                                    "raw_same_minus_different": raw_mean[flat_index],
                                    "raw_ci95_low": raw_low[flat_index],
                                    "raw_ci95_high": raw_high[flat_index],
                                    "oriented_recovery_advantage": improve_mean[flat_index],
                                    "advantage_ci95_low": improve_low[flat_index],
                                    "advantage_ci95_high": improve_high[flat_index],
                                    "same_significantly_better": improve_low[flat_index] > 0.0,
                                }
                            )

    recovery_rows: list[dict[str, Any]] = []
    recovery_change_rows: list[dict[str, Any]] = []
    internal_error_rows: list[dict[str, Any]] = []
    internal_error_change_rows: list[dict[str, Any]] = []
    for key, (path, result) in indexed.items():
        if result.get("representation_recovery") is None:
            continue
        mode, level, carrier, condition, identity = key
        for recovery_kind in ("representation_recovery", "prediction_recovery"):
            values = np.asarray(result[recovery_kind])
            flat_shape = values.shape[1:]
            means, lows, highs = paired_intervals(
                values.reshape(len(values), -1),
                samples=bootstrap_samples,
                seed=seed,
            )
            for timestep in range(values.shape[1]):
                for pcoder_offset, pcoder_index in enumerate(result["pcoder_indices"]):
                    for metric_offset, metric_name in enumerate(result["recovery_metric_names"]):
                        flat_index = np.ravel_multi_index(
                            (timestep, pcoder_offset, metric_offset), flat_shape
                        )
                        recovery_rows.append(
                            {
                                "degradation_mode": mode,
                                "degradation_level": level,
                                "memory_carrier": carrier,
                                "condition": condition,
                                "prime_identity": identity,
                                "recovery_kind": recovery_kind,
                                "pcoder": pcoder_index,
                                "timestep": timestep,
                                "metric": metric_name,
                                "mean": means[flat_index],
                                "ci95_low": lows[flat_index],
                                "ci95_high": highs[flat_index],
                                "source_file": str(path),
                            }
                        )
            last_timestep = values.shape[1] - 1
            raw_deltas = values[:, last_timestep] - values[:, 0]
            metric_signs = np.asarray(
                [
                    1.0 if name == "cosine" else -1.0
                    for name in result["recovery_metric_names"]
                ]
            )
            improvements = raw_deltas * metric_signs[None, None, :]
            change_shape = raw_deltas.shape[1:]
            delta_means, delta_lows, delta_highs = paired_intervals(
                raw_deltas.reshape(len(raw_deltas), -1),
                samples=bootstrap_samples,
                seed=seed,
            )
            improve_means, improve_lows, improve_highs = paired_intervals(
                improvements.reshape(len(improvements), -1),
                samples=bootstrap_samples,
                seed=seed,
            )
            for pcoder_offset, pcoder_index in enumerate(result["pcoder_indices"]):
                for metric_offset, metric_name in enumerate(result["recovery_metric_names"]):
                    flat_index = np.ravel_multi_index(
                        (pcoder_offset, metric_offset), change_shape
                    )
                    recovery_change_rows.append(
                        {
                            "degradation_mode": mode,
                            "degradation_level": level,
                            "memory_carrier": carrier,
                            "condition": condition,
                            "prime_identity": identity,
                            "recovery_kind": recovery_kind,
                            "pcoder": pcoder_index,
                            "metric": metric_name,
                            "initial_timestep": 0,
                            "final_timestep": last_timestep,
                            "raw_final_minus_initial": delta_means[flat_index],
                            "raw_ci95_low": delta_lows[flat_index],
                            "raw_ci95_high": delta_highs[flat_index],
                            "oriented_improvement": improve_means[flat_index],
                            "improvement_ci95_low": improve_lows[flat_index],
                            "improvement_ci95_high": improve_highs[flat_index],
                            "improved_significantly": improve_lows[flat_index] > 0.0,
                            "source_file": str(path),
                        }
                    )
        internal_errors = result.get("pcoder_internal_error")
        if internal_errors is not None:
            internal_errors = np.asarray(internal_errors)
            internal_shape = internal_errors.shape[1:]
            internal_means, internal_lows, internal_highs = paired_intervals(
                internal_errors.reshape(len(internal_errors), -1),
                samples=bootstrap_samples,
                seed=seed,
            )
            for timestep in range(internal_errors.shape[1]):
                for pcoder_offset, pcoder_index in enumerate(result["pcoder_indices"]):
                    flat_index = np.ravel_multi_index(
                        (timestep, pcoder_offset), internal_shape
                    )
                    internal_error_rows.append(
                        {
                            "degradation_mode": mode,
                            "degradation_level": level,
                            "memory_carrier": carrier,
                            "condition": condition,
                            "prime_identity": identity,
                            "pcoder": pcoder_index,
                            "timestep": timestep,
                            "mean_internal_error": internal_means[flat_index],
                            "ci95_low": internal_lows[flat_index],
                            "ci95_high": internal_highs[flat_index],
                            "source_file": str(path),
                        }
                    )
            last_timestep = internal_errors.shape[1] - 1
            raw_deltas = internal_errors[:, last_timestep] - internal_errors[:, 0]
            delta_means, delta_lows, delta_highs = paired_intervals(
                raw_deltas,
                samples=bootstrap_samples,
                seed=seed,
            )
            for pcoder_offset, pcoder_index in enumerate(result["pcoder_indices"]):
                internal_error_change_rows.append(
                    {
                        "degradation_mode": mode,
                        "degradation_level": level,
                        "memory_carrier": carrier,
                        "condition": condition,
                        "prime_identity": identity,
                        "pcoder": pcoder_index,
                        "initial_timestep": 0,
                        "final_timestep": last_timestep,
                        "raw_final_minus_initial": delta_means[pcoder_offset],
                        "raw_ci95_low": delta_lows[pcoder_offset],
                        "raw_ci95_high": delta_highs[pcoder_offset],
                        "oriented_improvement": -delta_means[pcoder_offset],
                        "improvement_ci95_low": -delta_highs[pcoder_offset],
                        "improvement_ci95_high": -delta_lows[pcoder_offset],
                        "improved_significantly": delta_highs[pcoder_offset] < 0.0,
                        "source_file": str(path),
                    }
                )

    feedforward_rows: list[dict[str, Any]] = []
    feedforward_indexed = (
        _load_results(feedforward_dir.glob("*.pckl"))
        if feedforward_dir is not None
        else {}
    )
    for key, (path, result) in indexed.items():
        if result.get("representation_recovery") is None:
            continue
        mode, level, carrier, condition, identity = key
        reference_key = (mode, level, "none", "target_only", "none")
        if reference_key not in feedforward_indexed:
            continue
        reference_path, reference = feedforward_indexed[reference_key]
        reference_indices = _aligned_reference_indices(result, reference)
        for recovery_kind in ("representation_recovery", "prediction_recovery"):
            values = np.asarray(result[recovery_kind])
            reference_values = np.asarray(reference[recovery_kind])[reference_indices, 0]
            raw_deltas = values - reference_values[:, None, :, :]
            metric_signs = np.asarray(
                [
                    1.0 if name == "cosine" else -1.0
                    for name in result["recovery_metric_names"]
                ]
            )
            improvements = raw_deltas * metric_signs[None, None, None, :]
            flat_shape = raw_deltas.shape[1:]
            delta_means, delta_lows, delta_highs = paired_intervals(
                raw_deltas.reshape(len(raw_deltas), -1),
                samples=bootstrap_samples,
                seed=seed,
            )
            improve_means, improve_lows, improve_highs = paired_intervals(
                improvements.reshape(len(improvements), -1),
                samples=bootstrap_samples,
                seed=seed,
            )
            for timestep in range(values.shape[1]):
                for pcoder_offset, pcoder_index in enumerate(result["pcoder_indices"]):
                    for metric_offset, metric_name in enumerate(result["recovery_metric_names"]):
                        flat_index = np.ravel_multi_index(
                            (timestep, pcoder_offset, metric_offset), flat_shape
                        )
                        feedforward_rows.append(
                            {
                                "degradation_mode": mode,
                                "degradation_level": level,
                                "memory_carrier": carrier,
                                "condition": condition,
                                "prime_identity": identity,
                                "recovery_kind": recovery_kind,
                                "pcoder": pcoder_index,
                                "timestep": timestep,
                                "metric": metric_name,
                                "raw_recurrent_minus_feedforward": delta_means[flat_index],
                                "raw_ci95_low": delta_lows[flat_index],
                                "raw_ci95_high": delta_highs[flat_index],
                                "oriented_improvement": improve_means[flat_index],
                                "improvement_ci95_low": improve_lows[flat_index],
                                "improvement_ci95_high": improve_highs[flat_index],
                                "better_than_feedforward_significantly": improve_lows[flat_index] > 0.0,
                                "source_file": str(path),
                                "feedforward_source_file": str(reference_path),
                            }
                        )

    output_paths = {
        "decoding": output_dir / "layer_timestep_decoding.csv",
        "effects": output_dir / "paper_condition_effects.csv",
        "causal": output_dir / "causal_same_vs_different.csv",
        "causal_recovery": output_dir / "causal_recovery_same_vs_different.csv",
        "recovery": output_dir / "clean_reference_recovery.csv",
        "recovery_change": output_dir / "recovery_change_t0_tlast.csv",
        "internal_error": output_dir / "pcoder_internal_error.csv",
        "internal_error_change": output_dir / "pcoder_internal_error_change_t0_tlast.csv",
        "feedforward": output_dir / "recurrent_vs_feedforward_recovery.csv",
    }
    _write_csv(output_paths["decoding"], decoding_rows)
    _write_csv(output_paths["effects"], effect_rows)
    _write_csv(output_paths["causal"], causal_rows)
    _write_csv(output_paths["causal_recovery"], causal_recovery_rows)
    _write_csv(output_paths["recovery"], recovery_rows)
    _write_csv(output_paths["recovery_change"], recovery_change_rows)
    _write_csv(output_paths["internal_error"], internal_error_rows)
    _write_csv(output_paths["internal_error_change"], internal_error_change_rows)
    _write_csv(output_paths["feedforward"], feedforward_rows)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze paper-matched result files")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--feedforward-dir",
        type=Path,
        help="Optional matched random-init=false clean-reference result directory",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = analyze_directory(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        folds=args.folds,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        feedforward_dir=(
            None if args.feedforward_dir is None else args.feedforward_dir.resolve()
        ),
    )
    for name, path in paths.items():
        print(f"[paper-analysis] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
