from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig, load_experiment_config
from .paper_analysis import (
    _aligned_reference_indices,
    cross_validated_linear_svm,
    paired_interval,
)
from .paper_parafoveal import run_paper_experiment


BASE_HYPERPARAMETERS = (
    {"ffm": 0.8, "fbm": 0.1, "erm": 0.01},
    {"ffm": 0.8, "fbm": 0.1, "erm": 0.01},
    {"ffm": 0.8, "fbm": 0.1, "erm": 0.01},
    {"ffm": 0.8, "fbm": 0.1, "erm": 0.01},
)

# PCoder5 has no top-down input. Its update is ffm*ff + (1-ffm)*memory
# - erm*error_gradient; fbm is recorded as zero to reflect that equation.
PCODER5_CANDIDATES = {
    "baseline_ff0p8_er0p01": {"ffm": 0.8, "fbm": 0.0, "erm": 0.01},
    "ff0p4_er0p01": {"ffm": 0.4, "fbm": 0.0, "erm": 0.01},
    "ff0p5_er0p01": {"ffm": 0.5, "fbm": 0.0, "erm": 0.01},
    "ff0p6_er0p01": {"ffm": 0.6, "fbm": 0.0, "erm": 0.01},
    "ff0p9_er0p01": {"ffm": 0.9, "fbm": 0.0, "erm": 0.01},
    "ff0p95_er0p01": {"ffm": 0.95, "fbm": 0.0, "erm": 0.01},
    "ff1p0_er0p01": {"ffm": 1.0, "fbm": 0.0, "erm": 0.01},
    "ff0p8_er0": {"ffm": 0.8, "fbm": 0.0, "erm": 0.0},
    "ff0p4_er0": {"ffm": 0.4, "fbm": 0.0, "erm": 0.0},
    "ff0p5_er0": {"ffm": 0.5, "fbm": 0.0, "erm": 0.0},
    "ff0p9_er0": {"ffm": 0.9, "fbm": 0.0, "erm": 0.0},
    "ff0p95_er0": {"ffm": 0.95, "fbm": 0.0, "erm": 0.0},
}


def candidate_hyperparameters(name: str) -> tuple[dict[str, float], ...]:
    if name not in PCODER5_CANDIDATES:
        raise ValueError(f"unknown calibration candidate: {name}")
    return (*BASE_HYPERPARAMETERS, dict(PCODER5_CANDIDATES[name]))


def _candidate_config(
    base: ExperimentConfig,
    *,
    candidate: str,
    output_dir: Path,
    target_offset: int,
    max_samples: int,
    targets_per_category: int,
) -> ExperimentConfig:
    model = replace(
        base.model,
        random_init=True,
        original_mode=True,
        hyperparameters=candidate_hyperparameters(candidate),
    )
    runtime = replace(base.runtime, max_samples=max_samples)
    paper = replace(
        base.paper,
        protocol="target_only",
        conditions=("no_prime",),
        prime_identities=("same",),
        memory_carriers=("none",),
        target_offset=target_offset,
        targets_per_category=targets_per_category,
        clean_reference_enabled=True,
    )
    return replace(
        base,
        paths=replace(base.paths, output_dir=output_dir),
        runtime=runtime,
        model=model,
        paper=paper,
    )


def _load_results(directory: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for path in directory.glob("*.pckl"):
        with path.open("rb") as handle:
            result = pickle.load(handle)
        metadata = result["run_metadata"]
        mode = str(metadata["degradation_mode"])
        if mode in results:
            raise ValueError(f"duplicate degradation mode in {directory}: {mode}")
        results[mode] = result
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_candidates(
    output_root: Path,
    feedforward_dir: Path,
    *,
    folds: int,
    seed: int,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    references = _load_results(feedforward_dir)
    recovery_rows: list[dict[str, Any]] = []
    decoding_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for candidate in PCODER5_CANDIDATES:
        candidate_dir = output_root / candidate
        if not candidate_dir.is_dir():
            continue
        results = _load_results(candidate_dir)
        if set(results) != set(references):
            # A sweep can be inspected while the next condition is still being
            # written; only rank complete degradation pairs.
            continue
        candidate_recovery_rows: list[dict[str, Any]] = []
        candidate_decoding_rows: list[dict[str, Any]] = []
        for mode, result in results.items():
            if mode not in references:
                raise ValueError(f"no feedforward reference for {mode}")
            reference = references[mode]
            reference_indices = _aligned_reference_indices(result, reference)
            values = np.asarray(result["representation_recovery"])
            reference_values = np.asarray(reference["representation_recovery"])[
                reference_indices, 0
            ]
            for pcoder_offset in (3, 4):
                for metric_offset, metric_name in ((0, "cosine"), (1, "normalized_l2")):
                    raw_delta = (
                        values[:, -1, pcoder_offset, metric_offset]
                        - reference_values[:, pcoder_offset, metric_offset]
                    )
                    improvement = raw_delta if metric_name == "cosine" else -raw_delta
                    mean, low, high = paired_interval(
                        improvement,
                        samples=bootstrap_samples,
                        seed=seed,
                    )
                    candidate_recovery_rows.append(
                        {
                            "candidate": candidate,
                            "degradation_mode": mode,
                            "pcoder": pcoder_offset + 1,
                            "metric": metric_name,
                            "oriented_improvement": mean,
                            "improvement_ci95_low": low,
                            "improvement_ci95_high": high,
                            "significantly_better_than_feedforward": low > 0.0,
                            "n_targets": len(improvement),
                        }
                    )

            categories = np.asarray(result["category_labels"])
            identities = np.asarray(result["object_identity_labels"])
            for layer in ("fc1", "fc2"):
                candidate_features = np.asarray(
                    result["target_representations"][layer][:, -1], dtype=np.float32
                )
                reference_features = np.asarray(
                    reference["target_representations"][layer][reference_indices, 0],
                    dtype=np.float32,
                )
                _, candidate_correct, candidate_accuracy = cross_validated_linear_svm(
                    candidate_features,
                    categories,
                    identities,
                    folds=folds,
                    seed=seed,
                )
                _, reference_correct, reference_accuracy = cross_validated_linear_svm(
                    reference_features,
                    categories,
                    identities,
                    folds=folds,
                    seed=seed,
                )
                delta, low, high = paired_interval(
                    candidate_correct.astype(float) - reference_correct.astype(float),
                    samples=bootstrap_samples,
                    seed=seed,
                    scale=100.0,
                )
                candidate_decoding_rows.append(
                    {
                        "candidate": candidate,
                        "degradation_mode": mode,
                        "layer": layer,
                        "recurrent_tlast_accuracy": candidate_accuracy,
                        "feedforward_accuracy": reference_accuracy,
                        "decoding_delta_pp": delta,
                        "decoding_ci95_low_pp": low,
                        "decoding_ci95_high_pp": high,
                        "significantly_worse_than_feedforward": high < 0.0,
                        "n_targets": len(categories),
                    }
                )

        recovery_rows.extend(candidate_recovery_rows)
        decoding_rows.extend(candidate_decoding_rows)
        cosine_rows = [row for row in candidate_recovery_rows if row["metric"] == "cosine"]
        l2_rows = [row for row in candidate_recovery_rows if row["metric"] == "normalized_l2"]
        ranking_rows.append(
            {
                "candidate": candidate,
                "all_deep_recovery_cells_pass": all(
                    row["significantly_better_than_feedforward"]
                    for row in candidate_recovery_rows
                ),
                "no_hidden_decoding_cell_significantly_worse": not any(
                    row["significantly_worse_than_feedforward"]
                    for row in candidate_decoding_rows
                ),
                "minimum_cosine_ci95_low": min(
                    float(row["improvement_ci95_low"]) for row in cosine_rows
                ),
                "mean_cosine_improvement": float(
                    np.mean([row["oriented_improvement"] for row in cosine_rows])
                ),
                "minimum_l2_ci95_low": min(
                    float(row["improvement_ci95_low"]) for row in l2_rows
                ),
                "mean_hidden_decoding_delta_pp": float(
                    np.mean([row["decoding_delta_pp"] for row in candidate_decoding_rows])
                ),
            }
        )
    ranking_rows.sort(
        key=lambda row: (
            bool(row["all_deep_recovery_cells_pass"]),
            bool(row["no_hidden_decoding_cell_significantly_worse"]),
            float(row["minimum_cosine_ci95_low"]),
            float(row["mean_cosine_improvement"]),
            float(row["mean_hidden_decoding_delta_pp"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank
    return recovery_rows, decoding_rows, ranking_rows


def run_calibration(args: argparse.Namespace) -> int:
    base = load_experiment_config(args.config).with_overrides(
        imagenet_root=args.imagenet_root,
        device=args.device,
    )
    candidates = (
        list(PCODER5_CANDIDATES)
        if args.phase == "sweep"
        else [args.candidate]
    )
    if args.phase == "confirm" and args.candidate not in PCODER5_CANDIDATES:
        raise ValueError("--candidate must name a registered candidate in confirm phase")
    target_offset = 0 if args.phase == "sweep" else args.holdout_offset
    max_samples = args.selection_samples if args.phase == "sweep" else args.holdout_samples
    phase_root = args.output_root / args.phase
    for candidate in candidates:
        candidate_dir = phase_root / candidate
        expected_files = len(base.stimulus.degradation_conditions())
        if len(list(candidate_dir.glob("*.pckl"))) == expected_files:
            print(f"[calibration] reuse complete candidate={candidate}")
            continue
        config = _candidate_config(
            base,
            candidate=candidate,
            output_dir=candidate_dir.resolve(),
            target_offset=target_offset,
            max_samples=max_samples,
            targets_per_category=args.targets_per_category,
        )
        print(
            f"[calibration] phase={args.phase} candidate={candidate} "
            f"offset={target_offset} n={max_samples}"
        )
        run_paper_experiment(config)

    recovery_rows, decoding_rows, ranking_rows = evaluate_candidates(
        phase_root,
        args.feedforward_dir.resolve(),
        folds=args.folds,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    analysis_dir = phase_root / "analysis"
    _write_csv(analysis_dir / "calibration_recovery_cells.csv", recovery_rows)
    _write_csv(analysis_dir / "calibration_hidden_decoding.csv", decoding_rows)
    _write_csv(analysis_dir / "calibration_ranking.csv", ranking_rows)
    if ranking_rows:
        print(f"[calibration] selected={ranking_rows[0]['candidate']}")
        print(f"[calibration] ranking={analysis_dir / 'calibration_ranking.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate deep PCoder recovery gains")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--imagenet-root", type=Path, required=True)
    parser.add_argument("--feedforward-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("sweep", "confirm"), required=True)
    parser.add_argument("--candidate", choices=tuple(PCODER5_CANDIDATES))
    parser.add_argument("--selection-samples", type=int, default=100)
    parser.add_argument("--holdout-offset", type=int, default=100)
    parser.add_argument("--holdout-samples", type=int, default=100)
    parser.add_argument("--targets-per-category", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "confirm" and args.candidate is None:
        raise ValueError("confirm phase requires --candidate")
    return run_calibration(args)


if __name__ == "__main__":
    raise SystemExit(main())
