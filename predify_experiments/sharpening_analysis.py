from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def normalized_trace_var(stats: dict[str, Any]) -> np.ndarray:
    """Return scale-invariant per-class variance for v3 or legacy summaries."""
    if "normalized_trace_var" in stats:
        return np.asarray(stats["normalized_trace_var"], dtype=np.float64)

    count = np.asarray(stats["count"], dtype=np.float64)
    within = np.asarray(stats["within_cosine"], dtype=np.float64)
    result = np.full_like(within, np.nan, dtype=np.float64)
    valid = count >= 2
    # For L2-normalized vectors, biased trace(Cov) and mean pairwise cosine
    # are algebraically equivalent. This makes v2 result files recoverable.
    result[valid] = (
        (count[valid] - 1.0) / count[valid] * (1.0 - within[valid])
    )
    return result


def _relative_effect(prime_value: float, control_value: float) -> float:
    if not np.isfinite(control_value) or abs(control_value) < 1e-12:
        return float("nan")
    return (prime_value / control_value - 1.0) * 100.0


def summarize_pair(
    prime: dict[str, Any],
    control: dict[str, Any],
    *,
    timestep: int = -1,
) -> dict[str, dict[str, float]]:
    """Summarize Prime-vs-NonPrime effects for one matched result pair."""
    layer_names = list(prime["repr_layer_names"])
    if layer_names != list(control["repr_layer_names"]):
        raise ValueError("Prime and NonPrime representation layers do not match")

    prime_activation = np.asarray(prime["layer_metrics_arr"], dtype=np.float64)[
        1::2, timestep, :, 0
    ]
    control_activation = np.asarray(control["layer_metrics_arr"], dtype=np.float64)[
        1::2, timestep, :, 0
    ]
    prime_stats = prime["repr_stats_timeline"][timestep]
    control_stats = control["repr_stats_timeline"][timestep]

    summary: dict[str, dict[str, float]] = {}
    for layer_index, layer in enumerate(layer_names):
        p_stats = prime_stats[layer]
        c_stats = control_stats[layer]
        p_norm_trace = normalized_trace_var(p_stats)
        c_norm_trace = normalized_trace_var(c_stats)
        valid_trace = np.isfinite(p_norm_trace) & np.isfinite(c_norm_trace)
        p_within = np.asarray(p_stats["within_cosine"], dtype=np.float64)
        c_within = np.asarray(c_stats["within_cosine"], dtype=np.float64)
        valid_within = np.isfinite(p_within) & np.isfinite(c_within)

        activation_effect = _relative_effect(
            float(prime_activation[:, layer_index].mean()),
            float(control_activation[:, layer_index].mean()),
        )
        normalized_trace_effect = _relative_effect(
            float(p_norm_trace[valid_trace].mean()),
            float(c_norm_trace[valid_trace].mean()),
        )
        within_delta = float((p_within[valid_within] - c_within[valid_within]).mean())

        if "normalized_fisher_ratio" in p_stats and "normalized_fisher_ratio" in c_stats:
            normalized_fisher_effect = _relative_effect(
                float(np.asarray(p_stats["normalized_fisher_ratio"])),
                float(np.asarray(c_stats["normalized_fisher_ratio"])),
            )
        else:
            normalized_fisher_effect = float("nan")

        summary[layer] = {
            "activation_effect_pct": activation_effect,
            "normalized_trace_effect_pct": normalized_trace_effect,
            "within_cosine_delta_x1e3": within_delta * 1000.0,
            "normalized_fisher_effect_pct": normalized_fisher_effect,
            "complete_sharpening": float(
                activation_effect < 0
                and normalized_trace_effect < 0
                and normalized_fisher_effect > 0
            ),
        }
    return summary


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _accuracy(result: dict[str, Any]) -> tuple[float, float]:
    """Return the recorded odd/repeated-trial Top-1 and Top-5 accuracy."""
    top1, top5 = result["acc_result"]
    return float(top1), float(top5)


def analyze_result_directories(
    hebb_dir: Path,
    nohebb_dir: Path,
    *,
    timestep: int = -1,
) -> list[dict[str, Any]]:
    """Calculate Hebbian × Prime interactions for all matched result files."""
    rows: list[dict[str, Any]] = []
    prime_paths = sorted(hebb_dir.glob("*_Prime.pckl"))
    if not prime_paths:
        raise ValueError(f"No Prime result files found in {hebb_dir}")

    for hebb_prime_path in prime_paths:
        filename = hebb_prime_path.name
        nonprime_filename = filename.replace("_Prime.pckl", "_nonPrime.pckl")
        paths = {
            "hebb_prime": hebb_prime_path,
            "hebb_control": hebb_dir / nonprime_filename,
            "nohebb_prime": nohebb_dir / filename,
            "nohebb_control": nohebb_dir / nonprime_filename,
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing matched results:\n- " + "\n- ".join(missing))

        loaded = {name: _load(path) for name, path in paths.items()}
        try:
            hebb_effect = summarize_pair(
                loaded["hebb_prime"], loaded["hebb_control"], timestep=timestep
            )
            nohebb_effect = summarize_pair(
                loaded["nohebb_prime"], loaded["nohebb_control"], timestep=timestep
            )
            metadata = loaded["hebb_prime"]["run_metadata"]
            accuracies = {
                name: _accuracy(result) for name, result in loaded.items()
            }
            for layer, effect in hebb_effect.items():
                baseline = nohebb_effect[layer]
                interaction_activation = (
                    effect["activation_effect_pct"]
                    - baseline["activation_effect_pct"]
                )
                interaction_trace = (
                    effect["normalized_trace_effect_pct"]
                    - baseline["normalized_trace_effect_pct"]
                )
                interaction_fisher = (
                    effect["normalized_fisher_effect_pct"]
                    - baseline["normalized_fisher_effect_pct"]
                )
                rows.append(
                    {
                        "degradation_mode": metadata["degradation_mode"],
                        "degradation_level": float(metadata["degradation_level"]),
                        "layer": layer,
                        "timestep": timestep,
                        "hebb_prime_top1": accuracies["hebb_prime"][0],
                        "hebb_prime_top5": accuracies["hebb_prime"][1],
                        "hebb_nonprime_top1": accuracies["hebb_control"][0],
                        "hebb_nonprime_top5": accuracies["hebb_control"][1],
                        "nohebb_prime_top1": accuracies["nohebb_prime"][0],
                        "nohebb_prime_top5": accuracies["nohebb_prime"][1],
                        "nohebb_nonprime_top1": accuracies["nohebb_control"][0],
                        "nohebb_nonprime_top5": accuracies["nohebb_control"][1],
                        **{f"hebb_{key}": value for key, value in effect.items()},
                        **{f"nohebb_{key}": value for key, value in baseline.items()},
                        "interaction_activation_pct": interaction_activation,
                        "interaction_normalized_trace_pct": interaction_trace,
                        "interaction_normalized_fisher_pct": interaction_fisher,
                        "interaction_complete_sharpening": float(
                            interaction_activation < 0
                            and interaction_trace < 0
                            and interaction_fisher > 0
                        ),
                        "hebb_prime_file": str(paths["hebb_prime"].resolve()),
                        "hebb_nonprime_file": str(paths["hebb_control"].resolve()),
                        "nohebb_prime_file": str(paths["nohebb_prime"].resolve()),
                        "nohebb_nonprime_file": str(paths["nohebb_control"].resolve()),
                    }
                )
        finally:
            loaded.clear()
    return rows


def write_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
    if not rows:
        raise ValueError("No analysis rows to write")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze normalized sharpening and Hebbian interactions."
    )
    parser.add_argument("--hebb-dir", type=Path, required=True)
    parser.add_argument("--nohebb-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--timestep", type=int, default=-1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = analyze_result_directories(
        args.hebb_dir.resolve(),
        args.nohebb_dir.resolve(),
        timestep=args.timestep,
    )
    write_csv(rows, args.output_csv.resolve())
    print(f"[analysis] wrote {len(rows)} rows to {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
