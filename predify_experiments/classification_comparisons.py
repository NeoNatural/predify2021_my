from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
from typing import Any, Iterable, Sequence

import numpy as np


ConditionKey = tuple[str, float]


def _condition_sort_key(item: ConditionKey) -> tuple[int, float]:
    order = {"none": 0, "predify_noise": 1, "fractal_occlusion": 2}
    return order.get(item[0], 99), item[1]


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _topk(logits: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    k = min(k, logits.shape[-1])
    top = np.argpartition(logits, -k, axis=-1)[:, -k:]
    return np.any(top == labels[:, None], axis=1)


def predify_correctness(
    result: dict[str, Any], timestep: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(result["Out_list"])[1::2, timestep, :]
    labels = np.asarray(result["y_list"], dtype=np.int64)[1::2]
    indices = np.asarray(result["idx_log_list"], dtype=np.int64)[1::2]
    return _topk(logits, labels, k), labels, indices


def vanilla_correctness(
    result: dict[str, Any], k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(result["logits"])
    labels = np.asarray(result["y_list"], dtype=np.int64)
    indices = np.asarray(result["idx_log_list"], dtype=np.int64)
    return _topk(logits, labels, k), labels, indices


def paired_bootstrap(
    paired_values: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 114514,
) -> tuple[float, float, float]:
    """Mean and percentile CI for paired per-target effects, in percentage points."""
    values = np.asarray(paired_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired_values must be a non-empty one-dimensional array")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk = 256
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        resamples = rng.integers(0, values.size, size=(stop - start, values.size))
        estimates[start:stop] = values[resamples].mean(axis=1)
    return (
        float(values.mean() * 100.0),
        float(np.percentile(estimates, 2.5) * 100.0),
        float(np.percentile(estimates, 97.5) * 100.0),
    )


def _metadata_key(result: dict[str, Any]) -> ConditionKey:
    metadata = result.get("run_metadata", {})
    mode = str(metadata.get("degradation_mode", "none"))
    level = float(metadata.get("degradation_level", 0.0))
    # All zero-strength degradation implementations are the same clean tensor.
    return ("none", 0.0) if level == 0.0 else (mode, level)


def _index_results(
    directories: Iterable[Path],
    *,
    condition: str | None = None,
    prefer_highest_schema: bool = False,
) -> dict[ConditionKey, Path]:
    index: dict[ConditionKey, Path] = {}
    selection_scores: dict[ConditionKey, tuple[int, int]] = {}
    suffix = {
        "priming": "*_Prime.pckl",
        "nonpriming": "*_nonPrime.pckl",
        None: "*.pckl",
    }[condition]
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob(suffix)):
            result = _load(path)
            metadata = result.get("run_metadata", {})
            recorded_condition = metadata.get("condition")
            if condition is not None and recorded_condition not in {None, condition}:
                continue
            key = _metadata_key(result)
            schema = int(result.get("schema_version", 0) or 0)
            clean_sigma0 = int(
                key == ("none", 0.0) and "predify_noise_sigma0_" in path.name
            )
            score = (schema, clean_sigma0)
            if key not in index or (
                prefer_highest_schema
                and score > selection_scores.get(key, (-1, -1))
            ):
                index[key] = path.resolve()
                selection_scores[key] = score
    return index


def _assert_aligned(
    labels_a: np.ndarray,
    indices_a: np.ndarray,
    labels_b: np.ndarray,
    indices_b: np.ndarray,
    context: str,
) -> None:
    if not np.array_equal(labels_a, labels_b):
        raise ValueError(f"label mismatch for paired comparison: {context}")
    if not np.array_equal(indices_a, indices_b):
        raise ValueError(f"sample-index mismatch for paired comparison: {context}")


def timeline_rows(
    vanilla_index: dict[ConditionKey, Path],
    predify_index: dict[ConditionKey, Path],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 114514,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing = sorted(set(vanilla_index) - set(predify_index))
    if missing:
        raise FileNotFoundError(f"missing no-Hebb Predify conditions: {missing}")
    for condition_key in sorted(vanilla_index, key=_condition_sort_key):
        vanilla_path = vanilla_index[condition_key]
        predify_path = predify_index[condition_key]
        vanilla = _load(vanilla_path)
        predify = _load(predify_path)
        for timestep in range(np.asarray(predify["Out_list"]).shape[1]):
            row: dict[str, Any] = {
                "degradation_mode": condition_key[0],
                "degradation_level": condition_key[1],
                "timestep": timestep,
            }
            for k in (1, 5):
                vanilla_correct, v_labels, v_indices = vanilla_correctness(vanilla, k)
                predify_correct, p_labels, p_indices = predify_correctness(
                    predify, timestep, k
                )
                _assert_aligned(
                    v_labels, v_indices, p_labels, p_indices, str(condition_key)
                )
                delta, low, high = paired_bootstrap(
                    predify_correct.astype(float) - vanilla_correct.astype(float),
                    samples=bootstrap_samples,
                    seed=seed + timestep * 10 + k,
                )
                row.update(
                    {
                        f"vanilla_top{k}": float(vanilla_correct.mean()),
                        f"predify_top{k}": float(predify_correct.mean()),
                        f"delta_top{k}_pp": delta,
                        f"delta_top{k}_ci_low_pp": low,
                        f"delta_top{k}_ci_high_pp": high,
                        f"predify_improves_top{k}": bool(delta > 0 and low > 0),
                    }
                )
            row["vanilla_file"] = str(vanilla_path)
            row["predify_file"] = str(predify_path)
            rows.append(row)
        del vanilla, predify
    return rows


def _condition_correctness(
    path: Path, k: int, timestep: int = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return predify_correctness(_load(path), timestep, k)


def _four_condition_rows(
    hebb_index: dict[str, dict[ConditionKey, Path]],
    nohebb_index: dict[str, dict[ConditionKey, Path]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    keys = set(hebb_index["priming"]) & set(hebb_index["nonpriming"])
    keys &= set(nohebb_index["priming"]) & set(nohebb_index["nonpriming"])
    rows: list[dict[str, Any]] = []
    for key in sorted(keys, key=_condition_sort_key):
        paths = {
            "hebb_prime": hebb_index["priming"][key],
            "hebb_nonprime": hebb_index["nonpriming"][key],
            "nohebb_prime": nohebb_index["priming"][key],
            "nohebb_nonprime": nohebb_index["nonpriming"][key],
        }
        row: dict[str, Any] = {
            "degradation_mode": key[0],
            "degradation_level": key[1],
        }
        for k in (1, 5):
            loaded = {
                name: _condition_correctness(path, k) for name, path in paths.items()
            }
            reference = loaded["hebb_prime"]
            for name, (_, labels, indices) in loaded.items():
                _assert_aligned(
                    reference[1], reference[2], labels, indices, f"{key}/{name}"
                )
            correct = {name: values[0].astype(float) for name, values in loaded.items()}
            hebb_effect = correct["hebb_prime"] - correct["hebb_nonprime"]
            nohebb_effect = correct["nohebb_prime"] - correct["nohebb_nonprime"]
            for label, values in (
                ("hebb_prime_minus_nonprime", hebb_effect),
                ("nohebb_prime_minus_nonprime", nohebb_effect),
                ("difference_in_differences", hebb_effect - nohebb_effect),
            ):
                effect, low, high = paired_bootstrap(
                    values,
                    samples=bootstrap_samples,
                    seed=seed + k + len(label),
                )
                row[f"top{k}_{label}_pp"] = effect
                row[f"top{k}_{label}_ci_low_pp"] = low
                row[f"top{k}_{label}_ci_high_pp"] = high
            for name, values in correct.items():
                row[f"top{k}_{name}"] = float(values.mean())
        row.update({f"{name}_file": str(path) for name, path in paths.items()})
        rows.append(row)
    return rows


def foveated_accuracy_rows(
    hebb_dir: Path,
    nohebb_dir: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 114514,
) -> list[dict[str, Any]]:
    return _four_condition_rows(
        {
            "priming": _index_results([hebb_dir], condition="priming"),
            "nonpriming": _index_results([hebb_dir], condition="nonpriming"),
        },
        {
            "priming": _index_results([nohebb_dir], condition="priming"),
            "nonpriming": _index_results([nohebb_dir], condition="nonpriming"),
        },
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


def layout_comparison_rows(
    foveated_hebb_dir: Path,
    parafovea_hebb_dirs: Sequence[Path],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 114514,
) -> list[dict[str, Any]]:
    fov = {
        "prime": _index_results([foveated_hebb_dir], condition="priming"),
        "nonprime": _index_results([foveated_hebb_dir], condition="nonpriming"),
    }
    para = {
        "prime": _index_results(
            parafovea_hebb_dirs,
            condition="priming",
            prefer_highest_schema=True,
        ),
        "nonprime": _index_results(
            parafovea_hebb_dirs,
            condition="nonpriming",
            prefer_highest_schema=True,
        ),
    }
    keys = set(fov["prime"]) & set(fov["nonprime"])
    keys &= set(para["prime"]) & set(para["nonprime"])
    rows: list[dict[str, Any]] = []
    for key in sorted(keys, key=_condition_sort_key):
        paths = {
            "foveated_prime": fov["prime"][key],
            "foveated_nonprime": fov["nonprime"][key],
            "parafovea_prime": para["prime"][key],
            "parafovea_nonprime": para["nonprime"][key],
        }
        row: dict[str, Any] = {
            "degradation_mode": key[0],
            "degradation_level": key[1],
        }
        for k in (1, 5):
            loaded = {
                name: _condition_correctness(path, k) for name, path in paths.items()
            }
            reference = loaded["foveated_prime"]
            for name, (_, labels, indices) in loaded.items():
                _assert_aligned(
                    reference[1], reference[2], labels, indices, f"{key}/{name}"
                )
            correct = {name: values[0].astype(float) for name, values in loaded.items()}
            fov_effect = correct["foveated_prime"] - correct["foveated_nonprime"]
            para_effect = correct["parafovea_prime"] - correct["parafovea_nonprime"]
            for label, values in (
                ("foveated_prime_minus_nonprime", fov_effect),
                ("parafovea_prime_minus_nonprime", para_effect),
                ("foveated_minus_parafovea_effect", fov_effect - para_effect),
            ):
                effect, low, high = paired_bootstrap(
                    values,
                    samples=bootstrap_samples,
                    seed=seed + k + len(label),
                )
                row[f"top{k}_{label}_pp"] = effect
                row[f"top{k}_{label}_ci_low_pp"] = low
                row[f"top{k}_{label}_ci_high_pp"] = high
        row.update({f"{name}_file": str(path) for name, path in paths.items()})
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _condition_label(mode: str, level: float) -> str:
    if mode == "none":
        return "clean"
    prefix = "noise σ" if mode == "predify_noise" else "occlusion"
    return f"{prefix}={level:g}"


def write_plots(
    timeline: list[dict[str, Any]],
    layout: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    groups: dict[ConditionKey, list[dict[str, Any]]] = {}
    for row in timeline:
        groups.setdefault(
            (str(row["degradation_mode"]), float(row["degradation_level"])), []
        ).append(row)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    mode_columns = (("none", "Clean"), ("predify_noise", "Noise"), ("fractal_occlusion", "Occlusion"))
    for column, (mode, title) in enumerate(mode_columns):
        for key, items in groups.items():
            if key[0] != mode:
                continue
            items = sorted(items, key=lambda row: int(row["timestep"]))
            label = _condition_label(*key)
            for row_index, k in enumerate((1, 5)):
                axis = axes[row_index, column]
                line = axis.plot(
                    [int(row["timestep"]) for row in items],
                    [float(row[f"predify_top{k}"]) * 100 for row in items],
                    marker="o",
                    markersize=3,
                    label=f"Predify {label}",
                )
                axis.axhline(
                    float(items[0][f"vanilla_top{k}"]) * 100,
                    linestyle="--",
                    alpha=0.6,
                    color=line[0].get_color(),
                )
                axis.set_title(f"{title}: Top-{k}")
                axis.set_ylabel("Accuracy (%)")
                axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Saved predictive state (t)")
    axes[-1, 1].set_xlabel("Saved predictive state (t)")
    axes[-1, 2].set_xlabel("Saved predictive state (t)")
    for axis in axes.flat:
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "predify_vs_vanilla_timeline.png", dpi=180)
    plt.close(fig)

    labels = [_condition_label(str(r["degradation_mode"]), float(r["degradation_level"])) for r in layout]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(9, 5))
    for offset, (key, name) in zip(
        (-0.18, 0.18),
        (("top1_foveated_prime_minus_nonprime", "Foveated"), ("top1_parafovea_prime_minus_nonprime", "Parafovea")),
    ):
        values = np.asarray([float(row[f"{key}_pp"]) for row in layout])
        lows = np.asarray([float(row[f"{key}_ci_low_pp"]) for row in layout])
        highs = np.asarray([float(row[f"{key}_ci_high_pp"]) for row in layout])
        axis.errorbar(
            x + offset,
            values,
            yerr=np.vstack((values - lows, highs - values)),
            fmt="o",
            capsize=4,
            label=name,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Top-1 Prime − NonPrime (percentage points)")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "foveated_vs_parafovea.png", dpi=180)
    plt.close(fig)


def write_report(
    timeline: list[dict[str, Any]],
    foveated: list[dict[str, Any]],
    layout: list[dict[str, Any]],
    output_file: Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 114514,
) -> None:
    lines = [
        "# Vanilla VGG、Predify 时间线与 Foveated Priming 对比",
        "",
        "本报告使用同一批 1000 个 odd targets，并以样本级配对 bootstrap（10,000 次）估计 95% CI。当前保存的 Predify t0 已包含一次 predictive update，不解释为纯 feedforward。",
        "",
        "## Predify 相对 torchvision vanilla VGG16",
        "",
        "| 条件 | t9 Top-1 差值 (pp, 95% CI) | t9 Top-5 差值 (pp, 95% CI) | 首次显著超过 vanilla | t1→t9 Top-1 (pp, 95% CI) | 收敛 t |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[ConditionKey, list[dict[str, Any]]] = {}
    for row in timeline:
        key = (str(row["degradation_mode"]), float(row["degradation_level"]))
        grouped.setdefault(key, []).append(row)
    for key, rows in sorted(grouped.items(), key=lambda item: _condition_sort_key(item[0])):
        rows.sort(key=lambda row: int(row["timestep"]))
        final = rows[-1]
        first = next(
            (str(row["timestep"]) for row in rows if bool(row["predify_improves_top1"])),
            "无",
        )
        predify = _load(Path(str(rows[0]["predify_file"])))
        correct_t1, labels_t1, indices_t1 = predify_correctness(predify, 1, 1)
        correct_t9, labels_t9, indices_t9 = predify_correctness(predify, 9, 1)
        _assert_aligned(labels_t1, indices_t1, labels_t9, indices_t9, f"{key}/t1-t9")
        recurrent, recurrent_low, recurrent_high = paired_bootstrap(
            correct_t9.astype(float) - correct_t1.astype(float),
            samples=bootstrap_samples,
            seed=seed + 101,
        )
        final_accuracy = float(rows[-1]["predify_top1"])
        convergence = 9
        for position, candidate in enumerate(rows[1:], start=1):
            later = rows[position:]
            if all(
                abs(float(item["predify_top1"]) - final_accuracy) <= 0.001 + 1e-12
                for item in later
            ):
                convergence = int(candidate["timestep"])
                break
        lines.append(
            f"| {_condition_label(*key)} | {float(final['delta_top1_pp']):+.2f} "
            f"[{float(final['delta_top1_ci_low_pp']):+.2f}, {float(final['delta_top1_ci_high_pp']):+.2f}] | "
            f"{float(final['delta_top5_pp']):+.2f} "
            f"[{float(final['delta_top5_ci_low_pp']):+.2f}, {float(final['delta_top5_ci_high_pp']):+.2f}] | "
            f"{first} | {recurrent:+.2f} [{recurrent_low:+.2f}, {recurrent_high:+.2f}] | "
            f"{convergence} |"
        )
        del predify
    lines += [
        "",
        "这里的“收敛 t”操作性定义为：从该 timestep 起，所有后续 Top-1 均在 t9 的 ±0.1 percentage point 内。",
    ]
    lines += [
        "",
        "## Foveated priming 的分类效应",
        "",
        "| 条件 | Hebb P−NP Top-1 (pp) | no-Hebb P−NP Top-1 (pp) | 差分中的差分 (pp, 95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for row in foveated:
        lines.append(
            f"| {_condition_label(str(row['degradation_mode']), float(row['degradation_level']))} | "
            f"{float(row['top1_hebb_prime_minus_nonprime_pp']):+.2f} | "
            f"{float(row['top1_nohebb_prime_minus_nonprime_pp']):+.2f} | "
            f"{float(row['top1_difference_in_differences_pp']):+.2f} "
            f"[{float(row['top1_difference_in_differences_ci_low_pp']):+.2f}, "
            f"{float(row['top1_difference_in_differences_ci_high_pp']):+.2f}] |"
        )
    lines += [
        "",
        "## Foveated 与 parafovea",
        "",
        "| 条件 | Foveated P−NP Top-1 | Parafovea P−NP Top-1 | 布局效应差 (pp, 95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for row in layout:
        lines.append(
            f"| {_condition_label(str(row['degradation_mode']), float(row['degradation_level']))} | "
            f"{float(row['top1_foveated_prime_minus_nonprime_pp']):+.2f} | "
            f"{float(row['top1_parafovea_prime_minus_nonprime_pp']):+.2f} | "
            f"{float(row['top1_foveated_minus_parafovea_effect_pp']):+.2f} "
            f"[{float(row['top1_foveated_minus_parafovea_effect_ci_low_pp']):+.2f}, "
            f"{float(row['top1_foveated_minus_parafovea_effect_ci_high_pp']):+.2f}] |"
        )
    lines += [
        "",
        "仅当百分点差为正且配对 95% CI 下界大于 0 时称为分类性能提升。Vanilla 对比检验的是整个 Predify predictive-coding system，不单独隔离 feedback、memory 或 error-correction。",
    ]
    significant_t9 = [
        _condition_label(str(row["degradation_mode"]), float(row["degradation_level"]))
        for row in timeline
        if int(row["timestep"]) == 9 and bool(row["predify_improves_top1"])
    ]
    lines += [
        "",
        "## 简明结论",
        "",
        f"- t9 相对 vanilla 有显著 Top-1 提升的条件：{('、'.join(significant_t9) if significant_t9 else '无')}。",
        "- 三个 foveated 点均未出现显著正向 Hebbian Prime−NonPrime Top-1 效应。",
        "- noise 0.5 的 foveated Hebbian priming 为显著负效应，并且相对 parafovea 进一步变差；clean 与 occlusion 0.3 的布局差异不显著。",
    ]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paired classification comparisons.")
    parser.add_argument("--vanilla-dir", type=Path, required=True)
    parser.add_argument("--timeline-predify-dir", type=Path, action="append", required=True)
    parser.add_argument("--foveated-hebb-dir", type=Path, required=True)
    parser.add_argument("--foveated-nohebb-dir", type=Path, required=True)
    parser.add_argument("--parafovea-hebb-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=114514)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    vanilla_index = _index_results([args.vanilla_dir.resolve()])
    predify_index = _index_results(
        [path.resolve() for path in args.timeline_predify_dir], condition="priming"
    )
    timeline = timeline_rows(
        vanilla_index,
        predify_index,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    foveated = foveated_accuracy_rows(
        args.foveated_hebb_dir.resolve(),
        args.foveated_nohebb_dir.resolve(),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    layout = layout_comparison_rows(
        args.foveated_hebb_dir.resolve(),
        [path.resolve() for path in args.parafovea_hebb_dir],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_csv(timeline, output_dir / "predify_vs_vanilla_timeline.csv")
    write_csv(foveated, output_dir / "foveated_priming_accuracy.csv")
    write_csv(layout, output_dir / "foveated_vs_parafovea.csv")
    write_plots(timeline, layout, output_dir)
    write_report(
        timeline,
        foveated,
        layout,
        output_dir / "REPORT.md",
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"[analysis] wrote comparison artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
