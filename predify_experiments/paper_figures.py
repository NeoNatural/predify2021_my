from __future__ import annotations

import argparse
import csv
from pathlib import Path


LAYER_COLORS = {
    "conv4": "#0072B2",
    "conv5": "#56B4E9",
    "fc1": "#E69F00",
    "fc2": "#D55E00",
}
PCODER_COLORS = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00"]
CARRIER_COLORS = {
    "hebb_only": "#0072B2",
    "pcoder_carry": "#E69F00",
    "hebb_and_pcoder": "#D55E00",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_clean_effects(effect_csv: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(effect_csv)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for column, condition in enumerate(("one_back", "two_back")):
        selected = [row for row in rows if row["condition_vs_no_prime"] == condition]
        for layer, color in LAYER_COLORS.items():
            layer_rows = sorted(
                (row for row in selected if row["layer"] == layer),
                key=lambda row: int(row["timestep"]),
            )
            x = [int(row["timestep"]) for row in layer_rows]
            axes[0, column].plot(
                x,
                [float(row["decoding_delta_pp"]) for row in layer_rows],
                marker="o",
                ms=3,
                color=color,
                label=layer,
            )
            axes[1, column].plot(
                x,
                [float(row["activation_delta_pct"]) for row in layer_rows],
                marker="o",
                ms=3,
                color=color,
                label=layer,
            )
        axes[0, column].axhline(0, color="black", lw=0.8)
        axes[1, column].axhline(0, color="black", lw=0.8)
        axes[0, column].set_title(condition.replace("_", " ").title())
        axes[1, column].set_xlabel("Target recurrent timestep")
    axes[0, 0].set_ylabel("Five-category decoding delta (pp)")
    axes[1, 0].set_ylabel("Mean absolute activation delta (%)")
    axes[0, 1].legend(frameon=False, ncol=2)
    fig.suptitle("Paper-matched parafoveal priming: Hebbian minus no-prime")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recovery(recovery_csv: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(recovery_csv)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    modes = ("predify_noise", "fractal_occlusion")
    kinds = ("representation_recovery", "prediction_recovery")
    for row_index, kind in enumerate(kinds):
        for column, mode in enumerate(modes):
            selected = [
                row
                for row in rows
                if row["degradation_mode"] == mode
                and row["recovery_kind"] == kind
                and row["metric"] == "cosine"
            ]
            for pcoder in range(1, 6):
                pcoder_rows = sorted(
                    (row for row in selected if int(row["pcoder"]) == pcoder),
                    key=lambda row: int(row["timestep"]),
                )
                axes[row_index, column].plot(
                    [int(row["timestep"]) for row in pcoder_rows],
                    [float(row["mean"]) for row in pcoder_rows],
                    marker="o",
                    ms=3,
                    color=PCODER_COLORS[pcoder - 1],
                    label=f"PCoder{pcoder}",
                )
            axes[row_index, column].set_title(
                f"{mode.replace('_', ' ')} — {kind.replace('_recovery', '')}"
            )
            axes[row_index, column].set_ylabel("Cosine to clean reference")
            if row_index == 1:
                axes[row_index, column].set_xlabel("Target recurrent timestep")
    axes[0, 1].legend(frameon=False, ncol=2)
    fig.suptitle("Target-only recurrent recovery toward clean reference")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feedforward_corrected_recovery(comparison_csv: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(comparison_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for axis, mode in zip(axes, ("predify_noise", "fractal_occlusion")):
        selected = [
            row
            for row in rows
            if row["degradation_mode"] == mode
            and row["recovery_kind"] == "representation_recovery"
            and row["metric"] == "cosine"
        ]
        for pcoder in range(1, 6):
            pcoder_rows = sorted(
                (row for row in selected if int(row["pcoder"]) == pcoder),
                key=lambda row: int(row["timestep"]),
            )
            axis.plot(
                [int(row["timestep"]) for row in pcoder_rows],
                [float(row["oriented_improvement"]) for row in pcoder_rows],
                marker="o",
                ms=3,
                color=PCODER_COLORS[pcoder - 1],
                label=f"PCoder{pcoder}",
            )
        axis.axhline(0, color="black", lw=0.8)
        axis.set_title(mode.replace("_", " "))
        axis.set_xlabel("Target recurrent timestep")
    axes[0].set_ylabel("Representation cosine minus feedforward")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Recurrent recovery corrected by matched feedforward baseline")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_causal_carrier_effects(
    causal_csv: Path,
    causal_recovery_csv: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    causal_rows = _read_csv(causal_csv)
    recovery_rows = _read_csv(causal_recovery_csv)
    modes = ("predify_noise", "fractal_occlusion")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for row_index, mode in enumerate(modes):
        for carrier, color in CARRIER_COLORS.items():
            fc2_rows = sorted(
                (
                    row
                    for row in causal_rows
                    if row["degradation_mode"] == mode
                    and row["memory_carrier"] == carrier
                    and row["layer"] == "fc2"
                ),
                key=lambda row: int(row["timestep"]),
            )
            recovery = sorted(
                (
                    row
                    for row in recovery_rows
                    if row["degradation_mode"] == mode
                    and row["memory_carrier"] == carrier
                    and row["recovery_kind"] == "representation_recovery"
                    and row["pcoder"] == "5"
                    and row["metric"] == "cosine"
                ),
                key=lambda row: int(row["timestep"]),
            )
            x = [int(row["timestep"]) for row in fc2_rows]
            axes[row_index, 0].plot(
                x,
                [float(row["decoding_delta_pp"]) for row in fc2_rows],
                marker="o",
                ms=3,
                color=color,
                label=carrier,
            )
            axes[row_index, 1].plot(
                x,
                [float(row["activation_delta_pct"]) for row in fc2_rows],
                marker="o",
                ms=3,
                color=color,
            )
            axes[row_index, 2].plot(
                [int(row["timestep"]) for row in recovery],
                [float(row["oriented_recovery_advantage"]) for row in recovery],
                marker="o",
                ms=3,
                color=color,
            )
        for column in range(3):
            axes[row_index, column].axhline(0, color="black", lw=0.8)
        axes[row_index, 0].set_ylabel(mode.replace("_", " "))
    axes[0, 0].set_title("FC2 category decoding delta (pp)")
    axes[0, 1].set_title("FC2 activation delta (%)")
    axes[0, 2].set_title("PCoder5 clean-recovery cosine advantage")
    for axis in axes[1]:
        axis.set_xlabel("Target recurrent timestep")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Same-prime minus Different-prime by memory carrier")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shared_decoder_effects(effect_csv: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _read_csv(effect_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for axis, mode in zip(axes, ("predify_noise", "fractal_occlusion")):
        for carrier, color in CARRIER_COLORS.items():
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["degradation_mode"] == mode
                    and row["memory_carrier"] == carrier
                    and row["layer"] == "fc2"
                ),
                key=lambda row: int(row["timestep"]),
            )
            x = [int(row["timestep"]) for row in selected]
            mean = [float(row["accuracy_delta_pp"]) for row in selected]
            low = [float(row["ci95_low_pp"]) for row in selected]
            high = [float(row["ci95_high_pp"]) for row in selected]
            axis.plot(x, mean, marker="o", ms=3, color=color, label=carrier)
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        axis.axhline(0, color="black", lw=0.8)
        axis.axvline(9, color="0.6", lw=0.8, ls="--")
        axis.set_title(mode.replace("_", " "))
        axis.set_xlabel("Target recurrent timestep")
    axes[0].set_ylabel("Frozen five-category accuracy delta (pp)")
    axes[1].legend(frameon=False)
    fig.suptitle("Shared readout: Same-prime minus Different-prime at FC2")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot paper-matched experiment summaries")
    parser.add_argument("--effects-csv", type=Path, required=True)
    parser.add_argument("--recovery-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feedforward-comparison-csv", type=Path)
    parser.add_argument("--causal-csv", type=Path)
    parser.add_argument("--causal-recovery-csv", type=Path)
    parser.add_argument("--shared-decoder-effects-csv", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    plot_clean_effects(args.effects_csv, args.output_dir / "clean_paper_effects.png")
    plot_recovery(args.recovery_csv, args.output_dir / "clean_reference_recovery.png")
    if args.feedforward_comparison_csv is not None:
        plot_feedforward_corrected_recovery(
            args.feedforward_comparison_csv,
            args.output_dir / "feedforward_corrected_recovery.png",
        )
    if (args.causal_csv is None) != (args.causal_recovery_csv is None):
        raise ValueError("causal figure requires both causal CSV inputs")
    if args.causal_csv is not None:
        plot_causal_carrier_effects(
            args.causal_csv,
            args.causal_recovery_csv,
            args.output_dir / "causal_memory_carriers.png",
        )
    if args.shared_decoder_effects_csv is not None:
        plot_shared_decoder_effects(
            args.shared_decoder_effects_csv,
            args.output_dir / "shared_decoder_fc2.png",
        )
    print(f"[paper-figures] wrote {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
