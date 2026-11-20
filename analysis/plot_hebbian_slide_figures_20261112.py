"""Export standalone presentation figures from the frozen Hebbian result tables."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.plot_adaptive_20261028 import Axis

DEST = ROOT / "deliverables/hebbian_slide_figures_20261112"
FC = "fc_sae_gated_six_20261106"
CONV = "conv_fc_transfer_20261106"
LOCAL = "conv_local_readouts_20261106"
DIAG = "conv_fc_transfer_diagnosis_20261106"
NEW = "hebb_post_pool_20261111"

DATASETS = {"imagenet": "ImageNet", "things": "THINGS"}
LAYERS = {"fc1": "FC1", "fc2": "FC2", "conv4": "Conv4 (PCoder4)",
          "conv5": "Conv5 (PCoder5)", "conv4_input": "Conv4 input", "fc2_input": "FC2 input"}
WINDOWS = ("early", "middle", "late")
WINDOW_LABELS = ("Early\n0–4", "Middle\n5–9", "Late\n10–14")
HISTORIES = ("exact", "different")
METRICS = ("within", "between", "fisher")
VARIANTS = ("direct_rep4", "direct_prd5", "sae_rep4", "sae_prd5")
VARIANT_LABELS = ("Native\nrep4", "Native\nprd5", "SAE\nrep4", "SAE\nprd5")
OLD_CASES = ("clip", "direct_rep4_pool", "direct_rep4_full", "direct_rep4_shuffle")
NEW_CASES = ("eps0", "eps005")

INK, MUTED, GRID = "#182633", "#526171", "#e2e7ec"
BRICK, INDIGO, PURPLE = "#ac4d3e", "#465b9c", "#755486"
GREEN, ORANGE, TEAL, GRAY = "#16724b", "#c06a28", "#17807a", "#7b8691"
STYLES = {
    "exact": ("Exact", "#111111", "o"), "different": ("Different category", ORANGE, "s"),
    "clip": ("No Hebbian", GRAY, "D"),
    "pool_only": ("Pool only", INDIGO, "o"), "full": ("Full", BRICK, "s"),
    "scale_both": ("Matched scale", TEAL, "^"), "shuffled": ("Shuffled", GRAY, "D"),
    "direct_rep4_pool": ("Pool only", INDIGO, "o"),
    "direct_rep4_full": ("Full", BRICK, "s"),
    "direct_rep4_shuffle": ("Shuffled", TEAL, "^"),
    "direct_rep4": ("Native rep4", BRICK, "o"), "direct_prd5": ("Native prd5", ORANGE, "s"),
    "sae_rep4": ("SAE rep4", INDIGO, "^"), "sae_prd5": ("SAE prd5", PURPLE, "D"),
    "eps0": ("ε = 0", INDIGO, "o"), "eps005": ("ε = 0.005", BRICK, "s"),
    "within": ("Within-class", INDIGO, "o"), "between": ("Between-class", ORANGE, "s"),
    "fisher": ("Fisher ratio", INK, "D"),
    "fc1": ("FC1", INDIGO, "o"), "fc2": ("FC2", BRICK, "s"),
    "conv4": ("Conv4", INDIGO, "o"), "conv5": ("Conv5", BRICK, "s"),
    "fixed": ("Fixed SVM", INDIGO, "o"), "shared": ("Shared refit SVM", BRICK, "s"),
    "kappa": ("Memory match κ", INDIGO, "o"), "threshold": ("Subtraction threshold", BRICK, "s"),
    "small_lambda": ("λ = 0.01", INDIGO, "o"), "large_lambda": ("λ = 0.03", BRICK, "s"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b9c4cf",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def select(frame, **criteria):
    for key, value in criteria.items():
        frame = frame[frame[key].eq(value)]
    return frame


def ordered(frame, key, values):
    part = frame.set_index(key)
    assert part.index.is_unique, key
    return part.loc[list(values)]


def outcome(frame, key, values, field, multiplier=1):
    return ordered(frame, key, values)[field].to_numpy(dtype=float) * multiplier


class Atlas:
    def __init__(self, pdf):
        self.pdf = pdf
        self.sources = {}
        self.index = []
        self.values = []
        self.axis_notes = []
        self.cache = {}

    def source(self, folder, name, ext="csv"):
        path = ROOT / "docs" / folder / f"{name}.{ext}"
        if path not in self.cache:
            self.sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.cache[path] = (pd.read_csv(path) if ext == "csv"
                                else json.loads(path.read_text(encoding="utf-8")))
        return self.cache[path]

    def figure(self, name, title, context, samples, footer, rows=2, legends=(), extra=(), legend_lines=False):
        fig, axes = plt.subplots(rows, 2, figsize=(40 / 3, 7.5), squeeze=False)
        fig.subplots_adjust(left=.095, right=.958, top=.732, bottom=.144,
                            wspace=.32, hspace=.66 if rows == 2 else .28)
        fig.text(.062, .928, title, fontsize=23, weight="semibold", va="center")
        fig.text(.063, .862, context, fontsize=13.3, color=MUTED, va="center")
        fig.text(.063, .817, "Clean targets", fontsize=11.3, color=GREEN, weight="semibold", va="center")
        fig.text(.172, .817, samples, fontsize=11.2, color=MUTED, va="center")
        fig.text(.063, .038, footer, fontsize=10.3, color=MUTED, va="center")
        if legends or extra:
            handles = [Line2D([], [], marker=STYLES[k][2], color=STYLES[k][1], lw=1.7,
                              linestyle=self.line_style(k) if legend_lines else "none",
                              markerfacecolor="none" if legend_lines and self.overlay(k) else STYLES[k][1],
                              markersize=5.6, label=STYLES[k][0]) for k in legends]
            fig.legend(handles=handles + list(extra), loc="center right", bbox_to_anchor=(.958, .815),
                       ncol=len(handles) + len(extra), frameon=False, fontsize=10,
                       handlelength=1.35, handletextpad=.45, columnspacing=1.05, borderaxespad=0)
        self.index.append(dict(file=name, title=title, context=context, footer=footer))
        self.name = name
        return fig, axes

    @staticmethod
    def overlay(key):
        return key in ("direct_rep4_full", "eps005")

    def line_style(self, key):
        return (0, (4, 2)) if self.overlay(key) else "-"

    def panel(self, ax, dataset, panel, x, labels, ylabel, *, zero=False, time=False,
              adaptive=False, reference=None):
        title = DATASETS[dataset] + (f"  /  {panel}" if panel else "")
        ax.set_title(title, loc="left", fontsize=13.1, pad=8, weight="semibold")
        ax.set_ylabel(ylabel, labelpad=7)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=3, labelsize=10.5, pad=4)
        if time:
            ax.set_xlim(-.18, 14.18)
            ax.set_xlabel("Target timestep", labelpad=4)
            ax.axvline(4.5, color="#d8dfe6", lw=.85, zorder=0)
            ax.axvline(9.5, color="#d8dfe6", lw=.85, zorder=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3g}"))
        ref = 0 if zero else reference
        if ref is not None and not adaptive:
            ax.axhline(ref, color=GRAY, lw=1, ls=(0, (3, 3)), zorder=0)
        if adaptive:
            # Only absolute accuracy/AUC panels use the prior data-dependent scale.
            values = np.concatenate([line.get_ydata() for line in ax.lines if len(line.get_ydata()) > 2] or
                                    [line.get_ydata() for line in ax.lines])
            scale = Axis(values, reference=reference, physical=(0, 100), compress=True)
            padding = .048 * np.ptp(scale.mapped)
            low, high = scale.inverse(scale.forward([min(values.min(), reference), values.max()]) + [-padding, padding])
            low, high = min(scale.lo, max(0, low)), max(scale.hi, min(100, high))
            for segment in scale.compressed:
                if segment["low"] == scale.lo:
                    segment["low"] = float(low)
                if segment["high"] == scale.hi:
                    segment["high"] = float(high)
            scale.lo, scale.hi = float(low), float(high)
            scale.apply(ax, unit="%")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3g}"))
            self.axis_notes.append(dict(figure=self.name, dataset=dataset, panel=panel, **scale.meta()))
        else:
            self.axis_notes.append(dict(figure=self.name, dataset=dataset, panel=panel,
                                        limits=[float(y) for y in ax.get_ylim()], compressed_segments=[]))
        if not time and len(x) > 1:
            ax.margins(x=.12)

    def draw(self, ax, dataset, panel, key, x, y, *, labels=None, limits=None,
             connected=False, hollow=False, variant=""):
        label, color, marker = STYLES[key]
        yy = np.asarray(y, float)
        xx = np.asarray(x)
        assert len(xx) == len(yy) and np.isfinite(yy).all(), (self.name, dataset, panel, key)
        low, high = (np.full(len(yy), np.nan),) * 2
        if limits is not None:
            low, high = np.asarray(limits, float).T
            assert np.isfinite([low, high]).all() and np.all(low <= high)
            ax.vlines(xx, low, high, color=color, linewidth=1.3, alpha=.7, zorder=2)
            ax.plot(xx, low, linestyle="none", marker="_", color=color, markersize=5, alpha=.7)
            ax.plot(xx, high, linestyle="none", marker="_", color=color, markersize=5, alpha=.7)
        marker_face = "none" if connected and self.overlay(key) else "white" if hollow else color
        line, = ax.plot(xx, yy, linestyle=self.line_style(key) if connected else "none", linewidth=1.85,
                        marker=marker, markersize=5.8, markeredgewidth=1.4,
                        color=color, markerfacecolor=marker_face, zorder=3)
        np.testing.assert_array_equal(line.get_ydata(), yy)
        for xi, yi, lo, hi in zip(labels if labels is not None else xx, yy, low, high):
            self.values.append(dict(figure=self.name, dataset=dataset, panel=panel, series=label, variant=variant,
                                    x=xi, value=yi, ci_low=lo, ci_high=hi))

    def save(self, fig):
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        renderer = fig.canvas.get_renderer()
        for artist in fig.texts + fig.legends:
            bbox = artist.get_window_extent(renderer)
            assert bbox.x0 >= 0 and bbox.x1 <= width and bbox.y0 >= 0 and bbox.y1 <= height, self.name
        panels = [ax.get_tightbbox(renderer) for ax in fig.axes]
        for idx, box in enumerate(panels):
            assert box.x0 >= 0 and box.x1 <= width and box.y0 >= 0 and box.y1 <= height, (self.name, idx)
            assert not any(box.overlaps(other) for other in panels[:idx]), (self.name, idx)
        for ext in ("png", "svg"):
            fig.savefig(DEST / f"{self.name}.{ext}", dpi=180)
        self.pdf.savefig(fig)
        plt.close(fig)


def geometry_panel(atlas, ax, dataset, title, labels, series, *, two_scopes=False, unit="Change from reference (%)"):
    x = np.arange(len(labels))
    for j, metric in enumerate(METRICS):
        for scope, estimates in series.items():
            y, ci = estimates[metric]
            step = (-.06 if scope == "local" else .06) if two_scopes else 0
            atlas.draw(ax, dataset, title, metric, x + (j - 1) * .21 + step, y,
                       labels=labels, limits=ci, hollow=scope == "local", variant=scope)
    atlas.panel(ax, dataset, title, x, labels, unit, zero=True)


def grouped_panel(atlas, ax, ds, title, labels, data, ylabel, *, zero=False, adaptive=False,
                  reference=None, connected=False):
    x = np.arange(len(labels))
    offsets = np.linspace(-.23, .23, len(data)) if len(data) > 1 and not connected else np.zeros(len(data))
    for shift, (key, (values, ci)) in zip(offsets, data.items()):
        atlas.draw(ax, ds, title, key, x + shift, values, labels=labels, limits=ci, connected=connected)
    atlas.panel(ax, ds, title, x, labels, ylabel, zero=zero, adaptive=adaptive, reference=reference)


def score_intervals(rows, scale=1):
    return rows["estimate"].to_numpy() * scale, rows[["ci_low", "ci_high"]].to_numpy() * scale


def no_ci(values):
    return np.asarray(values), None


def fc_previous(a):
    layers = ("fc1", "fc2")
    labels = tuple(LAYERS[k] for k in layers)
    methods = ("pool_only", "full", "scale_both", "shuffled")
    method_labels = ("Pool only", "Full", "Matched\nscale", "Shuffled")
    hist = a.source(FC, "history_contrasts")
    metrics = a.source(FC, "layer_metrics")
    readouts = a.source(FC, "readouts")

    fig, axes = a.figure("01_fc_previous_activity", "FC repetition survives pool-only suppression",
                         "Previous five-step FC cascade  /  frozen development batch", "ImageNet n = 16  ·  THINGS n = 18",
                         "Zero denotes equal activity across histories. Positive repetition suppression denotes a lower Exact response.",
                         legends=layers)
    for col, d in enumerate(DATASETS):
        frame = select(metrics, dataset=d)
        m = select(hist, dataset=d, history="exact")
        for row, (title, ylabel) in enumerate((("Raw mean activity", "Exact − Different (raw units)"),
                                               ("Paired repetition", "Repetition suppression (%)"))):
            keys = tuple(k for k in methods if row or k != "scale_both")
            ticks = tuple(method_labels[methods.index(k)] for k in keys)
            data = {}
            for layer in layers:
                if row == 0:
                    e, diff = [outcome(select(frame, layer=layer, history=h), "method", keys, "activation")
                               for h in HISTORIES]
                    values = e - diff
                else:
                    values = outcome(select(m, layer=layer), "name", keys, "exact_different_suppression")
                data[layer] = no_ci(values)
            grouped_panel(a, axes[row, col], d, title, ticks, data, ylabel, zero=True)
    a.save(fig)

    for name, title, comparisons in (
        ("02_fc_previous_full_vs_pool", "FC gain expands within-class dispersion",
         tuple((h.capitalize(), "full", h, "pool_only", h) for h in HISTORIES)),
        ("02_fc_previous_full_vs_scale", "FC geometry after matching mean activity",
         tuple((h.capitalize(), "full", h, "scale_both", h) for h in HISTORIES)),
        ("03_fc_previous_history_geometry", "FC geometry across preview histories",
         (("Exact vs no preview", "full", "exact", "full", "none"),
          ("Exact vs Different", "full", "exact", "full", "different"))),
    ):
        context = {
            "02_fc_previous_full_vs_pool": "Previous FC cascade  /  Full versus pool only",
            "02_fc_previous_full_vs_scale": "Previous FC cascade  /  Full versus matched scale",
            "03_fc_previous_history_geometry": "Previous FC cascade  /  Full rule",
        }[name]
        fig, axes = a.figure(name, title, context + "  /  normalized representations",
                             "ImageNet n = 16  ·  THINGS n = 18",
                             "Fisher ratio = between-class / within-class dispersion. Zero is the labeled reference.",
                             legends=METRICS)
        for row, (panel, method, h, control, ref) in enumerate(comparisons):
            for col, d in enumerate(DATASETS):
                frame = select(metrics, dataset=d)
                series = {}
                for metric in METRICS:
                    y, baseline = [outcome(select(frame, method=k, history=v), "layer", layers, metric)
                                   for k, v in ((method, h), (control, ref))]
                    series[metric] = no_ci(100 * (y / baseline - 1))
                geometry_panel(a, axes[row, col], d, panel, labels, {"actual": series})
        a.save(fig)

    f_local = a.source(FC, "fc2_local")
    fig, axes = a.figure("03_fc_previous_fc2_local", "FC2 output relative to its actual input",
                         "Previous FC cascade  /  each method retains its actual FC1-to-FC2 input",
                         "ImageNet n = 16  ·  THINGS n = 18",
                         "The subtraction stage reduces activity. The geometry comparison uses the same method’s FC2 input.",
                         legends=HISTORIES)
    for row, (field, label) in enumerate((("local_suppression", "Local suppression (%)"),
                                          ("local_fisher_change", "Local Fisher change (%)"))):
        for col, d in enumerate(DATASETS):
            f = select(f_local, dataset=d)
            data = {h: no_ci(outcome(select(f, history=h), "method", methods, field)) for h in HISTORIES}
            grouped_panel(a, axes[row, col], d, "FC2", method_labels, data, label, zero=True)
    a.save(fig)

    fig, axes = a.figure("03_fc_previous_response_profile", "FC responses concentrate on preferred units",
                         "Previous FC cascade  /  Exact versus no preview",
                         "ImageNet n = 16  ·  THINGS n = 18",
                         "Preferred units are the top quartile from training data. Activity density is the nonzero fraction.",
                         legends=layers)
    for row, (field, title) in enumerate((("preferred_pp", "Preferred activity share"),
                                          ("nonzero_pp", "Nonzero fraction"))):
        for col, d in enumerate(DATASETS):
            f = select(hist, dataset=d, history="exact")
            data = {l: no_ci(outcome(select(f, layer=l), "name", methods, field)) for l in layers}
            grouped_panel(a, axes[row, col], d, title, method_labels, data,
                          "Difference from no preview (pp)", zero=True)
    a.save(fig)

    for read_metric, title, ylabel, scale in (
        ("ba", "FC2 balanced accuracy across controls", "Balanced accuracy (%)", 100),
        ("auc", "FC2 macro AUC across controls", "Macro AUC (%)", 100),
        ("margin_change_vs_none", "FC2 margin relative to no preview", "True-class margin difference", 1),
    ):
        fig, axes = a.figure(f"04_fc_previous_readout_{read_metric}", title,
                             "Previous five-step FC cascade  /  frozen development batch", "ImageNet n = 16  ·  THINGS n = 18",
                             "Shared refit heads were fitted on training folds, separately for each method.",
                             legends=HISTORIES)
        for row, head in enumerate(("fixed", "shared")):
            for col, d in enumerate(DATASETS):
                f = select(readouts, dataset=d, layer="fc2", kind=head)
                data = {h: no_ci(outcome(select(f, history=h), "method", methods, read_metric, scale))
                        for h in HISTORIES}
                grouped_panel(a, axes[row, col], d, STYLES[head][0], method_labels, data, ylabel,
                              zero=read_metric.startswith("margin"))
        a.save(fig)

    train = a.source(FC, "training_screen")
    fig, axes = a.figure("05_fc_training_screen", "Training-fold screen for the previous FC rule",
                         "Six prespecified settings  /  minimum across FC1, FC2 and both histories",
                         "ImageNet n = 256  ·  THINGS n = 320",
                         "The frozen full setting used ε = 0.03 and λ = 0.03. The activity target started at 1%.",
                         legends=("small_lambda", "large_lambda"))
    for row, (field, label) in enumerate((("suppression", "Minimum suppression (%)"),
                                          ("fisher_change", "Minimum Fisher change (%)"))):
        for col, d in enumerate(DATASETS):
            frame = select(train, dataset=d).groupby("name")[field].min()
            data = {key: no_ci([frame[f"eps{eps}_lambda{lam}"] for eps in (0., .01, .03)])
                    for key, lam in (("small_lambda", .01), ("large_lambda", .03))}
            grouped_panel(a, axes[row, col], d, "", ("ε = 0", "ε = 0.01", "ε = 0.03"),
                          data, label, zero=row == 1)
            if row == 0:
                axes[row, col].axhline(1, color=GRAY, lw=1, ls="--")
    a.save(fig)

    diagnostics = a.source("fc_sae_gain_adjust_20261106", "diagnosis", "json")
    fig, axes = a.figure("05_fc_initial_gain_diagnostic", "Initial lightweight SAE gain and subtraction strength",
                         "Initial gain-only FC pilot  /  first lateral step  /  Exact",
                         "ImageNet n = 16  ·  THINGS n = 18",
                         "RMS uses active units. The subtraction threshold is shown relative to the active-unit mean.")
    diagnostics = pd.DataFrame(diagnostics["diagnostics"])
    for row, (field, label) in enumerate((("active_modulation_rms", "Active-unit modulation RMS (%)"),
                                          ("pool_over_active_mean", "Threshold / active mean (%)"))):
        for col, d in enumerate(DATASETS):
            values = outcome(select(diagnostics, dataset=d, history="exact"), "layer", layers, field, 100)
            grouped_panel(a, axes[row, col], d, "", labels, {"full": no_ci(values)}, label, zero=True)
    a.save(fig)


def timelines(a, frame, case_key, cases, name, branch, pairs, fields=("mean",)):
    x = np.arange(15)
    for pair in pairs:
        for field in fields:
            hist_baseline = "different_category" if case_key == "case" else "different"
            desc = {"mean": "Mean activation", "fisher": "Fisher ratio",
                    "within": "Within-class dispersion", "between": "Between-class dispersion"}[field]
            ylabel = "Exact − Different (raw units)" if field == "mean" else "Exact − Different"
            fig, axes = a.figure(f"{name}_{pair[0]}_{pair[1]}_{field}",
                                 f"{desc} after Exact and Different previews",
                                 branch + "  /  " + " and ".join(LAYERS[l] for l in pair),
                                 "ImageNet n = 16  ·  THINGS n = 20",
                                 "Zero is Different category at each timestep. Vertical guides mark the three analysis windows.",
                                 legends=cases, legend_lines=True)
            for row, layer in enumerate(pair):
                for col, d in enumerate(DATASETS):
                    ax = axes[row, col]
                    f = select(frame, dataset=d, layer=layer)
                    for k in cases:
                        e, diff = [outcome(select(f, **{case_key: k}, history=h), "t", x, field)
                                   for h in ("exact", hist_baseline)]
                        a.draw(ax, d, layer, k, x, e - diff, connected=True)
                    a.panel(ax, d, LAYERS[layer], x[::2], x[::2], ylabel, zero=True, time=True)
            a.save(fig)


def conv_previous(a):
    steps = a.source(CONV, "step_metrics")
    win = a.source(CONV, "window_metrics")
    local = a.source(LOCAL, "conv4_local_exact_all_windows")
    rep = a.source(LOCAL, "conv4_conv5_repetition_all_windows")
    ci = select(a.source(CONV, "paired_concept_bootstrap"), metric="mean", control="different_category")
    timelines(a, steps, "case", OLD_CASES, "06_conv_previous_timeline", "Previous recurrent native rep4 rule",
              (("conv4", "conv5"), ("fc1", "fc2")))
    timelines(a, steps, "case", OLD_CASES[:3], "07_conv_previous_timeline", "Previous recurrent native rep4 rule",
              (("conv4", "conv5"), ("fc1", "fc2")), fields=METRICS)

    fig, axes = a.figure("08_conv_previous_repetition", "Conv4 repetition suppression emerges after the early window",
                         "Previous recurrent native rep4 rule  /  Exact versus Different category", "ImageNet n = 16  ·  THINGS n = 20",
                         "Positive values denote lower Exact activity. Intervals show paired concept-bootstrap 95% ranges where archived.",
                         legends=OLD_CASES)
    for row, layer in enumerate(("conv4", "conv5")):
        for col, d in enumerate(DATASETS):
            data = {}
            for k in OLD_CASES:
                f = ordered(select(rep, dataset=d, case=k, layer=layer), "window", WINDOWS)
                c = select(ci, dataset=d, case=k, layer=layer)
                bars = None if c.empty else ordered(c, "window", WINDOWS)[["ci_low", "ci_high"]].to_numpy()
                if bars is not None:
                    np.testing.assert_allclose(f["suppression_ED_pct"], ordered(c, "window", WINDOWS)["effect"])
                data[k] = (f["suppression_ED_pct"], bars)
            grouped_panel(a, axes[row, col], d, LAYERS[layer], WINDOW_LABELS, data,
                          "Repetition suppression (%)", zero=True)
    a.save(fig)

    fig, axes = a.figure("09_conv_previous_branch_repetition", "Conv4 repetition across graph sources",
                         "Previous full rule  /  Exact versus Different category", "ImageNet n = 16  ·  THINGS n = 20",
                         "Points summarize separate timestep windows. Whiskers show paired concept-bootstrap 95% intervals.",
                         rows=1, legends=VARIANTS)
    for col, d in enumerate(DATASETS):
        data = {}
        for v in VARIANTS:
            rows = ordered(select(ci, dataset=d, case=v + "_full", layer="conv4"), "window", WINDOWS)
            data[v] = (rows["effect"], rows[["ci_low", "ci_high"]].to_numpy())
        grouped_panel(a, axes[0, col], d, "", WINDOW_LABELS, data,
                      "Repetition suppression (%)", zero=True)
    a.save(fig)

    fig, axes = a.figure("10_conv_previous_local", "Conv4 output relative to its actual input",
                         "Previous full rule  /  Exact  /  early window t0–4", "ImageNet n = 16  ·  THINGS n = 20",
                         "Top: local mean reduction. Bottom: geometry of unit-normalized Conv4 maps.",
                         legends=METRICS)
    for col, d in enumerate(DATASETS):
        f = ordered(select(local, dataset=d, window="early"), "case", [k + "_full" for k in VARIANTS])
        x = np.arange(len(VARIANTS))
        a.draw(axes[0, col], d, "Local suppression", "conv4", x, f["local_suppression_pct"], labels=VARIANT_LABELS)
        a.panel(axes[0, col], d, "Local suppression", x, VARIANT_LABELS, "Local suppression (%)", zero=True)
        data = {k: no_ci(f[k + "_vs_input_pct"]) for k in METRICS}
        geometry_panel(a, axes[1, col], d, "Local geometry", VARIANT_LABELS, {"actual": data})
    a.save(fig)

    for suffix, title, context, control in (
        ("pool", "Geometry after adding the previous recurrent gain",
         "Native rep4 graph  /  Full versus pool only under Exact", "direct_rep4_pool"),
        ("no_hebbian", "Geometry relative to the earlier no-Hebbian run",
         "Native rep4 graph  /  Full versus no Hebbian under Exact", "clip"),
    ):
        fig, axes = a.figure(f"11_conv_previous_full_vs_{suffix}", title, context,
                             "ImageNet n = 16  ·  THINGS n = 20",
                             "Comparisons use the actual recurrent trajectories. Fisher is computed per step and averaged in each window.",
                             legends=METRICS)
        for row, layer in enumerate(("conv4", "conv5")):
            for col, d in enumerate(DATASETS):
                f = select(win, dataset=d, layer=layer, history="exact")
                data = {}
                for metric in METRICS:
                    full, reference = [outcome(select(f, case=k), "window", WINDOWS, metric)
                                       for k in ("direct_rep4_full", control)]
                    data[metric] = no_ci(100 * (full / reference - 1))
                geometry_panel(a, axes[row, col], d, LAYERS[layer], WINDOW_LABELS, {"actual": data})
        a.save(fig)

    fig, axes = a.figure("11_conv_previous_fc2_geometry", "Downstream FC2 geometry across convolutional variants",
                         "Previous recurrent rule  /  Exact  /  early window t0–4",
                         "ImageNet n = 16  ·  THINGS n = 20",
                         "Full native-channel runs shrink between-class dispersion. Full SAE runs expand within-class dispersion.",
                         legends=METRICS)
    geometry = a.source(DIAG, "geometry_full_vs_controls")
    for row, (control, panel) in enumerate((("pool", "Full vs pool only"), ("scale", "Full vs matched scale"))):
        for col, d in enumerate(DATASETS):
            f = ordered(select(geometry, dataset=d, control=control), "branch_source", VARIANTS)
            data = {k: no_ci(f[k + "_full_vs_control_pct"]) for k in METRICS}
            geometry_panel(a, axes[row, col], d, panel, VARIANT_LABELS, {"actual": data})
    a.save(fig)

    fig, axes = a.figure("12_conv_previous_mechanism", "Gain magnitude, memory match and inhibition",
                         "Previous full rule  /  target t0–4  /  lateral step 5", "ImageNet n = 16  ·  THINGS n = 20",
                         "Gain RMS pools all units. Match and threshold effects use paired Exact/Different ratios.",
                         legends=("kappa", "threshold"))
    diagnostics = a.source(DIAG, "mechanism_summary")
    for col, d in enumerate(DATASETS):
        f = ordered(select(diagnostics, dataset=d), "branch_source", VARIANTS)
        x = np.arange(len(VARIANTS))
        a.draw(axes[0, col], d, "Actual modulation", "conv4", x, f["gain_RMS_pct"], labels=VARIANT_LABELS)
        a.panel(axes[0, col], d, "Actual modulation", x, VARIANT_LABELS, "Modulation RMS (%)", zero=True)
        data = {key: no_ci(f[field]) for key, field in (("kappa", "kappa_ED_pct"), ("threshold", "threshold_ED_pct"))}
        grouped_panel(a, axes[1, col], d, "History effect", VARIANT_LABELS, data,
                      "Exact relative to Different (%)", zero=True)
    a.save(fig)

    stages = ("conv4_input", "conv4", "conv5", "fc1", "fc2")
    fig, axes = a.figure("13_conv_previous_propagation", "Conv4 activity reduction weakens downstream",
                         "Previous full rule  /  Exact versus same-history no Hebbian  /  t0–4", "ImageNet n = 16  ·  THINGS n = 20",
                         "Relative mean changes use each layer’s own baseline. Points trace stages in the actual network.",
                         rows=1, legends=VARIANTS, legend_lines=True)
    change = a.source(DIAG, "propagation_mean_changes")
    for col, d in enumerate(DATASETS):
        f = ordered(select(change, dataset=d), "case", [k + "_full" for k in VARIANTS])
        labels = ("Conv4\ninput", "Conv4\noutput", "Conv5", "FC1", "FC2")
        data = {k: no_ci(f.loc[k + "_full", list(stages)]) for k in VARIANTS}
        grouped_panel(a, axes[0, col], d, "", labels, data, "Relative mean change (%)", zero=True, connected=True)
    a.save(fig)

    old_svm = a.source(CONV, "fixed_svm").rename(columns={"case": "condition",
        "time_average_accuracy": "accuracy", "macro_auc": "auc"}).replace({"different_category": "different"})
    fig, axes = a.figure("14_conv_previous_accuracy_variants", "FC2 accuracy across convolutional variants",
                         "Previous full rule  /  fixed SVM  /  early window t0–4",
                         "ImageNet n = 16  ·  THINGS n = 20",
                         "Differences are relative to the same-history no-Hebbian run. Accuracy is averaged across timesteps.")
    for row, h in enumerate(HISTORIES):
        for col, d in enumerate(DATASETS):
            f = select(old_svm, dataset=d, layer="fc2", history=h, window="early")
            full = outcome(f, "condition", [k + "_full" for k in VARIANTS], "accuracy")
            baseline = outcome(f, "condition", ["clip"], "accuracy")
            grouped_panel(a, axes[row, col], d, STYLES[h][0], VARIANT_LABELS,
                          {"full": no_ci(100 * (full - baseline))}, "Accuracy difference (pp)", zero=True)
    a.save(fig)

    readout_windows(a, old_svm, "14_conv_previous", OLD_CASES[:3],
                    "Previous native rep4 rule  /  fixed FC2 SVM")


def readout_windows(a, frame, prefix, cases, subtitle):
    f = select(frame, layer="fc2")
    for metric, title, label, mult in (
        ("accuracy", "FC2 time-averaged accuracy", "Accuracy (%)", 100),
        ("auc", "FC2 macro AUC", "Macro AUC (%)", 100),
        ("margin", "FC2 true-class margin", "True-class margin", 1),
    ):
        for mode in (("adaptive", "linear") if metric == "accuracy" else ("linear",)):
            adaptive = mode == "adaptive"
            footer = {
                "accuracy": "Window mean of timestep accuracy. Dashed line is chance (25% / 20%).",
                "auc": "Macro one-vs-rest AUC of the window-averaged scores. Dashed line is chance.",
                "margin": "Window mean of the true-class score minus the largest competing score.",
            }[metric]
            if adaptive:
                footer += " Gray bands mark axis compression."
            fig, axes = a.figure(f"{prefix}_{metric}_{mode}", title, subtitle,
                                 "ImageNet n = 16  ·  THINGS n = 20", footer, legends=cases)
            for row, h in enumerate(HISTORIES):
                for col, d in enumerate(DATASETS):
                    data = {}
                    for case in cases:
                        sel = select(f, dataset=d, condition=case, history=h)
                        if "metric" in sel:
                            data[case] = no_ci(outcome(select(sel, metric=metric), "window", WINDOWS, "estimate", mult))
                        else:
                            data[case] = no_ci(outcome(sel, "window", WINDOWS, metric, mult))
                    ref = {"imagenet": 25, "things": 20}[d] if metric == "accuracy" else 50 if metric == "auc" else None
                    grouped_panel(a, axes[row, col], d, STYLES[h][0], WINDOW_LABELS, data, label,
                                  adaptive=adaptive, reference=ref)
            a.save(fig)


def new_geometry(a, geo, branch):
    is_fc = branch == "fc"
    keys = ("fc1", "fc2") if is_fc else ("middle", "late")
    labels = ("FC1", "FC2") if is_fc else WINDOW_LABELS[1:]
    xkey = "layer" if is_fc else "window"
    title = "Single post-suppression gain in the FC cascade" if is_fc else "Single post-suppression gain in recurrent Conv4"
    stage_legend = [Line2D([], [], linestyle="none", marker="o", markerfacecolor=fill, color=INK,
                           label=label) for fill, label in (("white", "Local"), (INK, "Network"))]
    fig, axes = a.figure(f"15_new_{branch}_local_vs_network", title,
                         "ε = 0.005 versus ε = 0  /  matched local replay and actual network",
                         "ImageNet n = 16  ·  THINGS n = " + ("18" if is_fc else "20"),
                         "Whiskers show paired concept-bootstrap 95% intervals for Fisher. All points use normalized representations.",
                         legends=METRICS, extra=stage_legend)
    f = select(geo, branch=branch, unit="percent")
    if not is_fc:
        f = select(f, layer="conv4")
    for row, h in enumerate(HISTORIES):
        for col, d in enumerate(DATASETS):
            data = {}
            for scope, contrast in (("local", "local_gain"), ("network", "network_gain")):
                f_scope = select(f, comparison=contrast, dataset=d, history=h)
                values = {}
                for k in METRICS:
                    r = ordered(select(f_scope, metric=k), xkey, keys)
                    values[k] = score_intervals(r) if k == "fisher" else no_ci(r["estimate"])
                data[scope] = values
            geometry_panel(a, axes[row, col], d, STYLES[h][0], labels, data, two_scopes=True)
    a.save(fig)


def new_conv5_geometry(a, geo):
    fig, axes = a.figure("15_new_conv5_network_geometry", "Geometry in Conv5 after Conv4 post-suppression gain",
                         "ε = 0.005 versus ε = 0  /  downstream recurrent responses", "ImageNet n = 16  ·  THINGS n = 20",
                         "Whiskers show paired concept-bootstrap 95% intervals for Fisher. Zero is the pool-only run.",
                         legends=METRICS)
    f = select(geo, branch="conv", layer="conv5", unit="percent", comparison="network_gain")
    for row, h in enumerate(HISTORIES):
        for col, d in enumerate(DATASETS):
            estimates = {}
            for metric in METRICS:
                r = ordered(select(f, dataset=d, history=h, metric=metric), "window", WINDOWS)
                estimates[metric] = score_intervals(r) if metric == "fisher" else no_ci(r["estimate"])
            geometry_panel(a, axes[row, col], d, STYLES[h][0], WINDOW_LABELS, {"network": estimates})
    a.save(fig)


def new_repetition(a, rep, raw):
    for branch, layers, sample in (("fc", ("fc1", "fc2"), "18"), ("conv", ("conv4", "conv5"), "20")):
        fig, axes = a.figure(f"16_new_{branch}_repetition", "Repetition suppression after single post-suppression gain",
                             ("FC cascade" if branch == "fc" else "Native rep4 recurrence") + "  /  actual network",
                             "ImageNet n = 16  ·  THINGS n = " + sample,
                             "Positive repetition suppression denotes lower Exact activity. Whiskers show paired concept-bootstrap 95% intervals.",
                             legends=NEW_CASES)
        for col, d in enumerate(DATASETS):
            if branch == "fc":
                names = ("FC1", "FC2")
                f = select(rep, branch=branch, dataset=d)
                data = {case: score_intervals(ordered(select(f, condition=case, unit="percent"), "layer", layers))
                        for case in NEW_CASES}
                grouped_panel(a, axes[1, col], d, "Paired repetition", names, data,
                              "Repetition suppression (%)", zero=True)
                f = select(raw, branch=branch, dataset=d, metric="mean", window="feedforward")
                data = {}
                for case in NEW_CASES:
                    exact, diff = [outcome(select(f, condition=case, history=h), "layer", layers, "estimate")
                                   for h in HISTORIES]
                    data[case] = no_ci(exact - diff)
                grouped_panel(a, axes[0, col], d, "Raw mean activity", names, data,
                              "Exact − Different (raw units)", zero=True)
            else:
                for row, l in enumerate(layers):
                    f = select(rep, branch=branch, dataset=d, layer=l, unit="percent")
                    data = {case: score_intervals(ordered(select(f, condition=case), "window", WINDOWS))
                            for case in NEW_CASES}
                    grouped_panel(a, axes[row, col], d, LAYERS[l], WINDOW_LABELS, data,
                                  "Repetition suppression (%)", zero=True)
        a.save(fig)

    fig, axes = a.figure("17_new_conv_repetition_difference", "Small changes in repetition after recurrent propagation",
                         "Native rep4 recurrence  /  ε = 0.005 minus ε = 0", "ImageNet n = 16  ·  THINGS n = 20",
                         "Whiskers are paired concept-bootstrap 95% intervals. Each panel has an independently scaled linear axis.")
    for row, l in enumerate(("conv4", "conv5")):
        for col, d in enumerate(DATASETS):
            f = select(rep, branch="conv", dataset=d, layer=l, condition="eps005", unit="pp", reference="eps0")
            data = {l: score_intervals(ordered(f, "window", WINDOWS))}
            grouped_panel(a, axes[row, col], d, LAYERS[l], WINDOW_LABELS, data,
                          "Suppression difference (pp)", zero=True)
    a.save(fig)


def new_references(a, geo):
    for suffix, title, comparison, unit in (
        ("history", "Geometry after Exact versus Different previews", "exact_vs_different", "absolute"),
        ("baseline", "Geometry relative to the no-Hebbian baseline", "vs_no_hebb", "percent"),
    ):
        fig, axes = a.figure(f"18_new_geometry_{suffix}", title,
                             "Single post-suppression gain  /  ε = 0.005", "ImageNet n = 16  ·  THINGS n = 18–20",
                             "FC uses the no-preview feedforward baseline. Conv4 uses its same-history no-Hebbian baseline."
                             if suffix == "baseline" else "Each point is the raw difference between the Exact and Different history metrics.",
                             legends=METRICS)
        for row, (branch, key, keys, labels) in enumerate((
            ("fc", "layer", ("fc1", "fc2"), ("FC1", "FC2")),
            ("conv", "window", ("middle", "late"), WINDOW_LABELS[1:]),
        )):
            for col, d in enumerate(DATASETS):
                f = select(geo, dataset=d, branch=branch, comparison=comparison, unit=unit,
                           condition="eps005", history="exact")
                if branch == "conv":
                    f = select(f, layer="conv4")
                values = {k: no_ci(outcome(select(f, metric=k), key, keys, "estimate")) for k in METRICS}
                geometry_panel(a, axes[row, col], d, "FC cascade" if branch == "fc" else "Conv4",
                               labels, {"actual": values},
                               unit="Exact − Different" if unit == "absolute" else "Change from no Hebbian (%)")
        a.save(fig)


def new_fc_readouts(a, values):
    for metric, title, label, mult in (
        ("accuracy", "FC2 accuracy with single post-suppression gain", "Accuracy (%)", 100),
        ("ba", "FC2 balanced accuracy with single post-suppression gain", "Balanced accuracy (%)", 100),
        ("auc", "FC2 macro AUC with single post-suppression gain", "Macro AUC (%)", 100),
        ("margin", "FC2 true-class margin with single post-suppression gain", "True-class margin", 1),
    ):
        fig, axes = a.figure(f"19_new_fc_readout_{metric}", title, "FC cascade  /  fixed and shared refit SVM",
                             "ImageNet n = 16  ·  THINGS n = 18",
                             "A single training-fold refit head is shared across ε values and preview histories.",
                             legends=NEW_CASES)
        for row, head in enumerate(("fixed", "shared")):
            for col, d in enumerate(DATASETS):
                f = select(values, dataset=d, head=head, layer="fc2", metric=metric)
                data = {case: no_ci(outcome(select(f, condition=case), "history", HISTORIES, "estimate", mult))
                        for case in NEW_CASES}
                grouped_panel(a, axes[row, col], d, STYLES[head][0],
                              ("Exact", "Different category"), data, label)
        a.save(fig)


def local_check(a):
    fig, axes = a.figure("20_new_local_acceptance", "Numerical check of local activity conservation",
                         "Single post-suppression gain  /  same target input and frozen memory",
                         "ImageNet n = 16  ·  THINGS n = 18–20",
                         "Maxima span both histories. Conv4 includes all 15 steps. The conservation identity is exact in real arithmetic.")
    for col, d in enumerate(DATASETS):
        fc = a.source(NEW, f"fc_{d}_local", "json")
        conv = a.source(NEW, f"conv_{d}_local", "json")
        assert fc["passed"] and conv["passed"]
        layers = ("fc1", "fc2", "conv4")
        labels = ("FC1", "FC2", "Conv4")
        for row, (panel, ylabel, y) in enumerate((
            ("Activity sum", "Max relative error (×10⁻⁹)",
             [max(obj["local"][f"{h}/{l}"]["max_image_sum_rel"] for h in HISTORIES) * 1e9
              for l, obj in zip(layers, (fc, fc, conv))]),
            ("Repetition suppression", "Max difference (×10⁻⁷ pp)",
             [obj["repetition"][l]["max_abs_pp"] * 1e7 for l, obj in zip(layers, (fc, fc, conv))]),
        )):
            grouped_panel(a, axes[row, col], d, panel, labels, {"eps005": no_ci(y)}, ylabel, zero=True)
    a.save(fig)


def new_run(a):
    geo = a.source(NEW, "geometry_contrasts")
    raw = a.source(NEW, "layer_metrics")
    rep = a.source(NEW, "repetition")
    steps = select(a.source(NEW, "step_metrics"), branch="conv")
    values = a.source(NEW, "readouts")
    local_check(a)
    new_geometry(a, geo, "fc")
    new_geometry(a, geo, "conv")
    new_conv5_geometry(a, geo)
    new_repetition(a, rep, raw)
    timelines(a, steps, "condition", ("clip", *NEW_CASES), "21_new_conv_timeline",
              "Native rep4 recurrence with post-suppression gain", (("conv4", "conv5"), ("fc1", "fc2")))
    timelines(a, steps, "condition", ("clip", *NEW_CASES), "22_new_conv_timeline",
              "Native rep4 recurrence with post-suppression gain",
              (("conv4", "conv5"), ("fc1", "fc2")), fields=METRICS)
    new_references(a, geo)
    new_fc_readouts(a, select(values, branch="fc"))
    readout_windows(a, select(values, branch="conv", head="fixed"), "23_new_conv",
                    ("clip", *NEW_CASES), "Native rep4 recurrence with post-suppression gain  /  fixed FC2 SVM")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    pdf_path = DEST / "Hebbian_Results_Figure_Atlas.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Hebbian Results Figure Atlas"}) as pdf:
        atlas = Atlas(pdf)
        fc_previous(atlas)
        conv_previous(atlas)
        new_run(atlas)
    assert len({x["file"] for x in atlas.index}) == len(atlas.index)
    for rel, digest in atlas.sources.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest
    pd.DataFrame(atlas.values).to_csv(DEST / "plotted_values.csv", index=False, encoding="utf-8-sig")
    for name, value in (("figure_index", atlas.index), ("axis_manifest", atlas.axis_notes), ("sources", atlas.sources)):
        (DEST / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        "# Hebbian 实验汇报用图",
        "",
        "沿用英文 16:9 白底图，ImageNet 居左，THINGS 居右。每张图均可单独插入汇报。"
        "图集按实验组织，依次为既有 FC、既有卷积移植和新增单次增益。",
        "",
        "- 平均激活的历史对比使用 Exact − Different category 的原始单位差值，每个时间点单独计算。"
        "重复抑制另用逐目标 100 × (1 − Exact/Different) 汇总，正值表示 Exact 活动更低。",
        "- 几何分解保留类内、类间和 Fisher。增益与对照的相对差异使用百分比，"
        "新版历史几何图使用原始单位差值。对比图上的零线表示与标注的参考条件相等。",
        "- Accuracy 使用百分数，差值使用百分点。卷积准确率为窗口内逐时间步准确率的平均。"
        "卷积宏 AUC 使用窗口平均分数。带灰色区间的准确率图采用原有自适应压缩，另附线性核对图。",
        "- 置信区间来自已有概念块配对统计，仅在相应归档可用的图中展示。"
        "时间步按原顺序绘出，曲线保留全部实际点。",
        "",
        "## 汇报主图",
        "",
        "- [上一轮 FC 的类内与类间分解](02_fc_previous_full_vs_pool.svg)",
        "- [上一轮 Conv4 和 Conv5 的重复抑制](08_conv_previous_repetition.svg)",
        "- [上一轮卷积增益、匹配与抑制阈值](12_conv_previous_mechanism.svg)",
        "- [本轮 FC 的局部与网络几何](15_new_fc_local_vs_network.svg)",
        "- [本轮 Conv4 的局部与网络几何](15_new_conv_local_vs_network.svg)",
        "- [本轮 Conv4 和 Conv5 的重复抑制](16_new_conv_repetition.svg)",
        "- [本轮 FC2 平衡准确率](19_new_fc_readout_ba.svg) · [卷积实验的 FC2 准确率](23_new_conv_accuracy_adaptive.svg)",
        "",
        "## 图册索引",
        "",
        "| 页 | Figure | PNG | SVG |",
        "|---:|---|---|---|",
    ]
    for page, item in enumerate(atlas.index, 1):
        name, title = item["file"], item["title"]
        notes.append(f"| {page} | {title} | [PNG]({name}.png) | [SVG]({name}.svg) |")
    notes.extend(["", "逐点数值和已有区间见 [plotted_values.csv](plotted_values.csv)。"
                  "纵轴范围与压缩记录见 [axis_manifest.json](axis_manifest.json)。"
                  "输入路径与哈希见 [sources.json](sources.json)。"])
    (DEST / "图册索引.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    archive = DEST.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(DEST.iterdir()):
            if path.is_file():
                z.write(path, path.name)
    print(f"Exported {len(atlas.index)} figures and {len(atlas.values)} plotted values", flush=True)


if __name__ == "__main__":
    main()
