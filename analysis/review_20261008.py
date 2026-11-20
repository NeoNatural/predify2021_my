"""Re-audit saved experiments; no model fitting or GPU inference.

Run with the project's predify2021 Python environment from the repository root.
Only trusted, locally generated experiment pickles are loaded.
"""
from pathlib import Path
import csv
import json
import pickle

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "Hebbian-CNN/Log"
PAPER = LOG / "paper_parafoveal_v1"
OUT = ROOT / "docs/review_20261008"
SEED = 20261008


def load(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, scale=1.0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(values), size=(20000, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(delta=float(scale * values.mean()), low=float(scale * low), high=float(scale * high))


def paired(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    wins, losses = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return dict(n=len(a), accuracy_a_pct=100 * a.mean(), accuracy_b_pct=100 * b.mean(),
                **interval(a.astype(float) - b.astype(float), 100),
                wins=wins, losses=losses,
                exact_p=binomtest(wins, wins + losses).pvalue if wins + losses else 1.0)


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    correctness = load(PAPER / "shared_decoder/analysis/shared_decoder_predictions.pckl")["correctness"]
    carriers = {}
    audits = []
    for path in sorted((PAPER / "causal_pilot_calibrated").glob("*.pckl")):
        result = load(path)
        m = result["run_metadata"]
        key = m["degradation_mode"], m["memory_carrier"], m["prime_identity"]
        assert key not in carriers
        carriers[key] = result
        audits.append(dict(file=str(path.relative_to(ROOT)), n=len(result["dataset_indices"]),
                           unique_images=len(set(result["dataset_indices"])),
                           unique_synsets=len(set(result["imagenet_labels"])),
                           logits_finite=bool(np.isfinite(result["target_logits"]).all()),
                           recovery_finite=bool(np.isfinite(result["representation_recovery"]).all()),
                           balanced_categories=bool(np.all(np.bincount(result["category_labels"]) == 40)),
                           top1_t9_pct=100 * result["target_top1_correct"][:, 9].mean(),
                           top5_t9_pct=100 * result["target_top5_correct"][:, 9].mean()))
    assert len(carriers) == 14
    write_csv("carrier_inventory.csv", audits)
    contrast_rows, recovery_rows, category_rows, interaction_rows = [], [], [], []
    for mode in ("predify_noise", "fractal_occlusion"):
        none_key = next(k for k in carriers if k[0] == mode and k[1] == "none")
        baseline = carriers[none_key]
        for carrier in ("hebb_only", "pcoder_carry", "hebb_and_pcoder"):
            same, different = carriers[(mode, carrier, "same")], carriers[(mode, carrier, "different")]
            for condition in (same, different):
                for field in ("dataset_indices", "imagenet_labels", "category_labels"):
                    assert np.array_equal(condition[field], baseline[field])
                assert [x["degradation_seed"] for x in condition["trial_metadata"]] == [x["degradation_seed"] for x in baseline["trial_metadata"]]
            for timestep in range(10):
                s = correctness[(mode, carrier, "same", "fc2", timestep)]
                d = correctness[(mode, carrier, "different", "fc2", timestep)]
                n = correctness[(*none_key, "fc2", timestep)]
                for name, a, b in (("same-different", s, d), ("same-none", s, n), ("different-none", d, n)):
                    contrast_rows.append(dict(mode=mode, carrier=carrier, layer="fc2", timestep=timestep, contrast=name, **paired(a, b)))
                if timestep == 9:
                    for category in range(5):
                        mask = np.asarray(same["category_labels"]) == category
                        category_rows.append(dict(mode=mode, carrier=carrier, category=category,
                                                  category_name=same["supercategory_names"][category], **paired(s[mask], d[mask])))
            for name, a, b in (("same-different", same, different), ("same-none", same, baseline), ("different-none", different, baseline)):
                for pc in (3, 4):
                    values = a["representation_recovery"][:, 9, pc, 0] - b["representation_recovery"][:, 9, pc, 0]
                    recovery_rows.append(dict(mode=mode, carrier=carrier, pcoder=pc + 1, timestep=9, metric="cosine", contrast=name, **interval(values)))
        for timestep in range(10):
            effects = {}
            for carrier in ("hebb_only", "pcoder_carry", "hebb_and_pcoder"):
                effects[carrier] = (correctness[(mode, carrier, "same", "fc2", timestep)].astype(float)
                                    - correctness[(mode, carrier, "different", "fc2", timestep)].astype(float))
            interaction_rows.append(dict(mode=mode, timestep=timestep, contrast="joint_identity_effect-minus-hebb-minus-pcoder", **interval(effects["hebb_and_pcoder"] - effects["hebb_only"] - effects["pcoder_carry"], 100)))
    write_csv("fixed_fc2_baseline_contrasts.csv", contrast_rows)
    write_csv("recovery_baseline_contrasts.csv", recovery_rows)
    write_csv("fixed_fc2_category_contrasts.csv", category_rows)
    write_csv("carrier_interaction.csv", interaction_rows)

    with (PAPER / "shared_decoder/analysis/shared_decoder_same_vs_different.csv").open() as handle:
        multiple = list(csv.DictReader(handle))
    q = bh([float(row["exact_mcnemar_two_sided_p"]) for row in multiple])
    for row, adjusted in zip(multiple, q):
        row["bh_q_240_cells"] = float(adjusted)
    write_csv("fixed_head_multiplicity_audit.csv", multiple)

    split_rows = []
    for path in sorted((PAPER / "shared_decoder/train").glob("*.pckl")):
        train = load(path)
        test = carriers[(train["run_metadata"]["degradation_mode"], "hebb_and_pcoder", "same")]
        split_rows.append(dict(mode=train["run_metadata"]["degradation_mode"],
                               image_overlap=len(set(train["dataset_indices"]) & set(test["dataset_indices"])),
                               synset_overlap=len(set(train["imagenet_labels"]) & set(test["imagenet_labels"]))))
    write_csv("readout_split_audit.csv", split_rows)

    clean_controls = [load(path) for path in sorted((PAPER / "representations/nohebb").glob("*.pckl"))]
    assert len(clean_controls) == 3
    anchor = clean_controls[0]
    nohebb_equal = all(
        np.array_equal(anchor["dataset_indices"], other["dataset_indices"])
        and np.array_equal(anchor["target_logits"], other["target_logits"])
        and all(np.array_equal(anchor["target_representations"][layer], other["target_representations"][layer])
                for layer in anchor["target_representations"])
        for other in clean_controls[1:]
    )
    nohebb_rows = []
    for other in clean_controls[1:]:
        for field in ("target_logits", "target_top1_correct", "target_top5_correct"):
            a, b = np.asarray(anchor[field], dtype=float), np.asarray(other[field], dtype=float)
            nohebb_rows.append(dict(condition=other["run_metadata"]["condition"], field=field,
                                     exactly_equal=bool(np.array_equal(a, b)), max_abs_difference=float(np.max(np.abs(a-b)))))
        for layer in anchor["target_representations"]:
            a = np.asarray(anchor["target_representations"][layer], dtype=float)
            b = np.asarray(other["target_representations"][layer], dtype=float)
            nohebb_rows.append(dict(condition=other["run_metadata"]["condition"], field=layer,
                                     exactly_equal=bool(np.array_equal(a, b)), max_abs_difference=float(np.max(np.abs(a-b)))))
    write_csv("clean_nohebb_numerical_audit.csv", nohebb_rows)

    # Recompute benchmark endpoints from the exact source files referenced by the existing CSV.
    with (LOG / "comparisons_v1/predify_vs_vanilla_timeline.csv").open() as handle:
        timeline = [row for row in csv.DictReader(handle) if int(row["timestep"]) == 9]
    benchmark_rows = []
    for row in timeline:
        v, p = load(row["vanilla_file"]), load(row["predify_file"])
        labels = np.asarray(p["y_list"])[1::2]
        indices = np.asarray(p["idx_log_list"])[1::2]
        assert np.array_equal(labels, v["y_list"])
        assert np.array_equal(indices, v["idx_log_list"])
        vc = np.asarray(v["logits"]).argmax(-1) == labels
        pc = np.asarray(p["Out_list"])[1::2, 9].argmax(-1) == labels
        assert np.isclose(pc.mean(), float(row["predify_top1"]))
        assert np.isclose(vc.mean(), float(row["vanilla_top1"]))
        benchmark_rows.append(dict(mode=row["degradation_mode"], level=row["degradation_level"],
                                   unique_images=len(set(indices)), **paired(pc, vc)))
        del p, v
    for row, adjusted in zip(benchmark_rows, bh([x["exact_p"] for x in benchmark_rows])):
        row["bh_q_11_t9_cells"] = float(adjusted)
    write_csv("vanilla_t9_raw_audit.csv", benchmark_rows)

    summary = dict(seed=SEED, bootstrap_samples=20000, carrier_files=len(audits),
                   multiplicity_cells=len(multiple),
                   bootstrap_positive=sum(float(x["ci95_low_pp"]) > 0 for x in multiple),
                   exact_p_below_05=sum(float(x["exact_mcnemar_two_sided_p"]) < .05 for x in multiple),
                   bh_below_05=int(np.sum(q < .05)), minimum_q=float(q.min()),
                   benchmark_conditions=len(benchmark_rows), readout_splits=split_rows,
                   clean_nohebb_three_conditions_exactly_equal=nohebb_equal,
                   clean_nohebb_topk_correctness_equal=all(x["exactly_equal"] for x in nohebb_rows if x["field"] in ("target_top1_correct", "target_top5_correct")),
                   interval_scope="Conditional on saved checkpoint, seeds, readout and selected images; image-level bootstrap; exploratory, not confirmatory.")
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for row in contrast_rows:
        if row["timestep"] == 9:
            print(row)
    for row in interaction_rows:
        if row["timestep"] == 9:
            print("INTERACTION", row)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = ["#00897b", "#e08b25", "#546e7a"]
    for ax, mode, title in zip(axes, ("predify_noise", "fractal_occlusion"), ("Noise 0.5", "Occlusion 0.3")):
        for j, contrast in enumerate(("same-different", "same-none", "different-none")):
            selected = [x for x in contrast_rows if x["mode"] == mode and x["timestep"] == 9 and x["contrast"] == contrast]
            mean = np.array([x["delta"] for x in selected])
            low = np.array([x["low"] for x in selected]); high = np.array([x["high"] for x in selected])
            ax.errorbar(np.arange(3) + (j - 1) * .18, mean, yerr=[mean-low, high-mean], fmt="o", capsize=3, label=contrast, color=colors[j])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(3), ["Hebbian", "PCoder state", "Combined"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Fixed FC2 t9 accuracy difference (percentage points)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Same > Different does not establish a benefit over no memory", fontsize=12)
    fig.text(.5, .01, "Paired image bootstrap, 20,000 resamples; N=200; exploratory, unadjusted 95% intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.savefig(OUT / "fixed_readout_baselines.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
