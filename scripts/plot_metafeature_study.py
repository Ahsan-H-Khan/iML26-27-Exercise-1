"""Readable primary figures plus fixed-configuration and component evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from iml_interpretability.metric_development import verify_frozen

verify_frozen()
out = Path("results/metafeatures_v1")
figdir = out / "figures"
figdir.mkdir(exist_ok=True)
df = pd.read_csv(out / "candidates.csv")
manifest = json.loads((out / "condition_manifest.json").read_text(encoding="utf-8"))
aliases = [
    dict(dataset=m["dataset"], seed=m["seed"], condition=m["condition"], **a) for m in manifest for a in m["aliases"]
]
expanded = df.merge(pd.DataFrame(aliases), on=["dataset", "seed", "condition"], validate="many_to_many")
original = df[df.condition == "original"]
metrics = ["balanced_accuracy", "csc", "c_path", "c_encoding", "c_ensemble", "features_used", "total_nodes"]
fixed = expanded.merge(
    original[["dataset", "seed", "model"] + metrics],
    on=["dataset", "seed", "model"],
    suffixes=("", "_original"),
    validate="many_to_one",
)
for metric in metrics:
    fixed[f"delta_{metric}"] = fixed[metric] - fixed[f"{metric}_original"]
fixed.to_csv(out / "fixed_configuration_summary.csv", index=False)
selected = expanded[expanded.selected_csc].copy()
selected.to_csv(out / "selected_model_summary.csv", index=False)
# Paired imbalanced-minus-size-matched effects at the same retention level, model and seed.
balance = fixed[(fixed.experiment == "balance") & (fixed.level < 1)]
paired = balance[~balance.control].merge(
    balance[balance.control],
    on=["dataset", "seed", "model", "level"],
    suffixes=("_imbalanced", "_control"),
    validate="one_to_one",
)
paired = paired.copy()
for metric in metrics:
    paired[f"paired_delta_{metric}"] = paired[f"{metric}_imbalanced"] - paired[f"{metric}_control"]
paired.to_csv(out / "imbalance_paired_comparisons.csv", index=False)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
colors = {42: "#0072B2", 1337: "#D55E00", 2026: "#009E73"}
properties = {
    "samples": ("n_train", "Training examples"),
    "features": ("n_features_raw", "Available features"),
    "balance": ("minority_fraction", "Smallest class share in training (%)"),
}


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(figdir / f"{name}.{ext}", dpi=200)
    plt.close(fig)


for experiment, (x, label) in properties.items():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex="row")
    for axrow, dataset in zip(axes, ("iris", "breast_cancer"), strict=True):
        for seed, color in colors.items():
            for control in [False, True] if experiment == "balance" else [False]:
                part = selected[
                    (selected.dataset == dataset)
                    & (selected.seed == seed)
                    & (selected.experiment == experiment)
                    & (selected.control == control)
                ].sort_values(x)
                if part.empty:
                    continue
                xx = part[x] * (100 if experiment == "balance" else 1)
                ls = ":" if control else "-"
                marker = "s" if control else "o"
                axrow[0].plot(xx, part.balanced_accuracy * 100, color=color, marker=marker, ls=ls)
                axrow[1].plot(xx, part.csc, color=color, marker=marker, ls=ls)
                if not control:
                    axrow[0].plot(xx, part.validation_best_balanced_accuracy * 100, color=color, ls="--", alpha=0.6)
        for ax, ylabel in zip(axrow, ("Validation balanced accuracy (%)", "Selected model CSC"), strict=True):
            ax.set(xlabel=label, ylabel=ylabel, title=dataset.replace("_", " ").title())
            ax.grid(alpha=0.15)
        axrow[0].set_ylim(25, 102)
        axrow[1].set_ylim(0, max(12, selected.csc.max() + 1))
    handles = [Line2D([], [], color=c, marker="o", label=f"Seed {s}") for s, c in colors.items()]
    handles += [Line2D([], [], color=".4", ls="--", label="Best accuracy in condition")]
    if experiment == "balance":
        handles += [Line2D([], [], color=".4", marker="s", ls=":", label="Size-matched control")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, experiment)
    # Per-configuration evidence: one panel per model, three explicit seeds, no pooled inference.
    for dataset in ("iris", "breast_cancer"):
        for metric, ylabel in (("csc", "CSC score"), ("balanced_accuracy", "Validation balanced accuracy (%)")):
            fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
            for ax, model in zip(axes.flat, sorted(df.model.unique()), strict=True):
                for seed, color in colors.items():
                    for control in [False, True] if experiment == "balance" else [False]:
                        part = expanded[
                            (expanded.dataset == dataset)
                            & (expanded.seed == seed)
                            & (expanded.model == model)
                            & (expanded.experiment == experiment)
                            & (expanded.control == control)
                        ].sort_values(x)
                        ax.plot(
                            part[x] * (100 if experiment == "balance" else 1),
                            part[metric] * (100 if metric == "balanced_accuracy" else 1),
                            color=color,
                            marker="s" if control else "o",
                            ls=":" if control else "-",
                        )
                ax.set_title(model, fontsize=12)
                ax.set_xlabel(label, fontsize=10)
                ax.grid(alpha=0.15)
            fig.supylabel(ylabel)
            fig.suptitle(f"{dataset.replace('_', ' ').title()}: same configuration across conditions", fontsize=17)
            fig.legend(
                handles=handles[:3] + (handles[-1:] if experiment == "balance" else []),
                loc="upper center",
                bbox_to_anchor=(0.5, 0.96),
                ncol=4,
                frameon=False,
            )
            fig.tight_layout(rect=(0.02, 0, 1, 0.91))
            save(fig, f"fixed_{experiment}_{dataset}_{metric}")
# Endpoint changes for the selected models. Compare balance to the matched control, not full data alone.
endpoints = []
for (dataset, seed, experiment), group in selected.groupby(["dataset", "seed", "experiment"]):
    low = group[(group.level == 0.25) & ~group.control].iloc[0]
    high = group[
        (group.level == (0.25 if experiment == "balance" else 1.0)) & (group.control == (experiment == "balance"))
    ].iloc[0]
    r = dict(
        dataset=dataset,
        seed=seed,
        experiment=experiment,
        intervention_model=low.model,
        reference_model=high.model,
        family_changed=low.model_family != high.model_family,
        model_changed=low.model != high.model,
        meets_original_accuracy_reference=bool(low.meets_original_accuracy_reference),
    )
    for metric in metrics:
        r[f"delta_{metric}"] = low[metric] - high[metric]
    endpoints.append(r)
end = pd.DataFrame(endpoints)
end.to_csv(out / "selected_endpoint_comparisons.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, dataset in zip(axes, ("iris", "breast_cancer"), strict=True):
    means = (
        end[end.dataset == dataset]
        .groupby("experiment")[["delta_c_path", "delta_c_encoding", "delta_c_ensemble"]]
        .mean()
        .reindex(["samples", "features", "balance"])
    )
    for j, (column, name, color) in enumerate(
        zip(means.columns, ("Path", "Distinct rules", "Distinct trees"), ("#0072B2", "#D55E00", "#009E73"), strict=True)
    ):
        ax.bar([i + (j - 1) * 0.24 for i in range(3)], means[column], width=0.23, label=name, color=color)
    ax.axhline(0, color=".4", lw=1)
    ax.set_xticks(range(3), ["Fewer examples", "Fewer features", "More imbalance"])
    ax.set(title=dataset.replace("_", " ").title(), ylabel="Mean change in CSC component")
axes[0].legend(frameon=False)
fig.tight_layout()
save(fig, "component_changes")
lines = [
    "# Dataset meta-feature study",
    "",
    f"Completed {len(df)} candidate fits across {len(manifest)} distinct conditions, two datasets and "
    f"three seeds. Shared inputs are reused across experiment aliases. The original 72 candidates "
    f"reproduce their saved validation results, including R5/R7/R9. No additional test evaluations "
    f"were performed.",
    "",
    "## Reading the figures",
    "",
    "Solid lines show the minimum-CSC model among candidates beating the dummy and within two "
    "percentage points of the condition's best balanced accuracy. Dashed lines show that best "
    "accuracy. Colors identify seeds. Dotted square lines are size-matched class-balance controls. "
    "Axes use actual counts or actual smallest-class share. Model IDs and changes are listed below "
    "and in selected_model_summary.csv. These plots describe model selection as well as training "
    "effects.",
    "",
    "## Strongest intervention versus its reference",
    "",
    "Sample and feature results compare 25% availability with the original condition. Balance "
    "results compare 25% retention of the designated class with a stratified control of the same "
    "total size. Differences below are intervention minus reference; accuracy changes are percentage "
    "points.",
    "",
]
for r in endpoints:
    lines.append(
        f"- {r['dataset']} / seed {r['seed']} / {r['experiment']}: accuracy "
        f"{100 * r['delta_balanced_accuracy']:+.2f} pp; CSC {r['delta_csc']:+.2f}; "
        f"{r['reference_model']} to {r['intervention_model']}; meets original accuracy reference: "
        f"{r['meets_original_accuracy_reference']}."
    )
lines += [
    "",
    "## Fixed configurations and components",
    "",
    "The fixed_* figures hold model ID constant and display every seed. "
    "fixed_configuration_summary.csv includes per-component changes relative to the same "
    "configuration on original data. imbalance_paired_comparisons.csv compares identical "
    "configurations with their size-matched controls. component_changes shows mean selected-model "
    "component changes; its means are descriptive summaries of three overlapping splits, not "
    "independent estimates.",
    "",
]
for experiment in properties:
    for dataset in ("iris", "breast_cancer"):
        part = fixed[
            (fixed.dataset == dataset) & (fixed.experiment == experiment) & (fixed.level == 0.25) & ~fixed.control
        ]
        if experiment == "balance":
            part = paired[(paired.dataset == dataset) & (paired.level == 0.25)]
            a, c = part.paired_delta_balanced_accuracy, part.paired_delta_csc
        else:
            a, c = part.delta_balanced_accuracy, part.delta_csc
        es = end[(end.dataset == dataset) & (end.experiment == experiment)]
        lines.append(
            f"- {dataset}, {experiment}: fixed configurations (36 paired configuration/seed rows), "
            f"mean accuracy change {a.mean() * 100:+.2f} pp, mean CSC change {c.mean():+.2f}; CSC "
            f"fell in {int((c < 0).sum())}/36 rows. Selected CSC changes range "
            f"{es.delta_csc.min():+.2f} to {es.delta_csc.max():+.2f}; family changed in "
            f"{int(es.family_changed.sum())}/3 seeds."
        )
noeligible = df.groupby(["dataset", "seed", "condition"]).eligible.any()
reference = df.groupby(["dataset", "seed", "condition"]).meets_original_accuracy_reference.any()
lines += [
    "",
    "## Performance and limitations",
    "",
    f"No eligible model: {int((~noeligible).sum())}/{len(noeligible)} conditions. No candidate "
    f"reaches the original best accuracy minus two percentage points: "
    f"{int((~reference).sum())}/{len(reference)} conditions.",
    "",
    "Dataset properties affect the learned model, which then affects CSC. A smaller score can "
    "accompany a loss of useful information and accuracy. Feature removal changes feature content as "
    "well as count. Size matching limits the training-count confound but cannot hold individual "
    "examples identical. Seeds change data splits and random subsets together. With only two "
    "datasets and previously inspected validation data, these are exploratory within-dataset "
    "results, not universal causal laws or a fresh confirmation of interpretability. No human-effort "
    "unit or validated human threshold is claimed.",
]
(out / "report.md").write_text("\n".join(lines), encoding="utf-8")
(out / "presentation_summary.json").write_text(
    json.dumps(
        dict(
            candidate_fits=len(df),
            conditions=len(manifest),
            no_eligible=int((~noeligible).sum()),
            no_original_reference=int((~reference).sum()),
            endpoints=endpoints,
        ),
        indent=2,
        default=lambda v: v.item(),
    ),
    encoding="utf-8",
)
print("Saved figures and report", len(df), "candidates", len(manifest), "conditions")

# Wide, one-dataset figures for readable slides.
for experiment, (x, label) in properties.items():
    for dataset in ("iris", "breast_cancer"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
        for seed, color in colors.items():
            for control in [False, True] if experiment == "balance" else [False]:
                part = selected[
                    (selected.dataset == dataset)
                    & (selected.seed == seed)
                    & (selected.experiment == experiment)
                    & (selected.control == control)
                ].sort_values(x)
                xx = part[x] * (100 if experiment == "balance" else 1)
                for ax, metric in zip(axes, ("balanced_accuracy", "csc"), strict=True):
                    ax.plot(
                        xx,
                        part[metric] * (100 if metric == "balanced_accuracy" else 1),
                        color=color,
                        marker="s" if control else "o",
                        ls=":" if control else "-",
                    )
                if not control:
                    axes[0].plot(xx, part.validation_best_balanced_accuracy * 100, color=color, ls="--", alpha=0.6)
        axes[0].set(ylabel="Balanced accuracy (%)", ylim=(25, 102))
        axes[1].set(ylabel="CSC score", ylim=(0, 15))
        for ax in axes:
            ax.set_xlabel(label)
            ax.grid(alpha=0.15)
        handles = [Line2D([], [], color=c, marker="o", label=f"Seed {seed}") for seed, c in colors.items()]
        handles += [Line2D([], [], color=".4", ls="--", label="Best accuracy")]
        if experiment == "balance":
            handles += [Line2D([], [], color=".4", ls=":", marker="s", label="Matched control")]
        fig.legend(handles=handles, ncol=len(handles), loc="upper center", frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.87))
        save(fig, f"slide_{experiment}_{dataset}")

# Link the complete visual evidence and model switches in the written report.
report_path = out / "report.md"
visuals = ["", "## Figures", ""]
for name in ("samples", "features", "balance", "component_changes"):
    visuals.append(f"![{name.replace('_', ' ').title()}](figures/{name}.png)")
visuals += ["", "### Same configuration across conditions", ""]
for experiment in properties:
    for dataset in ("iris", "breast_cancer"):
        for metric in ("csc", "balanced_accuracy"):
            name = f"fixed_{experiment}_{dataset}_{metric}"
            visuals.append(f"- [{dataset}: {experiment}, {metric}](figures/{name}.png)")
visuals += ["", "[Selected model IDs and changes](selected_model_changes.md)", ""]
report_path.write_text(report_path.read_text(encoding="utf-8") + "\n".join(visuals), encoding="utf-8")
changes = [
    "# Selected models across conditions",
    "",
    "DT = decision tree; RF = random forest; GB = gradient boosting. d denotes depth; n denotes estimator count.",
    "",
    "| Dataset | Seed | Experiment | Level | Control | Training rows | Features | Model |",
    "|---|---:|---|---:|---|---:|---:|---|",
]
for r in selected.sort_values(["dataset", "seed", "experiment", "control", "level"]).itertuples():
    changes.append(
        f"| {r.dataset} | {r.seed} | {r.experiment} | {r.level:g} | {r.control} | "
        f"{r.n_train} | {r.n_features_raw} | {r.model} |"
    )
(out / "selected_model_changes.md").write_text("\n".join(changes) + "\n", encoding="utf-8")
