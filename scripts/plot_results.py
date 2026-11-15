"""Audit frozen scores and regenerate figures/reports; never fit or select new models."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

# Actual starter CSV schema, inspected before this polishing pass.
COLUMN_MAP = {"score_split": "split", "model": "configuration", "balanced_accuracy": "balanced_accuracy"}
KEY = ["dataset", "seed", "configuration"]
DATASETS = {"iris": "Iris", "breast_cancer": "Breast Cancer"}
FAMILIES = {
    "DecisionTreeClassifier": ("Decision tree", "#0072B2", "o"),
    "RandomForestClassifier": ("Random forest", "#D55E00", "s"),
    "GradientBoostingClassifier": ("Gradient boosting", "#009E73", "^"),
}
# Component colors deliberately use a separate grayscale encoding plus hatches.
COMPONENTS = [
    ("c_path", "Path", "#444444", ""),
    ("c_encoding", "Distinct splits", "#999999", "///"),
    ("c_ensemble", "Distinct trees", "#dddddd", "..."),
]
OUT = Path("results/figures")
CSC_LABEL = "CSC (lower = lower structural complexity)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#999999",
            "axes.labelcolor": "#26384a",
            "text.color": "#26384a",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def audit(data):
    require(len(data) == 78, "Expected 78 frozen rows")
    require(set(data.split) == {"validation", "test"}, "Unexpected split")
    require(set(data.dataset) == set(DATASETS), "Unexpected datasets")
    require(set(data.seed) == {42, 1337, 2026}, "Unexpected seeds")
    require(not data.duplicated(KEY + ["split"]).any(), "Duplicate dataset/seed/configuration/split keys")
    numeric = [
        "csc",
        "c_path",
        "c_encoding",
        "c_ensemble",
        "total_nodes",
        "total_leaves",
        "n_trees",
        "n_unique_trees",
        "n_unique_splits",
        "balanced_accuracy",
        "macro_f1",
        "budget_r5",
        "budget_r7",
        "budget_r9",
        "tepl",
    ]
    require(np.isfinite(data[numeric].to_numpy()).all(), "Non-finite numeric values")
    require(
        np.allclose(data.csc, data[["c_path", "c_encoding", "c_ensemble"]].sum(axis=1), rtol=1e-12, atol=1e-12),
        "CSC component mismatch",
    )
    for col in ["balanced_accuracy", "macro_f1", "budget_r5", "budget_r7", "budget_r9"]:
        require(data[col].between(0, 1).all(), f"Invalid {col}")
    for col in ["total_nodes", "total_leaves", "n_trees", "n_unique_trees"]:
        require((data[col] >= 1).all() and (data[col] % 1 == 0).all(), f"Invalid {col}")
    require((data.n_unique_trees <= data.n_trees).all(), "Unique trees exceed actual trees")
    require((data.total_nodes >= data.total_leaves).all(), "Invalid node counts")
    require(
        (data.budget_r5 >= data.budget_r7).all() and (data.budget_r7 >= data.budget_r9).all(),
        "Budgets are not monotone",
    )
    require(np.allclose(data.delta, 0.02), "Selection tolerance changed")
    require((data.above_proposed_budget == (data.budget_r7 > 0.5)).all(), "Stored budget flag disagrees")
    validation, test = data[data.split == "validation"].copy(), data[data.split == "test"].copy()
    require(len(validation) == 72 and len(test) == 6, "Wrong split counts")
    require((validation.groupby(["dataset", "seed"]).size() == 12).all(), "Wrong grid size")
    require(test.selected.all(), "Nonselected test row")
    require((test.groupby(["dataset", "seed"]).size() == 1).all(), "Test selection count mismatch")
    selected = validation.merge(test[KEY], on=KEY, how="inner", validate="one_to_one")
    require(
        len(selected) == 6 and selected.selected.all() and validation.selected.sum() == 6,
        "Frozen validation/test selections do not match",
    )
    for (dataset, seed), frame in validation.groupby(["dataset", "seed"]):
        # Verify the saved decision; do not replace it or use test performance.
        eligible = frame[frame.balanced_accuracy >= frame.balanced_accuracy.max() - 0.02]
        expected = eligible.sort_values(["csc", "configuration"]).iloc[0].configuration
        actual = selected[(selected.dataset == dataset) & (selected.seed == seed)].iloc[0].configuration
        require(actual == expected, "Saved selection violates the frozen validation rule")
    paired = selected.merge(test, on=KEY, suffixes=("_val", "_test"), validate="one_to_one")
    require((paired.hyperparameters_json_val == paired.hyperparameters_json_test).all(), "Hyperparameters changed")
    return validation, selected, test


def family_handles(selected=False):
    handles = [
        Line2D([], [], color=color, marker=marker, linestyle="", markersize=7, label=label)
        for label, color, marker in FAMILIES.values()
    ]
    if selected:
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markerfacecolor="none",
                linestyle="",
                markersize=10,
                label="Frozen validation selection",
            )
        )
    return handles


def points(ax, frame, x, y, percent=False):
    for family, (_, color, marker) in FAMILIES.items():
        group = frame[frame.model_family == family]
        ax.scatter(
            group[x],
            group[y] * (100 if percent else 1),
            s=48,
            c=color,
            marker=marker,
            alpha=0.65,
            linewidths=0,
            zorder=3,
        )
    ax.grid(axis="y", alpha=0.15, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.08, y=0.12)


def export(fig, name, slide=False):
    stem = OUT / (name + ("_slide" if slide else ""))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    if not slide:
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def scatter_layout(title, slide=False, bottom=0.20):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9 if slide else 6.6))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.17 if slide else bottom, top=0.84 if slide else 0.79, wspace=0.24
    )
    if not slide:
        fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)
    return fig, axes


def plot_accuracy(validation, selected, slide=False, shared_scale=False):
    fig, axes = scatter_layout("Validation balanced accuracy vs. CSC", slide)
    for ax, (dataset, label) in zip(axes, DATASETS.items(), strict=True):
        frame = validation[validation.dataset == dataset]
        chosen = selected[selected.dataset == dataset].sort_values("seed")
        points(ax, frame, "csc", "balanced_accuracy", percent=True)
        ax.scatter(
            chosen.csc,
            chosen.balanced_accuracy * 100,
            s=135,
            facecolors="none",
            edgecolors="black",
            linewidths=1.3,
            zorder=4,
        )
        offsets = [(18, -22), (28, 12), (22, -18)] if dataset == "iris" else [(25, -20), (25, 9), (18, 12)]
        for row, offset in zip(chosen.itertuples(), offsets, strict=True):
            ax.annotate(
                f"seed {row.seed}",
                (row.csc, row.balanced_accuracy * 100),
                xytext=offset,
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
                bbox={"boxstyle": "round,pad=.15", "fc": "white", "ec": "none", "alpha": 0.9},
            )
        ax.set(title=label, xlabel=CSC_LABEL, ylabel="Validation balanced accuracy (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        if shared_scale:
            ax.set_ylim(np.floor((validation.balanced_accuracy.min() * 100 - 2) / 5) * 5, 100)
    fig.legend(
        handles=family_handles(True),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99 if slide else 0.93),
        ncol=4,
        frameon=False,
        columnspacing=1.8,
    )
    if not slide:
        fig.text(
            0.08,
            0.075,
            "Selection: lowest validation CSC within 2 percentage points of best validation balanced accuracy,",
            fontsize=11,
        )
        fig.text(
            0.08,
            0.043,
            "separately for each dataset and seed. Seed labels identify frozen choices. "
            + ("Y-axis scale shared across datasets." if shared_scale else "Y-axis scales differ by dataset."),
            fontsize=11,
        )
    export(fig, "accuracy_csc_common_scale" if shared_scale else "accuracy_csc", slide)


def model_label(row):
    params = json.loads(row.hyperparameters_json)
    if row.model_family == "DecisionTreeClassifier":
        return f"DT (depth {params['max_depth']})"
    prefix = "RF" if row.model_family == "RandomForestClassifier" else "GB"
    return f"{prefix} ({params['n_estimators']} trees, depth {params['max_depth']})"


def plot_components(selected, slide=False):
    ordered = pd.concat([selected[selected.dataset == d].sort_values("seed") for d in DATASETS])
    fig, ax = plt.subplots(figsize=(13, 4.9 if slide else 6.6))
    fig.subplots_adjust(left=0.39, right=0.94, bottom=0.17 if slide else 0.20, top=0.86 if slide else 0.79)
    if not slide:
        fig.suptitle("Why selected models receive their CSC scores", fontsize=20, fontweight="bold", y=0.98)
    labels = [f"{DATASETS[r.dataset]} · seed {r.seed} · {model_label(r)}" for r in ordered.itertuples()]
    left = np.zeros(len(ordered))
    for col, label, color, hatch in COMPONENTS:
        ax.barh(
            np.arange(len(ordered)),
            ordered[col],
            left=left,
            color=color,
            label=label,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.7,
            height=0.62,
        )
        left += ordered[col].to_numpy()
    for i, total in enumerate(ordered.csc):
        ax.text(total + 0.12, i, f"{total:.2f}", va="center", fontsize=13, fontweight="bold")
    ax.set_yticks(np.arange(len(ordered)), labels, fontsize=12)
    ax.invert_yaxis()
    ax.set(xlabel="CSC (sum of three components)", xlim=(0, ordered.csc.max() * 1.14))
    ax.axhline(2.5, color="#bbbbbb", linewidth=0.6)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.legend(
        *ax.get_legend_handles_labels(),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.59, 0.99 if slide else 0.93),
        frameon=False,
    )
    if not slide:
        fig.text(
            0.05,
            0.08,
            "Validation-selected models; grouped by dataset, then seed. DT = Decision tree; RF = Random forest.",
            fontsize=11,
        )
        fig.text(
            0.05,
            0.045,
            "Path = mean path length; Distinct splits = log2(1 + S); Distinct trees = log2(U). "
            "Equal units are not calibrated human effort.",
            fontsize=10.5,
        )
    export(fig, "components", slide)


def plot_baselines(validation, correlations):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.69, wspace=0.24)
    fig.suptitle("CSC compared with conventional structural baselines", fontsize=20, fontweight="bold", y=0.98)
    for ax, (baseline, label) in zip(
        axes,
        [("n_trees", "Number of trees"), ("total_nodes", "Total nodes"), ("tepl", "Total expected path length (TEPL)")],
        strict=True,
    ):
        points(ax, validation, baseline, "csc")
        require((validation[baseline] > 0).all(), f"Cannot log-scale {baseline}")
        ax.set_xscale("log")
        ax.set(xlabel=label, ylabel="CSC")
        rho = correlations[correlations.baseline == baseline].set_index("dataset").spearman
        ax.set_title(
            f"Spearman ρ\nIris {rho['iris']:.3f}  |  Breast Cancer {rho['breast_cancer']:.3f}",
            fontsize=12,
            pad=12,
            fontweight="normal",
        )
    fig.legend(handles=family_handles(), ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.91), frameon=False)
    fig.text(
        0.06,
        0.085,
        "All 72 validation configurations. Each panel uses a log x-scale; correlations are "
        "calculated separately by dataset.",
        fontsize=12,
    )
    fig.text(
        0.06,
        0.04,
        "CSC is highly correlated with node count on this grid. Correlation does not "
        "establish superiority or human interpretability.",
        fontsize=12,
    )
    export(fig, "baselines")


def plot_budgets(validation, sensitivity, slide=False):
    fig, axes = scatter_layout("Hypothetical seven-rule budget vs. CSC", slide, bottom=0.36)
    fig.set_size_inches(13, 6.0 if slide else 7.6)
    fig.subplots_adjust(bottom=0.40 if slide else 0.42, top=0.82 if slide else 0.79)
    for ax, (dataset, label) in zip(axes, DATASETS.items(), strict=True):
        frame = validation[validation.dataset == dataset]
        points(ax, frame, "csc", "budget_r7")
        ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1, zorder=2)
        ax.set(
            title=f"{label}\nn = {len(frame)} candidate configurations",
            xlabel=CSC_LABEL,
            ylabel="R₇: fraction of predictions over budget",
            ylim=(-0.065, 1.065),
            yticks=[0, 0.25, 0.5, 0.75, 1],
        )
    handles = family_handles() + [
        Line2D([], [], color="#444444", linestyle="--", label="Hypothesized majority boundary (R₇ > 0.50)")
    ]
    fig.legend(
        handles=handles,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 if slide else 0.94),
        frameon=False,
        fontsize=11,
    )
    table_ax = fig.add_axes([0.12, 0.055 if slide else 0.14, 0.80, 0.15])
    table_ax.axis("off")
    rows = [[DATASETS[d], *[f"{sensitivity.loc[d, f'percent_r{b}']:.1f}%" for b in (5, 7, 9)]] for d in DATASETS]
    table = table_ax.table(
        cellText=rows,
        colLabels=["Validation configurations", "R₅ > 0.50", "R₇ > 0.50", "R₉ > 0.50"],
        loc="center",
        cellLoc="center",
        colWidths=[0.4, 0.2, 0.2, 0.2],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#eef1f4")
            cell.set_text_props(weight="bold")
    if not slide:
        fig.text(
            0.06,
            0.075,
            "Memory-inspired split-test budgets are design hypotheses, not validated human "
            "limits. Table: % of validation configurations.",
            fontsize=10.5,
        )
        fig.text(
            0.06,
            0.04,
            "Budgets count distinct tests per prediction and omit ensemble-output aggregation "
            "effort. Exact 0/1 values are not jittered.",
            fontsize=10.5,
        )
    export(fig, "rule_budgets", slide)


def write_summaries(validation, selected, test):
    test_table = test[KEY + ["balanced_accuracy", "csc", "macro_f1"]].copy()
    test_table["test_balanced_accuracy_percent"] = 100 * test_table.pop("balanced_accuracy")
    test_table.rename(
        columns={"configuration": "selected_model", "csc": "test_csc", "macro_f1": "test_macro_f1"}
    ).to_csv("results/selected_test_results.csv", index=False)
    selected[KEY + ["c_path", "c_encoding", "c_ensemble", "csc"]].to_csv(
        "results/selected_validation_components.csv", index=False
    )
    aggregate, sensitivity, correlations = [], [], []
    for dataset in DATASETS:
        frame, heldout = validation[validation.dataset == dataset], test[test.dataset == dataset]
        aggregate.append(
            dict(
                dataset=dataset,
                n_seeds=len(heldout),
                mean_test_balanced_accuracy_percent=heldout.balanced_accuracy.mean() * 100,
                sd_test_balanced_accuracy_pp=heldout.balanced_accuracy.std(ddof=1) * 100,
                mean_test_csc=heldout.csc.mean(),
                sd_test_csc=heldout.csc.std(ddof=1),
            )
        )
        sensitivity.append(
            dict(
                dataset=dataset,
                n_validation=len(frame),
                **{f"count_r{b}": int((frame[f"budget_r{b}"] > 0.5).sum()) for b in (5, 7, 9)},
                **{f"percent_r{b}": float((frame[f"budget_r{b}"] > 0.5).mean() * 100) for b in (5, 7, 9)},
            )
        )
        for baseline in ["n_trees", "total_nodes", "tepl", "c_path", "c_encoding", "c_ensemble"]:
            correlations.append(
                dict(dataset=dataset, baseline=baseline, spearman=frame.csc.corr(frame[baseline], method="spearman"))
            )
    pd.DataFrame(aggregate).to_csv("results/test_summary_presentation.csv", index=False)
    sensitivity = pd.DataFrame(sensitivity).set_index("dataset")
    sensitivity.to_csv("results/rule_budget_sensitivity.csv")
    correlations = pd.DataFrame(correlations)
    correlations.to_csv("results/correlations.csv", index=False)
    error = float((validation.csc - validation[["c_path", "c_encoding", "c_ensemble"]].sum(axis=1)).abs().max())
    facts = dict(
        validation_rows=len(validation),
        test_rows=len(test),
        test_summary=aggregate,
        selections=[
            dict(dataset=r.dataset, seed=int(r.seed), model=r.configuration, label=model_label(r))
            for r in selected.itertuples()
        ],
        budget_count=int((validation.budget_r7 > 0.5).sum()),
        budget_percent=float((validation.budget_r7 > 0.5).mean() * 100),
        budget_sensitivity=sensitivity.reset_index().to_dict("records"),
        node_correlations=correlations[correlations.baseline == "total_nodes"].to_dict("records"),
        max_component_sum_error=error,
        scores_sha256=hashlib.sha256(Path("results/scores.csv").read_bytes()).hexdigest(),
    )
    Path("results/presentation_summary.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    report = [
        "# Frozen-results post-processing audit",
        "",
        f"Validated {len(validation)} validation candidates and {len(test)} frozen selected test evaluations.",
        "Selection keys include dataset, seed and exact configuration; no test scores enter selection or charts.",
        f"Maximum validation CSC component-sum error: {error:.3g}.",
        "",
        "## Held-out test summary",
        "",
        "Mean +/- sample SD over three overlapping splits; descriptive spread, not confidence intervals.",
        "",
        "| Dataset | Balanced accuracy (%) +/- SD (pp) | CSC +/- SD |",
        "|---|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            f"| {DATASETS[row['dataset']]} | {row['mean_test_balanced_accuracy_percent']:.2f}"
            f" +/- {row['sd_test_balanced_accuracy_pp']:.2f} | {row['mean_test_csc']:.2f}"
            f" +/- {row['sd_test_csc']:.2f} |"
        )
    report.extend(
        [
            "",
            "## Hypothetical budget sensitivity",
            "",
            f"{facts['budget_count']}/{len(validation)} tested validation configurations "
            f"({facts['budget_percent']:.1f}%) satisfy strict R7 > 0.50.",
            "",
            "| Dataset | R5 > 0.50 | R7 > 0.50 | R9 > 0.50 |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        values = " | ".join(f"{sensitivity.loc[dataset, f'percent_r{b}']:.1f}%" for b in (5, 7, 9))
        report.append(f"| {DATASETS[dataset]} | {values} |")
    report.extend(
        [
            "",
            "## Evidence limits",
            "",
            "CSC is structural, not human-tested. Path length uses reference-data occupancy; split and tree counts "
            "describe global structure. The logarithmic distinct-split count is not an MDL bit code. "
            "Full signatures are representation-dependent, not a test of functional equivalence. "
            "Repeating an identical boosted tree can alter predictions without altering CSC. "
            "Rule budgets omit output aggregation arithmetic. Strong correlation with node count does not establish "
            "superiority. Two datasets and three overlapping splits limit generalization.",
            "",
            "## Frozen source",
            "",
            f"scores.csv SHA-256: `{facts['scores_sha256']}`",
            "",
            "The plotting command checks that scores.csv and the original summary.json remain byte-for-byte unchanged.",
        ]
    )
    Path("results/polishing_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return sensitivity, correlations, facts


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {
        p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ["results/scores.csv", "results/summary.json"]
    }
    data = pd.read_csv("results/scores.csv").rename(columns=COLUMN_MAP)
    validation, selected, test = audit(data)
    sensitivity, correlations, facts = write_summaries(validation, selected, test)
    for slide in (False, True):
        plot_accuracy(validation, selected, slide)
        plot_accuracy(validation, selected, slide, shared_scale=True)
        plot_components(selected, slide)
        plot_budgets(validation, sensitivity, slide)
    plot_baselines(validation, correlations)
    require(
        all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == value for p, value in frozen.items()),
        "Frozen input files changed",
    )
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
