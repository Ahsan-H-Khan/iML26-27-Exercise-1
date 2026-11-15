"""Generate registered controlled-study figures and a CSV-derived evidence report."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter, PercentFormatter, ScalarFormatter
from plot_results import style

from iml_interpretability.controlled_studies import OUT, provenance, verify_baseline

FIG = OUT / "figures"
NAMES = {"iris": "Iris", "breast_cancer": "Breast Cancer"}
BLUE, ORANGE = "#0072B2", "#D55E00"


def export(fig, name, slide=False):
    stem = FIG / (name + ("_slide" if slide else ""))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    if not slide:
        fig.savefig(stem.with_suffix(".pdf"))
        fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def tree_diagram(ax, chain=False):
    ax.axis("off")
    ax.set(xlim=(-0.02, 1.02), ylim=(0, 1.1))
    if chain:
        positions = [(0.14, 0.94), (0.04, 0.68), (0.43, 0.68), (0.32, 0.40), (0.73, 0.40), (0.62, 0.12), (0.95, 0.12)]
        labels = ["x ≤ .25", "0", "x ≤ .50", "1", "x ≤ .75", "0", "1"]
        edges = [(0, 1), (0, 2), (2, 3), (2, 4), (4, 5), (4, 6)]
    else:
        positions = [(0.5, 0.94), (0.25, 0.59), (0.75, 0.59), (0.10, 0.19), (0.39, 0.19), (0.61, 0.19), (0.90, 0.19)]
        labels = ["x ≤ .50", "x ≤ .25", "x ≤ .75", "0", "1", "0", "1"]
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for a, b in edges:
        xa, ya = positions[a]
        xb, yb = positions[b]
        ax.plot([xa, xb], [ya, yb], color="#8996a0", lw=1, zorder=1)
    for (x, y), label in zip(positions, labels, strict=True):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=.3", facecolor="#edf3f6" if len(label) > 1 else "white", edgecolor="#a2adb5"),
        )
    ax.set_title("Chain: leaf depths 1 / 2 / 3 / 3" if chain else "Balanced: all leaf depths 2", fontsize=14)


def plot_q1(data, slide=False):
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5 if slide else 8.5), gridspec_kw={"height_ratios": [1, 1.1]})
    fig.subplots_adjust(
        left=0.08, right=0.97, bottom=0.12 if slide else 0.17, top=0.88 if slide else 0.86, hspace=0.65, wspace=0.25
    )
    if not slide:
        fig.suptitle("Same function and size; different reference-weighted paths", fontsize=20, fontweight="bold")
    tree_diagram(axes[0, 0])
    tree_diagram(axes[0, 1], True)
    order = ["uniform", "early_exit", "deep_exit"]
    labels = ["Uniform", "Early exits", "Deep exits"]
    for ax, metric, label in [(axes[1, 0], "c_path", "Mean path length = TEPL"), (axes[1, 1], "csc", "CSC")]:
        for fixture, color, marker in [("balanced", BLUE, "o"), ("chain", ORANGE, "s")]:
            values = data[data.fixture == fixture].set_index("profile").loc[order, metric]
            ax.plot(range(3), values, marker=marker, color=color, label=fixture.title(), lw=1.7)
            for i, value in enumerate(values):
                ax.annotate(
                    f"{value:.2f}",
                    (i, value),
                    xytext=(0, (12 if i == 1 else -18) if fixture == "balanced" else (-18 if i == 1 else 12)),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                    color=color,
                )
        ax.set(xticks=range(3), xticklabels=labels, ylabel=label)
        ax.margins(y=0.4, x=0.15)
        ax.grid(axis="y", alpha=0.15)
    axes[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), ncol=2, frameon=False)
    if not slide:
        fig.text(
            0.06,
            0.065,
            "Constructed trees: 7 nodes, 4 leaves, S = 3, U = 1. Corresponding leaf outputs and predictions agree.",
            fontsize=12,
        )
        fig.text(
            0.06,
            0.025,
            "Occupancy reverses the ordering. This distinguishes CSC from node count; TEPL already "
            "captures the path effect.",
            fontsize=12,
        )
    export(fig, "q1_paths", slide)


def plot_q2(data, replication, slide=False):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.7 if slide else 7))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.38, top=0.85 if slide else 0.82, wspace=0.3)
    if not slide:
        fig.suptitle("Fixed ensemble size; distinct logic changes CSC", fontsize=20, fontweight="bold")
    for ax, col, label in zip(axes, ["csc", "total_nodes", "tepl"], ["CSC", "Total nodes", "TEPL"], strict=True):
        ax.plot(data.unique_stumps, data[col], color=BLUE, marker="o")
        ax.set(xlabel="Distinct stumps K", ylabel=label, xticks=[1, 3, 6, 12])
        ax.grid(axis="y", alpha=0.15)
        if col != "csc":
            ax.set_ylim(0, data[col].max() * 1.3)
    table_ax = fig.add_axes([0.08, 0.06 if slide else 0.13, 0.84, 0.16])
    table_ax.axis("off")
    rows = [
        [f"{int(r.replication)}x", int(r.n_trees), f"{r.csc:.2f}", int(r.total_nodes), f"{r.tepl:.0f}"]
        for r in replication.itertuples()
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=["Equal replication", "Trees M", "CSC", "Nodes", "TEPL"],
        bbox=[0, 0, 1, 1],
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d2dbe0")
        if r == 0:
            cell.set_facecolor("#edf3f6")
            cell.set_text_props(weight="bold")
    if not slide:
        fig.text(
            0.06,
            0.075,
            "Main controls: M = 12; S = U = K; C_path = 1. Artificial averaging ensembles, not trained random forests.",
            fontsize=11.5,
        )
        fig.text(
            0.06,
            0.035,
            "Table: replicate all three base stumps equally. Predictions and CSC stay fixed; total work grows.",
            fontsize=11.5,
        )
    export(fig, "q2_diversity", slide)


def paired_plot(data, slide=False):
    if slide:
        data = data.groupby(["dataset", "seed"], sort=False).head(1)
    data = data.reset_index(drop=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.8 if slide else 9), sharey=True)
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.16 if slide else 0.18, top=0.85 if slide else 0.88, wspace=0.35)
    if not slide:
        fig.suptitle("Near-best accuracy pairs: show the structural differences", fontsize=20, fontweight="bold")
    columns = [
        ("balanced_accuracy", "Validation BA (%)", 100),
        ("csc", "CSC", 1),
        ("total_nodes", "Total nodes", 1),
        ("tepl", "TEPL", 1),
    ]
    for ax, (col, title, factor) in zip(axes, columns, strict=True):
        for i, row in enumerate(data.to_dict("records")):
            a, b = row[f"{col}_a"] * factor, row[f"{col}_b"] * factor
            ax.plot([a, b], [i, i], color="#a5afb7", lw=1)
            ax.scatter(a, i, c=BLUE, marker="o", s=45, zorder=3)
            ax.scatter(b, i, c=ORANGE, marker="s", s=45, zorder=3)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="x", alpha=0.15)
        ax.margins(x=0.2)
        if col in ("total_nodes", "tepl"):
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300, 1000]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        if col == "balanced_accuracy":
            ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    labels = [
        f"{'Iris' if r.dataset == 'iris' else 'BC'} · seed {r.seed}\n{r.model_a} / {r.model_b}"
        for r in data.itertuples()
    ]
    axes[0].set_yticks(np.arange(len(data)), labels, fontsize=10 if not slide else 11)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, label="First listed model"),
        plt.Line2D([], [], marker="s", ls="", color=ORANGE, label="Second listed model"),
    ]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.62, 0.96), frameon=False)
    if not slide:
        fig.text(
            0.05,
            0.08,
            "Pairs share dataset and split; both are within 2 pp of best validation BA. Model IDs are "
            "ordered alphabetically.",
            fontsize=11.5,
        )
        fig.text(
            0.05,
            0.045,
            "Display: up to three pairs per group, closest accuracy first. Nodes/TEPL use log axes. All "
            "396 pairs are saved.",
            fontsize=11.5,
        )
    export(fig, "q3_pairs", slide)


def demand_plot(summary, curves, slide=False):
    chosen = summary[(summary.reference_split == "validation") & summary.selected].sort_values(["dataset", "seed"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7 if slide else 7.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.39, top=0.81 if slide else 0.80, wspace=0.24)
    if not slide:
        fig.suptitle("Rule-demand profiles, without an assumed human capacity", fontsize=20, fontweight="bold")
    colors = [BLUE, ORANGE, "#009E73"]
    line_styles = ["-", "--", ":"]
    for ax, (dataset, label) in zip(axes, NAMES.items(), strict=True):
        selected = chosen[chosen.dataset == dataset]
        max_b = int(selected.max_B.max())
        for color, ls, row in zip(colors, line_styles, selected.itertuples(), strict=True):
            c = curves[
                (curves.dataset == dataset)
                & (curves.seed == row.seed)
                & (curves.model == row.model)
                & (curves.reference_split == "validation")
            ]
            b = np.arange(max_b + 1)
            values = c.set_index("budget").exceedance.reindex(b, fill_value=0)
            ax.step(b, values, where="post", color=color, ls=ls, lw=2, label=f"seed {row.seed} · {row.model}")
            ax.scatter(c.budget, c.exceedance, color=color, s=13)
        ax.set(
            title=label,
            xlabel="Rule budget b (distinct tests)",
            ylabel="R(b): fraction with B(x) > b",
            ylim=(-0.04, 1.04),
            xlim=(-0.3, max_b + 0.3),
        )
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=10, frameon=False, loc="upper right")
    table_ax = fig.add_axes([0.15, 0.025 if slide else 0.11, 0.74, 0.23 if slide else 0.20])
    table_ax.axis("off")
    rows = [
        [f"{NAMES[r.dataset]} · {r.seed}", int(r.b90), int(r.b95), f"{r.mean_L:.2f}", int(r.n_trees)]
        for r in chosen.itertuples()
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=["Selected validation model", "b90", "b95", "Mean visits L", "Trees M"],
        bbox=[0, 0, 1, 1],
        cellLoc="center",
        colWidths=[0.43, 0.12, 0.12, 0.19, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5 if slide else 11)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d2dbe0")
        if r == 0:
            cell.set_facecolor("#edf3f6")
            cell.set_text_props(weight="bold")
    if not slide:
        fig.text(
            0.05,
            0.065,
            "b90 / b95: minimum integer budgets covering 90% / 95% of reference predictions. Curves "
            "include all integer budgets.",
            fontsize=11,
        )
        fig.text(
            0.05,
            0.03,
            "Six frozen validation selections. Overlapping curves are retained. Descriptive demand, not "
            "human ability; aggregation effort omitted.",
            fontsize=10.5,
        )
    export(fig, "rule_demand", slide)


def fixture_demands():
    q = pd.read_csv(OUT / "q2_redundancy_controls.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in q.itertuples():
        b = np.arange(13)
        ax.step(b, (b < r.unique_stumps).astype(float), where="post", label=f"K={r.unique_stumps}")
    ax.set(
        xlabel="Rule budget b", ylabel="R(b)", title="Constructed stumps: every row encounters exactly K distinct tests"
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    export(fig, "fixture_rule_demand")


def report(pairs, summary, reconciliation):
    near = pairs[pairs.near_best_pair]
    counts = {baseline: near[f"relation_{baseline}"].value_counts().to_dict() for baseline in ["total_nodes", "tepl"]}
    chosen = summary[(summary.reference_split == "validation") & summary.selected]
    facts = dict(
        all_pairs=len(pairs),
        matched_pairs=int(pairs.performance_matched.sum()),
        near_best_pairs=len(near),
        near_best_relations=counts,
        reconciled_rows=len(reconciliation),
        max_reconciliation_error=float(reconciliation.max_abs_error.max()),
        selected_demands=chosen[["dataset", "seed", "model", "b90", "b95", "mean_L", "n_trees"]].to_dict("records"),
    )
    (OUT / "summary.json").write_text(json.dumps(facts, indent=2))
    lines = [
        "# Controlled comparisons: evidence report",
        "",
        "CSC remains version 1.0. These studies test structural behavior, not human comprehension.",
        "",
        "## Q1: fixed size and function",
        "",
        "Both fixtures have 7 nodes, 4 leaves and 3 split tests, and agree in prediction. Balanced CSC is 4.00. "
        "Chain CSC is 4.25 (uniform), 3.30 (early exits), and 4.85 (deep exits). "
        "Node count misses the reversal; TEPL already captures it. This also demonstrates representation dependence.",
        "",
        "## Q2: fixed ensemble size",
        "",
        "With 12 stumps and 36 nodes, TEPL stays 12. CSC rises from 2.00 at K=1 to "
        f"{pd.read_csv(OUT / 'q2_redundancy_controls.csv').iloc[-1].csc:.2f} at K=12. "
        "Equal replication of a three-stump averaging ensemble preserves predictions and CSC while "
        "nodes and visits grow. "
        "The diversity controls are artificial; their predictive functions are not held fixed.",
        "",
        "## Q3: accuracy-matched validation pairs",
        "",
        f"{len(pairs)} unordered same-split pairs: {facts['matched_pairs']} within 2 percentage points of one another; "
        f"{len(near)} also have both members within 2 percentage points of their group best.",
        "",
        "| Baseline | Agree | Opposite ranking | Both tie | Baseline-only tie | CSC-only tie |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for baseline, c in counts.items():
        lines.append(
            f"| {baseline} | {c.get('agree', 0)} | {c.get('opposite', 0)} | {c.get('both_tie', 0)} | "
            f"{c.get('baseline_tie', 0)} | {c.get('csc_tie', 0)} |"
        )
    lines.extend(
        [
            "",
            "These are descriptive totals of dependent pairs, not independent statistical samples. "
            "Per-dataset/seed counts are in q3_pair_summary.csv. CSC mostly agrees with baselines. "
            "Disagreement adds descriptive information but does not show which ranking is more useful to people. "
            "No near-best pair has an equal baseline value and different CSC in this grid.",
            "",
            "## Rule-demand profiles",
            "",
            f"Reconstructed all {len(reconciliation)} original rows; maximum numeric discrepancy "
            f"{facts['max_reconciliation_error']:.3g}. "
            "The original CSV and summary were preserved. Per-row B, L and M, exact exceedance curves, split IDs "
            "and dataset hashes are recorded. Demand quantiles are integer coverage budgets, not cognitive limits.",
            "",
            "| Dataset / seed | Model | b90 | b95 | Mean visits L | Trees M |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for r in chosen.itertuples():
        lines.append(
            f"| {NAMES[r.dataset]} / {r.seed} | {r.model} | {r.b90} | {r.b95} | {r.mean_L:.2f} | {r.n_trees} |"
        )
    lines.extend(
        [
            "",
            "The old 5/7/9-rule results remain available as hypothetical sensitivity scenarios. "
            "Neither CSC=7 nor R7>0.50 establishes an interpretability cutoff. Aggregation effort is not modeled.",
            "",
            "## Scope and provenance",
            "",
            "Constructed controls and exploratory matched comparisons concern different questions. "
            "The real datasets were already inspected during development. New measurements are not independent "
            "confirmation. The formula was not reweighted. See manifest.json, split_manifest.csv, dataset_hashes.json "
            "and protocols/controlled_v1.md for provenance.",
            "",
            "## Figures",
            "",
            *[f"![{name}](figures/{name}.png)" for name in ["q1_paths", "q2_diversity", "q3_pairs", "rule_demand"]],
            "",
            "All figures also have vector PDF and SVG exports.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    provenance("reporting")
    style()
    FIG.mkdir(parents=True, exist_ok=True)
    q1 = pd.read_csv(OUT / "q1_path_controls.csv")
    q2 = pd.read_csv(OUT / "q2_redundancy_controls.csv")
    replication = pd.read_csv(OUT / "q2_replication_controls.csv")
    pairs = pd.read_csv(OUT / "q3_matched_pairs.csv")
    shown = pd.read_csv(OUT / "q3_displayed_pairs.csv")
    summary = pd.read_csv(OUT / "rule_demand_summary.csv")
    curves = pd.read_csv(OUT / "rule_demand_curves.csv")
    for slide in (False, True):
        plot_q1(q1, slide)
        plot_q2(q2, replication, slide)
        paired_plot(shown, slide)
        demand_plot(summary, curves, slide)
    fixture_demands()
    report(pairs, summary, pd.read_csv(OUT / "reconciliation.csv"))
    verify_baseline()
    print("Controlled-study report and figures generated from saved tables.", flush=True)


if __name__ == "__main__":
    main()
