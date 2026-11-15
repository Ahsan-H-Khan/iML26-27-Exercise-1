"""Render real-data results for the metric-development narrative, without controlled-study inputs."""

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

from iml_interpretability.metric_development import OUT, sha, verify_frozen

FIG = OUT / "figures"
NAMES = {"iris": "Iris", "breast_cancer": "Breast Cancer"}
FAMILIES = {
    "DecisionTreeClassifier": ("Decision tree", "#0072B2", "o"),
    "RandomForestClassifier": ("Random forest", "#D55E00", "s"),
    "GradientBoostingClassifier": ("Gradient boosting", "#009E73", "^"),
}
SEEDS = [42, 1337, 2026]


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=250, facecolor="white")
    plt.close(fig)


def axes_pair():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.80, wspace=0.16)
    for ax, name in zip(axes, NAMES.values(), strict=True):
        ax.set_title(name, fontsize=19, pad=12)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    return fig, axes


def performance(c, dummy):
    fig, axes = axes_pair()
    for ax, dataset in zip(axes, NAMES, strict=True):
        for family_index, (family, (_, color, marker)) in enumerate(FAMILIES.items()):
            for seed_index, seed in enumerate(SEEDS):
                rows = c[(c.dataset == dataset) & (c.model_family == family) & (c.seed == seed)].sort_values("model")
                x = family_index + (seed_index - 1) * 0.18 + np.linspace(-0.045, 0.045, len(rows))
                ax.scatter(x, rows.balanced_accuracy * 100, c=color, marker=marker, s=48, alpha=0.75)
        base = dummy[dummy.dataset == dataset].balanced_accuracy.unique()
        assert len(base) == 1
        ax.axhline(base[0] * 100, color="#68717a", ls="--", lw=1.5)
        ax.text(
            0.02,
            base[0] * 100 + 3,
            f"Dummy baseline: {base[0] * 100:.1f}%",
            transform=ax.get_yaxis_transform(),
            fontsize=13,
            color="#596776",
        )
        ax.set(
            xticks=range(3),
            xticklabels=["Decision\ntree", "Random\nforest", "Gradient\nboosting"],
            ylim=(25, 102),
            xlim=(-0.5, 2.5),
        )
    axes[0].set_ylabel("Balanced accuracy (%)", fontsize=14)
    save(fig, "predictive_performance")


def tradeoff(c):
    fig, axes = axes_pair()
    for ax, dataset in zip(axes, NAMES, strict=True):
        for family, (_, color, marker) in FAMILIES.items():
            rows = c[(c.dataset == dataset) & (c.model_family == family)]
            rejected = rows[~rows.eligible]
            good = rows[rows.eligible]
            ax.scatter(rejected.csc, rejected.balanced_accuracy * 100, color=color, marker=marker, alpha=0.2, s=42)
            ax.scatter(good.csc, good.balanced_accuracy * 100, color=color, marker=marker, alpha=0.9, s=54)
        selected = c[(c.dataset == dataset) & c.selected_csc]
        ax.scatter(
            selected.csc, selected.balanced_accuracy * 100, s=180, facecolors="none", edgecolors="#182531", lw=1.7
        )
        ax.set(xlabel="CSC score (lower = less complexity)", ylim=(63, 101), xlim=(0, 18))
    axes[0].set_ylabel("Balanced accuracy (%)", fontsize=14)
    handles = [Line2D([], [], marker=m, color=col, ls="", label=label) for label, col, m in FAMILIES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=14)
    save(fig, "accuracy_csc")


def components(c):
    rows = c[c.selected_csc].sort_values(["dataset", "seed"])
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.subplots_adjust(left=0.30, right=0.93, top=0.80, bottom=0.16)
    left = np.zeros(len(rows))
    for col, label, color in [
        ("c_path", "Average path length", "#40566b"),
        ("c_encoding", "Distinct-rule term", "#8b9baa"),
        ("c_ensemble", "Distinct-tree term", "#cbd3da"),
    ]:
        ax.barh(range(len(rows)), rows[col], left=left, color=color, label=label, height=0.62)
        left += rows[col].to_numpy()
    labels = [f"{NAMES[r.dataset]} · run {SEEDS.index(r.seed) + 1}" for r in rows.itertuples()]
    ax.set(yticks=range(len(rows)), yticklabels=labels, xlabel="CSC score", xlim=(0, 9))
    ax.invert_yaxis()
    for i, value in enumerate(left):
        ax.text(value + 0.12, i, f"{value:.2f}", va="center", fontsize=16)
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=14)
    save(fig, "components")


def comparisons(c):
    rows = []
    for (dataset, seed), group in c[c.eligible].groupby(["dataset", "seed"]):
        for metric in ["eepc_total", "ea", "total_nodes", "eepc_mean", "ea_total"]:
            defined = len(group) > 1 and group.csc.nunique() > 1 and group[metric].nunique() > 1
            rho = group.csc.rank().corr(group[metric].rank()) if defined else np.nan
            rows.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    metric=metric,
                    n_eligible=len(group),
                    spearman=rho,
                    status="defined" if defined else "insufficient_or_constant",
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "metric_comparisons.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.30, top=0.90)
    keys = [(d, s) for d in ["iris", "breast_cancer"] for s in SEEDS]
    columns = ["eepc_total", "ea", "total_nodes"]
    values = np.array(
        [
            [frame[(frame.dataset == d) & (frame.seed == s) & (frame.metric == m)].spearman.iloc[0] for d, s in keys]
            for m in columns
        ]
    )
    ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(6):
            v = values[i, j]
            ax.text(
                j,
                i,
                "n/a" if np.isnan(v) else f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=22,
                color="white" if v > 0.65 else "#26384a",
            )
    labels = [
        f"{NAMES[d].replace(chr(32), chr(10))}\nrun {SEEDS.index(s) + 1}\n"
        f"({int(((c.dataset == d) & (c.seed == s) & c.eligible).sum())} models)"
        for d, s in keys
    ]
    ax.set(yticks=range(3), yticklabels=["EEPC total", "EA", "Node count"], xticks=range(6), xticklabels=labels)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_title("Rank agreement with CSC: 1 = same order", fontsize=19, pad=16)
    save(fig, "metric_comparison")
    return frame


def main():
    verify_frozen()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#26384a",
            "axes.labelcolor": "#26384a",
        }
    )
    c = pd.read_csv(OUT / "candidates.csv")
    d = pd.read_csv(OUT / "dummy_baselines.csv")
    choices = pd.read_csv(OUT / "selections.csv")
    test = pd.read_csv(OUT / "selected_test_scores.csv")
    assert len(c) == 72 and not c.duplicated(["dataset", "seed", "model"]).any()
    assert not test.duplicated(["dataset", "seed", "model"]).any()
    assert (c.selected_csc <= c.eligible).all()
    performance(c, d)
    tradeoff(c)
    components(c)
    correlations = comparisons(c)
    performance_summary = (
        c.groupby(["dataset", "model_family"])
        .agg(
            n=("model", "size"),
            min_validation_ba=("balanced_accuracy", "min"),
            max_validation_ba=("balanced_accuracy", "max"),
            mean_validation_ba=("balanced_accuracy", "mean"),
            competitive=("eligible", "sum"),
        )
        .reset_index()
    )
    performance_summary.to_csv(OUT / "performance_summary.csv", index=False)
    selected = choices[choices.status == "selected"].merge(
        test, on=["dataset", "seed", "split_id", "model"], validate="many_to_one"
    )
    selected.to_csv(OUT / "selector_test_comparison.csv", index=False)
    test_summary = (
        selected.groupby(["dataset", "selector"])
        .agg(
            n=("model", "size"),
            mean_test_ba=("balanced_accuracy", "mean"),
            sd_test_ba=("balanced_accuracy", "std"),
            mean_test_csc=("csc", "mean"),
        )
        .reset_index()
    )
    test_summary.to_csv(OUT / "test_summary.csv", index=False)
    selected_val = c[c.selected_csc].sort_values(["dataset", "seed"])
    worked = selected_val[(selected_val.dataset == "iris") & (selected_val.seed == 42)].iloc[0]
    facts = {
        "candidate_count": len(c),
        "eligible_count": int(c.eligible.sum()),
        "above_dummy_count": int((c.balanced_accuracy > c.dummy_balanced_accuracy).sum()),
        "groups": [],
        "test_summary": test_summary.to_dict("records"),
        "selected_validation": selected_val.to_dict("records"),
        "worked_example": worked.to_dict(),
        "comparisons": correlations.to_dict("records"),
    }
    for (dataset, seed), g in c.groupby(["dataset", "seed"]):
        a = g[g.selected_accuracy].iloc[0]
        b = g[g.selected_csc].iloc[0]
        facts["groups"].append(
            dict(
                dataset=dataset,
                seed=int(seed),
                eligible=int(g.eligible.sum()),
                best_validation_ba=float(g.balanced_accuracy.max()),
                dummy_ba=float(g.dummy_balanced_accuracy.iloc[0]),
                csc_model=b.model,
                accuracy_model=a.model,
                validation_ba_cost_pp=float((a.balanced_accuracy - b.balanced_accuracy) * 100),
                validation_csc_reduction=float(a.csc - b.csc),
            )
        )
    facts = json.loads(json.dumps(facts), parse_constant=lambda _: None)
    (OUT / "presentation_summary.json").write_text(json.dumps(facts, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# CSC: metric development and real-data evaluation",
        "",
        "## Definitions",
        "",
        "EPL is the mean internal split count for one tree. EEPC_total sums tree EPL; EEPC_mean averages it. "
        "EA = EEPC_mean × log2(1 + features used); EA_total is a separate sensitivity. "
        "CSC = EEPC_mean + log2(1 + distinct splits) + log2(distinct full trees). "
        "These EA aggregation conventions were declared before this run. CSC is unchanged and "
        "MDL-inspired, not a formal code length.",
        "",
        "## 1. Check predictive performance",
        "",
        f"All {facts['above_dummy_count']}/72 candidates exceed the training-fitted dummy validation baseline. "
        f"{facts['eligible_count']}/72 also lie within two percentage points of their own dataset/seed best. "
        "These are competitive benchmark models, not a claim of application-level adequacy.",
        "",
        "| Dataset | Seed | Dummy BA | Best BA | Competitive |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in facts["groups"]:
        lines.append(
            f"| {NAMES[g['dataset']]} | {g['seed']} | {g['dummy_ba']:.1%} | "
            f"{g['best_validation_ba']:.1%} | {g['eligible']}/12 |"
        )
    lines += [
        "",
        "![Predictive performance](figures/predictive_performance.png)",
        "",
        "Every point is a validation candidate. Horizontal spread separates the three seeds; it is not a fitted trend. "
        "Balanced accuracy is mean recall across classes. Dummy predicts the most frequent training class.",
        "",
        "## 2. Apply CSC among competitive models",
        "",
        "![Accuracy and CSC](figures/accuracy_csc.png)",
        "",
        "Faint points fail the eligibility gate; solid points pass; rings mark minimum-CSC choices. "
        "Both datasets share axes. Each dataset/seed has its own best-accuracy reference.",
        "",
        "| Dataset | Seed | Minimum CSC | Accuracy-only | Validation BA cost (pp) | CSC reduction |",
        "|---|---:|---|---|---:|---:|",
    ]
    for g in facts["groups"]:
        lines.append(
            f"| {NAMES[g['dataset']]} | {g['seed']} | {g['csc_model']} | {g['accuracy_model']} | "
            f"{g['validation_ba_cost_pp']:.2f} | {g['validation_csc_reduction']:.2f} |"
        )
    lines += [
        "",
        "![CSC components](figures/components.png)",
        "",
        "Runs 1/2/3 correspond to seeds 42/1337/2026. Rule term = log2(1+S); tree term = log2(U). "
        "Equal component lengths are not calibrated equal human effort.",
        "",
        "## 3. Compare with earlier measures",
        "",
        "![Rank comparison](figures/metric_comparison.png)",
        "",
        "Spearman rank agreement is calculated separately within each dataset/seed among competitive candidates. "
        "A value of 1 means identical rank ordering; it does not establish that either measure is cognitively valid. "
        "Iris runs 2 and 3 contain only two competitive models, so perfect correlations there are weak evidence. "
        "The CSV also includes EEPC_mean and EA_total sensitivity. No independent-pair counts or "
        "significance claims are made.",
        "",
        "## 4. Evaluate frozen choices on test data",
        "",
        "| Dataset | Selector | Mean test BA | Sample SD (pp) | Mean test CSC |",
        "|---|---|---:|---:|---:|",
    ]
    for r in test_summary.itertuples():
        lines.append(
            f"| {NAMES[r.dataset]} | {r.selector} | {r.mean_test_ba:.2%} | {r.sd_test_ba * 100:.2f} | "
            f"{r.mean_test_csc:.2f} |"
        )
    lines += [
        "",
        "Choices were frozen on validation data, with no refit. Shared choices reuse the same test record. "
        "The three splits overlap: SD is descriptive, not a confidence interval. Test data were "
        "inspected in earlier work; "
        "this follow-up is not fresh confirmation. A validation tolerance does not guarantee a "
        "test-performance tolerance.",
        "",
        "## Interpretation and limits",
        "",
        "The evaluation tests predictive competitiveness and compares structural rankings. It does not measure human "
        "comprehension, prove superiority, or establish that MDL was numerically implemented. "
        "CSC's weights, logarithms and representation dependence remain assumptions. Original scores, formula and six "
        "original test choices were reconciled; see reconciliation.csv and manifest.json.",
        "",
        "## Hypothetical budget appendix",
        "",
        "Legacy R5/R7/R9 measurements are retained in candidates.csv. They describe distinct-test exceedance, "
        "not validated memory limits, and are not used in the primary selection or presentation.",
        "",
        "## Sources",
        "",
        "The two-part MDL explanation uses [Grünwald, A Tutorial Introduction to the Minimum "
        "Description Length Principle, "
        "sections 1.3–1.4](https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf). "
        "Tree path extraction is checked against [scikit-learn "
        "decision_path](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). "
        "EEPC, EA conventions and CSC are project definitions, not attributed to those sources.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = json.loads((OUT / "manifest.json").read_text())
    manifest["reporting_source_sha256"] = sha(__file__)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {k: v for k, v in facts.items() if k in ["candidate_count", "eligible_count", "groups", "test_summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
