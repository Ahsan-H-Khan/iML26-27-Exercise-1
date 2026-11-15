"""Investigate hypothetical rule budgets using frozen validation measurements."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from iml_interpretability.metric_development import sha, verify_frozen

verify_frozen()
source = Path("results/metric_development_v2/candidates.csv")
df = pd.read_csv(source)
out = Path("results/threshold_v1")
(out / "figures").mkdir(parents=True, exist_ok=True)
rows, switches = [], []
for (dataset, seed), group in df.groupby(["dataset", "seed"]):
    for scope, subset in (("all", group), ("eligible", group[group.eligible])):
        for b in (5, 7, 9):
            flag = subset[f"budget_r{b}"] > 0.5
            high, low = subset.loc[flag, "csc"], subset.loc[~flag, "csc"]
            assessable = len(high) > 0 and len(low) > 0
            rows.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    scope=scope,
                    budget=b,
                    candidate_denominator=len(subset),
                    validation_example_denominator=int(group.n_validation.iloc[0]),
                    flagged_count=int(flag.sum()),
                    flagged_fraction=float(flag.mean()) if len(flag) else None,
                    min_flagged_csc=high.min(),
                    max_flagged_csc=high.max(),
                    min_unflagged_csc=low.min(),
                    max_unflagged_csc=low.max(),
                    assessable=assessable,
                    increasing_cutoff_separates=bool(low.max() < high.min()) if assessable else None,
                    range_overlap_width=max(0.0, min(high.max(), low.max()) - max(high.min(), low.min()))
                    if assessable
                    else None,
                )
            )
        for a, b in ((5, 7), (7, 9), (5, 9)):
            switches.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    scope=scope,
                    budget_from=a,
                    budget_to=b,
                    candidate_denominator=len(subset),
                    changed_flags=int(((subset[f"budget_r{a}"] > 0.5) != (subset[f"budget_r{b}"] > 0.5)).sum()),
                )
            )
summary = pd.DataFrame(rows)
summary.to_csv(out / "threshold_summary.csv", index=False)
pd.DataFrame(switches).to_csv(out / "threshold_sensitivity.csv", index=False)
# Pooling seeds is descriptive; include it to assess a common boundary within each dataset.
pooled = []
for dataset, group in df.groupby("dataset"):
    for b in (5, 7, 9):
        flag = group[f"budget_r{b}"] > 0.5
        high, low = group.loc[flag, "csc"], group.loc[~flag, "csc"]
        pooled.append(
            dict(
                dataset=dataset,
                budget=b,
                flagged=int(flag.sum()),
                total=len(group),
                increasing_cutoff_separates=bool(low.max() < high.min()) if len(high) and len(low) else None,
                max_unflagged_csc=low.max(),
                min_flagged_csc=high.min(),
            )
        )
pd.DataFrame(pooled).to_csv(out / "pooled_thresholds.csv", index=False)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
colors = {
    "DecisionTreeClassifier": "#0072B2",
    "RandomForestClassifier": "#D55E00",
    "GradientBoostingClassifier": "#009E73",
}
markers = {42: "o", 1337: "s", 2026: "^"}
for axrow, (dataset, group) in zip(axes, df.groupby("dataset"), strict=True):
    for ax, b in zip(axrow, (5, 7, 9), strict=True):
        for (family, seed), part in group.groupby(["model_family", "seed"]):
            ax.scatter(part.csc, 100 * part[f"budget_r{b}"], color=colors[family], marker=markers[seed], alpha=0.7)
        ax.axhline(50, color=".4", ls="--", lw=1)
        ax.set(
            title=f"{dataset.replace('_', ' ').title()} | Budget: {b} tests",
            xlabel="CSC score",
            ylim=(-5, 105),
            xlim=(0, 18),
        )
        ax.grid(alpha=0.15)
    axrow[0].set_ylabel("Predictions exceeding budget (%)")

handles = [
    Line2D([], [], color=c, marker="o", ls="", label=n)
    for n, c in zip(("Decision tree", "Random forest", "Gradient boosting"), colors.values(), strict=True)
]
handles += [Line2D([], [], color=".4", marker=m, ls="", label=f"Seed {s}") for s, m in markers.items()]
fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for ext in ("png", "pdf", "svg"):
    fig.savefig(out / "figures" / f"threshold_sensitivity.{ext}", dpi=200)
lines = [
    "# Threshold investigation",
    "",
    "These are hypothetical rule-demand boundaries, not measured human limits.",
    "R_b is the fraction of validation examples with strictly more than b distinct encountered "
    "tests. A flag means R_b > 50%. CSC=7 does not mean seven tests.",
    "",
    "All 72 validation candidates are retained; eligible-only results are also saved. Seed groups "
    "have 12 candidates each. Reference-example denominators are explicit in threshold_summary.csv.",
    "",
    "## Across the three seeds (descriptive)",
    "",
]
for r in pooled:
    lines.append(
        f"- {r['dataset']}, budget {r['budget']}: {r['flagged']}/{r['total']} candidates flagged; an "
        f"increasing CSC cutoff separates these observed groups: {r['increasing_cutoff_separates']}."
    )
lines += [
    "",
    "Separation in this finite grid is not evidence of a universal cutoff. When ranges overlap, a "
    "single increasing cutoff cannot reproduce the rule-budget flags. Empty groups are unassessable. "
    "Both measures share model structure, so association is not independent human validation.",
    "",
    "The analysis reads the original R5/R7/R9 measurements; the meta-feature original-condition "
    "reruns independently reconcile them. No artificial controlled-study outputs are used.",
]
(out / "report.md").write_text("\n".join(lines), encoding="utf-8")
(out / "manifest.json").write_text(
    json.dumps(
        {"source": str(source), "sha256": sha(source), "script_sha256": sha(__file__), "candidates": len(df)}, indent=2
    ),
    encoding="utf-8",
)
print(pd.DataFrame(pooled).to_string(index=False))
