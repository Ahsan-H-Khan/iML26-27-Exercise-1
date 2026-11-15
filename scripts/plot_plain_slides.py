"""Audience-facing charts from unchanged controlled-v1 results."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("results/controlled_v1")
OUT = ROOT / "figures/plain_slides"
OUT.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 17,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#26384a",
        "text.color": "#26384a",
    }
)
BLUE, ORANGE = "#0072B2", "#D55E00"


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


q1 = pd.read_csv(ROOT / "q1_path_controls.csv")
fig, ax = plt.subplots(figsize=(11, 4.4))
fig.subplots_adjust(left=0.15, right=0.97, bottom=0.22, top=0.81)
for j, (fixture, label, color) in enumerate(
    [("balanced", "Balanced tree", BLUE), ("chain", "Chain-shaped tree", ORANGE)]
):
    values = [q1[(q1.fixture == fixture) & (q1.profile == p)].csc.iloc[0] for p in ("early_exit", "deep_exit")]
    bars = ax.bar(np.arange(2) + (j - 0.5) * 0.28, values, 0.28, label=label, color=color)
    ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=18)
ax.set(
    xticks=[0, 1],
    xticklabels=["Most examples exit early", "Most examples go deeper"],
    ylabel="CSC score",
    ylim=(0, 5.7),
)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
save(fig, "paths")

q2 = pd.read_csv(ROOT / "q2_redundancy_controls.csv")
fig, ax = plt.subplots(figsize=(10, 4.4))
fig.subplots_adjust(left=0.14, right=0.95, bottom=0.24, top=0.9)
ax.plot(q2.unique_stumps, q2.csc, marker="o", color=BLUE, linewidth=2.5, markersize=8)
ax.set(xlabel="Number of different rules", ylabel="CSC score", xticks=[1, 2, 3, 4, 6, 12], ylim=(0, 10))
for row in q2.iloc[[0, -1]].itertuples():
    ax.annotate(
        f"{row.csc:.2f}",
        (row.unique_stumps, row.csc),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=19,
    )
save(fig, "diversity")

summary = pd.read_csv(ROOT / "rule_demand_summary.csv")
selected = summary[(summary.reference_split == "validation") & summary.selected].copy()
selected = selected.sort_values(["dataset", "seed"])
fig, ax = plt.subplots(figsize=(11, 4.5))
fig.subplots_adjust(left=0.37, right=0.91, bottom=0.24, top=0.95)
labels = []
for row in selected.itertuples():
    dataset = "Iris" if row.dataset == "iris" else "Breast Cancer"
    labels.append(f"{dataset} · run {dict(zip([42, 1337, 2026], [1, 2, 3], strict=False))[row.seed]}")
bars = ax.barh(np.arange(6), selected.b90, color=BLUE, height=0.58)
ax.bar_label(bars, padding=8, fontsize=19)
ax.set(
    yticks=range(6),
    yticklabels=labels,
    xlabel="Different tests needed for 90% of examples",
    xlim=(0, 11),
    xticks=[0, 2, 4, 6, 8, 10],
)
ax.invert_yaxis()
save(fig, "demand")

pairs = pd.read_csv(ROOT / "q3_matched_pairs.csv")
r = pairs[
    (pairs.dataset == "breast_cancer")
    & (pairs.seed == 42)
    & (pairs.model_a == "gb_n50_d1")
    & (pairs.model_b == "rf_n5_d5")
].iloc[0]
assert r.near_best_pair and r.relation_total_nodes == "opposite"
(OUT / "example.json").write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
print("Plain-language charts generated; Q3 example verified against saved pairs.")
