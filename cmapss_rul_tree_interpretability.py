import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

DATA = Path(__file__).resolve().parent
OUT = DATA / "results"; OUT.mkdir(exist_ok=True)
DATASETS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}   # n operating regimes
CLIP, SEED, TOL, THRESH = 125, 42, 0.05, 30
cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
ops, all_sensors = ["op1", "op2", "op3"], cols[5:]


def load(kind, fd):
    return pd.read_csv(DATA / f"{kind}_{fd}.txt", sep=r"\s+", header=None, names=cols)


def regime_normalise(train, test, k):
    """Standardise sensors per operating regime (fit on train only)."""
    train, test = train.copy(), test.copy()
    if k == 1:
        train["reg"], test["reg"] = 0, 0
    else:
        km = KMeans(k, n_init=10, random_state=SEED).fit(train[ops])
        train["reg"], test["reg"] = km.labels_, km.predict(test[ops])
    stats = train.groupby("reg")[all_sensors].agg(["mean", "std"])
    for df in (train, test):
        for s in all_sensors:
            mu = df["reg"].map(stats[(s, "mean")])
            sd = df["reg"].map(stats[(s, "std")]).where(lambda v: v > 1e-6, 1.0)
            df[s] = (df[s] - mu) / sd
    return train, test


def add_features(df, sensors, w=10):
    df = df.copy()
    g = df.groupby("unit")[sensors]
    m = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
    sd = g.rolling(w, min_periods=2).std().reset_index(level=0, drop=True).fillna(0)
    for s in sensors:
        df[f"{s}_m"], df[f"{s}_sd"] = m[s], sd[s]
    return df


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def tree_signature(t, node=0, nd=6):
    tr = t.tree_
    if tr.children_left[node] == -1:
        return "L"
    return (f"({tr.feature[node]}@{round(tr.threshold[node], nd)}"
            f"{tree_signature(t, tr.children_left[node])}"
            f"{tree_signature(t, tr.children_right[node])})")


def csc(trees, Xref, nd=6):
    paths, tests, sigs = [], set(), set()
    for t in trees:
        tr = t.tree_
        dp = t.decision_path(Xref)
        paths.append((np.asarray(dp.sum(axis=1)).ravel() - 1).mean())
        for n in range(tr.node_count):
            if tr.children_left[n] != -1:
                tests.add((tr.feature[n], round(tr.threshold[n], nd)))
        sigs.add(tree_signature(t))
    mp, S, U = np.mean(paths), len(tests), len(sigs)
    return dict(mean_path=mp, S=S, U=U, M=len(trees),
                nodes=sum(t.tree_.node_count for t in trees),
                term_path=mp, term_S=np.log2(1 + S), term_U=np.log2(U),
                CSC=mp + np.log2(1 + S) + np.log2(U))


models = {
    "DT_d4":       lambda: DecisionTreeRegressor(max_depth=4, random_state=SEED),
    "DT_d8":       lambda: DecisionTreeRegressor(max_depth=8, random_state=SEED),
    "RF_n20_d6":   lambda: RandomForestRegressor(20, max_depth=6, n_jobs=-1, random_state=SEED),
    "RF_n100_d10": lambda: RandomForestRegressor(100, max_depth=10, n_jobs=-1, random_state=SEED),
    "GB_n50_d2":   lambda: GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=SEED),
    "GB_n200_d3":  lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=SEED),
}


def get_trees(m):
    if isinstance(m, DecisionTreeRegressor): return [m]
    if isinstance(m, RandomForestRegressor): return list(m.estimators_)
    return [e[0] for e in m.estimators_]


def run_dataset(fd, k):
    print(f"\n=== {fd} (regimes={k}) ===", flush=True)
    train, test = load("train", fd), load("test", fd)
    rul_true = pd.read_csv(DATA / f"RUL_{fd}.txt", header=None)[0].values
    train["RUL"] = (train.groupby("unit")["cycle"].transform("max") - train["cycle"]).clip(upper=CLIP)
    train, test = regime_normalise(train, test, k)
    sensors = [s for s in all_sensors if train[s].std() > 1e-6]
    train, test = add_features(train, sensors), add_features(test, sensors)
    feats = ops + sensors + [f"{s}_m" for s in sensors] + [f"{s}_sd" for s in sensors]
    X, y, groups = train[feats].values, train["RUL"].values, train["unit"].values
    X_test = test.groupby("unit").tail(1)[feats].values
    gkf, rows = GroupKFold(n_splits=5), []
    for name, make in models.items():
        cv = [np.sqrt(mean_squared_error(y[v], make().fit(X[t], y[t]).predict(X[v])))
              for t, v in gkf.split(X, y, groups)]
        m = make().fit(X, y)
        pred = np.clip(m.predict(X_test), 0, CLIP)
        r = csc(get_trees(m), X[::5])
        rows.append(dict(dataset=fd, model=name, cv_rmse=np.mean(cv),
                         test_rmse=np.sqrt(mean_squared_error(rul_true, pred)),
                         test_mae=mean_absolute_error(rul_true, pred),
                         nasa_score=nasa_score(rul_true, pred),
                         flagged=int((pred < THRESH).sum()), **r))
        print(f"  {name:12s} cv_rmse={np.mean(cv):6.2f} CSC={r['CSC']:6.2f}", flush=True)
    return pd.DataFrame(rows)


res = pd.concat([run_dataset(fd, k) for fd, k in DATASETS.items()], ignore_index=True)
res.to_csv(OUT / "all_results.csv", index=False)

winners = {}
for fd, d in res.groupby("dataset"):
    near = d[d.cv_rmse <= d.cv_rmse.min() * (1 + TOL)]
    winners[fd] = near.sort_values("CSC").iloc[0].model
summary = pd.DataFrame({"dataset": list(winners), "winner": list(winners.values())})
summary.to_csv(OUT / "winners.csv", index=False)
print("\nMost interpretable near-best model per dataset:\n", summary.to_string(index=False))

# ---------- styling ----------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120})
fam_col = {"DT": "#1f77b4", "RF": "#2ca02c", "GB": "#d62728"}
fam_mark = {"DT": "o", "RF": "s", "GB": "^"}
fam = lambda n: n.split("_")[0]
order = list(models)
fds = list(DATASETS)

# ---------- Figure 1: accuracy vs structural complexity ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
for ax, fd in zip(axes.ravel(), fds):
    d = res[res.dataset == fd].sort_values("CSC")
    for _, r in d.iterrows():
        f = fam(r.model)
        ax.scatter(r.CSC, r.cv_rmse, s=110, c=fam_col[f], marker=fam_mark[f], zorder=3,
                   edgecolor="k", linewidth=1.5 if r.model == winners[fd] else 0.3)
        ax.annotate(r.model, (r.CSC, r.cv_rmse), fontsize=8, xytext=(5, 5), textcoords="offset points")
    pf = d[d.cv_rmse <= d.cv_rmse.cummin()]
    ax.step(pf.CSC, pf.cv_rmse, where="post", c="grey", ls="--", lw=1, label="Pareto front")
    w = d[d.model == winners[fd]].iloc[0]
    ax.scatter(w.CSC, w.cv_rmse, s=420, facecolors="none", edgecolors="gold", linewidths=2.5, zorder=4)
    ax.axhline(d.cv_rmse.min() * (1 + TOL), c="orange", ls=":", lw=1)
    ax.set_title(f"{fd}  |  winner: {winners[fd]}")
    ax.set_xlabel("CSC (lower = more interpretable)"); ax.set_ylabel("Grouped CV RMSE [cycles]")
handles = [plt.Line2D([], [], marker=fam_mark[f], ls="", c=fam_col[f], label=l)
           for f, l in [("DT", "Decision tree"), ("RF", "Random forest"), ("GB", "Gradient boosting")]]
handles += [plt.Line2D([], [], c="orange", ls=":", label="+5% tolerance"),
            plt.Line2D([], [], marker="o", ls="", mfc="none", mec="gold", ms=12, mew=2, label="Selected")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
fig.suptitle("Fig. 1  Accuracy vs. structural complexity (C-MAPSS)", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0.05, 1, 0.96]); fig.savefig(OUT / "fig1_accuracy_vs_csc.png", dpi=250); plt.close(fig)

# ---------- Figure 2: what makes models complex (CSC decomposition) ----------
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
parts = [("term_path", "Mean path length", "#4c72b0"),
         ("term_S", r"$\log_2(1+S)$ distinct tests", "#dd8452"),
         ("term_U", r"$\log_2(U)$ distinct trees", "#55a868")]
for ax, fd in zip(axes, fds):
    d = res[res.dataset == fd].set_index("model").loc[order]
    bottom = np.zeros(len(d))
    for c, lab, col in parts:
        ax.bar(d.index, d[c], bottom=bottom, color=col, label=lab); bottom += d[c].values
    for i, v in enumerate(bottom):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_title(fd); ax.tick_params(axis="x", rotation=60)
axes[0].set_ylabel("CSC")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Fig. 2  CSC decomposition: path length vs. distinct logic", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "fig2_csc_decomposition.png", dpi=250); plt.close(fig)

# ---------- Figure 3: safety & generalisation ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
x, wd = np.arange(len(order)), 0.2
pal = plt.cm.viridis(np.linspace(0.1, 0.85, 4))
for ax, (col, ttl, log) in zip(axes, [("test_rmse", "Test RMSE [cycles]", False),
                                      ("nasa_score", "NASA score (log scale, lower = safer)", True)]):
    for i, fd in enumerate(fds):
        d = res[res.dataset == fd].set_index("model").loc[order]
        ax.bar(x + (i - 1.5) * wd, d[col], wd, label=fd, color=pal[i])
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right"); ax.set_title(ttl)
    if log: ax.set_yscale("log")
ax = axes[2]                                     # generalisation gap: CV vs test RMSE
for fd, c in zip(fds, pal):
    d = res[res.dataset == fd].set_index("model").loc[order]
    ax.plot(order, d.test_rmse - d.cv_rmse, "o-", c=c, label=fd)
ax.axhline(0, c="k", lw=0.8); ax.tick_params(axis="x", rotation=45)
ax.set_title("Generalisation gap (test RMSE − CV RMSE)")
axes[0].legend(frameon=False, ncol=2)
fig.suptitle("Fig. 3  Safety and generalisation across operating conditions", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / "fig3_safety_generalisation.png", dpi=250); plt.close(fig)
print(f"\nSaved CSV + 3 figures to {OUT}")