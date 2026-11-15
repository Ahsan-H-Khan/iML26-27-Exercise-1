"""Reproducible controlled studies and demand reporting; original results are read-only."""

import hashlib
import json
import platform
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .controlled_fixtures import FixtureEnsemble, factorial_reference, occupancy, path_trees, stump_bank
from .csc import CognitiveStructuralComplexity
from .data import load_benchmarks
from .models import get_models

OUT = Path("results/controlled_v1")
TOL = 1e-10
METRIC_VERSION = "CSC-1.0"
PROFILES = {"uniform": (25, 25, 25, 25), "early_exit": (80, 10, 5, 5), "deep_exit": (5, 5, 45, 45)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_baseline():
    baseline = json.loads(Path("protocols/baseline_hashes.json").read_text())
    mutable = {"iml_interpretability/csc.py", "iml_interpretability/tree_utils.py"}
    for path, expected in baseline.items():
        if path not in mutable and digest(path) != expected:
            raise ValueError(f"Frozen baseline changed: {path}")


def provenance(stage):
    verify_baseline()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        str(p).replace("\\", "/"): digest(p)
        for folder in ("iml_interpretability", "scripts")
        for p in sorted(Path(folder).glob("*.py"))
    }
    record = dict(
        stage=stage,
        metric_version=METRIC_VERSION,
        protocol_sha256=digest("protocols/controlled_v1.md"),
        baseline_scores_sha256=digest("results/scores.csv"),
        source_sha256=sources,
        package_versions={p: version(p) for p in ("numpy", "pandas", "scikit-learn")},
        python=platform.python_version(),
    )
    run_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
    record["run_id"] = run_id
    path = OUT / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[stage] = record
    path.write_text(json.dumps(manifest, indent=2))
    return run_id


def save(frame, filename, run_id):
    frame = frame.copy()
    frame["metric_version"], frame["run_id"] = METRIC_VERSION, run_id
    frame.to_csv(OUT / filename, index=False)


def coverage_budget(values, q):
    values = np.asarray(values)
    if len(values) == 0 or not 0 < q <= 1 or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Coverage requires nonnegative integer demands and q in (0,1]")
    return int(np.sort(values)[int(np.ceil(q * len(values))) - 1])


def demand_summary(details):
    b = details["B"]
    return dict(
        mean_B=float(np.mean(b)),
        median_B=float(np.median(b)),
        b90=coverage_budget(b, 0.9),
        b95=coverage_budget(b, 0.95),
        max_B=int(np.max(b)),
        mean_L=float(np.mean(details["L"])),
    )


def validate_details(components, details):
    b, visits = np.asarray(details["B"]), np.asarray(details["L"])
    if not ((b >= 0).all() and (b <= visits).all() and (b <= components["n_unique_splits"]).all()):
        raise ValueError("Invalid demand bounds")
    np.testing.assert_allclose(visits.mean(), components["tepl"], rtol=TOL, atol=TOL)
    for budget in (5, 7, 9):
        np.testing.assert_allclose(np.mean(b > budget), components[f"budget_r{budget}"], rtol=TOL, atol=TOL)


def detail_frame(details, **metadata):
    frame = pd.DataFrame(details)
    frame["reference_row_id"] = np.arange(len(frame))
    for key, value in metadata.items():
        frame[key] = value
    return frame


def run_controls():
    run_id = provenance("controls")
    metric = CognitiveStructuralComplexity()
    q1, q2, replication, all_details = [], [], [], []
    trees = path_trees()
    for profile, counts in PROFILES.items():
        X = occupancy(counts)
        np.testing.assert_array_equal(trees["balanced"].predict_proba(X), trees["chain"].predict_proba(X))
        for name, tree in trees.items():
            components, details = metric.measure(tree, X)
            expected_path = 2 if name == "balanced" else np.dot(counts, [1, 2, 3, 3]) / 100
            np.testing.assert_allclose(components["c_path"], expected_path, atol=TOL)
            assert (components["total_nodes"], components["total_leaves"], components["n_unique_splits"]) == (7, 4, 3)
            np.testing.assert_allclose(components["csc"], expected_path + 2, atol=TOL)
            validate_details(components, details)
            q1.append(
                dict(fixture=name, profile=profile, prediction_agreement=True, **components, **demand_summary(details))
            )
            all_details.append(
                detail_frame(
                    details, dataset="constructed", study="q1", model=name, seed=profile, reference_split="fixture"
                )
            )
    bank, X = stump_bank(), factorial_reference()
    for k in (1, 2, 3, 4, 6, 12):
        model = FixtureEnsemble(tuple(t for t in bank[:k] for _ in range(12 // k)))
        components, details = metric.measure(model, X)
        np.testing.assert_allclose(components["csc"], 1 + np.log2(1 + k) + np.log2(k), atol=TOL)
        assert (components["n_trees"], components["total_nodes"], components["tepl"]) == (12, 36, 12)
        assert (components["n_unique_trees"], components["n_unique_splits"]) == (k, k)
        np.testing.assert_array_equal(details["B"], np.full(len(X), k))
        validate_details(components, details)
        q2.append(dict(unique_stumps=k, **components, **demand_summary(details)))
        all_details.append(
            detail_frame(
                details, dataset="constructed", study="q2", model=f"K{k}", seed="factorial", reference_split="fixture"
            )
        )
    base = FixtureEnsemble(bank[:3])
    base_predictions = base.predict_proba(X)
    base_components = metric.components(base, X)
    for repeats in (1, 2, 4):
        model = FixtureEnsemble(bank[:3] * repeats)
        components, details = metric.measure(model, X)
        error = float(np.max(np.abs(base_predictions - model.predict_proba(X))))
        np.testing.assert_allclose(model.predict_proba(X), base_predictions, atol=TOL)
        np.testing.assert_allclose(components["csc"], base_components["csc"], atol=TOL)
        assert components["total_nodes"] == 9 * repeats
        validate_details(components, details)
        replication.append(
            dict(replication=repeats, prediction_max_abs_error=error, **components, **demand_summary(details))
        )
        all_details.append(
            detail_frame(
                details,
                dataset="constructed",
                study="replication",
                model=f"repeat{repeats}",
                seed="factorial",
                reference_split="fixture",
            )
        )
    for rows, filename in [
        (q1, "q1_path_controls.csv"),
        (q2, "q2_redundancy_controls.csv"),
        (replication, "q2_replication_controls.csv"),
    ]:
        save(pd.DataFrame(rows), filename, run_id)
    save(pd.concat(all_details, ignore_index=True), "fixture_reference_demands.csv", run_id)
    verify_baseline()
    print("Verified 6 path controls, 6 fixed-size diversity controls, and 3 replication controls.", flush=True)


def direction(value):
    return 0 if abs(value) <= TOL else (1 if value > 0 else -1)


def ranking_relation(delta_csc, delta_baseline):
    a, b = direction(delta_csc), direction(delta_baseline)
    if a == 0 and b == 0:
        return "both_tie"
    if a == 0:
        return "csc_tie"
    if b == 0:
        return "baseline_tie"
    return "agree" if a == b else "opposite"


def matched_pairs(validation):
    rows = []
    metrics = [
        "balanced_accuracy",
        "csc",
        "total_nodes",
        "tepl",
        "n_unique_splits",
        "n_unique_trees",
        "c_path",
        "c_encoding",
        "c_ensemble",
    ]
    for (dataset, seed), group in validation.groupby(["dataset", "seed"]):
        best = group.balanced_accuracy.max()
        for a, b in combinations(group.sort_values("model").to_dict("records"), 2):
            row = dict(dataset=dataset, seed=seed, model_a=a["model"], model_b=b["model"])
            for metric in metrics:
                row[f"{metric}_a"], row[f"{metric}_b"] = a[metric], b[metric]
                row[f"delta_{metric}"] = b[metric] - a[metric]
                row[f"abs_delta_{metric}"] = abs(b[metric] - a[metric])
            row["performance_matched"] = row["abs_delta_balanced_accuracy"] <= 0.02
            row["near_best_pair"] = (
                row["performance_matched"] and min(a["balanced_accuracy"], b["balanced_accuracy"]) >= best - 0.02
            )
            for baseline in ("total_nodes", "tepl"):
                row[f"relation_{baseline}"] = ranking_relation(row["delta_csc"], row[f"delta_{baseline}"])
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_pairs():
    run_id = provenance("matched_pairs")
    raw = pd.read_csv("results/scores.csv")
    pairs = matched_pairs(raw[raw.score_split == "validation"])
    assert len(pairs) == 396
    summaries = []
    for (dataset, seed), group in pairs.groupby(["dataset", "seed"]):
        for subset, selected in [
            ("all_matched", group[group.performance_matched]),
            ("near_best", group[group.near_best_pair]),
        ]:
            row = dict(dataset=dataset, seed=seed, subset=subset, n_pairs=len(selected))
            for baseline in ("total_nodes", "tepl"):
                for relation in ("agree", "opposite", "both_tie", "csc_tie", "baseline_tie"):
                    row[f"{baseline}_{relation}"] = int((selected[f"relation_{baseline}"] == relation).sum())
            summaries.append(row)
    displayed = (
        pairs[pairs.near_best_pair]
        .sort_values(["dataset", "seed", "abs_delta_balanced_accuracy", "model_a", "model_b"])
        .groupby(["dataset", "seed"])
        .head(3)
    )
    save(pairs, "q3_matched_pairs.csv", run_id)
    save(pd.DataFrame(summaries), "q3_pair_summary.csv", run_id)
    save(displayed, "q3_displayed_pairs.csv", run_id)
    print(
        f"{len(pairs)} pairs: {pairs.performance_matched.sum()} accuracy-matched; "
        f"{pairs.near_best_pair.sum()} near-best.",
        flush=True,
    )


def collect_demands():
    run_id = provenance("reference_demands")
    original = pd.read_csv("results/scores.csv")
    metric = CognitiveStructuralComplexity()
    details_rows, summaries, reconciliation, split_rows, curves, scenarios = [], [], [], [], [], []
    data_hashes = {}
    for dataset in load_benchmarks():
        data_hashes[dataset.name] = hashlib.sha256(
            pd.util.hash_pandas_object(dataset.X.assign(target=dataset.y), index=True).values.tobytes()
        ).hexdigest()
        for seed in (42, 1337, 2026):
            X_dev, X_test, y_dev, y_test = train_test_split(
                dataset.X, dataset.y, test_size=0.25, stratify=dataset.y, random_state=seed
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_dev, y_dev, test_size=1 / 3, stratify=y_dev, random_state=seed
            )
            for split, X in [("train", X_train), ("validation", X_val), ("test", X_test)]:
                split_rows.extend(dict(dataset=dataset.name, seed=seed, split=split, row_id=int(i)) for i in X.index)
            old_group = original[(original.dataset == dataset.name) & (original.seed == seed)]
            for name, estimator in get_models(seed).items():
                fitted = clone(estimator).fit(X_train, y_train)
                expected_rows = old_group[old_group.model == name]
                for expected in expected_rows.to_dict("records"):
                    split = expected["score_split"]
                    X, y = (X_val, y_val) if split == "validation" else (X_test, y_test)
                    components, details = metric.measure(fitted, X)
                    prediction = fitted.predict(X)
                    measured = dict(
                        components,
                        csc_rounded6=metric.components(fitted, X, decimals=6)["csc"],
                        accuracy=accuracy_score(y, prediction),
                        balanced_accuracy=balanced_accuracy_score(y, prediction),
                        macro_f1=f1_score(y, prediction, average="macro"),
                        seed=seed,
                        delta=0.02,
                        n_train=len(X_train),
                        n_validation=len(X_val),
                        n_test=len(X_test),
                        n_features_raw=dataset.X.shape[1],
                        n_features_encoded=dataset.X.shape[1],
                        n_classes=dataset.y.nunique(),
                    )
                    errors = []
                    for key, value in measured.items():
                        np.testing.assert_allclose(
                            value,
                            expected[key],
                            atol=TOL,
                            rtol=TOL,
                            err_msg=f"{dataset.name}/{seed}/{name}/{split}/{key}",
                        )
                        errors.append(abs(value - expected[key]))
                    if json.loads(expected["hyperparameters_json"]) != fitted.get_params():
                        raise ValueError("Reconstructed model parameters differ")
                    validate_details(components, details)
                    if bool(expected["above_proposed_budget"]) != (components["budget_r7"] > 0.5):
                        raise ValueError("Budget flag mismatch")
                    metadata = dict(
                        dataset=dataset.name,
                        study="original_reconstruction",
                        model=name,
                        seed=seed,
                        reference_split=split,
                    )
                    frame = detail_frame(details, **metadata)
                    frame["reference_row_id"] = X.index.to_numpy()
                    details_rows.append(frame)
                    summaries.append(
                        dict(metadata, selected=expected["selected"], **measured, **demand_summary(details))
                    )
                    reconciliation.append(
                        dict(metadata, numeric_fields_checked=len(measured), max_abs_error=max(errors))
                    )
                    b = details["B"]
                    r = np.array([np.mean(b > budget) for budget in range(int(b.max()) + 1)])
                    assert np.all(np.diff(r) <= 0) and r[-1] == 0
                    np.testing.assert_allclose(r.sum(), np.mean(b), atol=TOL)
                    curves.extend(
                        dict(metadata, budget=budget, exceedance=float(value)) for budget, value in enumerate(r)
                    )
                    if split == "validation":
                        scenarios.extend(
                            dict(metadata, budget=budget, alpha=alpha, flagged=bool(np.mean(b > budget) > alpha))
                            for budget in range(1, 16)
                            for alpha in (0.1, 0.25, 0.5)
                        )
            print(f"Reconciled demands: {dataset.name}, seed {seed}", flush=True)
    summary = pd.DataFrame(summaries)
    for (dataset, seed), group in summary[summary.reference_split == "validation"].groupby(["dataset", "seed"]):
        stored = original[(original.dataset == dataset) & (original.seed == seed)]
        np.testing.assert_allclose(
            stored.validation_best_balanced_accuracy, group.balanced_accuracy.max(), atol=TOL, rtol=TOL
        )
        eligible = group[group.balanced_accuracy >= group.balanced_accuracy.max() - 0.02]
        winner = eligible.sort_values(["csc", "model"]).iloc[0].model
        selected = group[group.selected]
        assert len(selected) == 1 and selected.iloc[0].model == winner
    assert len(summary) == 78
    for frame, name in [
        (pd.concat(details_rows, ignore_index=True), "reference_demands.csv"),
        (summary, "rule_demand_summary.csv"),
        (pd.DataFrame(reconciliation), "reconciliation.csv"),
        (pd.DataFrame(split_rows), "split_manifest.csv"),
        (pd.DataFrame(curves), "rule_demand_curves.csv"),
        (pd.DataFrame(scenarios), "hypothetical_scenarios.csv"),
    ]:
        save(frame, name, run_id)
    (OUT / "dataset_hashes.json").write_text(json.dumps(data_hashes, indent=2))
    verify_baseline()
    print("All 78 original rows and six selections reconciled; baseline files unchanged.", flush=True)
