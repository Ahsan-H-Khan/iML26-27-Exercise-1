"""Validation selection followed by untouched-test evaluation, without refitting."""

import json
import platform
from importlib.metadata import version
from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .csc import CognitiveStructuralComplexity
from .models import get_models


def run_protocol(datasets, seeds=(42, 1337, 2026), delta=0.02):
    rows = []
    versions = json.dumps(
        {**{p: version(p) for p in ("numpy", "pandas", "scikit-learn")}, "python": platform.python_version()},
        sort_keys=True,
    )
    metric = CognitiveStructuralComplexity()
    for dataset in datasets:
        for seed in seeds:
            X_dev, X_test, y_dev, y_test = train_test_split(
                dataset.X, dataset.y, test_size=0.25, stratify=dataset.y, random_state=seed
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_dev, y_dev, test_size=1 / 3, stratify=y_dev, random_state=seed
            )
            candidates = []
            for name, estimator in get_models(seed).items():
                start = perf_counter()
                model = clone(estimator).fit(X_train, y_train)
                metadata = dict(
                    dataset=dataset.name,
                    seed=seed,
                    model=name,
                    model_name=name,
                    model_family=type(model).__name__,
                    hyperparameters_json=json.dumps(model.get_params(), sort_keys=True),
                    n_train=len(X_train),
                    n_validation=len(X_val),
                    n_test=len(X_test),
                    n_features_raw=dataset.X.shape[1],
                    n_features_encoded=dataset.X.shape[1],
                    feature_names_json=json.dumps(dataset.feature_names),
                    n_classes=dataset.y.nunique(),
                    fit_seconds=perf_counter() - start,
                    package_versions_json=versions,
                    delta=delta,
                )
                row = score_model(model, X_val, y_val, metric, metadata, "validation")
                candidates.append((row, model, metadata))
            best = max(r["balanced_accuracy"] for r, _, _ in candidates)
            eligible = [item for item in candidates if item[0]["balanced_accuracy"] >= best - delta]
            winner, fitted, metadata = min(eligible, key=lambda item: (item[0]["csc"], item[0]["model"]))
            for row, _, _ in candidates:
                row.update(selected=row is winner, validation_best_balanced_accuracy=best)
                rows.append(row)
            test_row = score_model(fitted, X_test, y_test, metric, metadata, "test")
            test_row.update(selected=True, validation_best_balanced_accuracy=best)
            rows.append(test_row)
            print(
                f"{dataset.name} seed={seed}: {winner['model']}, test BA={test_row['balanced_accuracy']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def score_model(model, X, y, metric, metadata, split):
    start = perf_counter()
    components = metric.components(model, X)
    seconds = perf_counter() - start
    rounded = metric.components(model, X, decimals=6)
    prediction = model.predict(X)
    return dict(
        metadata,
        **components,
        csc_rounded6=rounded["csc"],
        metric_seconds=seconds,
        score_split=split,
        accuracy=accuracy_score(y, prediction),
        balanced_accuracy=balanced_accuracy_score(y, prediction),
        macro_f1=f1_score(y, prediction, average="macro"),
        above_proposed_budget=components["budget_r7"] > 0.5,
    )
