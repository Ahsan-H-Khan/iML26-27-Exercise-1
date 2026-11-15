"""Performance-first real-data evaluation, independent of historical controlled results."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .csc import CognitiveStructuralComplexity
from .data import load_benchmarks
from .experiment import score_model
from .models import get_models
from .path_metrics import path_metrics

OUT = Path("results/metric_development_v2")
PROTOCOL = Path("protocols/metric_development_v2.md")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen():
    hashes = json.loads(Path("protocols/metric_development_v2_hashes.json").read_text())
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f"Frozen file changed: {path}")
    return hashes


def split_ids(y, seed):
    dev, test = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed)
    train, val = train_test_split(dev, test_size=1 / 3, stratify=np.asarray(y)[dev], random_state=seed)
    return train, val, test


def choose_candidates(rows, dummy, delta=0.02):
    """No test inputs: mark eligibility and freeze deterministic validation choices."""
    best = max(r["balanced_accuracy"] for r in rows)
    for r in rows:
        r["dummy_balanced_accuracy"] = dummy
        r["validation_best_balanced_accuracy"] = best
        r["eligible"] = bool(r["balanced_accuracy"] > dummy and r["balanced_accuracy"] >= best - delta)
    eligible = [r for r in rows if r["eligible"]]
    chosen = {"accuracy_only": min(rows, key=lambda r: (-r["balanced_accuracy"], r["model"]))["model"]}
    if eligible:
        chosen["minimum_csc"] = min(eligible, key=lambda r: (r["csc"], r["model"]))["model"]
    for r in rows:
        r["selected_csc"] = r["model"] == chosen.get("minimum_csc")
        r["selected_accuracy"] = r["model"] == chosen["accuracy_only"]
    return chosen


def extended_score(model, X, y, metadata, split):
    row = score_model(model, X, y, CognitiveStructuralComplexity(), metadata, split)
    row.update(path_metrics(model, X))
    np.testing.assert_allclose(row["eepc_total"], row["tepl"], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(row["eepc_mean"], row["c_path"], atol=1e-10, rtol=1e-10)
    return row


def reconcile(row, old):
    errors = []
    for col in old.index:
        if col in {"fit_seconds", "metric_seconds", "selected"} or col not in row:
            continue
        expected = old[col]
        if isinstance(expected, int | float | np.number) and not isinstance(expected, bool | np.bool_):
            if pd.notna(expected):
                np.testing.assert_allclose(row[col], expected, rtol=1e-10, atol=1e-10, err_msg=col)
                errors.append(abs(float(row[col]) - float(expected)))
        elif col == "hyperparameters_json":
            assert json.loads(row[col]) == json.loads(expected)
    return max(errors, default=0.0)


def run():
    frozen = verify_frozen()
    old = pd.read_csv("results/scores.csv")
    candidates, baselines, choices, tests, partitions, reconciliation = [], [], [], [], [], []
    datasets = load_benchmarks()
    dataset_hashes = {}
    versions = {p: version(p) for p in ("numpy", "pandas", "scikit-learn")}
    versions["python"] = platform.python_version()
    for dataset in datasets:
        X, y = dataset.X, dataset.y
        dataset_hashes[dataset.name] = hashlib.sha256(
            pd.util.hash_pandas_object(pd.concat([X, y.rename("target")], axis=1), index=True).values.tobytes()
        ).hexdigest()
        for seed in (42, 1337, 2026):
            tr, va, te = split_ids(y, seed)
            partition = {"train": tr.tolist(), "validation": va.tolist(), "test": te.tolist()}
            split_id = hashlib.sha256(json.dumps(partition, sort_keys=True).encode()).hexdigest()[:16]
            partitions.append(dict(dataset=dataset.name, seed=seed, split_id=split_id, **partition))
            dummy = DummyClassifier(strategy="prior").fit(X.iloc[tr], y.iloc[tr])
            dp = dummy.predict(X.iloc[va])
            baseline = dict(
                dataset=dataset.name,
                seed=seed,
                split_id=split_id,
                score_split="validation",
                strategy="prior",
                balanced_accuracy=balanced_accuracy_score(y.iloc[va], dp),
                accuracy=accuracy_score(y.iloc[va], dp),
                macro_f1=f1_score(y.iloc[va], dp, average="macro"),
            )
            baselines.append(baseline)
            rows, fitted, metas = [], {}, {}
            for name, estimator in get_models(seed).items():
                model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
                metadata = dict(
                    dataset=dataset.name,
                    seed=seed,
                    split_id=split_id,
                    model=name,
                    model_family=type(model).__name__,
                    hyperparameters_json=json.dumps(model.get_params(), sort_keys=True),
                    n_train=len(tr),
                    n_validation=len(va),
                    n_test=len(te),
                    n_features_raw=X.shape[1],
                    n_features_encoded=X.shape[1],
                    n_classes=y.nunique(),
                    delta=0.02,
                    package_versions_json=json.dumps(versions, sort_keys=True),
                    metric_version="CSC-1.0",
                    protocol_version="metric_development_v2",
                )
                row = extended_score(model, X.iloc[va], y.iloc[va], metadata, "validation")
                rows.append(row)
                fitted[name], metas[name] = model, metadata
            chosen = choose_candidates(rows, baseline["balanced_accuracy"])
            for row in rows:
                previous = old[
                    (old.dataset == dataset.name)
                    & (old.seed == seed)
                    & (old.model == row["model"])
                    & (old.score_split == "validation")
                ].iloc[0]
                error = reconcile(row, previous)
                reconciliation.append(
                    dict(dataset=dataset.name, seed=seed, model=row["model"], split="validation", max_error=error)
                )
            candidates.extend(rows)
            # Selection table is fixed here, before any access to test predictions.
            for selector in ("minimum_csc", "accuracy_only"):
                name = chosen.get(selector)
                r = next((r for r in rows if r["model"] == name), None)
                choices.append(
                    dict(
                        dataset=dataset.name,
                        seed=seed,
                        split_id=split_id,
                        selector=selector,
                        model=name,
                        status="selected" if name else "no_eligible_model",
                        validation_balanced_accuracy=r["balanced_accuracy"] if r else np.nan,
                        validation_csc=r["csc"] if r else np.nan,
                    )
                )
            # Also reconcile any original frozen choice, even if the revised gate changes it.
            previous_tests = old[(old.dataset == dataset.name) & (old.seed == seed) & (old.score_split == "test")]
            to_score = set(chosen.values()) | set(previous_tests.model)
            for name in sorted(to_score):
                row = extended_score(fitted[name], X.iloc[te], y.iloc[te], metas[name], "test")
                row["validation_best_balanced_accuracy"] = rows[0]["validation_best_balanced_accuracy"]
                prev = previous_tests[previous_tests.model == name]
                if not prev.empty:
                    error = reconcile(row, prev.iloc[0])
                    reconciliation.append(
                        dict(dataset=dataset.name, seed=seed, model=name, split="test", max_error=error)
                    )
                if name in chosen.values():
                    row["selectors"] = ";".join(sorted(k for k, v in chosen.items() if v == name))
                    tests.append(row)
            print(
                f"{dataset.name} seed {seed}: {sum(r['eligible'] for r in rows)}/12 competitive; {chosen}", flush=True
            )
    verify_frozen()
    assert len(reconciliation) == len(old) == 78
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, data in [
        ("candidates", candidates),
        ("dummy_baselines", baselines),
        ("selections", choices),
        ("selected_test_scores", tests),
        ("reconciliation", reconciliation),
    ]:
        pd.DataFrame(data).to_csv(OUT / f"{filename}.csv", index=False)
    (OUT / "split_manifest.json").write_text(json.dumps(partitions, indent=2))
    sources = {str(p).replace("\\", "/"): sha(p) for p in sorted(Path("iml_interpretability").glob("*.py"))}
    sources.update({"scripts/run_metric_development.py": sha("scripts/run_metric_development.py")})
    manifest = dict(
        protocol_sha256=sha(PROTOCOL),
        frozen_hashes=frozen,
        dataset_hashes=dataset_hashes,
        versions=versions,
        source_hashes=sources,
        reconciled_rows=len(reconciliation),
        max_reconciliation_error=max(r["max_error"] for r in reconciliation),
        candidate_rows=len(candidates),
        unique_test_rows=len(tests),
        note="Existing benchmark splits, previously inspected; descriptive follow-up.",
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Reconciled {len(reconciliation)} old rows; saved {len(tests)} distinct selected test evaluations.")
