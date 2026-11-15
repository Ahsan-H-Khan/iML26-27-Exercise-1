"""Bounded validation-only meta-feature study with content-addressed condition caches."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score

from .data import load_benchmarks
from .dataset_variants import variants
from .metric_development import choose_candidates, extended_score, reconcile, sha, split_ids, verify_frozen
from .models import get_models

OUT = Path("results/metafeatures_v1")


def run(pilot=False):
    frozen = verify_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    old = pd.read_csv("results/metric_development_v2/candidates.csv")
    sources = {
        str(p): sha(p)
        for p in [
            Path("iml_interpretability/metafeature_study.py"),
            Path("iml_interpretability/metric_development.py"),
            Path("iml_interpretability/path_metrics.py"),
            Path("iml_interpretability/dataset_variants.py"),
            Path("iml_interpretability/models.py"),
            Path("protocols/remaining_tasks_v1.md"),
        ]
    }
    versions = {p: version(p) for p in ("numpy", "pandas", "scikit-learn")}
    rows, dummies, manifests, selections = [], [], [], []
    start = perf_counter()
    groups = [(d, s) for d in load_benchmarks() for s in (42, 1337, 2026)]
    for dataset, seed in groups[:1] if pilot else groups:
        X, y = dataset.X, dataset.y
        tr, va, te = split_ids(y, seed)
        datahash = hashlib.sha256(pd.util.hash_pandas_object(pd.concat([X, y], axis=1)).values.tobytes()).hexdigest()
        conditions = {}
        for variant in variants(y, tr, X.shape[1], seed):
            key = json.dumps([variant["train"], variant["columns"]])
            if key not in conditions:
                conditions[key] = dict(variant, aliases=[])
            conditions[key]["aliases"].append({k: variant[k] for k in ("experiment", "level", "control")})
        original = old[(old.dataset == dataset.name) & (old.seed == seed)]
        reference = original.balanced_accuracy.max() - 0.02
        for index, condition in enumerate(conditions.values()):
            full = condition["train"] == tr.tolist() and condition["columns"] == list(range(X.shape[1]))
            cid = "original" if full else f"condition_{index:02d}"
            spec = dict(
                dataset=dataset.name,
                seed=seed,
                condition=cid,
                **condition,
                validation=va.tolist(),
                test=te.tolist(),
                feature_names=X.columns[condition["columns"]].tolist(),
                data_sha256=datahash,
                versions=versions,
                source_sha256=sources,
            )
            digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            file = cache / f"{digest}.json"
            if file.exists():
                payload = json.loads(file.read_text(encoding="utf-8"))
                assert payload["digest"] == digest and len(payload["rows"]) == 12
            else:
                tx, ty = X.iloc[condition["train"], condition["columns"]], y.iloc[condition["train"]]
                vx, vy = X.iloc[va, condition["columns"]], y.iloc[va]
                dummy = float(balanced_accuracy_score(vy, DummyClassifier(strategy="prior").fit(tx, ty).predict(vx)))
                measured = []
                for name, estimator in get_models(seed).items():
                    model = clone(estimator).fit(tx, ty)
                    counts = list(condition["class_counts"].values())
                    meta = dict(
                        dataset=dataset.name,
                        seed=seed,
                        condition=cid,
                        model=name,
                        model_family=type(model).__name__,
                        n_train=len(tx),
                        n_validation=len(vx),
                        n_features_raw=tx.shape[1],
                        minority_fraction=min(counts) / sum(counts),
                        imbalance_ratio=max(counts) / min(counts),
                        hyperparameters_json=json.dumps(model.get_params(), sort_keys=True),
                    )
                    row = extended_score(model, vx, vy, meta, "validation")
                    row["meets_original_accuracy_reference"] = row["balanced_accuracy"] >= reference
                    row["original_accuracy_reference"] = reference
                    if full:
                        previous = original[original.model == name].iloc[0]
                        # The original row has metadata for the full dataset and identical validation metrics.
                        reconcile(row, previous)
                    measured.append(row)
                chosen = choose_candidates(measured, dummy)
                payload = dict(digest=digest, rows=measured, dummy=dummy, chosen=chosen)
                temporary = file.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(payload, indent=2, default=lambda value: value.item()), encoding="utf-8"
                )
                temporary.replace(file)
            rows.extend(payload["rows"])
            dummies.append(dict(dataset=dataset.name, seed=seed, condition=cid, balanced_accuracy=payload["dummy"]))
            for selector in ("minimum_csc", "accuracy_only"):
                chosen = payload["chosen"].get(selector)
                selections.append(
                    dict(
                        dataset=dataset.name,
                        seed=seed,
                        condition=cid,
                        selector=selector,
                        model=chosen,
                        status="selected" if chosen else "no_eligible_model",
                    )
                )
            manifests.append(spec)
            print(
                f"{dataset.name} seed={seed} {cid}: {len(rows)} candidate rows; {perf_counter() - start:.1f}s",
                flush=True,
            )
    prefix = "pilot_" if pilot else ""
    pd.DataFrame(rows).to_csv(OUT / f"{prefix}candidates.csv", index=False)
    pd.DataFrame(dummies).to_csv(OUT / f"{prefix}dummy_baselines.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / f"{prefix}selections.csv", index=False)
    (OUT / f"{prefix}condition_manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    (OUT / f"{prefix}manifest.json").write_text(
        json.dumps(
            dict(
                frozen=frozen,
                versions=versions,
                conditions=len(manifests),
                candidates=len(rows),
                elapsed_seconds=perf_counter() - start,
                sources=sources,
                test_evaluations=0,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    verify_frozen()
