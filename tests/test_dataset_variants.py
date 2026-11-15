import numpy as np

from iml_interpretability.data import load_benchmarks
from iml_interpretability.dataset_variants import allocation, variants
from iml_interpretability.metric_development import split_ids


def test_nested_fixed_partitions_and_matched_controls():
    for d in load_benchmarks():
        for seed in (42, 1337, 2026):
            tr, va, te = split_ids(d.y, seed)
            assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            conditions = variants(d.y, tr, d.X.shape[1], seed)
            assert conditions == variants(d.y, tr, d.X.shape[1], seed)
            for c in conditions:
                assert set(c["train"]) <= set(tr)
                assert sum(c["class_counts"].values()) == len(c["train"])
                assert min(c["class_counts"].values()) >= 2
                assert c["columns"] == sorted(c["columns"])
            for experiment, key in (("samples", "train"), ("features", "columns")):
                subset = [c for c in conditions if c["experiment"] == experiment]
                for a, b in zip(subset, subset[1:], strict=False):
                    assert set(a[key]) <= set(b[key])
                assert subset[-1]["train"] == tr.tolist()
                assert subset[-1]["columns"] == list(range(d.X.shape[1]))
            for fraction in (0.5, 0.25):
                imbalance, control = [c for c in conditions if c["experiment"] == "balance" and c["level"] == fraction]
                assert len(imbalance["train"]) == len(control["train"])
                assert not imbalance["control"] and control["control"]
                actual = np.array(list(control["class_counts"].values()))
                counts = np.unique(d.y.iloc[tr], return_counts=True)[1]
                assert np.max(np.abs(actual - counts / counts.sum() * actual.sum())) <= 1


def test_allocation_exact_total():
    assert allocation([25, 25, 24], 56).tolist() == [19, 19, 18]
