import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from iml_interpretability.metric_development import choose_candidates, reconcile, split_ids


def test_eligibility_strict_baseline_inclusive_tolerance_and_ties():
    rows = [
        dict(model="z", balanced_accuracy=0.96, csc=5),
        dict(model="a", balanced_accuracy=0.96, csc=5),
        dict(model="b", balanced_accuracy=0.94, csc=2),
        dict(model="c", balanced_accuracy=0.93, csc=1),
    ]
    chosen = choose_candidates(rows, 0.5)
    assert chosen == {"minimum_csc": "b", "accuracy_only": "a"}
    assert [r["eligible"] for r in rows] == [True, True, True, False]
    chosen = choose_candidates(rows, 0.96)
    assert chosen == {"accuracy_only": "a"}
    assert not any(r["eligible"] for r in rows)


def test_coincident_choices_and_test_information_not_used():
    rows = [
        dict(model="a", balanced_accuracy=0.9, csc=2, test_balanced_accuracy=0),
        dict(model="b", balanced_accuracy=0.89, csc=3, test_balanced_accuracy=1),
    ]
    selected = choose_candidates(rows, 0.5)
    assert set(selected.values()) == {"a"}
    rows[0]["test_balanced_accuracy"], rows[1]["test_balanced_accuracy"] = 1, 0
    assert choose_candidates(rows, 0.5) == selected


def test_splits_disjoint_complete_reproducible_and_stratified():
    _, y = load_iris(return_X_y=True)
    parts = split_ids(y, 42)
    assert set(np.concatenate(parts)) == set(range(len(y)))
    assert sum(map(len, parts)) == len(y)
    for a, b in zip(parts, split_ids(y, 42), strict=True):
        np.testing.assert_array_equal(a, b)
        assert len(set(y[a])) == 3


def test_reconciliation_rejects_changed_metrics():
    old = pd.Series(dict(csc=3.0, balanced_accuracy=0.9, metric_seconds=100))
    assert reconcile(dict(csc=3.0, balanced_accuracy=0.9, metric_seconds=1), old) == 0
    with pytest.raises(AssertionError):
        reconcile(dict(csc=3.1, balanced_accuracy=0.9), old)
