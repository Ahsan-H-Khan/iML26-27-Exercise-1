import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.controlled_fixtures import (
    FixtureEnsemble,
    FixtureTree,
    Leaf,
    Split,
    factorial_reference,
    occupancy,
    path_trees,
    stump_bank,
)
from iml_interpretability.controlled_studies import (
    PROFILES,
    coverage_budget,
    matched_pairs,
    ranking_relation,
    validate_details,
)
from iml_interpretability.csc import CognitiveStructuralComplexity

METRIC = CognitiveStructuralComplexity()


@pytest.mark.parametrize("profile,expected", [("uniform", 2.25), ("early_exit", 1.3), ("deep_exit", 2.85)])
def test_same_function_size_and_vocabulary(profile, expected):
    trees = path_trees()
    X = occupancy(PROFILES[profile])
    np.testing.assert_array_equal(trees["balanced"].predict_proba(X), trees["chain"].predict_proba(X))
    for name, tree in trees.items():
        c, d = METRIC.measure(tree, X)
        assert (c["total_nodes"], c["total_leaves"], c["n_unique_splits"], c["n_unique_trees"]) == (7, 4, 3, 1)
        assert c["c_path"] == pytest.approx(2 if name == "balanced" else expected)
        assert c["tepl"] == c["c_path"]
        assert c["csc"] == pytest.approx(c["c_path"] + 2)
        validate_details(c, d)


def test_boundary_routing_and_fixture_validation():
    X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    for tree in path_trees().values():
        np.testing.assert_array_equal(tree.predict(X), [0, 0, 1, 0, 1])
    with pytest.raises(ValueError):
        FixtureTree(Leaf((2.0, 0.0)), 1)
    with pytest.raises(ValueError):
        FixtureTree(Split(1, 0.0, Leaf((1.0, 0.0)), Leaf((0.0, 1.0))), 1)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 6, 12])
def test_fixed_size_diversity(k):
    bank = stump_bank()
    X = factorial_reference()
    model = FixtureEnsemble(tuple(t for t in bank[:k] for _ in range(12 // k)))
    c, d = METRIC.measure(model, X)
    assert (c["total_nodes"], c["total_leaves"], c["n_trees"], c["tepl"]) == (36, 24, 12, 12)
    assert c["csc"] == pytest.approx(1 + np.log2(1 + k) + np.log2(k))
    np.testing.assert_array_equal(d["B"], np.full(4096, k))
    validate_details(c, d)


def test_equal_replication_preserves_function_but_increases_work():
    bank = stump_bank()[:3]
    X = factorial_reference()
    base = FixtureEnsemble(bank)
    a, da = METRIC.measure(base, X)
    for repeats in (2, 4):
        copy = FixtureEnsemble(bank * repeats)
        b, db = METRIC.measure(copy, X)
        np.testing.assert_allclose(base.predict_proba(X), copy.predict_proba(X))
        assert b["csc"] == a["csc"]
        np.testing.assert_array_equal(da["B"], db["B"])
        np.testing.assert_array_equal(da["L"] * repeats, db["L"])


def test_details_and_quantiles_have_exact_semantics():
    values = np.array([0] * 8 + [2, 7])
    assert coverage_budget(values, 0.5) == 0
    assert coverage_budget(values, 0.9) == 2
    assert coverage_budget(values, 0.95) == 7
    assert np.mean(values > 2) == 0.1
    curve = np.array([np.mean(values > b) for b in range(8)])
    assert np.all(np.diff(curve) <= 0) and curve[-1] == 0
    assert curve.sum() == pytest.approx(values.mean())
    for values, q in [([], 0.9), ([-1], 0.9), ([0.5], 0.9), ([1], 0)]:
        with pytest.raises(ValueError):
            coverage_budget(values, q)
    model = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 0])
    c, d = METRIC.measure(model, [[0.0], [1.0]])
    assert c == METRIC.components(model, [[0.0], [1.0]])
    assert not d["B"].any() and not d["L"].any()


def test_sklearn_and_explicit_stump_use_identical_measurement():
    X = np.array([[-1.0], [1.0]])
    fitted = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, [0, 1])
    fixture = FixtureTree(Split(0, 0.0, Leaf((1.0, 0.0)), Leaf((0.0, 1.0))), 1)
    assert METRIC.components(fitted, X) == METRIC.components(fixture, X)


def test_pair_matching_is_within_split_and_ranking_direction():
    import pandas as pd

    rows = []
    for seed in (42, 1337):
        for name, accuracy, csc, nodes in [("a", 0.95, 3, 9), ("b", 0.96, 4, 7), ("c", 0.7, 2, 3)]:
            rows.append(
                dict(
                    dataset="test",
                    seed=seed,
                    model=name,
                    balanced_accuracy=accuracy,
                    csc=csc,
                    total_nodes=nodes,
                    tepl=2,
                    n_unique_splits=1,
                    n_unique_trees=1,
                    c_path=1,
                    c_encoding=1,
                    c_ensemble=0,
                )
            )
    pairs = matched_pairs(pd.DataFrame(rows))
    assert len(pairs) == 6 and pairs.near_best_pair.sum() == 2
    assert set(pairs[pairs.near_best_pair].relation_total_nodes) == {"opposite"}
    assert ranking_relation(1, 0) == "baseline_tie"
    assert ranking_relation(1e-12, 0) == "both_tie"
