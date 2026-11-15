from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from iml_interpretability.csc import CognitiveStructuralComplexity
from iml_interpretability.tree_utils import TreeView, full_signature, prediction_trees, traverse

METRIC = CognitiveStructuralComplexity()


def forest_with(trees, X, y):
    forest = RandomForestClassifier(n_estimators=1, random_state=0).fit(X, y)
    forest.estimators_ = trees
    return forest


def test_constant_stump_and_duplicates():
    X, y = np.array([[0.0], [1.0]]), [0, 1]
    constant = DecisionTreeClassifier().fit(X, [0, 0])
    stump = DecisionTreeClassifier().fit(X, y)
    assert METRIC.components(constant, X)["csc"] == 0
    assert METRIC.components(stump, X)["csc"] == 2
    duplicate = forest_with([stump, deepcopy(stump)], X, y)
    result = METRIC.components(duplicate, X)
    assert result["csc"] == 2
    assert result["tepl"] == 2
    assert result["n_unique_trees"] == 1


def test_distinct_features_and_outputs():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    a = DecisionTreeClassifier(max_depth=1).fit(X, [0, 0, 1, 1])
    b = DecisionTreeClassifier(max_depth=1).fit(X, [0, 1, 0, 1])
    model = forest_with([a, b], X, [0, 0, 1, 1])
    assert METRIC.components(model, X)["csc"] == pytest.approx(2 + np.log2(3))
    b = deepcopy(a)
    b.tree_.value[1] = b.tree_.value[1][..., ::-1]
    model.estimators_ = [a, b]
    result = METRIC.components(model, X)
    assert result["n_unique_splits"] == 1
    assert result["n_unique_trees"] == 2


def test_unbalanced_occupancy_and_permutation():
    X = np.arange(10.0).reshape(-1, 1)
    model = DecisionTreeRegressor(max_depth=3).fit(X, [0] * 7 + [10, 11, 12])
    view = prediction_trees(model)[0]
    _, paths = traverse(view, X)
    short = next(i for i, p in enumerate(paths) if len(p) == 1)
    deep = next(i for i, p in enumerate(paths) if len(p) == 3)
    reference = X[[short] * 9 + [deep]]
    result = METRIC.components(model, reference)
    assert result["c_path"] == pytest.approx(1.2)
    assert result == METRIC.components(model, reference[::-1])
    shifted = METRIC.components(model, X[[deep] * 10])
    assert shifted["c_path"] == 3
    assert shifted["n_unique_splits"] == result["n_unique_splits"]


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeClassifier(max_depth=4),
        RandomForestClassifier(n_estimators=5, max_depth=4, random_state=42),
        GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42),
    ],
)
def test_traversal_matches_sklearn(model):
    X, y = load_iris(return_X_y=True)
    model.fit(X, y)
    for view in prediction_trees(model):
        leaves, paths = traverse(view, X)
        np.testing.assert_array_equal(leaves, view.estimator.apply(X))
        np.testing.assert_array_equal(
            [len(p) for p in paths], np.asarray(view.estimator.decision_path(X).sum(axis=1)).ravel() - 1
        )
    result = METRIC.components(model, X)
    if isinstance(model, GradientBoostingClassifier):
        assert result["n_trees"] == 15
    assert result == METRIC.components(model, pd.DataFrame(X, columns=list("abcd")))


def test_missing_routing_and_threshold_precision():
    X = np.array([[0.0], [1.0], [np.nan], [2.0]])
    model = DecisionTreeClassifier().fit(X, [0, 1, 0, 1])
    view = prediction_trees(model)[0]
    leaves, _ = traverse(view, X)
    np.testing.assert_array_equal(leaves, model.apply(X))
    changed = deepcopy(model)
    changed.tree_.threshold[0] = np.nextafter(changed.tree_.threshold[0], np.inf)
    assert full_signature(view) != full_signature(TreeView(changed, (), 1.0))


def test_channels_scales_and_invalid_input():
    X = np.array([[0.0], [1.0]])
    model = DecisionTreeClassifier().fit(X, [0, 1])
    assert full_signature(TreeView(model, (0,), 1)) != full_signature(TreeView(model, (1,), 1))
    assert full_signature(TreeView(model, (), 1)) != full_signature(TreeView(model, (), 0.1))
    with pytest.raises(ValueError):
        METRIC.components(model, np.empty((0, 1)))
    with pytest.raises(ValueError):
        METRIC.components(model, [[np.inf]])


def test_budget_counts_unique_rules():
    X = np.arange(128.0).reshape(-1, 1)
    tree = DecisionTreeClassifier(random_state=42).fit(X, np.arange(128) % 2)
    single = METRIC.components(tree, X)
    duplicate = METRIC.components(forest_with([tree, deepcopy(tree)], X, np.arange(128) % 2), X)
    assert single["budget_r7"] > 0.5
    for b in (5, 7, 9):
        assert single[f"budget_r{b}"] == duplicate[f"budget_r{b}"]
