"""Exact structural identities and independent sklearn-compatible traversal."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from .controlled_fixtures import FixtureEnsemble, FixtureTree


@dataclass(frozen=True)
class TreeView:
    estimator: object
    channel: tuple
    scale: float


def prediction_trees(model):
    if isinstance(model, FixtureTree):
        return [TreeView(model, (), 1.0)]
    if isinstance(model, FixtureEnsemble):
        return [TreeView(t, (), 1.0 / len(model.trees)) for t in model.trees]
    check_is_fitted(model)
    if isinstance(model, DecisionTreeClassifier | DecisionTreeRegressor):
        return [TreeView(model, (), 1.0)]
    if isinstance(model, RandomForestClassifier | RandomForestRegressor):
        return [TreeView(t, (), 1.0 / len(model.estimators_)) for t in model.estimators_]
    if isinstance(model, GradientBoostingClassifier | GradientBoostingRegressor):
        if model.init is not None and model.init != "zero":
            raise TypeError("Custom boosting initializers are unsupported")
        return [TreeView(t, (j,), model.learning_rate) for stage in model.estimators_ for j, t in enumerate(stage)]
    raise TypeError(f"Unsupported model: {type(model).__name__}")


def tree_structure(view):
    return view.estimator.structure if isinstance(view.estimator, FixtureTree) else view.estimator.tree_


def split_id(tree, node, decimals=None):
    threshold = float(tree.threshold[node])
    if decimals is not None:
        threshold = round(threshold, decimals)
    route = int(tree.missing_go_to_left[node]) if hasattr(tree, "missing_go_to_left") else None
    return (int(tree.feature[node]), "<=", threshold.hex(), route)


def full_signature(view, decimals=None):
    """Classification payloads are normalized per-output probabilities."""
    estimator, tree = view.estimator, tree_structure(view)
    nodes = {}
    for node in range(tree.node_count - 1, -1, -1):
        if tree.children_left[node] == -1:
            value = tree.value[node].copy()
            if isinstance(estimator, DecisionTreeClassifier | FixtureTree):
                value /= value.sum(axis=1, keepdims=True)
            nodes[node] = ("leaf", tuple(value.shape), tuple(float(v).hex() for v in value.flat))
        else:
            nodes[node] = (
                "split",
                split_id(tree, node, decimals),
                nodes[tree.children_left[node]],
                nodes[tree.children_right[node]],
            )
    classes = estimator.classes if isinstance(estimator, FixtureTree) else getattr(estimator, "classes_", ())
    labels = repr(classes.tolist() if isinstance(classes, np.ndarray) else classes)
    return (view.channel, float(view.scale).hex(), labels, nodes[0])


def traverse(view, X, decimals=None):
    """Use float32 input conversion, just as sklearn prediction does."""
    X = np.asarray(X, dtype=np.float32)
    tree = tree_structure(view)
    features = view.estimator.n_features if isinstance(view.estimator, FixtureTree) else view.estimator.n_features_in_
    if X.ndim != 2 or not len(X) or X.shape[1] != features:
        raise ValueError("X_ref must be nonempty, 2D, and match the fitted feature count")
    if np.isinf(X).any():
        raise ValueError("Infinite reference values are unsupported")
    if np.isnan(X).any() and not hasattr(tree, "missing_go_to_left"):
        raise ValueError("This tree has no native missing-value routing")
    leaves, paths = [], []
    for row in X:
        node, tests = 0, []
        while tree.children_left[node] != -1:
            tests.append(split_id(tree, node, decimals))
            value = row[tree.feature[node]]
            left = bool(tree.missing_go_to_left[node]) if np.isnan(value) else value <= tree.threshold[node]
            node = tree.children_left[node] if left else tree.children_right[node]
        leaves.append(node)
        paths.append(tests)
    return np.array(leaves), paths
