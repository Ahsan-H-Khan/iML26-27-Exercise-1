import json

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability.csc import CognitiveStructuralComplexity
from iml_interpretability.path_metrics import path_metrics


@pytest.mark.parametrize(
    "estimator",
    [
        DecisionTreeClassifier(max_depth=3),
        RandomForestClassifier(n_estimators=3, max_depth=2, random_state=42),
        GradientBoostingClassifier(n_estimators=2, max_depth=2, random_state=42),
    ],
)
def test_epl_matches_sklearn_paths_and_csc(estimator):
    X, y = load_iris(return_X_y=True)
    model = estimator.fit(X, y)
    result = path_metrics(model, X)
    trees = [model] if hasattr(model, "tree_") else list(np.asarray(model.estimators_).flat)
    expected = [float(np.mean(t.decision_path(X).sum(axis=1) - 1)) for t in trees]
    assert json.loads(result["tree_epl_json"]) == pytest.approx(expected)
    comp = CognitiveStructuralComplexity().components(model, X)
    assert result["eepc_total"] == pytest.approx(sum(expected)) == pytest.approx(comp["tepl"])
    assert result["eepc_mean"] == pytest.approx(comp["c_path"])
    assert result["ea"] == pytest.approx(result["eepc_mean"] * np.log2(1 + result["features_used"]))
    assert result["ea_total"] == pytest.approx(result["ea"] * len(trees))
    if isinstance(model, GradientBoostingClassifier):
        assert len(expected) == 6  # three class-specific trees per stage


def test_feature_reuse_unused_inputs_and_leaf_only():
    X = np.column_stack([np.arange(8), np.zeros(8), np.zeros(8)])
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    model = DecisionTreeClassifier(random_state=0).fit(X, y)
    m = path_metrics(model, X)
    assert model.tree_.node_count > 3
    assert m["features_used"] == 1 and m["feature_burden"] == 1
    assert m["epl"] == m["eepc_total"] == m["eepc_mean"] == m["ea"]
    constant = DecisionTreeClassifier().fit(X, np.zeros(8))
    c = path_metrics(constant, X)
    assert c["features_used"] == c["epl"] == c["ea"] == 0
    with pytest.raises(ValueError, match="nonempty"):
        path_metrics(model, X[:0])
