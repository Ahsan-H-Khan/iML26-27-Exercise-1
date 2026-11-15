"""Inherently interpretable model candidates.

Extend or replace :func:`get_models` with the models you want to compare, e.g. trees of different
depths to probe the threshold at which a model stops being interpretable, or random forests and
gradient-boosted trees to cover a whole model class.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

    models = {
        f"dt_d{d}": DecisionTreeClassifier(max_depth=d, min_samples_leaf=2, random_state=seed) for d in (1, 2, 4, 8)
    }
    models.update(
        {
            f"rf_n{n}_d{d}": RandomForestClassifier(
                n_estimators=n, max_depth=d, min_samples_leaf=2, random_state=seed, n_jobs=1
            )
            for n in (5, 25)
            for d in (2, 5)
        }
    )
    models.update(
        {
            f"gb_n{n}_d{d}": GradientBoostingClassifier(
                n_estimators=n, max_depth=d, learning_rate=0.1, random_state=seed
            )
            for n in (10, 50)
            for d in (1, 2)
        }
    )
    return models
