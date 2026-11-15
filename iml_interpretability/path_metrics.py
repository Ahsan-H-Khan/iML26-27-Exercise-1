"""Path and feature measures for the metric-development narrative."""

import json

import numpy as np

from .tree_utils import prediction_trees, traverse, tree_structure


def path_metrics(model, X):
    """Empirical internal-node EPL; unweighted across actual prediction trees."""
    if len(X) == 0:
        raise ValueError("A nonempty reference set is required")
    views = prediction_trees(model)
    features, epls = set(), []
    for view in views:
        tree = tree_structure(view)
        features.update(int(tree.feature[i]) for i in range(tree.node_count) if tree.children_left[i] != -1)
        _, paths = traverse(view, X)
        epls.append(float(np.mean([len(p) for p in paths])))
    total, mean = float(np.sum(epls)), float(np.mean(epls))
    burden = float(np.log2(1 + len(features)))
    return dict(
        tree_epl_json=json.dumps(epls),
        epl=epls[0] if len(views) == 1 else np.nan,
        eepc_total=total,
        eepc_mean=mean,
        features_used=len(features),
        used_feature_positions_json=json.dumps(sorted(features)),
        feature_burden=burden,
        ea=mean * burden,
        ea_total=total * burden,
    )
