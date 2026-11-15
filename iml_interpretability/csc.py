"""Cognitive-Structural Complexity: an unvalidated structural proxy."""

import numpy as np

from .metric import InterpretabilityMetric
from .tree_utils import full_signature, prediction_trees, split_id, traverse, tree_structure


class CognitiveStructuralComplexity(InterpretabilityMetric):
    name = "csc"
    requires_reference = True

    def score(self, model, dataset):
        return self.components(model, dataset.X)["csc"]

    def components(self, model, X_ref, decimals=None):
        return self.measure(model, X_ref, decimals)[0]

    def reference_details(self, model, X_ref):
        """Return per-reference-row B (distinct tests), L (visits), and M (trees)."""
        return self.measure(model, X_ref)[1]

    def measure(self, model, X_ref, decimals=None):
        """Shared traversal for unchanged CSC and optional per-row demand reporting."""
        views = prediction_trees(model)
        rules, signatures = set(), set()
        encountered = [set() for _ in range(len(X_ref))]
        lengths = []
        total_nodes = total_leaves = max_depth = 0
        for view in views:
            tree = tree_structure(view)
            rules.update(split_id(tree, n, decimals) for n in range(tree.node_count) if tree.children_left[n] != -1)
            signatures.add(full_signature(view, decimals))
            _, paths = traverse(view, X_ref, decimals)
            lengths.append([len(p) for p in paths])
            for all_rules, path in zip(encountered, paths, strict=True):
                all_rules.update(path)
            total_nodes += tree.node_count
            total_leaves += tree.n_leaves
            max_depth = max(max_depth, tree.max_depth)
        c_path = float(np.mean(lengths))
        c_encoding = float(np.log2(1 + len(rules)))
        c_ensemble = float(np.log2(len(signatures)))
        budgets = np.array([len(s) for s in encountered])
        components = dict(
            csc=c_path + c_encoding + c_ensemble,
            c_path=c_path,
            c_encoding=c_encoding,
            c_ensemble=c_ensemble,
            n_trees=len(views),
            n_unique_trees=len(signatures),
            n_unique_splits=len(rules),
            total_nodes=total_nodes,
            total_leaves=total_leaves,
            max_depth=max_depth,
            tepl=c_path * len(views),
            **{f"budget_r{b}": float(np.mean(budgets > b)) for b in (5, 7, 9)},
        )

        details = dict(B=budgets, L=np.sum(lengths, axis=0), M=np.full(len(X_ref), len(views)))
        return components, details
