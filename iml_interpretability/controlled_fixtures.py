"""Explicit validated classification trees for mechanism tests, never disguised sklearn fits."""

from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class Leaf:
    probabilities: tuple[float, float]


@dataclass(frozen=True)
class Split:
    feature: int
    threshold: float
    left: object
    right: object


class FixtureTree:
    def __init__(self, root, n_features):
        if n_features < 1:
            raise ValueError("Feature count must be positive")
        self.n_features = n_features
        self.classes = np.array([0, 1])
        records = []

        def visit(node, depth):
            index = len(records)
            records.append(None)
            if isinstance(node, Leaf):
                p = np.asarray(node.probabilities, dtype=float)
                if p.shape != (2,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
                    raise ValueError("Leaves require two finite probabilities summing to one")
                records[index] = (-1, -1, -2, -2.0, p, depth)
            elif isinstance(node, Split):
                if not 0 <= node.feature < n_features or not np.isfinite(node.threshold):
                    raise ValueError("Invalid split")
                left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
                records[index] = (left, right, node.feature, node.threshold, np.zeros(2), depth)
            else:
                raise TypeError("Expected explicit Split or Leaf")
            return index

        visit(root, 0)
        left, right, feature, threshold, value, depth = zip(*records, strict=True)
        self.structure = SimpleNamespace(
            children_left=np.array(left),
            children_right=np.array(right),
            feature=np.array(feature),
            threshold=np.array(threshold),
            value=np.array(value)[:, None, :],
            missing_go_to_left=np.zeros(len(records), dtype=np.uint8),
            node_count=len(records),
            n_leaves=int(np.sum(np.array(left) == -1)),
            max_depth=max(depth),
        )
        for item in vars(self.structure).values():
            if isinstance(item, np.ndarray):
                item.flags.writeable = False

    def predict_proba(self, X):
        from .tree_utils import TreeView, traverse

        leaves, _ = traverse(TreeView(self, (), 1.0), X)
        return self.structure.value[leaves, 0, :]

    def predict(self, X):
        return self.classes[self.predict_proba(X).argmax(axis=1)]


@dataclass(frozen=True)
class FixtureEnsemble:
    trees: tuple[FixtureTree, ...]

    def __post_init__(self):
        if not self.trees or len({t.n_features for t in self.trees}) != 1:
            raise ValueError("Averaging ensemble needs compatible trees")

    def predict_proba(self, X):
        return np.mean([tree.predict_proba(X) for tree in self.trees], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def path_trees():
    a, b = Leaf((1.0, 0.0)), Leaf((0.0, 1.0))
    balanced = Split(0, 0.5, Split(0, 0.25, a, b), Split(0, 0.75, a, b))
    chain = Split(0, 0.25, a, Split(0, 0.5, b, Split(0, 0.75, a, b)))
    return {"balanced": FixtureTree(balanced, 1), "chain": FixtureTree(chain, 1)}


def occupancy(counts):
    return np.repeat([0.125, 0.375, 0.625, 0.875], counts).reshape(-1, 1)


def stump_bank():
    return tuple(FixtureTree(Split(i, 0.0, Leaf((1.0, 0.0)), Leaf((0.0, 1.0))), 12) for i in range(12))


def factorial_reference():
    return np.array(list(product((-1.0, 1.0), repeat=12)))
