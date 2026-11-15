"""Deterministic, nested training-only interventions."""

import numpy as np


def allocation(counts, total):
    counts = np.asarray(counts)
    raw = counts * total / counts.sum()
    out = np.minimum(counts, np.maximum(2, np.floor(raw).astype(int)))
    while out.sum() < total:
        i = max((i for i in range(len(out)) if out[i] < counts[i]), key=lambda i: raw[i] - out[i])
        out[i] += 1
    while out.sum() > total:
        i = max((i for i in range(len(out)) if out[i] > 2), key=lambda i: out[i] - raw[i])
        out[i] -= 1
    return out


def variants(y, train, n_features, seed):
    train = np.asarray(train)
    labels, counts = np.unique(np.asarray(y)[train], return_counts=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 811]))
    pools = [rng.permutation(train[np.asarray(y)[train] == c]) for c in labels]
    features = np.random.default_rng(np.random.SeedSequence([seed, 812])).permutation(n_features)
    target = int(np.argmin(counts))
    result = []

    def add(experiment, level, sizes, cols, control=False):
        chosen = set(np.concatenate([pool[: int(n)] for pool, n in zip(pools, sizes, strict=True)]).tolist())
        rows = [int(i) for i in train if i in chosen]
        result.append(
            dict(
                experiment=experiment,
                level=level,
                control=control,
                train=rows,
                columns=sorted(int(i) for i in cols),
                class_counts={str(c): int(n) for c, n in zip(labels, sizes, strict=True)},
                target_class=str(labels[target]),
            )
        )

    for fraction in (0.25, 0.5, 0.75, 1.0):
        add("samples", fraction, np.maximum(2, np.rint(counts * fraction).astype(int)), range(n_features))
        add("features", fraction, counts, features[: max(1, int(np.ceil(n_features * fraction)))])
    add("balance", 1.0, counts, range(n_features))
    for fraction in (0.5, 0.25):
        sizes = counts.copy()
        sizes[target] = max(2, int(round(counts[target] * fraction)))
        add("balance", fraction, sizes, range(n_features))
        add("balance", fraction, allocation(counts, int(sizes.sum())), range(n_features), True)
    return result
