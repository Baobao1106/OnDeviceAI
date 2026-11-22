import itertools

import numpy as np


def priority_order(n, score, pairs=None, seed=0):
    keys = sorted(pairs) if pairs is not None else list(itertools.combinations(range(n), 2))
    rng = np.random.default_rng(seed)
    partners = {i: [] for i in range(n)}
    for i, j in keys:
        partners[i].append(j)
        partners[j].append(i)
    ranked = {}
    for i in range(n):
        js = [partners[i][k] for k in rng.permutation(len(partners[i]))]
        js.sort(key=lambda j: -score.get((min(i, j), max(i, j)), 0))
        ranked[i] = [(min(i, j), max(i, j)) for j in js]
    out, seen = [], set()
    for r in range(max((len(v) for v in ranked.values()), default=0)):
        batch = [ranked[i][r] for i in range(n) if r < len(ranked[i])]
        batch.sort(key=lambda p: -score.get(p, 0))
        for p in batch:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def split_deadline(now, deadline, reserve):
    if deadline is None or reserve <= 0:
        return deadline
    return now + (1.0 - min(reserve, 0.9)) * max(deadline - now, 0.0)
