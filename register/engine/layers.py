from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree

LAYER_NAMES = ("bottom", "down", "horizon", "up", "top")
LAYER_PITCH = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
LAYER_COUNTS = np.array([1, 7, 8, 7, 1])
ROLL_SIGMA_DEG = 5.0
PITCH_SIGMA_DEG = 7.0
TRUNC = 3.0
GRID_POINTS = 20000


@dataclass(frozen=True)
class LayerAssignment:
    index: int
    layer: int | None
    pitch_deg: float | None
    component: int
    candidates: tuple
    inferred: bool

    def to_dict(self, names):
        return {"image": names[self.index], "index": self.index,
                "layer": None if self.layer is None else LAYER_NAMES[self.layer],
                "pitch_deg": None if self.pitch_deg is None else round(self.pitch_deg, 2),
                "component": self.component,
                "candidates": [LAYER_NAMES[k] for k in self.candidates], "inferred": self.inferred}


def absolute_rotations(n, pairs):
    acc = [p for p in pairs if p.accepted and p.rotation is not None]
    if not acc:
        return {}, np.arange(n)
    e = np.array([(p.i, p.j) for p in acc])
    w = np.array([1.0 / p.n_inliers for p in acc])
    g = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    tree = minimum_spanning_tree(g)
    n_comp, label = connected_components(g, directed=False)
    rot = {}
    for p in acc:
        rot[(p.i, p.j)] = p.rotation
        rot[(p.j, p.i)] = p.rotation.T
    sym = (tree + tree.T).tocsr()
    A = {}
    seen = set()
    for k in range(n):
        if k in seen or sym[k].nnz == 0:
            continue
        order, pred = breadth_first_order(sym, k, directed=False)
        A[k] = np.eye(3)
        for c in order[1:]:
            par = int(pred[c])
            A[int(c)] = A[par] @ rot[(par, int(c))]
        seen.update(int(c) for c in order)
    return A, label


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_GRID = _fibonacci_sphere(GRID_POINTS)


def _cost(U, X, D):
    roll = np.degrees(np.arcsin(np.clip(U @ X.T, -1, 1))) / ROLL_SIGMA_DEG
    pitch = np.degrees(np.arcsin(np.clip(U @ D.T, -1, 1)))
    layer = np.abs(pitch[..., None] - LAYER_PITCH).min(-1) / PITCH_SIGMA_DEG
    return (np.minimum(roll ** 2, TRUNC ** 2) + np.minimum(layer ** 2, TRUNC ** 2)).sum(1)


def fit_up_axis(X, Y, D):
    best = _GRID[np.argmin(_cost(_GRID, X, D))]
    a = np.cross(best, [1.0, 0.0, 0.0] if abs(best[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(best, a)
    off = np.radians(np.linspace(-2, 2, 41))
    gu, gv = np.meshgrid(off, off)
    cand = best + gu.ravel()[:, None] * a + gv.ravel()[:, None] * b
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    u = cand[np.argmin(_cost(cand, X, D))]
    return u if (Y @ u).sum() >= 0 else -u


def component_frames(n, pairs):
    A, label = absolute_rotations(n, pairs)
    U = {}
    for comp in sorted({int(label[k]) for k in A}):
        idx = [k for k in A if label[k] == comp]
        U[comp] = fit_up_axis(np.array([A[k][:, 0] for k in idx]), np.array([A[k][:, 1] for k in idx]),
                              np.array([A[k][:, 2] for k in idx]))
    return A, label, U


def assign_layers(names, pairs):
    n = len(names)
    A, label, U = component_frames(n, pairs)
    layer = [None] * n
    pitch = [None] * n
    for k, rot in A.items():
        p = float(np.degrees(np.arcsin(np.clip(rot[:, 2] @ U[int(label[k])], -1, 1))))
        pitch[k], layer[k] = p, int(np.abs(p - LAYER_PITCH).argmin())
    used = np.bincount([l for l in layer if l is not None], minlength=len(LAYER_NAMES))
    free = tuple(int(k) for k in np.flatnonzero(LAYER_COUNTS - used > 0)) or tuple(range(len(LAYER_NAMES)))
    out = []
    for k in range(n):
        if layer[k] is not None:
            out.append(LayerAssignment(k, layer[k], pitch[k], int(label[k]), (layer[k],), False))
        elif len(free) == 1:
            out.append(LayerAssignment(k, free[0], None, int(label[k]), free, True))
        else:
            out.append(LayerAssignment(k, None, None, int(label[k]), free, False))
    return out
