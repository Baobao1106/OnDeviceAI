"""Assign every photo to one of the five capture layers (bottom / down / horizon / up / top).

The 24 photos are taken on 5 pitch layers, 1 / 7 / 8 / 7 / 1 photos at about -90 / -45 / 0 / +45 / +90 deg
(roll within +-5 deg). Input: the accepted pairs of a PipelineResult (relative rotations, ray_i ~ R @ ray_j).

Per connected component of the pair graph:
  1. absolute rotations from a maximum-inlier spanning tree (root = identity);
  2. the world "up" axis u: with ~0 roll every camera's x axis is horizontal, so u is orthogonal to all x axes;
     the pitch layers (prior) pick u when the photos of a component do not fix it (e.g. a pair that differs only in
     pitch). The sign of u is the one that points along the cameras' up (y) axes;
  3. pitch_i = asin(d_i . u) with d_i the optical axis, layer = nearest of -90/-45/0/45/90.
Photos with no accepted pair cannot be placed: they get the layers whose 1/7/8/7/1 quota is not yet used up.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree

LAYER_NAMES = ("bottom", "down", "horizon", "up", "top")
LAYER_PITCH = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
LAYER_COUNTS = np.array([1, 7, 8, 7, 1])
ROLL_SIGMA_DEG = 5.0        # camera x axis is horizontal within the roll jitter
PITCH_SIGMA_DEG = 7.0       # photos sit within a few degrees of their layer pitch
TRUNC = 3.0                 # truncated quadratic: photos that fit neither term cost TRUNC**2
GRID_POINTS = 20000


@dataclass(frozen=True)
class LayerAssignment:
    index: int
    layer: int | None           # 0..4 (see LAYER_NAMES) or None when it could not be placed
    pitch_deg: float | None
    component: int
    candidates: tuple           # possible layers: (layer,) when placed, else the layers with quota left
    inferred: bool              # True when placed only by the quota argument (no matched pair)

    def to_dict(self, names):
        return {"image": names[self.index], "index": self.index,
                "layer": None if self.layer is None else LAYER_NAMES[self.layer],
                "pitch_deg": None if self.pitch_deg is None else round(self.pitch_deg, 2),
                "component": self.component,
                "candidates": [LAYER_NAMES[k] for k in self.candidates], "inferred": self.inferred}


def absolute_rotations(n, pairs):
    """Rotation image k -> frame of its component root (ray_root = A_k @ ray_k), over accepted pairs.
    Returns (A: dict k -> 3x3, component label per image). Isolated images get no entry."""
    acc = [p for p in pairs if p.accepted and p.rotation is not None]
    if not acc:
        return {}, np.arange(n)
    e = np.array([(p.i, p.j) for p in acc])
    w = np.array([1.0 / p.n_inliers for p in acc])
    g = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    tree = minimum_spanning_tree(g)                       # smallest 1/inliers = strongest pairs
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
            # ray_par = R[par, c] @ ray_c, so A_c = A_par @ R[par, c]
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
    """Cost of candidate up axes U (K x 3) given camera x axes X and optical axes D (m x 3)."""
    roll = np.degrees(np.arcsin(np.clip(U @ X.T, -1, 1))) / ROLL_SIGMA_DEG                  # K x m
    pitch = np.degrees(np.arcsin(np.clip(U @ D.T, -1, 1)))
    layer = np.abs(pitch[..., None] - LAYER_PITCH).min(-1) / PITCH_SIGMA_DEG
    return (np.minimum(roll ** 2, TRUNC ** 2) + np.minimum(layer ** 2, TRUNC ** 2)).sum(1)


def fit_up_axis(X, Y, D):
    """Up axis (unit 3-vector) of a set of cameras given their x / y / optical axes in a common frame."""
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
    """(A, label, U): absolute rotations per image, component label per image, up axis per component label."""
    A, label = absolute_rotations(n, pairs)
    U = {}
    for comp in sorted({int(label[k]) for k in A}):
        idx = [k for k in A if label[k] == comp]
        U[comp] = fit_up_axis(np.array([A[k][:, 0] for k in idx]), np.array([A[k][:, 1] for k in idx]),
                              np.array([A[k][:, 2] for k in idx]))
    return A, label, U


def assign_layers(names, pairs):
    """Layer assignment for every image. `pairs`: PairResult list (accepted ones are used)."""
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
