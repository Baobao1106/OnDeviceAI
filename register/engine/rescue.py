"""Rescue isolated photos with the pitch-layer prior.

A photo with no accepted pair usually has a few (4-9) RANSAC inliers with its true neighbours - too few to accept
on their own, as wrong pairs reach that level too. The capture structure (5 pitch layers, roll within +-5 deg)
filters them: a candidate pair places the photo at an absolute rotation in the frame of its neighbour's component,
and that rotation must (a) keep the camera x axis horizontal, (b) give a pitch near a layer whose quota is free.
A photo is rescued when the surviving candidates are strong (>= n_sure inliers) or two candidates of the same
component agree on its rotation.
"""
from dataclasses import dataclass

import numpy as np

from .layers import LAYER_PITCH, assign_layers, component_frames


@dataclass(frozen=True)
class RescueConfig:
    n_min: int = 6              # weakest candidate considered
    n_sure: int = 8             # a single candidate at least this strong is enough
    tilt_deg: float = 8.0       # camera x axis must be this close to horizontal
    pitch_tol_deg: float = 10.0
    agree_deg: float = 3.0      # two candidates agree when the photo's rotation differs by less than this


@dataclass(frozen=True)
class RescuedPair:
    i: int
    j: int
    n_inliers: int
    rotation: np.ndarray       # ray_i ~ R @ ray_j


def _angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rescue_isolated(names, pairs, cfg=None):
    cfg = cfg or RescueConfig()
    n = len(names)
    A, label, U = component_frames(n, pairs)
    free = {a.index: a.candidates for a in assign_layers(names, pairs)}
    out = []
    for k in range(n):
        if k in A:
            continue
        cands = []                                   # (comp, n, m, A_k, pair)
        for p in pairs:
            if p.cand_rotation is None or p.n_inliers < cfg.n_min or k not in (p.i, p.j):
                continue
            m = p.j if p.i == k else p.i
            if m not in A:
                continue
            r_mk = p.cand_rotation if p.i == m else p.cand_rotation.T       # image k -> image m
            a_k = A[m] @ r_mk
            u = U[int(label[m])]
            tilt = abs(np.degrees(np.arcsin(np.clip(a_k[:, 0] @ u, -1, 1))))
            pitch = np.degrees(np.arcsin(np.clip(a_k[:, 2] @ u, -1, 1)))
            near = min(abs(pitch - LAYER_PITCH[c]) for c in free[k])
            if tilt <= cfg.tilt_deg and near <= cfg.pitch_tol_deg:
                cands.append((int(label[m]), p.n_inliers, m, a_k, p))
        best, best_score = None, 0
        for c in cands:
            group = [d for d in cands if d[0] == c[0] and _angle_deg(d[3].T @ c[3]) <= cfg.agree_deg]
            score = sum(d[1] for d in group)
            if (len(group) >= 2 or c[1] >= cfg.n_sure) and score > best_score:
                best, best_score = group, score
        for _, _, _, _, p in best or []:
            out.append(RescuedPair(p.i, p.j, p.n_inliers, p.cand_rotation))
    return out
