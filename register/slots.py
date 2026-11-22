from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

import common

from .engine.layers import assign_layers, component_frames
from .engine.neighbors import yaw_deg

SIGMA_DEG = 7.0
CAP = 3.0
YAW_SIGN = 1.0
POLE_PITCH = 80.0
ALIGN_MARGIN = 3.0
ALIGN_FLOOR = 0.3


def nominal(slot):
    return common.capture_plan()[slot]


def _pair_objects(names, pairs):
    index = {n: k for k, n in enumerate(names)}
    out = []
    for p in pairs:
        if p.get("rotation") is None:
            continue
        out.append(SimpleNamespace(i=index[p["i"]], j=index[p["j"]], accepted=True, n_inliers=p["n_inliers"],
                                   rotation=np.asarray(p["rotation"], float), cand_rotation=None))
    return out


def _circ(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _costs(idx, slots, pitch, yaw, psi, Y, P):
    pi = np.array([pitch[k] for k in idx])[:, None]
    cost = np.minimum(((pi - P[None, slots]) / SIGMA_DEG) ** 2, CAP ** 2)
    ring_img = (np.abs(pi) < POLE_PITCH)
    ring_slot = (np.abs(P[slots]) < POLE_PITCH)[None, :]
    yi = np.array([yaw[k] for k in idx])[:, None]
    cost = cost + np.where(ring_img & ring_slot,
                           np.minimum((_circ(yi, Y[None, slots] + psi) / SIGMA_DEG) ** 2, CAP ** 2), 0.0)
    return cost


def _fit_component(idx, free, pitch, yaw, Y, P):
    slots = np.array(sorted(free))
    ring = sum(abs(pitch[k]) < POLE_PITCH for k in idx)

    def solve(psi):
        c = _costs(idx, slots, pitch, yaw, psi, Y, P)
        r, cidx = linear_sum_assignment(c)
        return c[r, cidx].sum(), dict(zip((idx[a] for a in r), (int(slots[b]) for b in cidx)))

    if ring < 2:
        total, assign = solve(0.0)
        return assign, 0.0, total, float("inf")
    grid = np.arange(0.0, 360.0, 1.0)
    vals = np.array([solve(g)[0] for g in grid])
    best = grid[int(np.argmin(vals))]
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, 0.1)
    psi = min(fine, key=lambda g: solve(g)[0])
    total, assign = solve(psi)
    far = _circ(grid, best) >= 10.0
    margin = float(vals[far].min() - total) if far.any() else float("inf")
    return assign, float(psi), total, max(margin, 0.0)


def assign_slots(images, pairs, deadline):
    names = sorted(images)
    objs = _pair_objects(names, pairs)
    A, label, U = component_frames(len(names), objs)
    lay = assign_layers(names, objs)
    pitch = {k: lay[k].pitch_deg for k in A}
    yaw = {k: YAW_SIGN * float(yaw_deg(A[k][:, 2], U[int(label[k])])[0]) for k in A}
    plan = common.capture_plan()
    Y = np.array([p[0] for p in plan])
    P = np.array([p[1] for p in plan])

    n_match = dict.fromkeys(range(len(names)), 0)
    for o in objs:
        n_match[o.i] += o.n_inliers
        n_match[o.j] += o.n_inliers

    comps = {}
    for k in A:
        comps.setdefault(int(label[k]), []).append(k)
    free = set(range(len(plan)))
    slot_of, conf = {}, {}
    for members in sorted(comps.values(), key=len, reverse=True):
        assign, psi, _, margin = _fit_component(members, free, pitch, yaw, Y, P)
        align = 1.0 if not np.isfinite(margin) else ALIGN_FLOOR + (1.0 - ALIGN_FLOOR) * min(1.0, margin / ALIGN_MARGIN)
        scale = 1.0 if len(members) >= 8 else min(1.0, (len(members) - 1) / 6.0) * 0.6 + 0.05
        for k, s in assign.items():
            dp = abs(pitch[k] - P[s])
            dy = 0.0 if (abs(pitch[k]) >= POLE_PITCH or abs(P[s]) >= POLE_PITCH) else float(_circ(yaw[k], Y[s] + psi))
            quality = max(0.0, 1.0 - 0.5 * max(dp, dy) / 10.0)
            slot_of[k], conf[k] = s, round(scale * quality * align * 0.95, 3)
            free.discard(s)
    left_imgs = [k for k in range(len(names)) if k not in slot_of]
    for k, s in zip(left_imgs, sorted(free)):
        slot_of[k], conf[k] = s, 0.05
    return {names[k]: {"slot": int(slot_of[k]), "yaw_deg": float(Y[slot_of[k]]), "pitch_deg": float(P[slot_of[k]]),
                       "confidence": float(conf[k]), "n_matches": int(n_match[k])} for k in range(len(names))}
