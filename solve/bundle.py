import heapq
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

import common

from .relative_rot import pixels_to_rays, relative_rotation

FOV_LO, FOV_HI = 70.0, 79.0
SIGMA_MATCH = 0.003
SIGMA_PRIOR = 0.15
POINTS_PER_PAIR = 60
FULL_WEIGHT_INLIERS = 40
PRUNE_MEDIAN_RAD = 0.02
ALIGN_REJECT_DEG = 25.0
MAX_NFEV = 300
NOMINAL_GATE_DEG = 35.0
POLE_PITCH_DEG = 80.0
POLE_VOTE_DEG = 8.0


class _Expired(Exception):
    pass


def _rot(v):
    return Rotation.from_rotvec(np.asarray(v, float).reshape(-1, 3)).as_matrix()


def _rotvec(R):
    return Rotation.from_matrix(np.asarray(R, float).reshape(-1, 3, 3)).as_rotvec()


def _project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def procrustes(est, nom, reject_deg=ALIGN_REJECT_DEG):
    keep = list(range(len(est)))
    A = np.eye(3)
    for _ in range(3):
        if not keep:
            break
        A = _project_so3(sum(nom[k] @ est[k].T for k in keep))
        far = [k for k in range(len(est)) if _angle_deg(A @ est[k] @ nom[k].T) > reject_deg]
        keep = [k for k in range(len(est)) if k not in far] or keep
    return A


def _nominal(names, slots):
    info = slots["images"]
    return [common.pose_to_R(info[n]["yaw_deg"], info[n]["pitch_deg"]) for n in names]


def _pair_rotation(pair, hfov, size):
    if pair.get("rotation") is not None:
        return np.asarray(pair["rotation"], float)
    return relative_rotation(pair, hfov, size)[0]


def _consistent_with_slots(pair, nom_i, nom_j, pole_i, pole_j, hfov, size):
    Rij = _pair_rotation(pair, hfov, size)
    N = nom_i.T @ nom_j
    if pole_i or pole_j:
        got = np.degrees(np.arccos(np.clip(Rij[2, 2], -1, 1)))
        want = np.degrees(np.arccos(np.clip(N[2, 2], -1, 1)))
        return abs(got - want) <= NOMINAL_GATE_DEG
    return _angle_deg(Rij.T @ N) <= NOMINAL_GATE_DEG


def _tree_rotations(n, edges):
    adj = {k: [] for k in range(n)}
    for i, j, w, Rij in edges:
        adj[i].append((w, i, j, Rij))
        adj[j].append((w, j, i, Rij.T))
    R, comps = {}, []
    for root in sorted(range(n), key=lambda k: -len(adj[k])):
        if root in R or not adj[root]:
            continue
        R[root] = np.eye(3)
        comp, heap = [root], [(-w, a, b, r) for w, a, b, r in adj[root]]
        heapq.heapify(heap)
        while heap:
            _, a, b, rab = heapq.heappop(heap)
            if b in R:
                continue
            R[b] = R[a] @ rab
            comp.append(b)
            for w, c, d, rcd in adj[b]:
                if d not in R:
                    heapq.heappush(heap, (-w, c, d, rcd))
        comps.append(comp)
    return R, comps


def _initial_poses(names, pairs, nom, hfov, size):
    index = {nm: k for k, nm in enumerate(names)}
    edges = [(index[p["i"]], index[p["j"]], p["n_inliers"], _pair_rotation(p, hfov, size)) for p in pairs]
    R, comps = _tree_rotations(len(names), edges)
    for comp in comps:
        A = procrustes([R[k] for k in comp], [nom[k] for k in comp])
        for k in comp:
            R[k] = A @ R[k]
    return R


def _collect(names, pairs, hfov, size, seed=0):
    index = {nm: k for k, nm in enumerate(names)}
    rng = np.random.default_rng(seed)
    ii, jj, pi, pj, w, pid = [], [], [], [], [], []
    for q, p in enumerate(pairs):
        n = len(p["pts_i"])
        take = np.sort(rng.choice(n, min(n, POINTS_PER_PAIR), replace=False))
        wt = min(1.0, p["n_inliers"] / FULL_WEIGHT_INLIERS) * (0.5 if p.get("rescued") else 1.0)
        ii += [index[p["i"]]] * len(take)
        jj += [index[p["j"]]] * len(take)
        pi.append(np.asarray(p["pts_i"], float)[take])
        pj.append(np.asarray(p["pts_j"], float)[take])
        w += [wt] * len(take)
        pid += [q] * len(take)
    return {"i": np.array(ii), "j": np.array(jj), "pi": np.vstack(pi), "pj": np.vstack(pj),
            "w": np.array(w), "pid": np.array(pid)}


def _match_residuals(R, fov, d, size):
    ri = np.einsum("nab,nb->na", R[d["i"]], pixels_to_rays(d["pi"], fov, size))
    rj = np.einsum("nab,nb->na", R[d["j"]], pixels_to_rays(d["pj"], fov, size))
    return ri - rj


def _sparsity(m, d, n_res):
    n_par = 3 * m + 1
    n = len(d["i"])
    rows = np.arange(n)
    r, c = [], []
    for img in (d["i"], d["j"]):
        for dr in range(3):
            for dc in range(3):
                r.append(3 * rows + dr)
                c.append(3 * img + dc)
    for dr in range(3):
        r.append(3 * rows + dr)
        c.append(np.full(n, n_par - 1))
    k = np.arange(3 * m)
    r.append(3 * n + k)
    c.append(k)
    r, c = np.concatenate(r), np.concatenate(c)
    return coo_matrix((np.ones(len(r), int), (r, c)), shape=(n_res, n_par)).tocsr()


def _optimise(R0, p0, fov0, d, size, deadline, max_nfev=MAX_NFEV):
    m = len(R0)
    state = {"x": np.concatenate([np.zeros(3 * m), [fov0]])}

    def fun(x):
        if time.time() > deadline:
            raise _Expired()
        state["x"] = x
        R = _rot(x[:-1]) @ R0
        res = (_match_residuals(R, x[-1], d, size) * d["w"][:, None] / SIGMA_MATCH).ravel()
        prior = ((p0 + x[:-1].reshape(m, 3)) / SIGMA_PRIOR).ravel()
        return np.concatenate([res, prior])

    n_res = 3 * len(d["i"]) + 3 * m
    lo = np.concatenate([np.full(3 * m, -np.inf), [FOV_LO]])
    hi = np.concatenate([np.full(3 * m, np.inf), [FOV_HI]])
    x0 = np.clip(state["x"], lo + 1e-6, hi - 1e-6)
    try:
        least_squares(fun, x0, bounds=(lo, hi), loss="huber", f_scale=2.5, jac_sparsity=_sparsity(m, d, n_res),
                      x_scale=np.concatenate([np.full(3 * m, 0.01), [0.5]]), max_nfev=max_nfev)
    except _Expired:
        pass
    x = state["x"]
    return _rot(x[:-1]) @ R0, float(np.clip(x[-1], FOV_LO, FOV_HI))


def _pair_medians(R, fov, d, size, n_pairs):
    err = np.linalg.norm(_match_residuals(R, fov, d, size), axis=1)
    return np.array([np.median(err[d["pid"] == q]) if (d["pid"] == q).any() else 0.0 for q in range(n_pairs)])


def _solve_once(names, pairs, nom, hfov, size, deadline):
    R0 = _initial_poses(names, pairs, nom, hfov, size)
    used = sorted(R0)
    pos = {k: a for a, k in enumerate(used)}
    d = _collect([names[k] for k in used], pairs, hfov, size)
    Rn = np.stack([nom[k] for k in used])
    Rs = np.stack([R0[k] for k in used])
    p0 = _rotvec(np.einsum("nab,ncb->nac", Rs, Rn))
    R, fov = _optimise(Rs, p0, hfov, d, size, deadline)
    meds = _pair_medians(R, fov, d, size, len(pairs))
    return {k: R[pos[k]] for k in used}, fov, meds


def _vote_pole_pairs(pairs, names, nom, pole, hfov, size):
    index = {nm: k for k, nm in enumerate(names)}
    ring = [p for p in pairs if not (pole[index[p["i"]]] or pole[index[p["j"]]])]
    R0 = _initial_poses(names, ring, nom, hfov, size) if ring else {}
    cand = {}
    for p in pairs:
        i, j = index[p["i"]], index[p["j"]]
        if pole[i] == pole[j]:
            continue
        Rij = _pair_rotation(p, hfov, size)
        k, q = (i, j) if pole[i] else (j, i)
        if q not in R0:
            continue
        cand.setdefault(k, []).append((id(p), R0[q] @ (Rij.T if k == i else Rij), p["n_inliers"]))
    drop = set()
    for k, items in cand.items():
        support = [sum(w2 for _, R2, w2 in items if _angle_deg(R1.T @ R2) <= POLE_VOTE_DEG) for _, R1, _ in items]
        best = items[int(np.argmax(support))][1]
        drop |= {pid for pid, R1, _ in items if _angle_deg(best.T @ R1) > POLE_VOTE_DEG}
    return [p for p in pairs if id(p) not in drop]


def _gate_pairs(pairs, names, nom, slots, hfov, size):
    index = {nm: k for k, nm in enumerate(names)}
    pole = [abs(slots["images"][n]["pitch_deg"]) >= POLE_PITCH_DEG for n in names]
    keep = []
    for p in pairs:
        i, j = index[p["i"]], index[p["j"]]
        if _consistent_with_slots(p, nom[i], nom[j], pole[i], pole[j], hfov, size):
            keep.append(p)
    return _vote_pole_pairs(keep, names, nom, pole, hfov, size)


def bundle_adjust(matches, slots, hfov_deg, deadline):
    names = sorted(matches["images"])
    size = matches["image_size"]
    nom = _nominal(names, slots)
    hfov = float(np.clip(hfov_deg, FOV_LO, FOV_HI))
    pairs = _gate_pairs([p for p in matches["pairs"] if len(p["pts_i"]) >= 3], names, nom, slots, hfov, size)
    Rs = {}
    if pairs:
        Rs, hfov, meds = _solve_once(names, pairs, nom, hfov, size, deadline)
        good = [p for p, m in zip(pairs, meds) if m <= PRUNE_MEDIAN_RAD]
        if len(good) < len(pairs) and good:
            Rs, hfov, _ = _solve_once(names, good, nom, hfov, size, deadline)
    return _finish(names, Rs, nom), hfov


def _finish(names, Rs, nom):
    reg = sorted(Rs)
    A = procrustes([Rs[k] for k in reg], [nom[k] for k in reg]) if reg else np.eye(3)
    out = {}
    for k, name in enumerate(names):
        if k in Rs:
            out[name] = {"R": Rs[k].tolist(), "registered": True, "source": "bundle"}
        else:
            out[name] = {"R": (A.T @ nom[k]).tolist(), "registered": False, "source": "slot_prior"}
    return out
