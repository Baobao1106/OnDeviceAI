import numpy as np

from .layers import LAYER_NAMES, assign_layers, component_frames

RING_LAYERS = (1, 2, 3)
N_CROSS = 2


def yaw_deg(d, u):
    e1 = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    d = np.atleast_2d(d)
    return np.degrees(np.arctan2(d @ e2, d @ e1))


def ring_order_neighbors(members, yaws):
    order = [m for _, m in sorted(zip(yaws, members))]
    n = len(order)
    if n == 1:
        return {order[0]: ()}
    if n == 2:
        return {order[0]: (order[1],), order[1]: (order[0],)}
    return {m: (order[k - 1], order[(k + 1) % n]) for k, m in enumerate(order)}


def build_neighbors(names, pairs):
    n = len(names)
    A, label, U = component_frames(n, pairs)
    lay = assign_layers(names, pairs)
    placed = [k for k in A if lay[k].layer is not None]
    comp = {k: int(label[k]) for k in placed}
    axis = {k: A[k][:, 2] for k in placed}
    out = {k: {"image": names[k], "index": k, "layer": None if lay[k].layer is None else LAYER_NAMES[lay[k].layer],
               "component": int(label[k]), "yaw_deg": None, "ring_neighbors": [], "layer_neighbors": {}}
           for k in range(n)}
    for c in sorted(set(comp.values())):
        for layer in range(5):
            mem = [k for k in placed if comp[k] == c and lay[k].layer == layer]
            if layer in RING_LAYERS and mem:
                yaws = yaw_deg(np.array([axis[k] for k in mem]), U[c])
                for k, y in zip(mem, yaws):
                    out[k]["yaw_deg"] = round(float(y), 2)
                for k, nb in ring_order_neighbors(mem, yaws).items():
                    out[k]["ring_neighbors"] = list(nb)
            for k in mem:
                for adj in (layer - 1, layer + 1):
                    if not 0 <= adj < 5:
                        continue
                    cand = [m for m in placed if comp[m] == c and lay[m].layer == adj]
                    if not cand:
                        continue
                    ang = [float(np.arccos(np.clip(axis[k] @ axis[m], -1, 1))) for m in cand]
                    keep = cand if layer in (0, 4) else [m for _, m in sorted(zip(ang, cand))][:N_CROSS]
                    out[k]["layer_neighbors"][LAYER_NAMES[adj]] = keep
    return [out[k] for k in range(n)]
