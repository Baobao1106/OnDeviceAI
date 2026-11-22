

import numpy as np

MIN_REGISTERED = 6
HUBER_RAD = 0.06
IRLS_ROUNDS = 8
Y = np.array([0.0, 1.0, 0.0])


def _constraints(poses, slots):
    rows, targets = [], []
    for name, p in poses.items():
        if not p.get("registered", True):
            continue
        R = np.asarray(p["R"], float)
        pitch = np.radians(slots["images"][name]["pitch_deg"])
        rows += [R[:, 2], R[:, 1], R[:, 0]]
        targets += [np.sin(pitch), np.cos(pitch), 0.0]
    return np.array(rows), np.array(targets)


def estimate_up(poses, slots):
    A, t = _constraints(poses, slots)
    if len(t) < 3 * MIN_REGISTERED:
        return Y.copy()
    w = np.ones(len(t))
    u = Y.copy()
    for _ in range(IRLS_ROUNDS):
        sol, *_ = np.linalg.lstsq(A * w[:, None], t * w, rcond=None)
        u = sol / np.linalg.norm(sol)
        err = np.abs(A @ u - t)
        w = np.sqrt(np.where(err <= HUBER_RAD, 1.0, HUBER_RAD / np.maximum(err, 1e-12)))
    return u


def _rotation_taking(u, v):
    c = float(np.clip(u @ v, -1.0, 1.0))
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def level_poses(poses, slots):
    up = estimate_up(poses, slots)
    Rl = _rotation_taking(up, Y)
    new = {name: {**p, "R": (Rl @ np.asarray(p["R"], float)).tolist()} for name, p in poses.items()}
    tilt = float(np.degrees(np.arccos(np.clip(up @ Y, -1.0, 1.0))))
    return new, up, tilt
