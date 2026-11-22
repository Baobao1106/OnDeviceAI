import numpy as np

import common


def pixels_to_rays(pts, hfov_deg, size):
    w, h = size
    f, cx, cy = common.intrinsics(w, h, hfov_deg)
    pts = np.asarray(pts, float)
    d = np.stack([(pts[:, 0] - cx) / f, -(pts[:, 1] - cy) / f, np.ones(len(pts))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def rotation_from_rays(a, b):
    H = b.T @ a
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def angular_errors(a, b, R):
    return np.degrees(np.arccos(np.clip(np.sum(a * (b @ R.T), axis=1), -1.0, 1.0)))


def fit_rotation(a, b, keep=0.8, rounds=3):
    R = rotation_from_rays(a, b)
    for _ in range(rounds):
        err = angular_errors(a, b, R)
        idx = np.argsort(err)[:max(3, int(len(err) * keep))]
        R = rotation_from_rays(a[idx], b[idx])
    err = angular_errors(a, b, R)
    return R, float(np.sort(err)[:max(3, int(len(err) * keep))].mean())


def relative_rotation(pair, hfov_deg, size, thresh_deg=0.4, iters=150, seed=0):
    if len(pair["pts_i"]) < 3:
        return np.eye(3), 0, 180.0
    a = pixels_to_rays(pair["pts_i"], hfov_deg, size)
    b = pixels_to_rays(pair["pts_j"], hfov_deg, size)
    rng = np.random.default_rng(seed)
    best_R, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(len(a), 3, replace=False)
        R = rotation_from_rays(a[idx], b[idx])
        n = int((angular_errors(a, b, R) < thresh_deg).sum())
        if n > best_n:
            best_R, best_n = R, n
        if best_n == len(a):
            break
    R = best_R
    for _ in range(3):
        inl = angular_errors(a, b, R) < thresh_deg
        if inl.sum() < 3:
            break
        R = rotation_from_rays(a[inl], b[inl])
    err = angular_errors(a, b, R)
    inl = err < thresh_deg
    return R, int(inl.sum()), float(err[inl].mean()) if inl.any() else 180.0
