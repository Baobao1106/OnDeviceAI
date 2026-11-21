"""Synthetic capture for solve tests: true poses = capture plan + hand-shake jitter, matches from random 3D directions."""
import numpy as np

import common

SIZE = (864, 648)


def make_scene(seed=0, hfov=74.0, noise_px=0.3, jitter=5.0, n_dirs=6000, min_pts=12, drop=()):
    """Return (matches, slots, truth) in the formats of docs/INTERFACES.md. `drop`: image indices with no pair at all."""
    rng = np.random.default_rng(seed)
    plan = common.capture_plan()
    names = [f"img{k:02d}.jpg" for k in range(24)]
    rots = [common.pose_to_R(y + rng.uniform(-jitter, jitter),
                             p + (0 if abs(p) == 90 else rng.uniform(-jitter, jitter)),
                             rng.uniform(-jitter, jitter)) for y, p in plan]
    dirs = rng.normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    f, cx, cy = common.intrinsics(*SIZE, hfov)
    proj = [_project(R, dirs, f, cx, cy, SIZE) for R in rots]
    pairs = []
    for a in range(24):
        for b in range(a + 1, 24):
            if a in drop or b in drop:
                continue
            both = np.flatnonzero(~np.isnan(proj[a][:, 0]) & ~np.isnan(proj[b][:, 0]))
            if len(both) < min_pts:
                continue
            pa = proj[a][both] + rng.normal(0, noise_px, (len(both), 2))
            pb = proj[b][both] + rng.normal(0, noise_px, (len(both), 2))
            pairs.append({"i": names[a], "j": names[b], "pts_i": pa.tolist(), "pts_j": pb.tolist(),
                          "n_inliers": len(both), "rotation": (rots[a].T @ rots[b]).tolist(), "rescued": False})
    matches = {"image_size": list(SIZE), "images": names, "pairs": pairs}
    slots = {"images": {n: {"slot": k, "yaw_deg": plan[k][0], "pitch_deg": plan[k][1], "confidence": 0.9, "n_matches": 50}
                        for k, n in enumerate(names)}}
    truth = {"hfov": hfov, "R": dict(zip(names, rots))}
    return matches, slots, truth


def _project(R, dirs, f, cx, cy, size):
    c = dirs @ R                                    # camera coordinates: R.T @ d for each row d
    ok = c[:, 2] > 0.2
    x = f * c[:, 0] / np.where(ok, c[:, 2], 1) + cx
    y = cy - f * c[:, 1] / np.where(ok, c[:, 2], 1)
    ok &= (x >= 0) & (x < size[0]) & (y >= 0) & (y < size[1])
    out = np.stack([x, y], 1)
    out[~ok] = np.nan
    return out


def pose_errors(poses, truth):
    """Per-image angle (deg) between estimated and true rotation after the best global rotation (yaw/tilt free)."""
    names = sorted(truth["R"])
    est = {n: np.asarray(poses[n]["R"], float) for n in names}
    M = sum(est[n] @ truth["R"][n].T for n in names)
    U, _, Vt = np.linalg.svd(M)
    A = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return {n: float(np.degrees(np.arccos(np.clip((np.trace(est[n].T @ A @ truth["R"][n]) - 1) / 2, -1, 1))))
            for n in names}, A


def tilt_deg(A):
    """Angle between the estimated frame up axis (0,1,0) and the true up expressed in that frame (A @ (0,1,0))."""
    up = A @ np.array([0.0, 1.0, 0.0])
    return float(np.degrees(np.arccos(np.clip(up[1], -1, 1))))
