"""RANSAC for a pure camera rotation with a shared focal length (3 DoF + f) instead of a free homography (8 DoF).

Match (a in image A, b in image B) satisfies  ray_a ~ R * ray_b  with ray = K^-1 [x, y, 1].
Minimal sample = 2 matches (Kabsch on 2 bearing pairs). f is scanned over the allowed hfov range.
Returns the inlier mask, R (B -> A), and the focal length chosen. `hfovs` overrides the scanned range
(pass a single value to use one shared focal length for every pair).
"""
import numpy as np

IMG_W, IMG_H = 864, 648
CX, CY = (IMG_W - 1) / 2, (IMG_H - 1) / 2
HFOVS = np.arange(71.0, 78.01, 0.5)   # slightly wider than the 72-77 deg of the task


def _bearings(pts, f):
    v = np.stack([(pts[:, 0] - CX) / f, -(pts[:, 1] - CY) / f, np.ones(len(pts))], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _kabsch_batch(A, B):
    """Rotations R_k with R_k @ B_k ~ A_k for stacks of correspondences A, B of shape (K, m, 3)."""
    M = np.einsum("kmi,kmj->kij", A, B)
    U, _, Vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros_like(U[..., :1, :].repeat(3, axis=-2))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    return U @ D @ Vt


def _errors(R, a_pts, B, f):
    """Pixel error in image A of matches rotated by R (inf where behind the camera). R: (K,3,3), B: (n,3)."""
    rb = np.einsum("kij,nj->kni", R, B)
    z = rb[..., 2]
    ok = z > 1e-6
    zs = np.where(ok, z, 1.0)
    px = rb[..., 0] / zs * f + CX
    py = -rb[..., 1] / zs * f + CY
    err = np.hypot(px - a_pts[None, :, 0], py - a_pts[None, :, 1])
    return np.where(ok, err, np.inf)


def rotation_ransac(pa, pb, px_thr=3.0, iters=200, seed=0, hfovs=None):
    n = len(pa)
    empty = (np.zeros(n, bool), None, None)
    if n < 6:
        return empty
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(iters, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    best_cnt, best = 0, empty
    for hfov in (HFOVS if hfovs is None else np.atleast_1d(hfovs)):
        f = (IMG_W / 2) / np.tan(np.radians(hfov) / 2)
        A, B = _bearings(pa.astype(np.float64), f), _bearings(pb.astype(np.float64), f)
        R = _kabsch_batch(A[idx], B[idx])
        cnt = (_errors(R, pa, B, f) < px_thr).sum(1)
        k = int(cnt.argmax())
        if cnt[k] > best_cnt:
            mask = _errors(R[k:k + 1], pa, B, f)[0] < px_thr
            Rk = R[k]
            for _ in range(2):   # least-squares refinement on the inliers
                if mask.sum() < 3:
                    break
                Rk = _kabsch_batch(A[mask][None], B[mask][None])[0]
                mask = _errors(Rk[None], pa, B, f)[0] < px_thr
            if mask.sum() > best_cnt:
                best_cnt, best = int(mask.sum()), (mask, Rk, float(f))
    return best


def _inlier_count(args):
    """Inliers of one pair at one fixed hfov (top-level so a process pool can run it)."""
    pa, pb, px_thr, hfov = args
    return int(rotation_ransac(pa, pb, px_thr, hfovs=hfov)[0].sum())


def run_ransac(args):
    """rotation_ransac(*args) with positional args (top-level so a process pool can run it)."""
    pa, pb, px_thr, hfovs = args
    return rotation_ransac(pa, pb, px_thr, hfovs=hfovs)


def estimate_shared_focal(pair_matches, px_thr=3.0, top_k=40, coarse=0.5, fine=0.1, pool=None):
    """One focal length (as hfov, deg) for all pairs: the value that maximises the total number of rotation-model
    inliers over the strongest pairs. pair_matches: list of (pa, pb) arrays, ideally already-verified pairs.
    Coarse scan over the allowed range, then a fine scan around the best coarse value. None if no pair given.
    `pool`: optional concurrent.futures executor; same result, evaluated in parallel."""
    if not pair_matches:
        return None
    strongest = sorted(pair_matches, key=lambda m: -len(m[0]))[:top_k]
    mapper = map if pool is None else (lambda fn, it: pool.map(fn, it, chunksize=8))

    def totals(grid):
        jobs = [(pa, pb, px_thr, float(h)) for h in grid for pa, pb in strongest]
        counts = list(mapper(_inlier_count, jobs))
        k = len(strongest)
        return [sum(counts[g * k:(g + 1) * k]) for g in range(len(grid))]

    grid = np.arange(HFOVS[0], HFOVS[-1] + 1e-9, coarse)
    t = totals(grid)
    best = grid[int(np.argmax(t))]
    grid = np.arange(best - coarse, best + coarse + 1e-9, fine)
    t = totals(grid)
    return float(grid[int(np.argmax(t))])
