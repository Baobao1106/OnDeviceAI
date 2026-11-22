

import numpy as np
from scipy.optimize import minimize_scalar

from .relative_rot import fit_rotation, pixels_to_rays

MAX_PAIRS = 40
MAX_POINTS = 150
FALLBACK_FOV = 74.5


def _select_pairs(matches, seed=0):
    rng = np.random.default_rng(seed)
    pairs = sorted(matches["pairs"], key=lambda p: -p["n_inliers"])[:MAX_PAIRS]
    chosen = []
    for p in pairs:
        n = len(p["pts_i"])
        if n < 8:
            continue
        idx = np.sort(rng.choice(n, min(n, MAX_POINTS), replace=False))
        chosen.append((np.asarray(p["pts_i"], float)[idx], np.asarray(p["pts_j"], float)[idx]))
    return chosen


def _cost(fov_deg, chosen, size):
    total = 0.0
    for pi, pj in chosen:
        _, err = fit_rotation(pixels_to_rays(pi, fov_deg, size), pixels_to_rays(pj, fov_deg, size))
        total += err * np.sqrt(len(pi))
    return total / sum(np.sqrt(len(pi)) for pi, _ in chosen)


def estimate_fov(matches, slots, lo=70.0, hi=79.0):
    chosen = _select_pairs(matches)
    if not chosen:
        return float(np.clip(FALLBACK_FOV, lo, hi))
    size = matches["image_size"]
    grid = np.arange(lo, hi + 1e-9, 0.5)
    costs = [_cost(f, chosen, size) for f in grid]
    best = float(grid[int(np.argmin(costs))])
    res = minimize_scalar(lambda f: _cost(f, chosen, size), bounds=(max(lo, best - 0.5), min(hi, best + 0.5)),
                          method="bounded", options={"xatol": 0.02})
    return float(np.clip(res.x, lo, hi))
