import itertools
import time


def match_pair(kp_i, desc_i, kp_j, desc_j):
    from .engine import get_engine
    return get_engine().pair_inliers(desc_i, desc_j)


def candidate_pairs(names):
    return list(itertools.combinations(sorted(names), 2))


def match_all(images, feats, deadline):
    from .engine import get_engine
    names = sorted(n for n in images if n in feats)
    index = {n: k for k, n in enumerate(names)}
    only = {tuple(sorted((index[a], index[b]))) for a, b in candidate_pairs(names)}
    result = get_engine().match(names, [feats[n][1] for n in names], deadline, only, [images[n] for n in names])
    pairs = []
    for p in result.accepted():
        pts_i, pts_j = result.inlier_points[(p.i, p.j)]
        pairs.append({"i": names[p.i], "j": names[p.j], "pts_i": pts_i.tolist(), "pts_j": pts_j.tolist(),
                      "n_inliers": p.n_inliers, "rotation": p.rotation.round(6).tolist(), "rescued": p.rescued})
    return pairs
