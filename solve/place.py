import os
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import common
from render import vignette, warp

PANO_W, PANO_H = 512, 256
SMALL_W = 96
MIN_REGISTERED = 6
MIN_OVERLAP = 0.15
MIN_SCORE = 0.4
POSE_BOUND_DEG = 6.0
COARSE_STEP_DEG = 3.0
PRIOR_PENALTY = 0.05
POLE_PITCH = 80.0
REFINE_STEPS = (2.0, 1.0, 0.5)


def _gray(img_bgr, width):
    g = vignette.remove_vignette(img_bgr).mean(axis=2)
    h, w = g.shape
    return cv2.resize(g, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def _build_pano(images_dir, poses, hfov):
    acc = np.zeros((PANO_H, PANO_W), np.float32)
    wt = np.zeros((PANO_H, PANO_W), np.float32)
    for name, p in poses.items():
        if not p.get("registered", True):
            continue
        img = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
        erp, w = warp.warp_to_erp(vignette.remove_vignette(img), np.asarray(p["R"], float), hfov, PANO_W, PANO_H)
        acc += erp.mean(axis=2) * w
        wt += w
    pano = np.where(wt > 0, acc / np.maximum(wt, 1e-6), 0).astype(np.float32)
    pad = 8
    return (cv2.copyMakeBorder(pano, 0, 0, pad, pad, cv2.BORDER_WRAP),
            cv2.copyMakeBorder((wt > 0).astype(np.float32), 0, 0, pad, pad, cv2.BORDER_WRAP), pad)


def _camera_dirs(h, w, hfov):
    f, cx, cy = common.intrinsics(w, h, hfov)
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    d = np.stack([(xs - cx) / f, -(ys - cy) / f, np.ones_like(xs, float)], -1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _ncc(R, dirs, gray, pano):
    ext, cov, pad = pano
    u, v = common.dirs_to_erp_uv(dirs @ R.T, PANO_W, PANO_H)
    mu, mv = u.reshape(-1, 1) + pad, v.reshape(-1, 1)
    s = cv2.remap(ext, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[:, 0]
    c = cv2.remap(cov, mu, mv, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)[:, 0] > 0.5
    if c.sum() < MIN_OVERLAP * len(c):
        return -1.0
    a = gray.ravel()[c]
    b = s[c]
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-6 else -1.0


def _objective(frame, slot_pose, delta, pole, dirs, gray, pano):
    yaw, pitch, roll = delta
    yaw0, pitch0 = slot_pose
    if pole:
        dev = np.array([0.0, pitch, roll])
        R = frame @ common.pose_to_R(yaw, pitch0 + pitch, roll)
    else:
        dev = np.array([yaw, pitch, roll])
        R = frame @ common.pose_to_R(yaw0 + yaw, pitch0 + pitch, roll)
    return _ncc(R, dirs, gray, pano) - PRIOR_PENALTY * float((dev ** 2).sum()) / POSE_BOUND_DEG ** 2


def _candidates(pole):
    steps = np.arange(-POSE_BOUND_DEG, POSE_BOUND_DEG + 1e-9, COARSE_STEP_DEG)
    yaws = np.arange(0.0, 360.0, 4.0) if pole else steps
    return [(y, p, r) for y in yaws for p in steps for r in ([0.0] if pole else steps)]


def _refine(start, frame, slot_pose, pole, dirs, gray, pano):
    cur = np.array(start, float)
    best = _objective(frame, slot_pose, cur, pole, dirs, gray, pano)
    for step in REFINE_STEPS:
        improved = True
        while improved:
            improved = False
            for k in range(3):
                for sign in (-1.0, 1.0):
                    trial = cur.copy()
                    trial[k] += sign * step
                    if k and abs(trial[k]) > POSE_BOUND_DEG * 1.5:
                        continue
                    val = _objective(frame, slot_pose, trial, pole, dirs, gray, pano)
                    if val > best + 1e-6:
                        cur, best, improved = trial, val, True
    return cur, best


def place_unregistered(images_dir, poses, slots, hfov, deadline):
    unreg = sorted(n for n, p in poses.items() if not p.get("registered", True))
    registered = [n for n, p in poses.items() if p.get("registered", True)]
    if not unreg or len(registered) < MIN_REGISTERED:
        return poses
    plan = common.capture_plan()
    slot_of = {n: slots["images"][n]["slot"] for n in unreg}
    free = sorted(set(slot_of.values()))
    first = unreg[0]
    frame = np.asarray(poses[first]["R"], float) @ common.pose_to_R(*plan[slot_of[first]]).T
    pano = _build_pano(images_dir, poses, hfov)

    cache = {}
    for n in unreg:
        gray = _gray(cv2.imread(os.path.join(images_dir, n), cv2.IMREAD_COLOR), SMALL_W)
        cache[n] = (gray, _camera_dirs(gray.shape[0], gray.shape[1], hfov))

    table = np.full((len(unreg), len(free)), -9.0)
    best = {}
    for a, n in enumerate(unreg):
        gray, dirs = cache[n]
        for b, s in enumerate(free):
            if time.time() > deadline:
                return poses
            pole = abs(plan[s][1]) >= POLE_PITCH
            scored = [(_objective(frame, plan[s], c, pole, dirs, gray, pano), c) for c in _candidates(pole)]
            val, start = max(scored, key=lambda t: t[0])
            delta, val = _refine(start, frame, plan[s], pole, dirs, gray, pano)
            table[a, b], best[(a, b)] = val, delta

    out = dict(poses)
    rows, cols = linear_sum_assignment(-table)
    for a, b in zip(rows, cols):
        if table[a, b] < MIN_SCORE:
            continue
        n, s = unreg[a], free[b]
        pole = abs(plan[s][1]) >= POLE_PITCH
        yaw, pitch, roll = best[(a, b)]
        R = frame @ common.pose_to_R(yaw if pole else plan[s][0] + yaw, plan[s][1] + pitch, roll)
        out[n] = {**poses[n], "R": R.tolist(), "source": "photometric"}
    return out
