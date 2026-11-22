import itertools
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace

import numpy as np

from .matchers import IMG_W, build, locnorm_gray, preprocess
from .order import priority_order, split_deadline
from .rescue import rescue_isolated
from .rotransac import estimate_shared_focal, rotation_ransac, run_ransac

CAND_MIN_INLIERS = 4


@dataclass(frozen=True)
class PipelineConfig:
    device: str = "auto"
    use_superpoint: bool = True
    use_sift: bool = True
    sp_max_keypoints: int = 2048
    sp_clahe: bool = True
    min_inliers: int = 10
    ransac_px: float = 3.0
    shared_focal: bool = True
    strong_inliers: int = 30
    workers: int = 0
    rescue: bool = True
    lowtex_pass: bool = True
    lowtex_degree: int = 2
    priority_order: bool = True
    lowtex_reserve: float = 0.15


@dataclass(frozen=True)
class PairResult:
    i: int
    j: int
    n_raw: int
    n_inliers: int
    accepted: bool
    confidence: float
    hfov_deg: float | None
    rotation: np.ndarray | None
    cand_rotation: np.ndarray | None = None
    rescued: bool = False
    boosted: bool = False


@dataclass(frozen=True)
class PipelineResult:
    names: list
    pairs: list
    inlier_points: dict
    timing: dict
    hfov_estimate: float | None
    config: PipelineConfig

    def accepted(self):
        return [p for p in self.pairs if p.accepted]


class PairEngine:
    def __init__(self, config=None):
        self.cfg = config or PipelineConfig()
        device = self.cfg.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        t0 = time.perf_counter()
        self.sources = []
        if self.cfg.use_superpoint:
            self.sources.append((build("superpoint_lightglue_rot", device, max_kp=self.cfg.sp_max_keypoints),
                                 self.cfg.sp_clahe))
        if self.cfg.use_sift:
            self.sources.append((build("rootsift_lowc", device), False))
        if not self.sources:
            raise ValueError("cần bật ít nhất một bộ khớp")
        self.load_time = time.perf_counter() - t0

    def extract(self, bgr):
        return [m.extract(preprocess(bgr, True, clahe)) for m, clahe in self.sources]

    def extract_lowtex(self, bgr):
        return [m.extract(locnorm_gray(bgr)) for m, _ in self.sources]

    def _pooled(self, fi, fj, cached=None):
        parts = [cached[s] if cached and s in cached else m.match(a, b)
                 for s, ((m, _), a, b) in enumerate(zip(self.sources, fi, fj))]
        a = np.concatenate([p[0] for p in parts]).astype(np.float64)
        b = np.concatenate([p[1] for p in parts]).astype(np.float64)
        return a, b

    def pair_inliers(self, fi, fj):
        a, b = self._pooled(fi, fj)
        mask, _, _ = rotation_ransac(a, b, self.cfg.ransac_px)
        return a[mask].astype(np.float32), b[mask].astype(np.float32), int(mask.sum())

    def match(self, names, feats, deadline=None, only=None, images=None):
        t_start = time.perf_counter()
        if len(names) < 2:
            return PipelineResult(list(names), [], {}, {}, None, self.cfg)
        n_workers = self.cfg.workers or min(6, os.cpu_count() or 1)
        pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
        try:
            return self._match(names, feats, deadline, only, images, pool, t_start)
        except BrokenProcessPool:
            return self._match(names, feats, deadline, only, images, None, t_start)
        finally:
            if pool is not None:
                pool.shutdown()

    def _match(self, names, feats, deadline, only, images, pool, t_start):
        t1 = time.perf_counter()
        raw, jobs = {}, {}
        order = sorted(only) if only is not None else list(itertools.combinations(range(len(names)), 2))
        np.random.default_rng(0).shuffle(order)
        main_deadline = split_deadline(time.time(), deadline,
                                       self.cfg.lowtex_reserve if self.cfg.lowtex_pass and images is not None else 0.0)
        cache = {}
        sift_idx = next((s for s, (m, _) in enumerate(self.sources) if m.name.startswith("rootsift")), None)
        if self.cfg.priority_order and sift_idx is not None and len(self.sources) > 1:
            score = {}
            for i, j in sorted(order):
                if main_deadline is not None and time.time() > main_deadline:
                    break
                try:
                    res = self.sources[sift_idx][0].match(feats[i][sift_idx], feats[j][sift_idx])
                except Exception:
                    continue
                cache[(i, j)] = {sift_idx: res}
                score[(i, j)] = len(res[0])
            order = priority_order(len(names), score, order)
        for i, j in order:
            if main_deadline is not None and time.time() > main_deadline:
                break
            try:
                a, b = self._pooled(feats[i], feats[j], cache.get((i, j)))
            except Exception:
                print(f"[match] bỏ cặp ({names[i]}, {names[j]}):", file=sys.stderr)
                traceback.print_exc()
                continue
            raw[(i, j)] = (a, b)
            args = (a, b, self.cfg.ransac_px, None)
            jobs[(i, j)] = pool.submit(run_ransac, args) if pool else run_ransac(args)
        raw = dict(sorted(raw.items()))
        first = {k: (jobs[k].result() if pool else jobs[k]) for k in raw}
        t_match = time.perf_counter() - t1

        t2 = time.perf_counter()
        shared = None
        if self.cfg.shared_focal:
            strong = [raw[k] for k, v in first.items() if v[0].sum() >= self.cfg.strong_inliers]
            shared = estimate_shared_focal(strong, self.cfg.ransac_px, pool=pool)
        second = first
        if shared is not None:
            keys = list(raw)
            mapper = map if pool is None else (lambda fn, it: pool.map(fn, it, chunksize=8))
            second = dict(zip(keys, mapper(run_ransac, [(*raw[k], self.cfg.ransac_px, shared) for k in keys])))
        pairs, inlier_points, cand_points = [], {}, {}
        for (i, j), (a, b) in raw.items():
            mask, rot, focal = second[(i, j)]
            n_inl = int(mask.sum())
            ok = n_inl >= self.cfg.min_inliers and rot is not None
            hfov = float(np.degrees(2 * np.arctan((IMG_W / 2) / focal))) if focal else None
            pairs.append(PairResult(i, j, len(a), n_inl, ok, min(1.0, n_inl / self.cfg.strong_inliers) if ok else 0.0,
                                    hfov if ok else None, rot if ok else None,
                                    rot if (not ok and rot is not None and n_inl >= CAND_MIN_INLIERS) else None))
            if ok:
                inlier_points[(i, j)] = (a[mask].astype(np.float32), b[mask].astype(np.float32))
            elif pairs[-1].cand_rotation is not None:
                cand_points[(i, j)] = (a[mask].astype(np.float32), b[mask].astype(np.float32))
        t_low = time.perf_counter()
        if self.cfg.lowtex_pass and images is not None and shared is not None:
            pairs = self._lowtex_pass(names, feats, images, pairs, shared, only, deadline, pool, inlier_points,
                                      cand_points)
        t_low = time.perf_counter() - t_low
        if self.cfg.rescue:
            by_key = {(p.i, p.j): k for k, p in enumerate(pairs)}
            for r in rescue_isolated(names, pairs):
                k = by_key[(r.i, r.j)]
                pairs[k] = replace(pairs[k], accepted=True, rescued=True, rotation=r.rotation,
                                   confidence=min(1.0, r.n_inliers / self.cfg.strong_inliers),
                                   hfov_deg=shared if shared is not None else pairs[k].hfov_deg)
                inlier_points[(r.i, r.j)] = cand_points[(r.i, r.j)]
        timing = {"match_and_pass1": round(t_match, 2), "shared_focal_and_pass2": round(time.perf_counter() - t2, 2),
                  "lowtex_pass": round(t_low, 2), "total_match": round(time.perf_counter() - t_start, 2)}
        return PipelineResult(list(names), pairs, inlier_points, timing, shared, self.cfg)

    def _lowtex_pass(self, names, feats, images, pairs, shared, only, deadline, pool, inlier_points, cand_points):
        n = len(names)
        deg = np.zeros(n, int)
        for p in pairs:
            if p.accepted:
                deg[p.i] += 1
                deg[p.j] += 1
        weak = [k for k in range(n) if deg[k] < self.cfg.lowtex_degree]
        if not weak or len(weak) == n or (deadline is not None and time.time() > deadline):
            return pairs
        keys = sorted({(min(w, o), max(w, o)) for w in weak for o in range(n) if o != w
                       and (only is None or (min(w, o), max(w, o)) in only)})
        feats_low = {}
        by_key = {(p.i, p.j): p for p in pairs}
        old_raw = {}
        jobs = {}
        for i, j in keys:
            if deadline is not None and time.time() > deadline:
                break
            try:
                for k in (i, j):
                    if k not in feats_low:
                        feats_low[k] = self.extract_lowtex(images[k])
                a, b = self._pooled(feats_low[i], feats_low[j])
            except Exception:
                print(f"[lowtex] bỏ cặp ({names[i]}, {names[j]}):", file=sys.stderr)
                traceback.print_exc()
                continue
            args = (a, b, self.cfg.ransac_px, shared)
            jobs[(i, j)] = ((a, b), pool.submit(run_ransac, args) if pool else run_ransac(args))
        out = dict(by_key)
        for key, ((a, b), job) in sorted(jobs.items()):
            mask, rot, focal = job.result() if pool else job
            n_inl = int(mask.sum())
            old = by_key.get(key)
            if rot is None or n_inl < CAND_MIN_INLIERS or (old is not None and (old.accepted or n_inl <= old.n_inliers)):
                continue
            ok = n_inl >= self.cfg.min_inliers
            hfov = float(np.degrees(2 * np.arctan((IMG_W / 2) / focal))) if focal else None
            n_raw = len(a) + (old.n_raw if old is not None else 0)
            out[key] = PairResult(key[0], key[1], n_raw, n_inl, ok, min(1.0, n_inl / self.cfg.strong_inliers) if ok else 0.0,
                                  hfov if ok else None, rot if ok else None, None if ok else rot, boosted=ok)
            pts = (a[mask].astype(np.float32), b[mask].astype(np.float32))
            if ok:
                inlier_points[key] = pts
                cand_points.pop(key, None)
            else:
                cand_points[key] = pts
        return [out[k] for k in sorted(out)]
