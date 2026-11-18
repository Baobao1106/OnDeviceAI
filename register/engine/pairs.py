"""Lõi ghép cặp: đặc trưng từng ảnh -> khớp mọi cặp -> RANSAC phép quay thuần với FOV chung -> cứu ảnh cô lập.

Đây là bản của `matching/pipeline.py` (code đã đo trên 50 bộ train) tách thành các bước mà `register/` gọi:
  PairEngine.extract(bgr)               đặc trưng của MỘT ảnh (gọi cho từng ảnh)
  PairEngine.pair_inliers(fi, fj)       khớp MỘT cặp + RANSAC (FOV tự do), cho `match.match_pair`
  PairEngine.match(names, feats, dl)    khớp cả bộ: lượt 1, FOV chung, lượt 2, cứu ảnh cô lập
Thuật toán và tham số giữ nguyên; chỉ thêm kiểm tra `deadline`.
"""
import itertools
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace

import numpy as np

from .matchers import IMG_W, build, preprocess
from .rescue import rescue_isolated
from .rotransac import estimate_shared_focal, rotation_ransac, run_ransac

CAND_MIN_INLIERS = 4            # cặp bị loại có ít nhất chừng này inlier giữ lại góc quay để cứu (rescue)


@dataclass(frozen=True)
class PipelineConfig:
    device: str = "auto"            # "auto" -> cuda nếu có
    use_superpoint: bool = True
    use_sift: bool = True
    sp_max_keypoints: int = 2048
    sp_clahe: bool = True
    min_inliers: int = 10           # ngưỡng chấp nhận một cặp (chọn trên 10 bộ train)
    ransac_px: float = 3.0
    shared_focal: bool = True       # fit một tiêu cự chung trên các cặp mạnh rồi ước lượng lại mọi cặp
    strong_inliers: int = 30        # độ tin cậy bão hòa ở đây
    workers: int = 0                # số tiến trình cho các bước RANSAC (0 = số nhân CPU, tối đa 6; 1 = không dùng pool)
    rescue: bool = True             # chấp nhận cặp yếu của ảnh cô lập khi tiền nghiệm tầng đồng ý


@dataclass(frozen=True)
class PairResult:
    i: int
    j: int
    n_raw: int
    n_inliers: int
    accepted: bool
    confidence: float
    hfov_deg: float | None
    rotation: np.ndarray | None     # R với ray_i ~ R @ ray_j (ảnh j -> ảnh i)
    cand_rotation: np.ndarray | None = None
    rescued: bool = False


@dataclass(frozen=True)
class PipelineResult:
    names: list
    pairs: list                      # list[PairResult]
    inlier_points: dict              # (i, j) -> (pts_i, pts_j) inlier RANSAC, chỉ cặp được chấp nhận
    timing: dict
    hfov_estimate: float | None      # FOV chung đã fit (độ), None nếu không có cặp mạnh nào
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
        self.sources = []           # (matcher, clahe)
        if self.cfg.use_superpoint:
            self.sources.append((build("superpoint_lightglue_rot", device, max_kp=self.cfg.sp_max_keypoints),
                                 self.cfg.sp_clahe))
        if self.cfg.use_sift:
            self.sources.append((build("rootsift_lowc", device), False))
        if not self.sources:
            raise ValueError("cần bật ít nhất một bộ khớp")
        self.load_time = time.perf_counter() - t0

    def extract(self, bgr):
        """Đặc trưng của một ảnh: list theo từng nguồn (SuperPoint 4 hướng xoay, RootSIFT)."""
        return [m.extract(preprocess(bgr, True, clahe)) for m, clahe in self.sources]

    def _pooled(self, fi, fj):
        parts = [m.match(a, b) for (m, _), a, b in zip(self.sources, fi, fj)]
        a = np.concatenate([p[0] for p in parts]).astype(np.float64)
        b = np.concatenate([p[1] for p in parts]).astype(np.float64)
        return a, b

    def pair_inliers(self, fi, fj):
        """Khớp một cặp + RANSAC phép quay (tiêu cự tự do). Trả về (pts_i, pts_j, n_inliers)."""
        a, b = self._pooled(fi, fj)
        mask, _, _ = rotation_ransac(a, b, self.cfg.ransac_px)
        return a[mask].astype(np.float32), b[mask].astype(np.float32), int(mask.sum())

    def match(self, names, feats, deadline=None, only=None):
        """feats[k] = self.extract(ảnh names[k]). `only`: tập cặp chỉ số (i, j), i < j, cần khớp (mặc định: tất cả).
        Trả về PipelineResult."""
        t_start = time.perf_counter()
        if len(names) < 2:
            return PipelineResult(list(names), [], {}, {}, None, self.cfg)
        n_workers = self.cfg.workers or min(6, os.cpu_count() or 1)
        pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
        try:
            return self._match(names, feats, deadline, only, pool, t_start)
        except BrokenProcessPool:     # không tạo được tiến trình con (ví dụ chạy từ stdin): làm tuần tự, kết quả như nhau
            return self._match(names, feats, deadline, only, None, t_start)
        finally:
            if pool is not None:
                pool.shutdown()

    def _match(self, names, feats, deadline, only, pool, t_start):
        t1 = time.perf_counter()
        raw, jobs = {}, {}
        order = sorted(only) if only is not None else list(itertools.combinations(range(len(names)), 2))
        np.random.default_rng(0).shuffle(order)         # nếu hết giờ giữa chừng, mọi ảnh đều đã có một phần cặp
        for i, j in order:
            if deadline is not None and time.time() > deadline:
                break                                   # dừng sớm: các cặp chưa khớp bị bỏ
            try:
                a, b = self._pooled(feats[i], feats[j])
            except Exception:                           # một cặp lỗi (ví dụ hết bộ nhớ GPU) không làm hỏng cả bộ
                print(f"[match] bỏ cặp ({names[i]}, {names[j]}):", file=sys.stderr)
                traceback.print_exc()
                continue
            raw[(i, j)] = (a, b)
            args = (a, b, self.cfg.ransac_px, None)                        # lượt 1: tiêu cự tự do theo từng cặp
            jobs[(i, j)] = pool.submit(run_ransac, args) if pool else run_ransac(args)   # CPU chạy chồng GPU
        raw = dict(sorted(raw.items()))                 # thứ tự cố định, kết quả không phụ thuộc thứ tự xử lý
        first = {k: (jobs[k].result() if pool else jobs[k]) for k in raw}
        t_match = time.perf_counter() - t1

        t2 = time.perf_counter()
        shared = None
        if self.cfg.shared_focal:   # lượt 2: tiêu cự chung cho mọi ảnh nên fit một lần rồi cố định
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
        if self.cfg.rescue:
            by_key = {(p.i, p.j): k for k, p in enumerate(pairs)}
            for r in rescue_isolated(names, pairs):
                k = by_key[(r.i, r.j)]
                pairs[k] = replace(pairs[k], accepted=True, rescued=True, rotation=r.rotation,
                                   confidence=min(1.0, r.n_inliers / self.cfg.strong_inliers),
                                   hfov_deg=shared if shared is not None else pairs[k].hfov_deg)
                inlier_points[(r.i, r.j)] = cand_points[(r.i, r.j)]
        timing = {"match_and_pass1": round(t_match, 2), "shared_focal_and_pass2": round(time.perf_counter() - t2, 2),
                  "total_match": round(time.perf_counter() - t_start, 2)}
        return PipelineResult(list(names), pairs, inlier_points, timing, shared, self.cfg)
