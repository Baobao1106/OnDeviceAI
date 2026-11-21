"""Thứ tự khớp cặp khi thiếu giờ, và cách chia hạn giữa lượt khớp chính với lượt ghép lại ảnh ít texture.

Hàm thuần (không đụng GPU/ảnh) để kiểm thử độc lập. Chỉ dùng khi `PipelineConfig.priority_order` / `lowtex_reserve` bật;
khi đủ giờ, thứ tự xử lý không đổi kết quả (kết quả được sắp lại theo thứ tự cố định trước khi RANSAC lượt 2).
"""
import itertools

import numpy as np


def priority_order(n, score, pairs=None, seed=0):
    """Sắp các cặp (i < j) để cặp có triển vọng được khớp trước mà mọi ảnh đều được lo trước khi ảnh nào được lo nhiều.

    `score[(i, j)]`: điểm rẻ của cặp (số điểm khớp RootSIFT); cặp không có điểm coi là 0. Mỗi ảnh xếp các cặp chứa nó
    theo điểm giảm dần (hòa thì ngẫu nhiên có hạt giống cố định, nên ảnh mù RootSIFT vẫn có thứ tự như cách ngẫu nhiên
    cũ). Vòng r lấy cặp thứ r của mỗi ảnh, trong vòng xếp theo điểm giảm dần; cặp đã lấy không lấy lại.
    Trả về list gồm đúng mỗi cặp một lần.
    """
    keys = sorted(pairs) if pairs is not None else list(itertools.combinations(range(n), 2))
    rng = np.random.default_rng(seed)
    partners = {i: [] for i in range(n)}
    for i, j in keys:
        partners[i].append(j)
        partners[j].append(i)
    ranked = {}
    for i in range(n):
        js = [partners[i][k] for k in rng.permutation(len(partners[i]))]      # hòa điểm: ngẫu nhiên cố định
        js.sort(key=lambda j: -score.get((min(i, j), max(i, j)), 0))          # sort ổn định giữ thứ tự ngẫu nhiên
        ranked[i] = [(min(i, j), max(i, j)) for j in js]
    out, seen = [], set()
    for r in range(max((len(v) for v in ranked.values()), default=0)):
        batch = [ranked[i][r] for i in range(n) if r < len(ranked[i])]
        batch.sort(key=lambda p: -score.get(p, 0))
        for p in batch:
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def split_deadline(now, deadline, reserve):
    """Hạn của lượt khớp chính: dành `reserve` (0..0,9) phần thời gian còn lại cho lượt ghép lại ảnh ít texture.
    Không có hạn hoặc reserve <= 0 thì giữ nguyên `deadline`."""
    if deadline is None or reserve <= 0:
        return deadline
    return now + (1.0 - min(reserve, 0.9)) * max(deadline - now, 0.0)
