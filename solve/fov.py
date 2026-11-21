"""Ước lượng FOV ngang chung của 24 ảnh. NGƯỜI PHỤ TRÁCH: A.

Đề bài: FOV giống nhau trong cả bộ, nằm trong 72–77° (hãy quét rộng hơn một chút, ví dụ 70–79°, để an toàn).
FOV sai là đường nối bị lệch, rất tốn PSNR.

VIỆC CẦN LÀM:
  1. `estimate_fov(matches, slots)`: với mỗi FOV ứng viên, tính phép quay tương đối của các cặp tốt nhất
     (`relative_rot.relative_rotation`) rồi đo sai số khớp trung bình. Chọn FOV có sai số nhỏ nhất.
     Gợi ý: quét thô bước 1°, sau đó golden-section hoặc scipy.optimize.minimize_scalar trong khoảng hẹp.
  2. Có thể dùng vòng ngang kín: tổng 8 phép quay yaw đi hết một vòng phải bằng 360°, đây là ràng buộc rất mạnh
     về FOV khi các ảnh có khớp tốt.
  3. FOV chỉ là điểm khởi đầu; `bundle.py` sẽ tinh chỉnh cùng các pose.

Cách làm: lấy các cặp mạnh nhất, với mỗi FOV ứng viên (lưới 0,5°) khớp phép quay của từng cặp bằng Kabsch có cắt rồi cộng
sai số góc (trọng số theo căn số điểm); chọn FOV tốt nhất rồi tinh chỉnh trong ±0,5° bằng minimize_scalar. Vòng ngang kín
nằm ở `bundle.py` (tối ưu chung 24 phép quay), nên ở đây không cần.

XONG KHI: trên 10 bộ train, FOV ước lượng sai không quá ~0,5° so với `hfov_deg` trong
sets/tXX/hidden/true_poses.json (D đo bằng tools/pose_diff.py).
"""


import numpy as np
from scipy.optimize import minimize_scalar

from .relative_rot import fit_rotation, pixels_to_rays

MAX_PAIRS = 40          # chỉ các cặp mạnh nhất
MAX_POINTS = 150        # điểm mỗi cặp (lấy mẫu cố định)
FALLBACK_FOV = 74.5     # giữa khoảng 72-77 của đề bài


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
    """Trả về hfov_deg (float) trong [lo, hi]."""
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
