"""Phép quay giữa hai ảnh từ điểm khớp. NGƯỜI PHỤ TRÁCH: A.

Có `pixels_to_rays`, `rotation_from_rays` (Kabsch), `angular_errors`, `fit_rotation` (Kabsch cắt bỏ điểm lệch, không RANSAC,
dùng khi các điểm đã là inlier của một phép quay) và `relative_rotation` (RANSAC + Kabsch cho cả một cặp).

VIỆC CẦN LÀM:
  1. `relative_rotation(pair, hfov_deg, size)`: với một cặp ảnh trong matches.json, đổi điểm khớp thành tia 3D
     rồi tìm R sao cho  ray_i ≈ R @ ray_j. Bọc trong RANSAC (chọn 3 cặp tia, giải Kabsch, đếm điểm có góc
     lệch nhỏ như 0,3–0,5°). Trả về (R, số inlier, sai số góc trung bình).
  2. Kiểm tra dấu và quy ước với common.py (pitch > 0 nhìn lên, roll > 0 ngược chiều kim đồng hồ).
     Cách kiểm nhanh: với pose thật trong sets/tXX/hidden/true_poses.json, R_i.T @ R_j phải khớp với kết quả.
  3. Với FOV sai, sai số khớp sẽ tăng, đó là tín hiệu để `fov.py` ước lượng FOV.

XONG KHI: với FOV thật, phép quay tương đối của các cặp tốt sai lệch pose thật dưới ~0,5° (dùng tools/pose_diff.py).
"""
import numpy as np

import common


def pixels_to_rays(pts, hfov_deg, size):
    """pts [N,2] pixel (x, y) -> tia đơn vị [N,3] trong hệ camera (x phải, y lên, z trước)."""
    w, h = size
    f, cx, cy = common.intrinsics(w, h, hfov_deg)
    pts = np.asarray(pts, float)
    d = np.stack([(pts[:, 0] - cx) / f, -(pts[:, 1] - cy) / f, np.ones(len(pts))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def rotation_from_rays(a, b):
    """Phép quay R tốt nhất (bình phương tối thiểu, Kabsch) sao cho a ≈ R @ b. a, b: [N,3]."""
    H = b.T @ a
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def angular_errors(a, b, R):
    """Góc (độ) giữa a và R @ b cho từng cặp tia. a, b: [N,3]."""
    return np.degrees(np.arccos(np.clip(np.sum(a * (b @ R.T), axis=1), -1.0, 1.0)))


def fit_rotation(a, b, keep=0.8, rounds=3):
    """Kabsch có cắt: lặp lại, mỗi vòng giữ `keep` phần điểm có góc lệch nhỏ nhất. Trả về (R, sai số TB trên phần giữ)."""
    R = rotation_from_rays(a, b)
    for _ in range(rounds):
        err = angular_errors(a, b, R)
        idx = np.argsort(err)[:max(3, int(len(err) * keep))]
        R = rotation_from_rays(a[idx], b[idx])
    err = angular_errors(a, b, R)
    return R, float(np.sort(err)[:max(3, int(len(err) * keep))].mean())


def relative_rotation(pair, hfov_deg, size, thresh_deg=0.4, iters=150, seed=0):
    """Trả về (R_ij, n_inliers, mean_angle_err_deg) cho một cặp trong matches.json (ray_i ≈ R_ij @ ray_j)."""
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
    for _ in range(3):                                    # tinh chỉnh bằng toàn bộ inlier hiện tại
        inl = angular_errors(a, b, R) < thresh_deg
        if inl.sum() < 3:
            break
        R = rotation_from_rays(a[inl], b[inl])
    err = angular_errors(a, b, R)
    inl = err < thresh_deg
    return R, int(inl.sum()), float(err[inl].mean()) if inl.any() else 180.0
