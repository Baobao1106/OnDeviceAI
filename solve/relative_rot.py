"""Phép quay giữa hai ảnh từ điểm khớp. NGƯỜI PHỤ TRÁCH: A.

Đã viết sẵn hai hàm nhỏ (`pixels_to_rays`, `rotation_from_rays`). A viết thêm RANSAC và hàm cho cả cặp.

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


def relative_rotation(pair, hfov_deg, size):
    """Trả về (R_ij, n_inliers, mean_angle_err_deg) cho một cặp trong matches.json."""
    raise NotImplementedError("A: RANSAC + Kabsch cho một cặp ảnh")
