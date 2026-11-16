"""Chiếu một ảnh phối cảnh lên ảnh ERP. NGƯỜI PHỤ TRÁCH: C.

Với mỗi pixel ERP có tia 3D `d` (common.erp_dirs). Trong hệ camera, tia đó là  R^T @ d  (R: camera -> world).
Chiếu vào ảnh: x = f·X/Z + cx,  y = -f·Y/Z + cy  (common.intrinsics). Chỉ lấy pixel có Z > 0 và nằm trong ảnh.

Đã viết sẵn bản tham chiếu chạy trên CPU (cv2.remap). VIỆC CẦN LÀM (C):
  1. Kiểm tra đúng quy ước bằng tools/oracle_render.py (pose thật), nếu ảnh bị lật hoặc lệch hướng thì sửa dấu.
  2. Chuyển sang GPU (PyTorch grid_sample hoặc cv2.cuda) vì đây là khâu tốn thời gian nhất: 24 ảnh × 2 triệu pixel ERP.
     Gợi ý tăng tốc: chỉ tính cho khối chữ nhật chứa ảnh trên ERP, tính tia 3D một lần dùng cho cả 24 ảnh.
  3. Trọng số `weight`: bản đầu dùng khoảng cách tới mép ảnh (giảm dần ra rìa). Thử thêm cos của góc so với tâm ảnh,
     hoặc giảm trọng số vùng rìa nơi bù vignetting khuếch đại nhiễu.
  4. Độ nội suy: INTER_LINEAR cho nhanh, INTER_CUBIC cho nét hơn (LPIPS nhạy với mờ). Đo thử cả hai.

Đầu ra: (erp float32 [H,W,3] BGR theo thang 0–255, weight float32 [H,W], 0 nơi không có ảnh).
"""
import cv2
import numpy as np

import common


def warp_to_erp(img, R, hfov_deg, W=2048, H=1024, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    f, cx, cy = common.intrinsics(w, h, hfov_deg)
    d = common.erp_dirs(W, H)                 # [H,W,3] tia world
    dc = d @ R                                # = (R^T d) viết theo hàng
    z = dc[..., 2]
    zs = np.where(z > 1e-6, z, 1.0)
    x = f * dc[..., 0] / zs + cx
    y = -f * dc[..., 1] / zs + cy
    valid = (z > 1e-6) & (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    erp = cv2.remap(img.astype(np.float32), x.astype(np.float32), y.astype(np.float32), interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    edge = np.minimum.reduce([x, w - 1 - x, y, h - 1 - y])          # khoảng cách tới mép ảnh (pixel)
    weight = np.where(valid, np.clip(edge, 0, None) + 1.0, 0.0).astype(np.float32)
    erp[~valid] = 0
    return erp, weight
