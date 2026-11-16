"""Bù vignetting (ảnh tối dần ra rìa). NGƯỜI PHỤ TRÁCH: C.

Mô hình ống kính của đề đã biết (lấy từ make_set.py, dùng chung cho mọi ảnh):
    xs, ys = meshgrid(linspace(-1, 1, w), linspace(-1, 1, h) * h / w)
    vig = 1 - 0.18 * (xs**2 + ys**2)
Ảnh thu được = ảnh gốc × vig × gain × cân bằng trắng + nhiễu. Chia cho `vig` để bỏ phần tối rìa.

Đã viết sẵn bản chạy được. VIỆC CẦN LÀM (C):
  1. Kiểm tra bằng mắt và bằng số: dùng pose thật (tools/oracle_render.py), so điểm khi bật và tắt bù vignetting.
     Nếu mô hình đúng thì đường nối phải biến mất rõ rệt ở các vùng chồng lấn.
  2. Chia vig ở không gian tuyến tính hay gamma? Thử cả hai, chọn cái cho PSNR tốt hơn.
  3. Tránh khuếch đại nhiễu quá nhiều ở rìa (vig nhỏ nhất ~0,64). Thử hạ trọng số pixel rìa trong warp.py.
"""
import numpy as np


def vignette_map(w, h):
    xs, ys = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h) * h / w)
    return (1.0 - 0.18 * (xs ** 2 + ys ** 2)).astype(np.float32)


def remove_vignette(img_bgr):
    h, w = img_bgr.shape[:2]
    return img_bgr.astype(np.float32) / vignette_map(w, h)[..., None]
