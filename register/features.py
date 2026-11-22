"""Trích đặc trưng từ ảnh. NGƯỜI PHỤ TRÁCH: B.

VIỆC CẦN LÀM:
  1. `load_images`: đọc 24 ảnh trong thư mục thành dict {tên file: ảnh BGR}. (đã viết sẵn)
  2. `extract`: trả về (keypoints Nx2 theo pixel, descriptors NxD).
     - Bản đầu: OpenCV SIFT rồi chuyển sang RootSIFT (L1-chuẩn hóa, lấy căn bậc hai).
     - Ảnh trơn: thử CLAHE (cv2.createCLAHE) trước khi trích, hạ ngưỡng contrastThreshold, tăng số điểm.
     - Về sau: thử SuperPoint hoặc DISK nếu chạy được trên Jetson trong ngân sách thời gian.
  3. Nếu trích ở độ phân giải thấp hơn cho nhanh, nhớ nhân toạ độ về lại 864x648 trước khi trả về.

XONG KHI: chạy trên 10 bộ train, mỗi ảnh có đủ điểm (gợi ý ≥ 300), và thời gian trích đặc trưng cả 24 ảnh
đủ nhỏ (D sẽ đo bằng tools/bench.py).
"""
import os

import cv2


def load_images(images_dir):
    names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    out = {}
    for n in names:
        img = cv2.imread(os.path.join(images_dir, n), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"không đọc được {n}")
        out[n] = img
    return out


def extract(img_bgr):
    """Trả về (keypoints: float32 [N,2] theo pixel, descriptors).

    Dùng lõi `register.engine` (SuperPoint 4 hướng xoay + RootSIFT ngưỡng tương phản thấp, bù vignetting, CLAHE).
    `keypoints` là điểm RootSIFT (để xem/đếm); `descriptors` là gói đặc trưng của lõi (cả hai nguồn), chỉ
    `match.py` dùng nên không cần là mảng [N,D].
    """
    import numpy as np

    from .engine import get_engine
    feats = get_engine().extract(img_bgr)
    kp = next((f["kp"] for f in feats if isinstance(f, dict) and "kp" in f), np.zeros((0, 2), np.float32))
    return kp, feats
