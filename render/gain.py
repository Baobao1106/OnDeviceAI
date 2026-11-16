"""Cân độ sáng và màu giữa các ảnh. NGƯỜI PHỤ TRÁCH: C.

Mỗi ảnh có gain riêng 0,85–1,15 và lệch cân bằng trắng ±4% mỗi kênh. Chấm điểm chỉ bù một gain chung
cho cả ảnh ERP, nên sự khác biệt giữa các ảnh vẫn trừ điểm (đường nối lộ ra, PSNR và LPIPS giảm).

Hiện tại `estimate_gains` trả về toàn 1 (không bù). VIỆC CẦN LÀM (C):
  1. Với mỗi cặp ảnh chồng lên nhau (xem vùng cả hai trọng số > 0 trong ERP), tính tỉ lệ trung bình mỗi kênh màu
     trong vùng chồng lấn (bỏ pixel cháy sáng hoặc quá tối).
  2. Giải hệ bình phương tối thiểu cho một gain mỗi ảnh mỗi kênh (BGR) sao cho các cặp chồng lấn khớp nhau
     (log-gain biến thành hệ tuyến tính). Thêm ràng buộc mềm trung bình gain bằng 1 để tránh trôi toàn cục.
  3. Trả về mảng [N,3] theo thứ tự `layers`. Gain mỗi kênh nên ở khoảng 0,8–1,25.
  4. Chỉ so sánh pixel ở vùng giữa ảnh (trọng số cao), vì pose có sai số nhỏ làm rìa khác nhau.

XONG KHI: với pose thật (oracle_render), điểm tăng khi bật gain.py so với tắt, và đường nối giữa ảnh không còn
nhìn thấy bằng mắt.
"""
import numpy as np


def estimate_gains(layers, weights):
    """layers: list uint8 [H,W,3] (BGR), weights: list float32 [H,W]. Trả về [N,3] (BGR)."""
    return np.ones((len(layers), 3), np.float32)
