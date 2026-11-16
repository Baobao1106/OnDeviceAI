"""Hòa trộn các lớp ảnh thành một ảnh ERP. NGƯỜI PHỤ TRÁCH: C.

Đã viết sẵn bản trung bình có trọng số (đường nối mờ nhẹ, màu chưa khớp hoàn toàn).

VIỆC CẦN LÀM (C), theo thứ tự nên thử:
  1. Hòa trộn nhiều băng tần (Laplacian pyramid): trộn tần số thấp bằng mặt nạ mềm, tần số cao bằng mặt nạ cứng.
     Giữ chi tiết mà vẫn ẩn đường nối. Tránh làm mờ đều vì LPIPS rất nhạy với mờ.
  2. Chọn đường nối tối ưu (graph cut hoặc dynamic programming) để đường nối đi qua vùng ít chi tiết.
  3. Ưu tiên pixel gần tâm ảnh (ít méo, ít vignetting, ít nhiễu khuếch đại).
  4. Đảm bảo mép trái và mép phải của ERP liền mạch (cột 0 và cột cuối). `score.py --seam_png` xuất ảnh cuộn
     nửa chiều ngang để nhìn bằng mắt.
  5. Bộ nhớ: không dựng 24 pyramid float cùng lúc nếu không cần.

Trả về (erp uint8 [H,W,3], covered bool [H,W]): `covered` = pixel nào có ít nhất một ảnh phủ.

XONG KHI: với pose thật (oracle_render), điểm cao hơn bản trung bình có trọng số và đường nối không nhìn thấy.
"""
import numpy as np


def blend(layers, weights):
    acc = np.zeros(layers[0].shape, np.float32)
    wsum = np.zeros(layers[0].shape[:2], np.float32)
    for lay, w in zip(layers, weights):
        acc += lay.astype(np.float32) * w[..., None]
        wsum += w
    covered = wsum > 0
    out = acc / np.maximum(wsum, 1e-6)[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), covered
