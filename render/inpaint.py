"""Lấp chỗ trống (pixel ERP không có ảnh nào phủ). NGƯỜI PHỤ TRÁCH: C.

Chỗ trống xuất hiện khi ảnh không đặt được, hoặc ở sàn dưới −60° (không bị chấm điểm nên không cần lấp kỹ).

Đã viết sẵn bản đơn giản (cv2.inpaint ở độ phân giải thấp rồi phóng lên). VIỆC CẦN LÀM (C):
  1. Chỉ tập trung vào lỗ trên −60° (nhìn qua mặt nạ vĩ độ), bỏ qua vùng sàn.
  2. Thay bằng cách tốt hơn: lấp theo pyramid, hoặc mô hình inpainting nhỏ nếu còn ngân sách thời gian trên Jetson.
     Mục tiêu chính là tránh mảng mờ hoặc đen (LPIPS và PSNR đều phạt).
  3. Với vùng trơn (tường, trần), màu trung bình của vùng lân cận thường tốt hơn một mảng đen.

Trả về ảnh uint8 [H,W,3] đã lấp.
"""
import cv2
import numpy as np


def fill_holes(erp, covered):
    holes = ~covered
    if not holes.any():
        return erp
    H, W = erp.shape[:2]
    sw, sh = W // 4, H // 4
    small = cv2.resize(erp, (sw, sh), interpolation=cv2.INTER_AREA)
    m = cv2.resize(holes.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(small, m, 5, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC)
    out = erp.copy()
    out[holes] = filled[holes]
    return out
