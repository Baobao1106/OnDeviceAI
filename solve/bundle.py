"""Tối ưu toàn cục pose của 24 ảnh và FOV (bundle adjustment). NGƯỜI PHỤ TRÁCH: A.

VIỆC CẦN LÀM:
  1. Khởi tạo: dựng cây khung có độ tin cậy cao trên đồ thị ảnh (cạnh = cặp khớp, trọng số = n_inliers),
     nhân dần các phép quay tương đối từ ảnh gốc (ảnh ở ô 0) để có R ban đầu cho mọi ảnh nối được.
  2. Tối ưu (scipy.optimize.least_squares, tham số hóa phép quay bằng vector 3 chiều): hàm lỗi là góc giữa
     R_i @ ray_i và R_j @ ray_j của mọi cặp khớp, có hàm mất mát bền vững (loss="huber" hoặc "soft_l1").
     Tối ưu cùng lúc 24 phép quay và FOV.
  3. Ràng buộc mềm từ mẫu chụp: pose thật chỉ lệch ±5° so với góc danh định của ô (slots.json). Thêm một số hạng
     phạt nhẹ để ảnh ít khớp không "trôi" khỏi vị trí của nó.
  4. Ảnh không có khớp tốt: đặt pose bằng góc danh định của ô (`source="slot_prior"`, `registered=False`).
  5. Trả về poses theo định dạng poses.json: {tên ảnh: {"R": [[...]], "registered": bool, "source": str}}
     và hfov đã tinh chỉnh. Kiểm tra time.time() < deadline giữa các lần lặp.
     Lưu ý: chưa cân bằng ngang ở bước này (`level.py` làm bước đó).

XONG KHI: sai số pose tương đối so với pose thật (tools/pose_diff.py) dưới ~0,5° cho các ảnh `registered`
trên đa số bộ train, và thời gian tối ưu chỉ vài giây.
"""


def bundle_adjust(matches, slots, hfov_deg, deadline):
    """Trả về (poses: dict, hfov_deg: float)."""
    raise NotImplementedError("A: bundle adjustment")
