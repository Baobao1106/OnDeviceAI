"""Gán 24 ảnh vào 24 ô của mẫu chụp. NGƯỜI PHỤ TRÁCH: B.

Mẫu chụp (common.capture_plan()), chỉ số 0 đến 23:
   0–7   vòng ngang, pitch 0°, yaw cách nhau 45° (ô 0 là ảnh 01: chính diện, chụp cẩn thận)
   8–14  vòng trên, pitch +45°, 7 ảnh cách nhau ~51°
  15–21  vòng dưới, pitch −45°, 7 ảnh cách nhau ~51°
  22     trần (pitch +90°)      23  sàn (pitch −90°)
Tay run: mỗi ảnh lệch ±5° (yaw, pitch, roll), riêng ô 0 chỉ ±2°.

Tên file bị xáo ngẫu nhiên nên không biết ảnh nào ở ô nào. Phải suy ra từ điểm khớp.

VIỆC CẦN LÀM:
  1. `assign_slots`: dựng đồ thị các ảnh (cạnh = cặp khớp tốt) rồi gán ảnh vào ô.
     Gợi ý: tìm vòng ngang trước (các ảnh nối thành chu trình 8 ảnh), xác định hướng, rồi gán vòng trên/dưới,
     cuối cùng trần và sàn (ảnh trần nối được với cả vòng +45°, sàn nối với vòng −45°).
     Có thể dùng bài toán gán Hungarian (scipy.optimize.linear_sum_assignment) với chi phí = mức không phù hợp
     giữa ảnh và ô, xét thêm sự phù hợp giữa các ảnh láng giềng.
  2. Trả về dict {tên ảnh: {"slot", "yaw_deg", "pitch_deg", "confidence", "n_matches"}}, đủ 24 ảnh,
     không ô nào trùng (schema.validate_slots sẽ kiểm tra).
  3. Ảnh khớp yếu vẫn phải có ô dự đoán, với `confidence` thấp (`textureless.py` sẽ cải thiện tiếp).
  4. Có thể ước lượng nhanh vị trí bằng độ sáng: ảnh trần thường sáng đều, ảnh sàn tối hơn, như tín hiệu phụ.

XONG KHI: so với pose thật (D sẽ viết công cụ), phần trăm ảnh gán đúng ô trên 10 bộ train gần 100% với ảnh có hoa văn.
"""
import common


def nominal(slot):
    """(yaw_deg, pitch_deg) danh định của ô."""
    return common.capture_plan()[slot]


def assign_slots(images, pairs, deadline):
    """Trả về {tên ảnh: {"slot", "yaw_deg", "pitch_deg", "confidence", "n_matches"}}."""
    raise NotImplementedError("B: viết gán ảnh vào ô của mẫu chụp")
