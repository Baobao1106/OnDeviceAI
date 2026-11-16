"""Cân bằng ngang: tìm hướng "lên" thật và xoay mọi pose cho đúng. NGƯỜI PHỤ TRÁCH: A.

QUAN TRỌNG NHẤT: nghiêng ≥ 5° là 0 điểm cả bộ. Nên có tín hiệu thứ hai để kiểm tra chéo.

VIỆC CẦN LÀM:
  1. `estimate_up(poses, slots)`: ước lượng hướng lên trong hệ pose hiện tại. Hai nguồn:
       a) Cấu trúc mẫu chụp: vòng ngang pitch 0° (ô 0 đến 7) có các trục quang học nằm trên một mặt phẳng nằm ngang,
          pháp tuyến của mặt phẳng đó (khớp bằng bình phương tối thiểu) là hướng lên. Vòng ±45° đối xứng quanh nó.
       b) Điểm triệt tiêu thẳng đứng (đường thẳng đứng trong phòng: cạnh cửa, góc tường). Dùng nếu đủ chắc chắn.
     So sánh hai nguồn; nếu lệch nhiều thì tin nguồn (a) vì nó không phụ thuộc vào nội dung cảnh.
  2. `level_poses(poses, slots)`: xoay mọi R sao cho hướng lên trùng trục y (0, 1, 0), trả về
     (poses mới, up, tilt_deg_est). Giữ nguyên yaw tự do (hướng chính diện được tự căn lúc chấm điểm).
  3. Cẩn thận dấu: pitch > 0 là nhìn lên, roll > 0 xoay ảnh ngược chiều kim đồng hồ.

XONG KHI: với 10 bộ train, `tilt_deg_est` thật sự gần 0° khi so với pose thật (D kiểm bằng score.py: trường tilt_deg),
tilt < 1° ở hầu hết bộ, và không bộ nào gần 5°.
"""


def estimate_up(poses, slots):
    """Trả về vector lên (3,) trong hệ pose hiện tại."""
    raise NotImplementedError("A: ước lượng hướng lên")


def level_poses(poses, slots):
    """Trả về (poses đã xoay, up, tilt_deg_est)."""
    raise NotImplementedError("A: cân bằng ngang")
