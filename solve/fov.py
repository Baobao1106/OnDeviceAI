"""Ước lượng FOV ngang chung của 24 ảnh. NGƯỜI PHỤ TRÁCH: A.

Đề bài: FOV giống nhau trong cả bộ, nằm trong 72–77° (hãy quét rộng hơn một chút, ví dụ 70–79°, để an toàn).
FOV sai là đường nối bị lệch, rất tốn PSNR.

VIỆC CẦN LÀM:
  1. `estimate_fov(matches, slots)`: với mỗi FOV ứng viên, tính phép quay tương đối của các cặp tốt nhất
     (`relative_rot.relative_rotation`) rồi đo sai số khớp trung bình. Chọn FOV có sai số nhỏ nhất.
     Gợi ý: quét thô bước 1°, sau đó golden-section hoặc scipy.optimize.minimize_scalar trong khoảng hẹp.
  2. Có thể dùng vòng ngang kín: tổng 8 phép quay yaw đi hết một vòng phải bằng 360°, đây là ràng buộc rất mạnh
     về FOV khi các ảnh có khớp tốt.
  3. FOV chỉ là điểm khởi đầu; `bundle.py` sẽ tinh chỉnh cùng các pose.

XONG KHI: trên 10 bộ train, FOV ước lượng sai không quá ~0,5° so với `hfov_deg` trong
sets/tXX/hidden/true_poses.json (D đo bằng tools/pose_diff.py).
"""


def estimate_fov(matches, slots, lo=70.0, hi=79.0):
    """Trả về hfov_deg (float)."""
    raise NotImplementedError("A: quét FOV và chọn giá trị cho sai số khớp thấp nhất")
