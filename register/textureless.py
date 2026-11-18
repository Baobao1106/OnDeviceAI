"""Xử lý ảnh trơn (tường trắng, trần, sàn). NGƯỜI PHỤ TRÁCH: B.

Đây là chỗ nhóm có thể hơn các đội dùng cách cổ điển. Slide ví dụ t05: chỉ 13/24 ảnh đặt được, phần trơn bị
lấp bằng vùng mờ, nên PSNR và LPIPS đều thấp.

VIỆC CẦN LÀM:
  1. `find_textureless`: tìm ảnh có ít đặc trưng hoặc ít cặp khớp tốt (ví dụ số keypoint thấp, độ lệch chuẩn của
     gradient thấp, hoặc đồ thị khớp có bậc < 2).
  2. `place_textureless`: với mỗi ảnh trơn, chọn ô còn trống phù hợp nhất:
       - ô còn lại sau khi các ảnh khác đã gán (loại trừ);
       - ô kề với các ảnh đã đặt chắc chắn, đối chiếu với vùng chồng lấn (dù ít hoa văn, đường biên, cạnh tường,
         và chuyển sắc độ sáng thường vẫn khớp được ở độ phân giải thấp);
       - thử bộ khớp học sâu (LightGlue, LoFTR) chỉ cho các ảnh này, vì chúng khớp tốt hơn SIFT ở vùng ít hoa văn.
  3. Cập nhật `slot_info` (confidence phản ánh độ chắc chắn), trả về dict mới.
  4. Nếu thật sự không biết: để A dùng pose từ mẫu chụp (`source="slot_prior"`). Chấp nhận sai vài độ còn hơn
     để trống ô.

XONG KHI: trên các bộ có nhiều tường trơn (như t05), số ảnh đặt đúng ô tăng rõ so với chỉ dùng SIFT, và điểm
của bộ đó (run_eval.py --score) tăng.
"""


def find_textureless(images, pairs):
    """Trả về list tên ảnh trơn: ảnh có ít hơn 2 cặp khớp được chấp nhận (đồ thị khớp bậc < 2)."""
    degree = dict.fromkeys(images, 0)
    for p in pairs:
        degree[p["i"]] += 1
        degree[p["j"]] += 1
    return sorted(n for n, d in degree.items() if d < 2)


def place_textureless(images, pairs, slot_info, deadline):
    """Trả về slot_info đã cập nhật cho các ảnh trơn."""
    raise NotImplementedError("B: đặt ảnh trơn vào ô")
