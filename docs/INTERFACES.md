# Giao diện giữa các module (3 file JSON)

Đây là "hợp đồng" giữa A, B và C. Ai muốn đổi phải báo A. `schema.py` kiểm tra tự động.
Tất cả các file nằm trong thư mục làm việc `work_dir` (mặc định `work/<tên bộ>/`).

Quy ước toạ độ lấy từ `common.py`: x phải, y lên, z tiến tới; yaw > 0 quay phải, pitch > 0 nhìn lên,
roll > 0 xoay ảnh ngược chiều kim đồng hồ. Ma trận `R` là camera → world: `R = Ry(yaw) @ Rx(pitch) @ Rz(roll)`.
Chỉ số ô (slot) là chỉ số 0 đến 23 trong `common.capture_plan()`.

## 1. `matches.json`  (B ghi, A đọc)

```json
{
  "image_size": [864, 648],
  "images": ["0a5e.jpg", "1380.jpg", "..."],
  "pairs": [
    {
      "i": "0a5e.jpg",
      "j": "1380.jpg",
      "pts_i": [[x, y], [x, y]],
      "pts_j": [[x, y], [x, y]],
      "n_inliers": 57
    }
  ]
}
```

- Toạ độ điểm theo pixel của ảnh gốc 864×648 (nếu khớp ở độ phân giải thấp hơn thì nhân lại).
- `pts_i[k]` khớp với `pts_j[k]`. Chỉ gửi các điểm đã qua lọc (inlier).
- B lọc bằng RANSAC dạng không cần FOV (homography hoặc ma trận cơ bản). Phép quay chính xác và FOV do A tính.
- Chỉ ghi các cặp có đủ điểm khớp tốt (gợi ý: ≥ 15). Cặp tệ thì bỏ.

## 2. `slots.json`  (B ghi, A đọc)

```json
{
  "images": {
    "0a5e.jpg": {"slot": 3, "yaw_deg": 135.0, "pitch_deg": 0.0,
                 "confidence": 0.92, "n_matches": 120}
  }
}
```

- Đủ 24 ảnh, mỗi ảnh một `slot` khác nhau (0 đến 23). Slot 0 là ảnh 01 (hướng chính diện, chụp cẩn thận).
- `yaw_deg`, `pitch_deg` là góc danh định của ô trong mẫu chụp (lấy từ `capture_plan()`), thực tế lệch tới ±5°.
- `confidence` từ 0 đến 1. Ảnh trơn không khớp được vẫn phải có slot dự đoán, với `confidence` thấp.

## 3. `poses.json`  (A ghi, C đọc)

```json
{
  "hfov_deg": 74.2,
  "image_size": [864, 648],
  "up": [0.0, 1.0, 0.0],
  "tilt_deg_est": 0.7,
  "images": {
    "0a5e.jpg": {"R": [[1,0,0],[0,1,0],[0,0,1]], "registered": true, "source": "bundle"}
  }
}
```

- `R` đã được cân bằng ngang: dựng ERP trực tiếp từ `R` phải cho ảnh đúng chiều lên.
- `source`: `"bundle"` (từ tối ưu), `"slot_prior"` (suy từ mẫu chụp vì không khớp được), hoặc `"oracle"` (từ pose thật).
- `registered=false` nghĩa là pose chỉ là dự đoán. C vẫn nên dựng nhưng cho trọng số thấp.
- Đủ 24 ảnh. Ảnh nào cũng phải có `R`.
