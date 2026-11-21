# Bảng điểm theo phiên bản

NGƯỜI PHỤ TRÁCH: D. Sau mỗi lần merge vào `main`, thêm một dòng. Lệnh tạo số liệu:

```bash
python run_eval.py --sets sets --out runs/<tên phiên bản> \
    --cmd "python run.py --images {images} --out {out}" --score
```

Đọc `runs/<tên>/summary.json` (mean_quality, mean_seconds). Điểm cuối = 0,8 × chất lượng + điểm tốc độ,
tốc độ so với đội nhanh nhất (20·log(300/t)/log(300/t_nhanh_nhất)).

| Ngày | Phiên bản (commit) | Ai | Chất lượng TB | Thời gian TB (s) | Bộ tệ nhất | Ghi chú |
|---|---|---|---|---|---|---|
| | khung rỗng (ảnh xám) | | | | | điểm gốc thấp nhất |
| 2026-11-21 | `b/register-engine` @ 3135bfc + solve (chưa commit) + render cơ sở | A/B | 76,79 | 67,6 | t01 (69,26) | 10 bộ thật t01-t10, laptop RTX 2050; chưa đo trên Jetson; chưa tính điểm tốc độ |

## Điểm từng bộ, cả pipeline (2026-11-21)

`b/register-engine` @ 3135bfc (register bật ưu tiên + dành 15%) + solve mới (chưa commit) + render cơ sở, hạn mặc định 240 s,
laptop RTX 2050 (chưa phải Jetson). Mỗi bộ chạy `run.py` rồi `score.py` (cột "Điểm" là điểm chất lượng 0 đến 100).

| Bộ | Thời gian (s) | Register | Solve | Render | PSNR (dB) | LPIPS | Nghiêng (°) | Điểm |
|---|---|---|---|---|---|---|---|---|
| t01 | 77,3 | 67,6 | 0,7 | 8,0 | 25,10 | 0,0599 | 0,34 | 69,26 |
| t02 | 57,0 | 46,9 | 1,4 | 7,8 | 25,55 | 0,0473 | 0,78 | 71,64 |
| t03 | 65,3 | 54,8 | 1,8 | 7,9 | 26,78 | 0,0345 | 0,55 | 76,01 |
| t04 | 57,7 | 47,5 | 1,4 | 7,9 | 25,71 | 0,0397 | 0,38 | 72,79 |
| t05 | 61,5 | 51,1 | 1,7 | 7,8 | 28,39 | 0,0290 | 0,69 | 80,56 |
| t06 | 79,1 | 68,4 | 1,8 | 8,0 | 27,97 | 0,0393 | 0,39 | 78,49 |
| t07 | 70,4 | 59,7 | 1,9 | 7,9 | 27,94 | 0,0272 | 0,83 | 79,64 |
| t08 | 65,3 | 54,7 | 1,8 | 7,8 | 29,29 | 0,0274 | 1,12 | 82,98 |
| t09 | 69,1 | 58,9 | 0,9 | 8,3 | 28,82 | 0,0333 | 0,37 | 81,23 |
| t10 | 73,4 | 63,6 | 1,0 | 7,8 | 26,40 | 0,0316 | 0,96 | 75,34 |
| **TB** | **67,6** | | | | **27,19** | **0,0369** | max 1,12 | **76,79** |

Bộ thấp nhất `t01`: ảnh ở cực chỉ có một cặp yếu nên solve quay sai hướng (lệch 178° quanh trục đứng). Chạy render bằng pose thật
(`tools/oracle_render.py`, chấm bằng `score.py`) cho `t01` được 75,04 và `t04` được 73,42, tức trần của render cơ sở khoảng 73 đến 75.

## Trần điểm của stage render (pose thật)

Chạy `python tools/oracle_render.py --set sets/tXX --out runs/oracle/tXX.png`.

| Bộ | PSNR nhanh (dB) | Ghi chú |
|---|---|---|
| t01 | | |

## Sai số pose so với pose thật

Chạy `python tools/pose_diff.py --poses work/tXX/poses.json --truth sets/tXX/hidden/true_poses.json`.

| Bộ | FOV lệch (°) | Pose TB (°) | Pose max (°) | Số ảnh gán đúng ô |
|---|---|---|---|---|
| t01 | | | | |
