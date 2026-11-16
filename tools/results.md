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
