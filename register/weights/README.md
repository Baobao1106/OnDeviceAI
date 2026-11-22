# Trọng số SuperPoint / LightGlue

Hai file này KHÔNG nằm trong git (`*.pth` bị `.gitignore` loại, theo quy tắc nhóm "không commit model nặng").
Chúng phải có trong `register/weights/checkpoints/` (hoặc thư mục đặt bằng biến môi trường `REGISTER_WEIGHTS`)
trước khi chạy `run.py`:

| File | Dung lượng | SHA-256 |
|---|---|---|
| `superpoint_v1.pth` | 5.2 MB | `52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e` |
| `superpoint_lightglue_v0-1_arxiv.pth` | 47.5 MB | `6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a` |

Cách lấy: từ gốc repo chạy

```bash
python -m register.fetch_weights
```

Lệnh chép từ thư mục đệm của PyTorch nếu đã có file ở đó, không thì tải từ trang phát hành chính thức của
`cvg/LightGlue` (cần mạng) và kiểm tra SHA-256. Máy không có mạng: chép cả thư mục `register/weights/` từ máy khác.

Nếu thiếu file, `register.run` không tải gì mà báo lỗi rõ một lần và rơi về phương án dự phòng (không có cặp khớp,
ô gán theo thứ tự tên file, độ tin cậy 0).
