# Cài đặt và đo stage 1 (`register`) trên Jetson Orin Nano

Hướng dẫn này chỉ cho stage `register`. Stage `solve` và `render` chưa có nên chưa đo được cả pipeline.
Thay `teamN` và `<ip>` bằng tài khoản và địa chỉ Tailscale của đội.

## Nguyên tắc

- Chạy qua SSH (Tailscale), không có sudo. Jetson là ARM (aarch64), ổ đĩa 100 GB mỗi đội.
- Cài đặt và chép tệp là việc nhẹ. **Đo thời gian là việc nặng: chỉ làm trong lượt đã đặt lịch** (tối đa 3 giờ mỗi lượt),
  và dừng hết job khi hết lượt.
- **Khi chấm không có internet**: mọi tệp (kể cả trọng số mô hình) phải có sẵn trong thư mục đội.
- Không cần cài TensorRT: stage 1 chạy bằng PyTorch thuần.

## Bước 1: Môi trường (trên Jetson)

```bash
ssh teamN@<ip>
```

Cài Miniforge nếu chưa có (theo slide của ban tổ chức):

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
```

```bash
bash Miniforge3-Linux-aarch64.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && exec bash
```

Tạo môi trường (chỉ Python; các gói còn lại cài bằng pip ở bước 2 để không bị cài chồng `numpy`/`opencv`):

```bash
mamba create -n erp python=3.12 -y
```

Cài PyTorch GPU từ index `cu130` (khuyến nghị CUDA 13 theo slide):

```bash
mamba run -n erp pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Bước 2: Lấy code và cài gói (trên Jetson)

```bash
git clone -b b/register-engine https://github.com/Baobao1106/OnDeviceAI.git
```

Nếu Jetson không có mạng: trên laptop nén repo (bỏ `.git`, `work`, `.venv`), `scp` sang rồi giải nén.

```bash
cd OnDeviceAI
```

```bash
mamba run -n erp pip install -r requirements.txt
```

Cần mạng, vì `lightglue` được cài từ GitHub (ghim đúng commit đã đo). Dòng `torch>=2.0` trong `requirements.txt`
đã thoả bởi bản cài ở bước 1; nếu pip định cài lại torch từ PyPI thì dừng lại và cài các gói còn lại từng dòng,
bỏ dòng torch.

## Bước 3: Chép trọng số (trên laptop, PowerShell)

Hai tệp `.pth` không nằm trong git (quy tắc nhóm: không commit mô hình nặng):

| Tệp | Dung lượng |
|---|---|
| `superpoint_v1.pth` | 5,2 MB |
| `superpoint_lightglue_v0-1_arxiv.pth` | 47,5 MB |

```powershell
scp -r D:\OnDeviceAI\register\weights\checkpoints teamN@<ip>:~/OnDeviceAI/register/weights/
```

Kiểm tra trên Jetson (không dùng mạng, so kích thước và SHA-256):

```bash
mamba run -n erp python -m register.fetch_weights --no-download
```

Cả hai tệp phải báo đúng. Nếu thiếu hoặc sai, **đừng chạy tiếp**: code không báo lỗi to mà rơi về chế độ dự phòng
(không có cặp khớp, độ tin cậy 0), điểm sẽ tụt mạnh mà khó nhận ra.

## Bước 4: Kiểm tra môi trường (trên Jetson)

Python từ 3.10 trở lên:

```bash
mamba run -n erp python --version
```

GPU chạy được phép tính thật (không chỉ `is_available()`):

```bash
mamba run -n erp python -c "import torch; x=torch.randn(2000,2000,device='cuda'); print(torch.cuda.get_device_name(), (x@x).sum().item())"
```

Bộ test (vài giây, phải đạt cả 32 test):

```bash
mamba run -n erp python -m unittest discover -s register/tests
```

Torch trên laptop là 2.9.1+cu126, trên Jetson là bản CUDA 13: bước này để chắc hai bản cho cùng kết quả.

## Bước 5: Đo thời gian (chỉ trong lượt đã đặt lịch)

Chạy 5 set đại diện (`t01`, `t04`, `s20`, `s34`, `s35`) bằng cấu hình mặc định, **không thêm cờ hạn**. `nohup` để rớt SSH
thì job vẫn chạy:

```bash
nohup mamba run -n erp python tools/run_register_all.py --only t01 t04 s20 s34 s35 --work_name jetson --out work/jetson_check.json > work/jetson_check.txt 2>&1 &
```

Xem tiến độ (mỗi set xong thêm một dòng):

```bash
tail -f work/jetson_check.txt
```

Dự kiến: nếu Jetson chậm khoảng 4 lần, mỗi set 3–4 phút (có trần hạn), 5 set khoảng 20 phút, cộng một lần nạp mô hình.

Tuỳ chọn khi còn dư giờ, đo bản chỉ RootSIFT để biết Jetson nhanh cỡ nào khi không có LightGlue:

```bash
REGISTER_SIFT_ONLY=1 mamba run -n erp python tools/run_register_all.py --only t01 t04 s20 --work_name jetson_sift --out work/jetson_sift.json
```

## Bước 6: Đọc kết quả

Mỗi dòng của `work/jetson_check.txt` là một set:

| Cột | Nghĩa |
|---|---|
| thời gian (`39.0 s`) | thời gian stage 1 của set đó |
| `placed` | số ảnh (trong 24) được xác định vị trí; 24 là đủ |
| `layer` | số ảnh đã xếp được gán đúng tầng góc ngẩng |
| `exact(placed)` | số ảnh đã xếp được gán đúng ô trong 24 ô kế hoạch chụp |
| `exact(all)` | số ảnh (trong 24) được gán đúng ô, kể cả ảnh chưa xếp; tối đa 24, đây là số chấm chính |
| `yaw<=10` | số ảnh đã xếp có góc xoay ngang lệch đáp án không quá 10° / số ảnh đem so |

Cần chú ý:

1. Thời gian mỗi set so với trần của stage 1 (khoảng 120 s). Dưới 120 s: khớp đủ 276 cặp như trên laptop.
   Gần hoặc trên 120 s: máy đang ở chế độ thiếu giờ (hai tính năng ưu tiên cặp và dành 15% hạn cho ảnh ít texture
   phát huy tác dụng).
2. `placed` và `exact(all)` để biết đã mất bao nhiêu ảnh vì thiếu giờ.
3. Dòng `TOTAL` ở cuối (thời gian trung bình, trung vị, lớn nhất, tổng ô đúng).
4. Phần vượt hạn: trên laptop, set chậm nhất từng vượt trần khoảng 9 s sau khi hết hạn; trên Jetson cần xem phần này
   có lớn hơn không, vì hạn 240 s còn phải chia cho `solve` và `render`.

## Số tham chiếu trên laptop (RTX 2050, 50 set)

| Hạn | Ô đúng (tối đa 1200) | Thời gian TB mỗi set |
|---|---|---|
| 240 s (ngân sách thật của `run.py`) | 1200 | 73,2 s |
| 75 s (mô phỏng máy chậm khoảng 4 lần) | 1190 | 41,6 s |

Hạn 240 và 75 là tham số mô phỏng trên laptop (`--deadline`), không phải hai phiên bản code. Trên Jetson chạy bình thường,
máy tự rơi vào chế độ phù hợp.

## Tắt các tính năng mới nếu cần

Nếu số đo trên Jetson cho thấy cần đổi, đặt trong `PipelineConfig`:
`priority_order=False` (khớp cặp theo thứ tự ngẫu nhiên như cũ), `lowtex_reserve=0.0` (không dành hạn riêng cho lượt
ghép lại ảnh ít texture). Công cụ `tools/run_register_all.py` có cờ `--no_priority` và `--reserve`.

---

# Phần 2: Chạy cả pipeline khi đã gộp `register` + `solve` + `render`

Dùng khi ba nhánh (`b/register-engine`, `a/geometry`, `c/render`) đã được gộp vào một nhánh (ví dụ `main`).
`run.py` chạy lần lượt register, solve, render; stage nào lỗi hoặc chưa viết thì pipeline ghi ảnh xám dự phòng 2048x1024
để không bị 0 điểm vì thiếu hoặc sai kích thước.

## 2.1 Cập nhật code và gói trên Jetson

Lấy nhánh đã gộp (thay `main` bằng nhánh thực tế):

```bash
cd ~/OnDeviceAI && git fetch origin && git checkout main && git pull
```

Các bạn phụ trách `solve` và `render` có thể thêm thư viện mới, nên cài lại (cần mạng):

```bash
mamba run -n erp pip install -r requirements.txt
```

Kiểm tra lại trọng số và bộ test như bước 3 và 4 ở trên (nếu trọng số bị mất, pipeline rơi về chế độ dự phòng mà
không báo lỗi to).

## 2.2 Chạy thử một set

```bash
mamba run -n erp python run.py --images sets/t01/public/images --out work/full/t01.png --work work/full_t01
```

- Truyền `--work` rõ ràng. Nếu không, thư mục làm việc mặc định lấy tên thư mục cha của `images`, tức luôn là
  `work/public` cho mọi set.
- Thời gian từng stage nằm trong `work/full_t01/timing.json`; ảnh kết quả là `work/full/t01.png`.
- Đo thời gian thật của `solve` và `render` ở đây: cần để chỉnh trần thời gian của stage 1 (xem 2.5).

## 2.3 Chạy nhiều set như cách ban tổ chức chấm

`run_eval.py` chạy lệnh cho từng set, đo thời gian từ lúc khởi động đến khi thoát (kể cả nạp mô hình), dừng set nào
vượt 300 s (tính là 0 điểm), và có thể chấm điểm bằng `--score`:

```bash
nohup mamba run -n erp python run_eval.py --sets sets --out runs/jetson_full --cmd "python run.py --images {images} --out {out}" --limit 300 --score > runs/jetson_full.txt 2>&1 &
```

Xem tiến độ:

```bash
tail -f runs/jetson_full.txt
```

Lưu ý:

- Sau mỗi set, `run_eval.py` **dừng mọi tiến trình mới sinh ra của cùng tài khoản**. Trong lúc nó chạy, đừng khởi động
  việc khác bằng cùng tài khoản.
- `--only t01 t04 s20` để chỉ chạy vài set (tiết kiệm giờ đặt lịch). Với `sets_s3d` đổi `--sets sets_s3d`.
- Thư mục làm việc mặc định là `work/public` cho mọi set; các set chạy tuần tự nên không đè nhau.
  `run.py` bỏ qua `solve`/`render` khi `register` lỗi, nên không dùng nhầm dữ liệu set trước.
- Khối lệnh trên đã kích hoạt môi trường nhờ `mamba run -n erp` ở ngoài. Nếu `run_eval.py` bị bọc trong `mamba run`
  thì lệnh trong `--cmd` dùng `python` được; còn chạy `run_eval.py` bằng Python hệ thống thì đổi `--cmd` thành
  `"mamba run -n erp python run.py --images {images} --out {out}"`.

## 2.4 Lệnh nộp cho ban tổ chức

Gửi TA: tên môi trường (`erp`) và một dòng lệnh. TA chạy trong thư mục đội với môi trường đã kích hoạt:

```
mamba run -n erp python OnDeviceAI/run.py --images {images} --out {out}
```

Kiểm tra trước khi nộp:

- Trọng số `.pth` có trong `OnDeviceAI/register/weights/checkpoints/` (không có internet khi chấm).
- Kết quả là PNG 2048x1024 RGB 8 bit; mỗi set chạy dưới 300 s trên Jetson (đo bằng `run_eval.py` ở 2.3).
- Mọi tệp mô hình hoặc dữ liệu mà `solve`/`render` cần đều có sẵn trong thư mục đội, không tải từ mạng lúc chạy.
- Hạn nộp: thứ Sáu 2026-12-04.

## 2.5 Chỉnh trần thời gian cho stage 1

`run.py` cho cả pipeline 240 s trên 300 s. Stage 1 chỉ được dùng một phần: `register.REGISTER_SHARE` (hiện 0,5, tức
khoảng nửa thời gian còn lại, khoảng 120 s). Sau khi đo `solve` và `render` trên Jetson (2.2), nếu tổng hai stage đó là
`T` giây, mức an toàn thô là: thời gian stage 1 + 1,3 x `T` nhỏ hơn 240 s. Nếu `T` lớn thì giảm `REGISTER_SHARE` để
dành chỗ cho chúng; nếu `T` nhỏ thì có thể tăng để stage 1 khớp được nhiều cặp hơn. Chỉnh xong phải đo lại ít nhất vài
set bằng 2.3.
