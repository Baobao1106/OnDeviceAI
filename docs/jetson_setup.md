# Cài đặt, đo và chạy cả pipeline trên Jetson Orin Nano

Thay `teamN` và `<ip>` bằng tài khoản và địa chỉ Tailscale của đội. Mỗi khối lệnh ghi rõ chạy ở **laptop** hay **Jetson**:
nhầm chỗ là lỗi hay gặp nhất (xem phần "Lỗi thường gặp" cuối tài liệu).

## Nguyên tắc

- Chạy qua SSH (Tailscale), không có sudo. Jetson là ARM (aarch64), ổ đĩa 100 GB mỗi đội.
- Cài đặt và chép tệp là việc nhẹ. **Đo thời gian là việc nặng: chỉ làm trong lượt đã đặt lịch** (tối đa 3 giờ mỗi lượt),
  và dừng hết job khi hết lượt.
- **Khi chấm không có internet**: mọi tệp (kể cả trọng số mô hình) phải có sẵn trong thư mục đội.
- Không cần cài TensorRT: pipeline chạy bằng PyTorch thuần.
- Tệp đã chép sang Jetson (trọng số, bộ ảnh) nằm lại trên ổ đĩa của Jetson. Không cần chép lại ở các lượt sau, trừ khi
  có dữ liệu mới hoặc Jetson bị dọn.

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

Lưu ý: khi chạy, PyTorch có thể in cảnh báo "Found GPU0 Orin which is of compute capability (CC) 8.7 ... No published
PyTorch CUDA builds ... support this GPU". Đó chỉ là cảnh báo; GPU vẫn chạy bình thường (đã đo). Dấu hiệu GPU hỏng thật là
lỗi `no kernel image is available`, khi đó cần đổi bản torch.

## Bước 2: Lấy code và cài gói (trên Jetson)

Lấy nhánh `main` (đã gộp register, solve, render):

```bash
git clone https://github.com/Baobao1106/OnDeviceAI.git
```

Nếu Jetson không có mạng: trên laptop nén repo (bỏ `.git`, `work`, `.venv`), `scp` sang rồi giải nén.

```bash
cd OnDeviceAI
```

```bash
mamba run -n erp pip install -r requirements.txt
```

Cần mạng, vì `lightglue` được cài từ GitHub (ghim đúng commit đã đo). Hai chỗ có thể vấp:

- Nếu pip định cài lại `torch` từ PyPI (dòng `Collecting torch`), bấm `Ctrl+C` rồi cài các gói còn lại bằng danh sách đã bỏ
  dòng torch.
- Nếu pip báo `Cannot uninstall opencv-python ... installed by conda` (và có thể đã gỡ mất `scipy`), bỏ cả `opencv` khỏi
  danh sách, vì bản conda đã có sẵn và dùng được.

Một lệnh xử lý cả hai trường hợp (cũng cài lại `scipy` nếu bị gỡ):

```bash
grep -v -i -E '^(torch|opencv)' requirements.txt > /tmp/req_clean.txt && mamba run -n erp pip install -r /tmp/req_clean.txt
```

Kiểm tra torch vẫn dùng được GPU và các thư viện import được (kết quả mong đợi: `True`):

```bash
mamba run -n erp python -c "import torch, cv2, scipy, kornia, lightglue; print(torch.__version__, torch.cuda.is_available(), cv2.__version__, scipy.__version__)"
```

Nhắc lại: dấu nhắc `(base)` là môi trường gốc, không phải `erp`. Luôn dùng `mamba run -n erp ...` để chắc chạy đúng môi trường.

## Bước 3: Chép trọng số (trên laptop, PowerShell)

Hai tệp `.pth` không nằm trong git (quy tắc nhóm: không commit mô hình nặng):

| Tệp | Dung lượng |
|---|---|
| `superpoint_v1.pth` | 5,2 MB |
| `superpoint_lightglue_v0-1_arxiv.pth` | 47,5 MB |

Chạy ở **PowerShell trên laptop** (dấu nhắc kiểu `PS C:\...>`), không phải ở cửa sổ SSH vào Jetson: đường dẫn `D:\...` chỉ có
trên laptop.

```powershell
scp -r D:\OnDeviceAI\register\weights\checkpoints teamN@<ip>:~/OnDeviceAI/register/weights/
```

Lần đầu kết nối, laptop hỏi "Are you sure you want to continue connecting": gõ `yes`, rồi nhập mật khẩu của tài khoản.

Kiểm tra trên Jetson (không dùng mạng, so kích thước và SHA-256):

```bash
mamba run -n erp python -m register.fetch_weights --no-download
```

Cả hai tệp phải báo đúng. Nếu thiếu hoặc sai, **đừng chạy tiếp**: code không báo lỗi to mà rơi về chế độ dự phòng
(không có cặp khớp, độ tin cậy 0), điểm sẽ tụt mạnh mà khó nhận ra. Đổi nhánh hay `git pull` không đụng đến trọng số
(git không theo dõi chúng), nên không phải chép lại.

## Bước 4: Kiểm tra môi trường (trên Jetson)

Python từ 3.10 trở lên:

```bash
mamba run -n erp python --version
```

GPU chạy được phép tính thật (không chỉ `is_available()`):

```bash
mamba run -n erp python -c "import torch; x=torch.randn(2000,2000,device='cuda'); print(torch.cuda.get_device_name(), (x@x).sum().item())"
```

Hai bộ test (vài giây, phải đạt cả 32 test `register` và 23 test `solve`):

```bash
mamba run -n erp python -m unittest discover -s register/tests
```

```bash
mamba run -n erp python -m unittest discover -s solve/tests
```

## Bước 5: Chép bộ ảnh để đo

Bộ ảnh thật `sets/t01..t10` nằm trong git nên có sẵn sau `git clone`. Các bộ giữ riêng `sets_extra` (ảnh Laval chuyển
đổi, tạo bằng `make_set.py`) không nằm trong git; chép từ laptop khi cần (PowerShell trên laptop, khoảng 4 MB mỗi bộ):

```powershell
scp -r D:\OnDeviceAI\sets_extra teamN@<ip>:~/OnDeviceAI/
```

Kiểm tra trên Jetson:

```bash
ls sets_extra
```

```bash
ls sets_extra/x0011/public/images | wc -l
```

Lệnh thứ hai phải ra `24`. Lệnh chạy nền ghi kết quả vào thư mục `runs/`, nên tạo trước (nếu không, shell báo
`No such file or directory` và lệnh không chạy):

```bash
mkdir -p runs
```

## Bước 6: Đo thời gian (chỉ trong lượt đã đặt lịch)

### 6.1 Một bộ

```bash
time mamba run -n erp python run.py --images sets/t01/public/images --out work/t01.png --work work/full_t01
```

- Truyền `--work` rõ ràng. Nếu không, thư mục làm việc mặc định lấy tên thư mục cha của `images`, tức luôn là
  `work/public` cho mọi set.
- Thời gian từng stage nằm trong `work/full_t01/timing.json`; ảnh kết quả là `work/t01.png`.
- Chấm điểm bằng `score.py` của ban tổ chức (cần thư mục `hidden/` của bộ):

```bash
mamba run -n erp python score.py --set sets/t01 --pred work/t01.png
```

### 6.2 Nhiều bộ như ban tổ chức chấm

`run_eval.py` (nằm ở **thư mục gốc** của repo, không phải `tools/`) chạy lệnh cho từng set, đo thời gian từ lúc khởi động
đến khi thoát (kể cả nạp mô hình), dừng set nào vượt 300 s (tính là 0 điểm), và chấm điểm bằng `--score`. Chạy nền bằng
`nohup` để rớt SSH thì job vẫn chạy:

```bash
nohup mamba run -n erp python run_eval.py --sets sets --out runs/jetson_full --cmd "python run.py --images {images} --out {out}" --limit 300 --score > runs/jetson_full.txt 2>&1 &
```

Chỉ vài bộ (tiết kiệm giờ đặt lịch), ví dụ 10 bộ `x0011..x0020`:

```bash
nohup mamba run -n erp python run_eval.py --sets sets_extra --only x0011 x0012 x0013 x0014 x0015 x0016 x0017 x0018 x0019 x0020 --out runs/jetson_new10 --cmd "python run.py --images {images} --out {out}" --limit 300 --score > runs/jetson_new10.txt 2>&1 &
```

Xem tiến độ: **dùng `tail` một lần (không có `-f`)** và gõ lại sau vài phút. Lý do: sau mỗi set, `run_eval.py` dừng mọi
tiến trình mới sinh ra của cùng tài khoản, nên `tail -f` bị giết và in `Killed` (bài chạy nền vẫn sống).

```bash
tail runs/jetson_full.txt
```

Kiểm tra bài chạy còn sống:

```bash
ps aux | grep run_eval | grep -v grep
```

Khi xong, cuối tệp có dòng `mean time ... mean quality ...`. Mỗi bộ chạy khoảng 100–140 s, nên 10 bộ khoảng 20–25 phút.

Lưu ý:

- Trong lúc `run_eval.py` chạy, đừng khởi động việc khác bằng cùng tài khoản.
- Với `sets_s3d` đổi `--sets sets_s3d`: đáp án của chúng chỉ 1024x512 nên `score.py` không chấm trực tiếp được.
- Khối lệnh trên đã kích hoạt môi trường nhờ `mamba run -n erp` ở ngoài, nên `--cmd` dùng `python` được; còn chạy
  `run_eval.py` bằng Python hệ thống thì đổi `--cmd` thành
  `"mamba run -n erp python run.py --images {images} --out {out}"`.

### 6.3 Xem ảnh kết quả

Ảnh nằm ở `runs/<tên thư mục --out>/<tên bộ>.png` trên Jetson. Chép về laptop để xem (PowerShell trên laptop; chạy từ laptop
nên không bị `run_eval.py` giết):

```powershell
mkdir D:\OnDeviceAI\work\jetson_out
```

```powershell
scp -r teamN@<ip>:~/OnDeviceAI/runs/jetson_full D:\OnDeviceAI\work\jetson_out\
```

Chờ bộ đó chạy xong rồi mới chép, kẻo chép phải tệp đang ghi dở.

## Bước 7: Đọc kết quả

Thang đọc cho `run_eval.py`: mỗi dòng là một bộ, gồm trạng thái (`ok`), thời gian và `quality` (điểm chất lượng 0–100).

1. Thời gian mỗi bộ phải dưới 300 s; xem bộ **chậm nhất**, không chỉ trung bình. Dưới khoảng 230 s là an toàn; gần 300 s thì
   cần giảm thời gian (xem 8.2).
2. Điểm trung bình so với laptop để biết Jetson có làm giảm chất lượng không (xem bảng số tham chiếu).
3. Với `run.py` trực tiếp, `timing.json` cho thời gian từng stage; stage register bị chặn ở khoảng 120 s.

### Số tham chiếu (điểm chất lượng, chưa tính điểm tốc độ)

| Nơi đo | Tập bộ | Điểm TB | Thời gian TB | Chậm nhất |
|---|---|---|---|---|
| Jetson | `t01..t10` (ảnh thật) | 95,31 | 116,7 s | 139,4 s |
| Jetson | `x0011..x0020` (giữ riêng) | 94,93 | 117,2 s | 141,1 s |
| Laptop (RTX 2050) | `t01..t10` | 95,48 | khoảng 70 s | khoảng 79 s |
| Laptop (RTX 2050) | `x0001..x0010` | 97,03 | | |

Số Jetson ở trên đo trước khi bundle tăng lên 300 lần lặp (`MAX_NFEV`). Bản 300 lần lặp chậm thêm ước tính khoảng 30–40 s
trung bình trên Jetson (solve chạy trên CPU, chậm khoảng 3 lần laptop); cần đo lại trên Jetson, xem 8.2.

Stage register trên Jetson chạm trần khoảng 120 s (`REGISTER_SHARE`); điểm vẫn gần bằng laptop nên việc bị cắt giờ gần như
không làm mất chất lượng.

## Bước 8: Chỉnh khi cần

### 8.1 Trần thời gian cho register

`run.py` cho cả pipeline 240 s trên 300 s. Register chỉ được dùng một phần: `register.REGISTER_SHARE` (hiện 0,5, tức
khoảng nửa thời gian còn lại, khoảng 120 s). Tổng thời gian register + 1,3 x (solve + render) nên nhỏ hơn 240 s. Chỉnh xong
phải đo lại ít nhất vài bộ.

Hai tính năng ưu tiên khớp cặp và dành 15% hạn cho ảnh ít texture có thể tắt trong `PipelineConfig`:
`priority_order=False`, `lowtex_reserve=0.0`.

### 8.2 Số lần lặp của bundle

`solve/bundle.py` có hằng số `MAX_NFEV` (hiện 300). Tăng số lần lặp cho pose chính xác hơn ở những bộ khó (ví dụ x0014
86 lên 98 điểm) nhưng làm solve chậm hơn. Nếu bộ chậm nhất trên Jetson vượt khoảng 230 s thì hạ xuống 150 (vẫn lấy được
phần lớn lợi ích). Solve có hạn giờ riêng: hết giờ thì dừng ở nghiệm hiện có, không treo.

## Lệnh nộp cho ban tổ chức

Gửi TA: tên môi trường (`erp`) và một dòng lệnh. TA chạy trong thư mục đội với môi trường đã kích hoạt:

```
mamba run -n erp python OnDeviceAI/run.py --images {images} --out {out}
```

`run_eval.py` chỉ là công cụ để mình tự đo, không phải lệnh nộp. Kiểm tra trước khi nộp:

- Trọng số `.pth` có trong `OnDeviceAI/register/weights/checkpoints/` (không có internet khi chấm).
- Kết quả là PNG 2048x1024 RGB 8 bit; mỗi set chạy dưới 300 s trên Jetson (đo bằng `run_eval.py`).
- Mọi tệp mô hình hoặc dữ liệu mà các stage cần đều có sẵn trong thư mục đội, không tải từ mạng lúc chạy.
- Hạn nộp: thứ Sáu 2026-12-04.

## Cập nhật code về sau

Khi `main` có code mới (chạy trên Jetson):

```bash
cd ~/OnDeviceAI && git status --short && git pull origin main
```

`git status` phải sạch (không có tệp bị sửa) thì pull mới an toàn. Trọng số, `sets_extra`, `runs/` và `work/` do git không theo
dõi nên giữ nguyên. Nếu đang ở nhánh khác (ví dụ `b/register-engine`):

```bash
cd ~/OnDeviceAI && git fetch origin && git checkout main && git pull origin main
```

```bash
git branch --show-current && git log --oneline -1
```

## Thoát phiên SSH

Trước khi thoát, kiểm tra không còn bài chạy dở (không có dòng nào là ổn):

```bash
ps aux | grep -E "run_eval|run.py" | grep -v grep
```

Thoát bằng `exit` (hoặc `Ctrl+D`). Bài chạy nền bằng `nohup` vẫn tiếp tục sau khi thoát; muốn dừng hẳn thì
`pkill -u teamN -f run_eval`. Không tắt máy Jetson (máy dùng chung); tệp trên ổ đĩa vẫn còn ở lượt sau.

## Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| Gõ `scp ... D:\...` ở cửa sổ Jetson, hỏi fingerprint hoặc không thấy tệp | `scp` đang chạy trên Jetson. Bấm `Ctrl+C`, mở PowerShell **trên laptop** rồi chạy lại |
| `No such file or directory: tools/run_eval.py` | `run_eval.py` nằm ở thư mục gốc repo, không phải `tools/` |
| `runs/jetson_full.txt: No such file or directory` | Chưa tạo thư mục: chạy `mkdir -p runs` |
| `FileNotFoundError: 'sets_extra'` | Chưa chép `sets_extra` sang Jetson (bước 5) |
| `tail -f` in `Killed` | `run_eval.py` giết tiến trình mới sau mỗi bộ. Dùng `tail` không có `-f` |
| `Cannot uninstall opencv-python ... installed by conda` | Bỏ `opencv` khỏi danh sách cài (bước 2) |
| Cảnh báo compute capability 8.7 của PyTorch | Chỉ cảnh báo, GPU chạy được. Lỗi thật là `no kernel image is available` |
| Điểm tụt mạnh dù không lỗi | Có thể thiếu trọng số, chạy `fetch_weights --no-download` |
