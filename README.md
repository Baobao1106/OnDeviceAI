# Jetson ERP: 24 ảnh phối cảnh thành 1 ảnh 360° (ERP)

Đầu vào: 24 ảnh 864×648 (không biết thứ tự, góc, FOV).
Đầu ra: 1 ảnh ERP 2048×1024 PNG, đúng chiều ngang (nghiêng ≥ 5° là 0 điểm).
Chạy trên Jetson Orin Nano Super 8GB, tối đa 300 giây mỗi bộ (quá là 0 điểm).
Điểm = 0,8 × chất lượng (PSNR 50 + LPIPS 50) + tốc độ (20).

## Sơ đồ thư mục và người phụ trách

```
run.py              A   điểm vào, nối 3 module, bắt lỗi, luôn xuất PNG hợp lệ
schema.py           A   kiểm tra định dạng 3 file JSON giữa các module
register/           B   tìm điểm khớp, gán ảnh vào ô của mẫu chụp
  features.py           trích đặc trưng
  match.py              khớp cặp ảnh
  slots.py              gán 24 ảnh vào 24 ô của mẫu chụp
  textureless.py        xử lý ảnh trơn (tường, trần trắng)
solve/              A   tính pose
  relative_rot.py       phép quay giữa 2 ảnh từ điểm khớp
  fov.py                ước lượng FOV chung
  bundle.py             tối ưu toàn cục
  level.py              cân bằng ngang
render/             C   dựng ảnh ERP
  vignette.py           bù vignetting
  gain.py               cân độ sáng và màu giữa các ảnh
  warp.py               chiếu 1 ảnh lên ERP
  blend.py              hòa trộn các lớp
  inpaint.py            lấp chỗ trống
tools/              D   đo đạc
  bench.py              đo thời gian từng khâu
  pose_diff.py          so poses.json với pose thật
  gen_sets.py           tạo thêm bộ train từ ERP công khai
  oracle_render.py      dựng ảnh bằng pose thật (để C kiểm thử một mình)
  results.md            bảng điểm theo phiên bản
common.py, make_set.py, score.py, run_eval.py   có sẵn của ban tổ chức, KHÔNG SỬA
```

Luồng dữ liệu (chi tiết định dạng trong `docs/INTERFACES.md`):

```
ảnh -> register/ -> matches.json + slots.json -> solve/ -> poses.json -> render/ -> result.png
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Dữ liệu cuộc thi (`sets/`, `panos/`) là dữ liệu riêng của ban tổ chức, chỉ dùng cho cuộc thi và
không được công bố ra ngoài. Vì vậy chúng nằm trong `.gitignore`: giải nén file zip của ban tổ chức
rồi chép thư mục `sets/` và `panos/` vào gốc repo trên máy mình, không đẩy lên git.

## Chạy

```bash
# một bộ
python run.py --images sets/t01/public/images --out runs/dev/t01.png

# cả 10 bộ, có đo thời gian và chấm điểm
python run_eval.py --sets sets --out runs/dev \
    --cmd "python run.py --images {images} --out {out}" --score
```

Hiện tại các module còn là khung rỗng: `run.py` chạy được nhưng sẽ xuất ảnh xám vì chưa có ai viết
phần của mình. Mỗi file có chú thích ở đầu nói rõ người nào làm gì, vào ra là gì, khi nào gọi là xong.

Kiểm thử quy ước toạ độ và phần dựng ảnh khi chưa có pose thật từ A và B:

```bash
python tools/oracle_render.py --set sets/t01 --out runs/oracle/t01.png
```

## Quy tắc làm việc nhóm

1. `main` luôn chạy được và không bao giờ làm hỏng điểm. Mỗi người làm trên nhánh riêng
   (`a/...`, `b/...`, `c/...`, `d/...`) rồi gửi pull request nhỏ vào `main`.
2. Chỉ sửa thư mục của mình. Cần đổi file của người khác thì nhắn người đó.
3. Các module chỉ nói chuyện với nhau qua 3 file JSON (`docs/INTERFACES.md`). Muốn đổi định dạng thì
   báo A để cập nhật `schema.py` và thông báo cho cả nhóm.
4. Trước khi merge: chạy `run_eval.py` trên 10 bộ, ghi điểm vào `tools/results.md`. Không merge nếu điểm
   trung bình giảm hoặc có bộ lỗi, timeout hay nghiêng.
5. Mọi stage phải kiểm tra `time.time() < deadline` và có phương án dự phòng. `run.py` tự cắt ở
   khoảng 240 giây.
6. Không commit ảnh kết quả, `runs/`, `work/`, hay model nặng. Chúng đã nằm trong `.gitignore`.
