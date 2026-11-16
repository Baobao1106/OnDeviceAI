"""Điểm vào của cả pipeline. NGƯỜI PHỤ TRÁCH: A.

    python run.py --images <thư mục 24 ảnh> --out <result.png>

Luồng: register (B) -> solve (A) -> render (C). Mỗi stage đọc/ghi JSON trong thư mục làm việc.
Định dạng: docs/INTERFACES.md.

Quy tắc an toàn (đừng bỏ):
  - Luôn ghi ra một PNG 2048x1024 hợp lệ, kể cả khi stage lỗi (sai kích thước là 0 điểm, thiếu file là lỗi).
  - Có 'deadline' (mặc định 240 giây kể từ lúc chạy). Mỗi stage phải kiểm tra time.time() < deadline và
    tự dừng sớm với kết quả tốt nhất hiện có, vì quá 300 giây là 0 điểm.
  - Stage chưa viết (NotImplementedError) thì bỏ qua, rồi dùng phương án dự phòng.

VIỆC CẦN LÀM (A):
  1. Chốt và giữ ổn định giao diện 3 file JSON (schema.py).
  2. Khi register/solve/render có code thật, kiểm tra chạy hết 10 bộ không lỗi.
  3. Khi có pose nhưng stage render lỗi, cân nhắc phương án dự phòng tốt hơn ảnh xám.
"""
import argparse
import os
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tools.bench import Timer  # noqa: E402

OUT_W, OUT_H = 2048, 1024
TIME_BUDGET_S = 240.0


def run_stage(name, fn, timer, *args):
    """Chạy một stage, trả về True nếu thành công. Không để lỗi làm sập cả pipeline."""
    try:
        with timer(name):
            fn(*args)
        return True
    except NotImplementedError as e:
        print(f"[{name}] chưa được viết: {e}", file=sys.stderr)
    except Exception:
        print(f"[{name}] LỖI:", file=sys.stderr)
        traceback.print_exc()
    return False


def write_fallback(out_png):
    """Ảnh xám đúng kích thước, để không bị 0 điểm vì sai kích thước hoặc thiếu file."""
    img = np.full((OUT_H, OUT_W, 3), 128, np.uint8)
    cv2.imwrite(out_png, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="thư mục chứa 24 ảnh")
    ap.add_argument("--out", required=True, help="đường dẫn PNG kết quả")
    ap.add_argument("--work", default=None, help="thư mục tạm (mặc định: work/<tên thư mục cha của images>)")
    ap.add_argument("--budget", type=float, default=TIME_BUDGET_S, help="giây tối đa trước khi các stage phải dừng")
    a = ap.parse_args()

    start = time.time()
    deadline = start + a.budget
    work = a.work or os.path.join("work", os.path.basename(os.path.dirname(os.path.abspath(a.images.rstrip("/")))))
    os.makedirs(work, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    import register
    import solve
    import render

    timer = Timer()
    ok = run_stage("register", register.run, timer, a.images, work, deadline)
    ok = ok and run_stage("solve", solve.run, timer, a.images, work, deadline)
    ok = ok and run_stage("render", render.run, timer, a.images, work, a.out, deadline)

    if not (ok and os.path.exists(a.out)):
        print("Dùng ảnh dự phòng (xám).", file=sys.stderr)
        write_fallback(a.out)

    img = cv2.imread(a.out)
    if img is None or img.shape[:2] != (OUT_H, OUT_W):
        print("Kết quả sai kích thước, ghi lại ảnh dự phòng.", file=sys.stderr)
        write_fallback(a.out)

    timer.report(os.path.join(work, "timing.json"))


if __name__ == "__main__":
    main()
