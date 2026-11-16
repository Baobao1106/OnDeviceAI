"""Tạo thêm bộ train từ ảnh ERP công khai. NGƯỜI PHỤ TRÁCH: D.

    python tools/gen_sets.py --erp_dir extra_panos --out sets_extra --per_pano 3 --seed0 5000

Với mỗi ảnh ERP trong `--erp_dir`, gọi `make_set.py` với nhiều seed khác nhau (đổi hướng chính diện, FOV 72–77°,
độ rung tay, ánh sáng). Tên bộ: x0001, x0002, ... Kết quả cùng cấu trúc với sets/ nên `run_eval.py --sets sets_extra`
dùng được ngay.

Nguồn ERP (kiểm tra giấy phép từng tập, đa số chỉ cho nghiên cứu và phải xin quyền):
  - Structured3D: có sẵn ảnh ERP (dễ nhất).
  - Matterport3D: đổi skybox sang ERP.
  - Replica, HM3D: dựng ERP bằng Habitat-sim.
Ưu tiên cảnh trơn (tường trắng, ít hoa văn) vì đó là điểm yếu chính của bài toán.

VIỆC CẦN LÀM (D):
  1. Lấy 30–100 ảnh ERP phòng trong nhà, chia hai phần: phần phát triển và phần kiểm chứng (không dùng để chỉnh tham số).
  2. Chạy script này để tạo bộ.
  3. Lưu ý: nhiều bộ từ cùng một ERP không tăng độ đa dạng cảnh. Chỉ thêm độ đa dạng về góc chụp, rung tay, ánh sáng.
  4. Dữ liệu của ban tổ chức (sets/, panos/) không được đẩy lên git hay công bố ra ngoài.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--erp_dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per_pano", type=int, default=1)
ap.add_argument("--seed0", type=int, default=5000)
a = ap.parse_args()

erps = sorted(f for f in os.listdir(a.erp_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
k = 0
for f in erps:
    for _ in range(a.per_pano):
        k += 1
        dst = os.path.join(a.out, f"x{k:04d}")
        subprocess.run([sys.executable, os.path.join(ROOT, "make_set.py"), "--erp", os.path.join(a.erp_dir, f),
                        "--out", dst, "--seed", str(a.seed0 + k)], check=True, cwd=ROOT)
print(f"đã tạo {k} bộ trong {a.out}")
