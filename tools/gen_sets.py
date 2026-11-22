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
