"""Stage 3: dựng ảnh ERP 2048x1024. NGƯỜI PHỤ TRÁCH: C.

Đầu vào : thư mục ảnh + work_dir/poses.json (từ A: FOV chung, ma trận quay R của từng ảnh, đã cân bằng ngang).
Đầu ra  : ảnh PNG 2048x1024 (đường dẫn `out_png`).

Điểm PSNR và LPIPS được đo ở stage này nên đường nối, màu, độ nét quyết định phần lớn điểm.

Đã có bản chạy được đơn giản để C bắt đầu từ đó (bù vignetting -> chiếu lên ERP -> trung bình có trọng số ->
lấp lỗ). C cải thiện từng bước trong 5 file của thư mục này:
  vignette.py  ->  gain.py  ->  warp.py  ->  blend.py  ->  inpaint.py

Cách kiểm thử một mình, không cần chờ A và B (dùng pose thật):
    python tools/oracle_render.py --set sets/t01 --out runs/oracle/t01.png
Điểm của lệnh này là TRẦN ĐIỂM của stage render: nếu thấp thì lỗi nằm ở đây, không phải ở pose.

Lưu ý bộ nhớ: Jetson chỉ có 8GB dùng chung CPU và GPU. Đừng giữ 24 ảnh float64 2048x1024 cùng lúc
(mỗi ảnh ~50MB). Dưới đây lớp ảnh lưu uint8 và trọng số float32.
"""
import os
import time

import cv2
import numpy as np

import common
from schema import validate_poses

OUT_W, OUT_H = 2048, 1024


def run(images_dir, work_dir, out_png, deadline):
    from . import blend, gain, inpaint, vignette, warp

    poses = validate_poses(common.load_json(os.path.join(work_dir, "poses.json")))
    hfov = poses["hfov_deg"]

    layers, weights, names = [], [], []
    for name, p in sorted(poses["images"].items()):
        if time.time() > deadline:
            break
        img = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
        img = vignette.remove_vignette(img)
        R = np.asarray(p["R"], float)
        erp, wgt = warp.warp_to_erp(img, R, hfov, OUT_W, OUT_H)
        if not p.get("registered", True):
            wgt = wgt * 0.25  # pose chỉ là dự đoán: cho trọng số thấp
        layers.append(np.clip(erp + 0.5, 0, 255).astype(np.uint8))
        weights.append(wgt.astype(np.float32))
        names.append(name)

    gains = gain.estimate_gains(layers, weights)  # [N,3], mặc định toàn 1
    layers = [np.clip(l.astype(np.float32) * g[None, None, :] + 0.5, 0, 255).astype(np.uint8)
              for l, g in zip(layers, gains)]

    erp, covered = blend.blend(layers, weights)
    erp = inpaint.fill_holes(erp, covered)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    cv2.imwrite(out_png, erp)
