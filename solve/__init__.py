"""Stage 2: tính pose của 24 ảnh. NGƯỜI PHỤ TRÁCH: A.

Đầu vào : work_dir/matches.json, work_dir/slots.json (từ B).
Đầu ra  : work_dir/poses.json (cho C): FOV chung, ma trận quay R của 24 ảnh, đã cân bằng ngang.

Ý tưởng: 24 ảnh chụp từ cùng một điểm, camera chỉ xoay. Nên mỗi ảnh chỉ khác nhau một phép quay 3D và
chung một FOV. Điểm P ở ảnh i khớp với điểm Q ở ảnh j thì  R_i @ ray(P) ≈ R_j @ ray(Q).

Thứ tự công việc (xem chú thích trong từng file):
  fov.py -> relative_rot.py -> bundle.py -> level.py
"""
import os

import common
from schema import validate_matches, validate_slots, validate_poses


def run(images_dir, work_dir, deadline):
    from . import bundle, fov, level, relative_rot  # noqa: F401

    matches = validate_matches(common.load_json(os.path.join(work_dir, "matches.json")))
    slots = validate_slots(common.load_json(os.path.join(work_dir, "slots.json")))

    hfov = fov.estimate_fov(matches, slots)
    poses, hfov = bundle.bundle_adjust(matches, slots, hfov, deadline)
    poses, up, tilt = level.level_poses(poses, slots)

    out = {
        "hfov_deg": float(hfov),
        "image_size": matches["image_size"],
        "up": [float(x) for x in up],
        "tilt_deg_est": float(tilt),
        "images": poses,
    }
    common.save_json(validate_poses(out), os.path.join(work_dir, "poses.json"))
