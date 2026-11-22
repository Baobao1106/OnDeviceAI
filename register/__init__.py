"""Stage 1: khớp ảnh và gán ô. NGƯỜI PHỤ TRÁCH: B.

Đầu vào : thư mục 24 ảnh (864x648, tên ngẫu nhiên, không biết thứ tự).
Đầu ra  : work_dir/matches.json và work_dir/slots.json (định dạng: docs/INTERFACES.md).

Hàm `run` dưới đây chỉ nối các bước. B viết ruột ở 4 file trong thư mục này rồi sửa `run` cho khớp.
Gọi `schema.validate_matches/validate_slots` trước khi ghi file.

MỤC TIÊU CỦA B (xem chi tiết trong chú thích đầu mỗi file):
  - Đặt được cả 24 ảnh, kể cả ảnh tường trắng, trần trơn (điểm yếu lớn nhất của cách cổ điển).
  - Gán đúng mỗi ảnh vào đúng ô của mẫu chụp (common.capture_plan()).
  - Nhanh: không khớp 276 cặp một cách mù quáng, chỉ khớp các cặp có thể chồng lên nhau.

BỀN LỖI VÀ HẾT GIỜ: `run` luôn ghi ra `matches.json` và `slots.json` hợp lệ (trừ khi không đọc được ảnh).
  - Một bước lỗi: ảnh trích đặc trưng lỗi thì coi là ảnh cô lập; một cặp khớp lỗi thì bỏ cặp đó; `match_all` lỗi thì
    không có cặp nào; `assign_slots` lỗi thì gán ô theo thứ tự tên file (confidence 0). `solve` vẫn chạy được và
    đặt các ảnh theo góc danh định của ô.
  - Hết giờ: stage này chỉ dùng REGISTER_SHARE thời gian còn lại, để lại phần cho các stage sau. Quá hạn thì dừng
    trích đặc trưng / khớp thêm, giữ những gì đã có (các cặp được khớp theo thứ tự ngẫu nhiên nên mọi ảnh đều có
    một phần cặp).
"""
import os
import sys
import time
import traceback

import common
from schema import validate_matches, validate_slots  # noqa: F401

REGISTER_SHARE = 0.5     # phần thời gian còn lại mà stage này được dùng; phần còn lại nhường cho solve và render


def _safe(stage, fn, default):
    """Chạy `fn`; lỗi thì ghi traceback ra stderr và trả về `default` (không để một bước làm hỏng cả stage)."""
    try:
        return fn()
    except Exception:
        print(f"[register/{stage}] LỖI, dùng phương án dự phòng:", file=sys.stderr)
        traceback.print_exc()
        return default


def default_slots(names):
    """Phương án dự phòng: ảnh theo thứ tự tên file vào ô 0, 1, 2, ... (không có thông tin nên confidence = 0)."""
    plan = common.capture_plan()
    return {n: {"slot": k, "yaw_deg": plan[k][0], "pitch_deg": plan[k][1], "confidence": 0.0, "n_matches": 0}
            for k, n in enumerate(sorted(names))}


def run(images_dir, work_dir, deadline):
    from . import features, match, slots, textureless  # noqa: F401

    images = features.load_images(images_dir)           # không đọc được ảnh thì không có gì để làm: để lỗi nổi lên
    now = time.time()
    stage_deadline = now + REGISTER_SHARE * max(deadline - now, 0.0)

    feats = {}
    for name, img in images.items():
        if time.time() >= stage_deadline:
            print(f"[register/features] hết giờ, bỏ {len(images) - len(feats)} ảnh chưa trích đặc trưng", file=sys.stderr)
            break
        got = _safe(f"extract {name}", lambda: features.extract(img), None)
        if got is not None:
            feats[name] = got

    pairs = _safe("match", lambda: match.match_all(images, feats, stage_deadline), [])
    h, w = next(iter(images.values())).shape[:2]
    matches = {"image_size": [w, h], "images": sorted(images), "pairs": pairs}
    if _safe("validate_matches", lambda: validate_matches(matches), None) is None:
        pairs = []
        matches["pairs"] = pairs

    slot_info = _safe("slots", lambda: slots.assign_slots(images, pairs, deadline), None)
    if slot_info is not None:
        slot_info = _safe("textureless", lambda: textureless.place_textureless(images, pairs, slot_info, deadline),
                          slot_info)
    if slot_info is None or _safe("validate_slots", lambda: validate_slots({"images": slot_info}), None) is None:
        slot_info = default_slots(images)

    common.save_json(matches, os.path.join(work_dir, "matches.json"))
    common.save_json(validate_slots({"images": slot_info}), os.path.join(work_dir, "slots.json"))
