"""Stage 1: khớp ảnh và gán ô. NGƯỜI PHỤ TRÁCH: B.

Đầu vào : thư mục 24 ảnh (864x648, tên ngẫu nhiên, không biết thứ tự).
Đầu ra  : work_dir/matches.json và work_dir/slots.json (định dạng: docs/INTERFACES.md).

Hàm `run` dưới đây chỉ nối các bước. B viết ruột ở 4 file trong thư mục này rồi sửa `run` cho khớp.
Gọi `schema.validate_matches/validate_slots` trước khi ghi file.

MỤC TIÊU CỦA B (xem chi tiết trong chú thích đầu mỗi file):
  - Đặt được cả 24 ảnh, kể cả ảnh tường trắng, trần trơn (điểm yếu lớn nhất của cách cổ điển).
  - Gán đúng mỗi ảnh vào đúng ô của mẫu chụp (common.capture_plan()).
  - Nhanh: không khớp 276 cặp một cách mù quáng, chỉ khớp các cặp có thể chồng lên nhau.
"""
import os

import common
from schema import validate_matches, validate_slots  # noqa: F401


def run(images_dir, work_dir, deadline):
    from . import features, match, slots, textureless  # noqa: F401

    images = features.load_images(images_dir)
    feats = {name: features.extract(img) for name, img in images.items()}
    pairs = match.match_all(images, feats, deadline)
    slot_info = slots.assign_slots(images, pairs, deadline)
    slot_info = textureless.place_textureless(images, pairs, slot_info, deadline)

    h, w = next(iter(images.values())).shape[:2]
    matches = {"image_size": [w, h], "images": sorted(images), "pairs": pairs}
    common.save_json(validate_matches(matches), os.path.join(work_dir, "matches.json"))
    common.save_json(validate_slots({"images": slot_info}), os.path.join(work_dir, "slots.json"))
