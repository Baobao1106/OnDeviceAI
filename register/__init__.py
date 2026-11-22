import os
import sys
import time
import traceback

import common
from schema import validate_matches, validate_slots  # noqa: F401

REGISTER_SHARE = 0.5


def _safe(stage, fn, default):
    try:
        return fn()
    except Exception:
        print(f"[register/{stage}] LỖI, dùng phương án dự phòng:", file=sys.stderr)
        traceback.print_exc()
        return default


def default_slots(names):
    plan = common.capture_plan()
    return {n: {"slot": k, "yaw_deg": plan[k][0], "pitch_deg": plan[k][1], "confidence": 0.0, "n_matches": 0}
            for k, n in enumerate(sorted(names))}


def run(images_dir, work_dir, deadline):
    from . import features, match, slots, textureless  # noqa: F401

    images = features.load_images(images_dir)
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
