import os

import common
from schema import validate_matches, validate_slots, validate_poses


def run(images_dir, work_dir, deadline):
    import sys
    import traceback

    from . import bundle, fov, level, place, relative_rot  # noqa: F401

    matches = validate_matches(common.load_json(os.path.join(work_dir, "matches.json")))
    slots = validate_slots(common.load_json(os.path.join(work_dir, "slots.json")))

    hfov = fov.estimate_fov(matches, slots)
    poses, hfov = bundle.bundle_adjust(matches, slots, hfov, deadline)
    try:
        poses = place.place_unregistered(images_dir, poses, slots, hfov, deadline)
    except Exception:
        print("[solve/place] LỖI, giữ vị trí dự phòng:", file=sys.stderr)
        traceback.print_exc()
    poses, up, tilt = level.level_poses(poses, slots)

    out = {
        "hfov_deg": float(hfov),
        "image_size": matches["image_size"],
        "up": [float(x) for x in up],
        "tilt_deg_est": float(tilt),
        "images": poses,
    }
    common.save_json(validate_poses(out), os.path.join(work_dir, "poses.json"))
