"""Kiểm tra định dạng 3 file JSON giữa các module. NGƯỜI PHỤ TRÁCH: A.

Định dạng đầy đủ: docs/INTERFACES.md.

Cách dùng (mỗi module gọi ngay trước khi ghi file và ngay sau khi đọc file):
    from schema import validate_matches, validate_slots, validate_poses
    validate_poses(d)          # raise ValueError nếu sai, nói rõ sai ở đâu

VIỆC CẦN LÀM (A): nếu đổi định dạng JSON thì sửa ở đây và báo cả nhóm.
"""
import numpy as np


def _need(d, key, where):
    if key not in d:
        raise ValueError(f"{where}: thiếu khóa '{key}'")
    return d[key]


def validate_matches(d):
    _need(d, "image_size", "matches")
    names = _need(d, "images", "matches")
    for k, p in enumerate(_need(d, "pairs", "matches")):
        for key in ("i", "j", "pts_i", "pts_j", "n_inliers"):
            _need(p, key, f"matches.pairs[{k}]")
        if p["i"] not in names or p["j"] not in names:
            raise ValueError(f"matches.pairs[{k}]: tên ảnh không có trong 'images'")
        if len(p["pts_i"]) != len(p["pts_j"]):
            raise ValueError(f"matches.pairs[{k}]: pts_i và pts_j khác độ dài")
    return d


def validate_slots(d):
    imgs = _need(d, "images", "slots")
    used = set()
    for name, s in imgs.items():
        for key in ("slot", "yaw_deg", "pitch_deg", "confidence"):
            _need(s, key, f"slots.images[{name}]")
        if not 0 <= s["slot"] < 24:
            raise ValueError(f"slots.images[{name}]: slot phải từ 0 đến 23")
        if s["slot"] in used:
            raise ValueError(f"slots.images[{name}]: slot {s['slot']} bị trùng")
        used.add(s["slot"])
    return d


def validate_poses(d):
    _need(d, "hfov_deg", "poses")
    _need(d, "image_size", "poses")
    for name, p in _need(d, "images", "poses").items():
        R = np.asarray(_need(p, "R", f"poses.images[{name}]"), float)
        if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
            raise ValueError(f"poses.images[{name}]: R phải là ma trận quay 3x3")
        _need(p, "registered", f"poses.images[{name}]")
    return d
