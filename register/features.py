import os

import cv2


def load_images(images_dir):
    names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    out = {}
    for n in names:
        img = cv2.imread(os.path.join(images_dir, n), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"không đọc được {n}")
        out[n] = img
    return out


def extract(img_bgr):
    import numpy as np

    from .engine import get_engine
    feats = get_engine().extract(img_bgr)
    kp = next((f["kp"] for f in feats if isinstance(f, dict) and "kp" in f), np.zeros((0, 2), np.float32))
    return kp, feats
