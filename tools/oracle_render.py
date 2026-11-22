import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402
import render  # noqa: E402


def write_oracle_poses(set_dir, work_dir):
    truth = common.load_json(os.path.join(set_dir, "hidden", "true_poses.json"))
    imgs = {}
    for s in truth["shots"]:
        R = common.pose_to_R(s["yaw_deg"], s["pitch_deg"], s["roll_deg"])
        imgs[s["image"]] = {"R": R.tolist(), "registered": True, "source": "oracle"}
    poses = {"hfov_deg": truth["hfov_deg"], "image_size": [864, 648], "up": [0, 1, 0], "tilt_deg_est": 0.0,
             "images": imgs}
    os.makedirs(work_dir, exist_ok=True)
    common.save_json(poses, os.path.join(work_dir, "poses.json"))


def quick_psnr(pred_png, set_dir):
    gt = cv2.imread(os.path.join(set_dir, "hidden", "gt.png")).astype(np.float64)
    mask = cv2.imread(os.path.join(set_dir, "hidden", "mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    pred = cv2.imread(pred_png).astype(np.float64)
    if pred.shape != gt.shape:
        return None
    for c in range(3):
        p, t = pred[..., c][mask], gt[..., c][mask]
        pred[..., c] *= (p * t).sum() / max((p * p).sum(), 1e-9)
    mse = ((np.clip(pred, 0, 255) - gt)[mask] ** 2).mean()
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="vd sets/t01")
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    work = os.path.join("work", "oracle_" + os.path.basename(a.set.rstrip("/")))
    write_oracle_poses(a.set, work)
    import time
    render.run(os.path.join(a.set, "public", "images"), work, a.out, time.time() + 1e9)
    p = quick_psnr(a.out, a.set)
    print(f"PSNR nhanh (đã bù gain, chưa căn yaw vì đã đúng hệ): {p:.2f} dB" if p else "sai kích thước")
    if a.score:
        subprocess.run([sys.executable, os.path.join(ROOT, "score.py"), "--set", a.set, "--pred", a.out])


if __name__ == "__main__":
    main()
