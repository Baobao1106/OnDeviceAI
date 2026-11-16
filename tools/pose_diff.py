"""So sánh pose ước lượng với pose thật. NGƯỜI PHỤ TRÁCH: D.

Dùng để biết khi điểm thấp thì lỗi nằm ở pose (A, B) hay ở dựng ảnh (C).

    python tools/pose_diff.py --poses work/t01/poses.json --truth sets/t01/hidden/true_poses.json

Cách tính không phụ thuộc hệ quy chiếu (yaw tự do): tìm phép quay toàn cục khớp nhất giữa hai bộ pose,
rồi đo góc lệch của từng ảnh. In ra sai số từng ảnh, trung bình, lớn nhất, và sai số FOV.

Đã viết sẵn. VIỆC CẦN LÀM (D):
  - Chạy cho cả 10 bộ rồi tổng hợp vào tools/results.md (cột: bộ, FOV lỗi, sai số pose TB, sai số pose max).
  - Thêm mục kiểm tra ô: so `slots.json` với ô thật (tìm ô gần nhất của pose thật trong common.capture_plan())
    để đếm bao nhiêu ảnh B gán đúng ô.
  - Thêm cột nghiêng so với pose thật nếu muốn (score.py đã in tilt_deg).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def true_rotations(truth):
    return {s["image"]: common.pose_to_R(s["yaw_deg"], s["pitch_deg"], s["roll_deg"]) for s in truth["shots"]}


def compare(poses, truth):
    est = {n: np.asarray(p["R"], float) for n, p in poses["images"].items()}
    tru = true_rotations(truth)
    names = sorted(set(est) & set(tru))
    M = sum(est[n] @ tru[n].T for n in names)
    A = project_so3(M)                      # est ≈ A @ tru
    rows = [(n, angle_deg(est[n].T @ A @ tru[n]), poses["images"][n].get("registered", True)) for n in names]
    return rows, poses["hfov_deg"] - truth["hfov_deg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    poses, truth = common.load_json(a.poses), common.load_json(a.truth)
    rows, dfov = compare(poses, truth)
    for n, e, reg in sorted(rows, key=lambda r: -r[1]):
        print(f"{n:10s} {e:7.3f} deg {'' if reg else '(không registered)'}")
    errs = [e for _, e, _ in rows]
    print(f"FOV lệch: {dfov:+.3f} deg | pose TB: {np.mean(errs):.3f} deg | max: {np.max(errs):.3f} deg | {len(rows)} ảnh")


if __name__ == "__main__":
    main()
