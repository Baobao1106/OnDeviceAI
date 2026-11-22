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
    A = project_so3(M)
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
