import argparse
import os
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import common  # noqa: E402
import solve  # noqa: E402
from solve.bundle import procrustes  # noqa: E402
from tools.pose_diff import true_rotations  # noqa: E402


def truth_path(name):
    for sets in ("sets", "sets_s3d", "sets_extra"):
        p = os.path.join(ROOT, sets, name, "hidden", "true_poses.json")
        if os.path.exists(p):
            return p
    return None


def evaluate(poses, truth):
    tru = true_rotations(truth)
    reg = sorted(n for n, p in poses["images"].items() if p.get("registered", True) and n in tru)
    est = {n: np.asarray(poses["images"][n]["R"], float) for n in poses["images"]}
    A = procrustes([est[n] for n in reg], [tru[n] for n in reg], reject_deg=10.0)
    err = {n: float(np.degrees(Rotation.from_matrix(est[n].T @ A.T @ tru[n]).magnitude())) for n in tru}
    up = A.T @ np.array([0.0, 1.0, 0.0])
    return err, reg, float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))), poses["hfov_deg"] - truth["hfov_deg"]


def run_one(name, src, out):
    work = os.path.join(out, name)
    os.makedirs(work, exist_ok=True)
    for f in ("matches.json", "slots.json"):
        shutil.copy(os.path.join(src, name, f), os.path.join(work, f))
    t = time.perf_counter()
    solve.run("unused", work, time.time() + 240)
    dt = time.perf_counter() - t
    poses = common.load_json(os.path.join(work, "poses.json"))
    err, reg, tilt, dfov = evaluate(poses, common.load_json(truth_path(name)))
    e = np.array([err[n] for n in reg]) if reg else np.array([np.nan])
    return {"set": name, "time": dt, "dfov": dfov, "tilt": tilt, "n_reg": len(reg), "mean": float(np.mean(e)),
            "max": float(np.max(e)), "bad": int((e > 2.0).sum())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_work", required=True, help="thư mục chứa <bộ>/matches.json và slots.json (từ register)")
    ap.add_argument("--out", default=os.path.join(ROOT, "work", "solve_eval"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    src = os.path.abspath(a.from_work)
    names = a.only or sorted(n for n in os.listdir(src) if truth_path(n))
    rows = []
    print("set    time  dFOV    tilt  reg  pose_mean pose_max  >2deg")
    for n in names:
        r = run_one(n, src, a.out)
        rows.append(r)
        print(f"{n:5s} {r['time']:5.1f} {r['dfov']:+6.2f} {r['tilt']:6.2f} {r['n_reg']:4d} {r['mean']:9.3f} {r['max']:8.3f} "
              f"{r['bad']:5d}", flush=True)
    t = np.array([r["tilt"] for r in rows])
    print(f"TOTAL {len(rows)} bộ: thời gian TB {np.mean([r['time'] for r in rows]):.1f} s | |dFOV| TB "
          f"{np.mean([abs(r['dfov']) for r in rows]):.2f}° max {max(abs(r['dfov']) for r in rows):.2f}° | "
          f"nghiêng TB {t.mean():.2f}° max {t.max():.2f}° ; >=1°: {(t >= 1).sum()} ; >=5°: {(t >= 5).sum()} | "
          f"pose TB {np.nanmean([r['mean'] for r in rows]):.3f}° | ảnh đăng ký lệch >2°: {sum(r['bad'] for r in rows)}")


if __name__ == "__main__":
    main()
