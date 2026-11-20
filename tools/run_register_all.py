"""Run register.run (repo = the folder holding tools/) on every training set and score slots.json against the truth.

    python tools/run_register_all.py [--repo .] [--out runs/register50.json]
Times are wall-clock seconds of register.run per set (the first set also pays the one-off model load, reported
separately). Work files go to <repo>/work/all50/<set> (git-ignored).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--out", default="runs/register50.json")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--sets", nargs="*", default=["sets", "sets_s3d"])
    ap.add_argument("--work_name", default="all50")
    ap.add_argument("--no_lowtex", action="store_true", help="baseline: switch the low-texture re-matching off")
    ap.add_argument("--no_priority", action="store_true", help="match pairs in random order instead of RootSIFT-score order (the order matters only when time runs out)")
    ap.add_argument("--reserve", type=float, default=None, help="share of the remaining time kept for the low-texture re-matching pass (default: engine's)")
    ap.add_argument("--deadline", type=float, default=240.0,
                    help="seconds given to register.run; the stage itself may use half (REGISTER_SHARE). Smaller values "
                         "simulate a slower machine, e.g. 150 -> 75 s for the stage = a machine ~2x slower than 150 s")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    import common
    import register
    from register.engine import get_engine
    from schema import validate_matches, validate_slots

    if a.no_lowtex or a.no_priority or a.reserve is not None:
        import register.engine as engine_pkg
        from register.engine.pairs import PairEngine, PipelineConfig
        defaults = PipelineConfig()
        engine_pkg._ENGINE = PairEngine(PipelineConfig(
            lowtex_pass=not a.no_lowtex, priority_order=not a.no_priority,
            lowtex_reserve=defaults.lowtex_reserve if a.reserve is None else a.reserve))
    t0 = time.perf_counter()
    get_engine()
    load_s = time.perf_counter() - t0
    print(f"model load {load_s:.1f} s", flush=True)
    plan = common.capture_plan()
    layer = lambda p: int(np.abs(p - np.array([-90.0, -45.0, 0.0, 45.0, 90.0])).argmin())
    rows = []
    for sets in a.sets:
        for name in sorted(os.listdir(f"{ROOT}/{sets}")):
            if a.only and name not in a.only:
                continue
            images = f"{ROOT}/{sets}/{name}/public/images"
            if not os.path.isdir(images):
                continue
            work = f"{a.repo}/work/{a.work_name}/{name}"
            os.makedirs(work, exist_ok=True)
            t = time.perf_counter()
            register.run(images, work, time.time() + a.deadline)
            dt = time.perf_counter() - t
            validate_matches(json.load(open(f"{work}/matches.json")))
            s = validate_slots(json.load(open(f"{work}/slots.json")))["images"]
            truth = json.load(open(f"{ROOT}/{sets}/{name}/hidden/true_poses.json"))["shots"]
            ts = {x["image"]: k for k, x in enumerate(truth)}
            ty = {x["image"]: x["yaw_deg"] for x in truth}
            placed = [n for n, v in s.items() if v["confidence"] > 0.05]
            ring = [n for n in placed if abs(plan[ts[n]][1]) < 80 and abs(plan[s[n]["slot"]][1]) < 80]
            d = np.array([(s[n]["yaw_deg"] - ty[n] + 180) % 360 - 180 for n in ring])
            shift = float(np.degrees(np.angle(np.exp(1j * np.radians(d)).mean()))) if len(d) else 0.0
            r = np.abs((d - shift + 180) % 360 - 180)
            rows.append({"set": name, "time_s": round(dt, 2), "placed": len(placed),
                         "layer_ok": sum(layer(plan[s[n]["slot"]][1]) == layer(plan[ts[n]][1]) for n in placed),
                         "exact": sum(s[n]["slot"] == ts[n] for n in placed),
                         "exact_all": sum(v["slot"] == ts[n] for n, v in s.items()),
                         "yaw_within10": int((r <= 10).sum()), "ring": len(ring)})
            x = rows[-1]
            print(f"{name:4s} {dt:6.1f} s  placed {x['placed']:2d}  layer {x['layer_ok']:2d}  exact(placed) {x['exact']:2d}  "
                  f"exact(all) {x['exact_all']:2d}  yaw<=10 {x['yaw_within10']}/{x['ring']}", flush=True)
            json.dump({"model_load_s": round(load_s, 2), "rows": rows}, open(f"{ROOT}/{a.out}", "w"), indent=1)
    t = np.array([r["time_s"] for r in rows])
    print(f"TOTAL sets {len(rows)}: time per set mean {t.mean():.1f} s, median {np.median(t):.1f}, max {t.max():.1f} "
          f"({rows[int(t.argmax())]['set']}), min {t.min():.1f}; total {t.sum() / 60:.1f} min; model load once {load_s:.1f} s")
    print(f"placed {sum(r['placed'] for r in rows)}, layer ok {sum(r['layer_ok'] for r in rows)}, "
          f"exact(placed) {sum(r['exact'] for r in rows)}, exact(all 24) {sum(r['exact_all'] for r in rows)}/{24 * len(rows)}, "
          f"yaw within 10: {sum(r['yaw_within10'] for r in rows)}/{sum(r['ring'] for r in rows)}")


if __name__ == "__main__":
    main()
