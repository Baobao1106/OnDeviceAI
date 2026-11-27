import argparse
import json
import math
import sys

QUALITY_WEIGHT = 0.8
SPEED_MAX = 20.0
DEFAULT_LIMIT = 300.0
DEFAULT_BEST = 60.0


def speed_score(seconds, best, limit):
    if seconds >= limit:
        return 0.0
    return SPEED_MAX * math.log(limit / seconds) / math.log(limit / best)


def total_score(row, best, limit):
    if row.get("status") != "ok":
        return 0.0, 0.0
    speed = min(SPEED_MAX, speed_score(row["seconds"], best, limit))
    return QUALITY_WEIGHT * row["score"] + speed, speed


def main():
    ap = argparse.ArgumentParser(description="Điểm tổng = 0,8 x chất lượng + điểm tốc độ, từ summary.json của run_eval.py")
    ap.add_argument("summary", help="đường dẫn runs/<tên>/summary.json")
    ap.add_argument("--best", type=float, nargs="+", default=[DEFAULT_BEST],
                    help="thời gian nhanh nhất của các đội (giây); đưa nhiều giá trị để so sánh")
    ap.add_argument("--limit", type=float, default=DEFAULT_LIMIT)
    a = ap.parse_args()

    with open(a.summary, encoding="utf-8") as f:
        rows = json.load(f)["sets"]
    if not rows:
        sys.exit("summary.json không có bộ nào")
    if any(b <= 0 or b >= a.limit for b in a.best):
        sys.exit("--best phải nằm trong (0, limit)")

    for best in a.best:
        print(f"nhanh nhất = {best:g} s")
        totals = []
        for r in rows:
            total, speed = total_score(r, best, a.limit)
            totals.append(total)
            print(f"  {r['set']:8s} {r['seconds']:7.1f} s  chất lượng {r.get('score', 0.0):6.2f}  tốc độ {speed:5.2f}  tổng {total:6.2f}")
        print(f"  TRUNG BÌNH tổng {sum(totals) / len(totals):.2f}  (thấp nhất {min(totals):.2f})")


if __name__ == "__main__":
    main()
