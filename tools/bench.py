"""Đo thời gian từng khâu. NGƯỜI PHỤ TRÁCH: D.

Đã có sẵn bộ đo thời gian đơn giản để cả nhóm dùng ngay. `run.py` đã bọc từng stage bằng `Timer`
và ghi `work/<bộ>/timing.json`.

Cách dùng trong code của mọi người:
    from tools.bench import Timer
    timer = Timer()
    with timer("extract_features"):
        ...
    timer.report()

VIỆC CẦN LÀM (D):
  - Chạy cả 10 bộ trên Jetson (MAXN SUPER), tổng hợp timing.json thành bảng: khâu nào chiếm bao nhiêu giây.
  - Báo cáo cho cả nhóm 2–3 khâu chậm nhất, rồi cùng chủ module đó giảm (giảm độ phân giải khi khớp,
    chạy GPU, cache, FP16/TensorRT nếu dùng mạng học sâu).
  - Nhớ thời gian chấm tính cả load model, nên đo cả phần import và load model.
"""
import json
import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.stages = {}

    @contextmanager
    def __call__(self, name):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t

    def report(self, path=None):
        total = time.perf_counter() - self.t0
        out = {"total_s": round(total, 3), "stages_s": {k: round(v, 3) for k, v in self.stages.items()}}
        if path:
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
        for k, v in out["stages_s"].items():
            print(f"  {k:24s} {v:7.2f} s")
        print(f"  {'total':24s} {total:7.2f} s")
        return out
