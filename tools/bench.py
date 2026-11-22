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
