import os
import sys

_ENGINE = None
_ERROR = None
_MODE = None
SIFT_ONLY_ENV = "REGISTER_SIFT_ONLY"


def engine_mode():
    return _MODE


def get_engine():
    global _ENGINE, _ERROR, _MODE
    if _ENGINE is not None:
        return _ENGINE
    if _ERROR is not None:
        raise _ERROR
    try:
        from . import weights
        from .pairs import PairEngine, PipelineConfig
        if os.environ.get(SIFT_ONLY_ENV) == "1":
            _ENGINE, _MODE = PairEngine(PipelineConfig(use_superpoint=False)), "sift-only"
        else:
            try:
                with weights.use():
                    _ENGINE, _MODE = PairEngine(), "full"
            except weights.WeightsMissing as e:
                print(f"[register/engine] CẢNH BÁO: {e}", file=sys.stderr)
                print("[register/engine] Dùng bản dự phòng chỉ RootSIFT.", file=sys.stderr)
                _ENGINE, _MODE = PairEngine(PipelineConfig(use_superpoint=False)), "sift-only"
    except Exception as e:
        _ERROR = e
        raise
    return _ENGINE
