"""Lõi ghép đặc trưng (SuperPoint + LightGlue + RootSIFT, RANSAC phép quay, cứu ảnh cô lập, tầng, ảnh kề).

Code của thư mục `matching/` (đã đo trên 50 bộ train) chép vào đây để repo tự chạy được; `register/` chỉ gọi vào.
Mô hình nặng tải một lần: `get_engine()`. Trọng số lấy từ `register/weights/` (xem `weights.py`); thiếu thì lỗi được
ghi nhớ và ném lại ngay cho các lần gọi sau (không thử lại cho từng ảnh). Riêng thiếu trọng số thì dùng bản dự phòng
chỉ RootSIFT (ghi cảnh báo một lần) thay vì để mọi ảnh lỗi.
"""
import os
import sys

_ENGINE = None
_ERROR = None
_MODE = None                      # "full" (SuperPoint + LightGlue + RootSIFT) hoặc "sift-only" (dự phòng)
SIFT_ONLY_ENV = "REGISTER_SIFT_ONLY"    # đặt = 1 để ép dùng bản chỉ RootSIFT (đo bản dự phòng, hoặc máy không có torch)


def engine_mode():
    """"full" hoặc "sift-only" sau khi `get_engine()` đã chạy; None nếu chưa."""
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
                # thiếu trọng số: vẫn ghép được bằng RootSIFT (kém hơn, nhất là ảnh ít texture) thay vì không có cặp nào
                print(f"[register/engine] CẢNH BÁO: {e}", file=sys.stderr)
                print("[register/engine] Dùng bản dự phòng chỉ RootSIFT.", file=sys.stderr)
                _ENGINE, _MODE = PairEngine(PipelineConfig(use_superpoint=False)), "sift-only"
    except Exception as e:
        _ERROR = e
        raise
    return _ENGINE
