"""Lõi ghép đặc trưng (SuperPoint + LightGlue + RootSIFT, RANSAC phép quay, cứu ảnh cô lập, tầng, ảnh kề).

Code của thư mục `matching/` (đã đo trên 50 bộ train) chép vào đây để repo tự chạy được; `register/` chỉ gọi vào.
Mô hình nặng tải một lần: `get_engine()`. Trọng số lấy từ `register/weights/` (xem `weights.py`); thiếu thì lỗi được
ghi nhớ và ném lại ngay cho các lần gọi sau (không thử lại cho từng ảnh).
"""
_ENGINE = None
_ERROR = None


def get_engine():
    global _ENGINE, _ERROR
    if _ENGINE is not None:
        return _ENGINE
    if _ERROR is not None:
        raise _ERROR
    try:
        from . import weights
        from .pairs import PairEngine
        with weights.use():
            _ENGINE = PairEngine()
    except Exception as e:
        _ERROR = e
        raise
    return _ENGINE
