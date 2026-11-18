"""Lõi ghép đặc trưng (SuperPoint + LightGlue + RootSIFT, RANSAC phép quay, cứu ảnh cô lập, tầng, ảnh kề).

Code của thư mục `matching/` (đã đo trên 50 bộ train) chép vào đây để repo tự chạy được; `register/` chỉ gọi vào.
Mô hình nặng tải một lần: `get_engine()`.
"""
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from .pairs import PairEngine
        _ENGINE = PairEngine()
    return _ENGINE
