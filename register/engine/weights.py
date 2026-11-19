"""Trọng số SuperPoint / LightGlue: tìm trong thư mục của dự án, không tải từ mạng khi không cần.

`lightglue` tải trọng số bằng `torch.hub.load_state_dict_from_url`, hàm này xem thư mục đệm của PyTorch
(`<hub_dir>/checkpoints/<tên file>`) trước rồi mới tải. Trong lúc tạo mô hình, ta trỏ thư mục đệm vào `register/weights/` (hoặc thư mục đặt
bằng biến môi trường REGISTER_WEIGHTS) khi hai file có ở đó. Không có thì dùng thư mục đệm mặc định của PyTorch nếu đã
có file ở đó; còn không thì báo lỗi rõ ràng một lần (không thử mạng).
Lấy file: `python -m register.fetch_weights`.
"""
import contextlib
import os

# tên file trong thư mục `checkpoints/`, kích thước và SHA-256 của bản đang dùng, URL chính thức
WEIGHTS = {
    "superpoint_v1.pth": (
        5206086, "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"),
    "superpoint_lightglue_v0-1_arxiv.pth": (
        47500279, "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a",
        "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"),
}
ENV_VAR = "REGISTER_WEIGHTS"


def weights_dir():
    """Thư mục gốc (chứa `checkpoints/`) của trọng số của dự án."""
    return os.environ.get(ENV_VAR) or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "weights")


def _present(directory):
    return all(os.path.isfile(os.path.join(directory, "checkpoints", name)) for name in WEIGHTS)


class WeightsMissing(RuntimeError):
    pass


@contextlib.contextmanager
def use():
    """Trong khối `with`, PyTorch tìm trọng số ở nơi đúng; ra khỏi khối thì trả thư mục đệm về như cũ (để không ảnh
    hưởng đoạn code khác của nhóm dùng torch.hub). Ném WeightsMissing nếu không có trọng số ở đâu cả."""
    import torch
    own = weights_dir()
    default = torch.hub.get_dir()
    if _present(own):
        saved = default
        torch.hub.set_dir(own)
        try:
            yield own
        finally:
            torch.hub.set_dir(saved)
    elif _present(default):
        yield default
    else:
        raise WeightsMissing(
            "Thiếu trọng số SuperPoint/LightGlue (" + ", ".join(WEIGHTS) + f"). Tìm trong "
            f"{os.path.join(own, 'checkpoints')} và {os.path.join(default, 'checkpoints')} đều không đủ. Chạy "
            f"`python -m register.fetch_weights` (cần mạng một lần, hoặc chép thư mục `{own}` từ máy khác).")
