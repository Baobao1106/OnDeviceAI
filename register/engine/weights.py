import contextlib
import os

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
    return os.environ.get(ENV_VAR) or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "weights")


def _present(directory):
    return all(os.path.isfile(os.path.join(directory, "checkpoints", name)) for name in WEIGHTS)


class WeightsMissing(RuntimeError):
    pass


@contextlib.contextmanager
def use():
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
