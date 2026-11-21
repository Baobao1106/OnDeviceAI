"""Lấy trọng số SuperPoint / LightGlue vào `register/weights/checkpoints/` (hoặc $REGISTER_WEIGHTS).

    python -m register.fetch_weights            # chép từ thư mục đệm của PyTorch nếu đã có, không thì tải
    python -m register.fetch_weights --no-download

Mỗi file được kiểm tra kích thước và SHA-256. File đã đúng thì bỏ qua. Cần mạng chỉ khi file chưa có trong thư mục
đệm của PyTorch; máy không có mạng thì chép thư mục `register/weights/` từ máy khác.
"""
import argparse
import hashlib
import os
import shutil
import sys
import urllib.request

from .engine.weights import WEIGHTS, weights_dir


def _ok(path, size, sha):
    if not os.path.isfile(path) or os.path.getsize(path) != size:
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true", help="chỉ chép từ thư mục đệm PyTorch, không dùng mạng")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):          # console Windows có thể không in được tiếng Việt: không để in lỗi làm hỏng
        stream.reconfigure(errors="replace")
    import torch
    dest_dir = os.path.join(weights_dir(), "checkpoints")
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(dest_dir, exist_ok=True)
    failed = 0
    for name, (size, sha, url) in WEIGHTS.items():
        dest = os.path.join(dest_dir, name)
        cached = os.path.join(cache_dir, name)
        if _ok(dest, size, sha):
            print(f"[ok]         {dest}")
            continue
        if _ok(cached, size, sha):
            shutil.copyfile(cached, dest)
            print(f"[copied]    {cached} -> {dest}")
            continue
        if a.no_download:
            print(f"[MISSING]   {name}: không có trong {cache_dir} và --no-download", file=sys.stderr)
            failed += 1
            continue
        tmp = dest + ".part"
        print(f"[download] {url} ({size / 1e6:.1f} MB)")
        try:
            urllib.request.urlretrieve(url, tmp)
        except OSError as e:
            print(f"[ERROR]    không tải được {name}: {e}", file=sys.stderr)
            failed += 1
            continue
        if _ok(tmp, size, sha):
            os.replace(tmp, dest)
        else:
            os.remove(tmp)
            print(f"[ERROR]    {name}: kích thước hoặc SHA-256 không khớp", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
