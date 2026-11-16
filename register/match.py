"""Khớp đặc trưng giữa các cặp ảnh. NGƯỜI PHỤ TRÁCH: B.

VIỆC CẦN LÀM:
  1. `match_pair`: khớp 2 ảnh (FLANN hoặc brute force + kiểm tra tỉ lệ Lowe 0,75–0,8, kiểm tra 2 chiều),
     rồi lọc bằng RANSAC không cần FOV: cv2.findHomography (ảnh chỉ xoay nên homography hợp lý)
     hoặc cv2.findFundamentalMat. Trả về chỉ các điểm inlier theo pixel ảnh gốc.
  2. `candidate_pairs`: đừng khớp hết 276 cặp. 24 ảnh theo mẫu chụp chỉ có vài láng giềng thật sự chồng lên nhau
     (cùng vòng ± láng giềng, vòng trên/dưới với vòng ngang, trần/sàn với vòng ±45°). Bước đầu có thể khớp tất cả
     rồi đo xem cặp nào hữu ích để biết cần giữ lại cặp nào.
  3. `match_all`: gọi `match_pair` cho các cặp, bỏ cặp ít hơn ~15 điểm inlier, trả về list theo định dạng
     matches.json (xem docs/INTERFACES.md): {"i", "j", "pts_i", "pts_j", "n_inliers"}.
  4. Kiểm tra `time.time() < deadline` trong vòng lặp để dừng sớm.

CẠM BẪY: cảnh có hoa văn lặp lại (gạch, cửa sổ giống nhau) cho ra khớp sai mà RANSAC vẫn chấp nhận. Dùng thêm
điều kiện chặt (số inlier tối thiểu, tỉ lệ inlier) và đối chiếu với láng giềng theo mẫu chụp.

XONG KHI: tools/pose_diff.py (D) cho thấy các cặp B đưa ra hầu như không có khớp sai, và tỉ lệ ảnh có ít nhất
2 cặp tốt trên 10 bộ train là cao.
"""
import itertools
import time


def match_pair(kp_i, desc_i, kp_j, desc_j):
    """Trả về (pts_i [M,2], pts_j [M,2], n_inliers) theo pixel ảnh gốc."""
    raise NotImplementedError("B: viết khớp một cặp ảnh + RANSAC")


def candidate_pairs(names):
    """Danh sách cặp (tên_i, tên_j) đáng khớp. Bản mặc định: tất cả các cặp."""
    return list(itertools.combinations(sorted(names), 2))


def match_all(images, feats, deadline):
    pairs = []
    for a, b in candidate_pairs(images):
        if time.time() > deadline:
            break
        pts_i, pts_j, n_in = match_pair(*feats[a], *feats[b])
        if n_in >= 15:
            pairs.append({"i": a, "j": b, "pts_i": pts_i.tolist(), "pts_j": pts_j.tolist(), "n_inliers": int(n_in)})
    return pairs
