"""Cân bằng ngang: tìm hướng "lên" thật và xoay mọi pose cho đúng. NGƯỜI PHỤ TRÁCH: A.

QUAN TRỌNG NHẤT: nghiêng ≥ 5° là 0 điểm cả bộ. Nên có tín hiệu thứ hai để kiểm tra chéo.

VIỆC CẦN LÀM:
  1. `estimate_up(poses, slots)`: ước lượng hướng lên trong hệ pose hiện tại. Hai nguồn:
       a) Cấu trúc mẫu chụp: vòng ngang pitch 0° (ô 0 đến 7) có các trục quang học nằm trên một mặt phẳng nằm ngang,
          pháp tuyến của mặt phẳng đó (khớp bằng bình phương tối thiểu) là hướng lên. Vòng ±45° đối xứng quanh nó.
       b) Điểm triệt tiêu thẳng đứng (đường thẳng đứng trong phòng: cạnh cửa, góc tường). Dùng nếu đủ chắc chắn.
     So sánh hai nguồn; nếu lệch nhiều thì tin nguồn (a) vì nó không phụ thuộc vào nội dung cảnh.
  2. `level_poses(poses, slots)`: xoay mọi R sao cho hướng lên trùng trục y (0, 1, 0), trả về
     (poses mới, up, tilt_deg_est). Giữ nguyên yaw tự do (hướng chính diện được tự căn lúc chấm điểm).
  3. Cẩn thận dấu: pitch > 0 là nhìn lên, roll > 0 xoay ảnh ngược chiều kim đồng hồ.

Cách làm (nguồn a): mỗi ảnh đã đăng ký, ở góc ngẩng danh định p, có (trong hệ thế giới, u là hướng lên)
    trục quang  z·u = sin p,   trục y của ảnh  y·u = cos p,   trục x của ảnh  x·u = 0
(sai lệch cầm tay ±5° mỗi trục, đối xứng nên triệt tiêu khi lấy trung bình trên ~22 ảnh). Giải bình phương tối thiểu có
trọng số Huber (IRLS) để ảnh gán nhầm ô không kéo lệch. Nguồn (b) điểm triệt tiêu chưa làm.

XONG KHI: với 10 bộ train, `tilt_deg_est` thật sự gần 0° khi so với pose thật (D kiểm bằng score.py: trường tilt_deg),
tilt < 1° ở hầu hết bộ, và không bộ nào gần 5°.
"""


import numpy as np

MIN_REGISTERED = 6          # ít hơn thì không đủ để ước lượng hướng lên: giữ nguyên hệ (trục y)
HUBER_RAD = 0.06            # ~3,4°: sai số tới mức này tính đủ trọng số
IRLS_ROUNDS = 8
Y = np.array([0.0, 1.0, 0.0])


def _constraints(poses, slots):
    """Hàng A (k x 3) và đích t (k) sao cho A @ u ≈ t, từ mọi ảnh đã đăng ký."""
    rows, targets = [], []
    for name, p in poses.items():
        if not p.get("registered", True):
            continue
        R = np.asarray(p["R"], float)
        pitch = np.radians(slots["images"][name]["pitch_deg"])
        rows += [R[:, 2], R[:, 1], R[:, 0]]
        targets += [np.sin(pitch), np.cos(pitch), 0.0]
    return np.array(rows), np.array(targets)


def estimate_up(poses, slots):
    """Trả về vector lên (3,) trong hệ pose hiện tại."""
    A, t = _constraints(poses, slots)
    if len(t) < 3 * MIN_REGISTERED:
        return Y.copy()
    w = np.ones(len(t))
    u = Y.copy()
    for _ in range(IRLS_ROUNDS):
        sol, *_ = np.linalg.lstsq(A * w[:, None], t * w, rcond=None)
        u = sol / np.linalg.norm(sol)
        err = np.abs(A @ u - t)
        w = np.sqrt(np.where(err <= HUBER_RAD, 1.0, HUBER_RAD / np.maximum(err, 1e-12)))
    return u


def _rotation_taking(u, v):
    """Phép quay nhỏ nhất đưa vector đơn vị u thành v (Rodrigues)."""
    c = float(np.clip(u @ v, -1.0, 1.0))
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.eye(3)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def level_poses(poses, slots):
    """Trả về (poses đã xoay, up, tilt_deg_est). tilt = góc giữa hướng lên ước lượng và trục y của hệ cũ."""
    up = estimate_up(poses, slots)
    Rl = _rotation_taking(up, Y)
    new = {name: {**p, "R": (Rl @ np.asarray(p["R"], float)).tolist()} for name, p in poses.items()}
    tilt = float(np.degrees(np.arccos(np.clip(up @ Y, -1.0, 1.0))))
    return new, up, tilt
