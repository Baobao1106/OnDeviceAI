"""Gán 24 ảnh vào 24 ô của mẫu chụp. NGƯỜI PHỤ TRÁCH: B.

Mẫu chụp (common.capture_plan()), chỉ số 0 đến 23:
   0–7   vòng ngang, pitch 0°, yaw cách nhau 45° (ô 0 là ảnh 01: chính diện, chụp cẩn thận)
   8–14  vòng trên, pitch +45°, 7 ảnh cách nhau ~51°
  15–21  vòng dưới, pitch −45°, 7 ảnh cách nhau ~51°
  22     trần (pitch +90°)      23  sàn (pitch −90°)
Tay run: mỗi ảnh lệch ±5° (yaw, pitch, roll), riêng ô 0 chỉ ±2°.

Tên file bị xáo ngẫu nhiên nên không biết ảnh nào ở ô nào. Phải suy ra từ điểm khớp.

VIỆC CẦN LÀM:
  1. `assign_slots`: dựng đồ thị các ảnh (cạnh = cặp khớp tốt) rồi gán ảnh vào ô.
     Gợi ý: tìm vòng ngang trước (các ảnh nối thành chu trình 8 ảnh), xác định hướng, rồi gán vòng trên/dưới,
     cuối cùng trần và sàn (ảnh trần nối được với cả vòng +45°, sàn nối với vòng −45°).
     Có thể dùng bài toán gán Hungarian (scipy.optimize.linear_sum_assignment) với chi phí = mức không phù hợp
     giữa ảnh và ô, xét thêm sự phù hợp giữa các ảnh láng giềng.
  2. Trả về dict {tên ảnh: {"slot", "yaw_deg", "pitch_deg", "confidence", "n_matches"}}, đủ 24 ảnh,
     không ô nào trùng (schema.validate_slots sẽ kiểm tra).
  3. Ảnh khớp yếu vẫn phải có ô dự đoán, với `confidence` thấp (`textureless.py` sẽ cải thiện tiếp).
  4. Có thể ước lượng nhanh vị trí bằng độ sáng: ảnh trần thường sáng đều, ảnh sàn tối hơn, như tín hiệu phụ.

XONG KHI: so với pose thật (D sẽ viết công cụ), phần trăm ảnh gán đúng ô trên 10 bộ train gần 100% với ảnh có hoa văn.
"""
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

import common

from .engine.layers import assign_layers, component_frames
from .engine.neighbors import yaw_deg

SIGMA_DEG = 7.0                 # tay run +-5 độ: sai lệch quanh góc danh định cỡ chừng này
CAP = 3.0                       # mỗi số hạng chi phí bị chặn ở CAP**2 (ảnh lệch quá nhiều không kéo cả bộ)
YAW_SIGN = 1.0                  # chiều yaw của lõi so với common.py (yaw > 0 quay phải); đã kiểm với pose thật
POLE_PITCH = 80.0               # |pitch| lớn hơn: ảnh trần/sàn, không có yaw
ALIGN_MARGIN = 3.0              # margin (chi phí) từ mức này trở lên: cách đặt nhãn của thành phần coi là chắc
ALIGN_FLOOR = 0.3               # hệ số confidence thấp nhất khi hai cách đặt nhãn hoà hẳn


def nominal(slot):
    """(yaw_deg, pitch_deg) danh định của ô."""
    return common.capture_plan()[slot]


def _pair_objects(names, pairs):
    """Cặp của matches.json (có khóa `rotation`) -> đối tượng mà lõi tầng dùng. Cặp thiếu `rotation` bị bỏ."""
    index = {n: k for k, n in enumerate(names)}
    out = []
    for p in pairs:
        if p.get("rotation") is None:
            continue
        out.append(SimpleNamespace(i=index[p["i"]], j=index[p["j"]], accepted=True, n_inliers=p["n_inliers"],
                                   rotation=np.asarray(p["rotation"], float), cand_rotation=None))
    return out


def _circ(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _costs(idx, slots, pitch, yaw, psi, Y, P):
    """Chi phí [ảnh, ô]: lệch pitch + lệch yaw (yaw của ảnh trong khung của thành phần cộng `psi`)."""
    pi = np.array([pitch[k] for k in idx])[:, None]
    cost = np.minimum(((pi - P[None, slots]) / SIGMA_DEG) ** 2, CAP ** 2)
    ring_img = (np.abs(pi) < POLE_PITCH)
    ring_slot = (np.abs(P[slots]) < POLE_PITCH)[None, :]
    yi = np.array([yaw[k] for k in idx])[:, None]
    cost = cost + np.where(ring_img & ring_slot,
                           np.minimum((_circ(yi, Y[None, slots] + psi) / SIGMA_DEG) ** 2, CAP ** 2), 0.0)
    return cost


def _fit_component(idx, free, pitch, yaw, Y, P):
    """Gán các ảnh của một thành phần vào các ô còn trống. Chọn độ lệch yaw chung `psi` của thành phần (khung của
    nó có hướng 0 tùy ý) sao cho tổng chi phí gán Hungarian nhỏ nhất.

    Trả về ({ảnh: ô}, psi, chi phí, margin). `margin` = chi phí của cách đặt nhãn tốt nhì (cách xa >= 10 độ) trừ chi
    phí tốt nhất: nhỏ nghĩa là hai cách gần như hoà (vòng ngang 8 ảnh và vòng trên/dưới 7 ảnh chỉ khác nhau ~6 độ,
    ngang độ rung tay), nên chỉ số ô có thể lệch cả bộ một nấc. `inf` khi không có sự nhập nhằng để đo.
    """
    slots = np.array(sorted(free))
    ring = sum(abs(pitch[k]) < POLE_PITCH for k in idx)

    def solve(psi):
        c = _costs(idx, slots, pitch, yaw, psi, Y, P)
        r, cidx = linear_sum_assignment(c)
        return c[r, cidx].sum(), dict(zip((idx[a] for a in r), (int(slots[b]) for b in cidx)))

    if ring < 2:                                          # một ảnh vòng thì yaw không phân biệt được ô nào
        total, assign = solve(0.0)
        return assign, 0.0, total, float("inf")
    grid = np.arange(0.0, 360.0, 1.0)
    vals = np.array([solve(g)[0] for g in grid])
    best = grid[int(np.argmin(vals))]
    fine = np.arange(best - 1.0, best + 1.0 + 1e-9, 0.1)
    psi = min(fine, key=lambda g: solve(g)[0])
    total, assign = solve(psi)
    far = _circ(grid, best) >= 10.0
    margin = float(vals[far].min() - total) if far.any() else float("inf")
    return assign, float(psi), total, max(margin, 0.0)


def assign_slots(images, pairs, deadline):
    """Trả về {tên ảnh: {"slot", "yaw_deg", "pitch_deg", "confidence", "n_matches"}}.

    Ảnh có cặp khớp: tầng và yaw (trong khung của thành phần liên thông chứa nó) từ phép quay tương đối của các cặp,
    rồi gán Hungarian vào ô; thành phần lớn nhất gán trước, các thành phần nhỏ gán vào ô còn lại.
    Ảnh không có cặp nào: nhận ô còn lại với `confidence` thấp (không có thông tin để chọn ô cụ thể).
    `confidence` của ảnh có cặp = độ khớp tầng/yaw x cỡ thành phần x độ chắc của cách đặt nhãn (`margin`: khoảng cách
    chi phí tới cách đặt nhãn tốt nhì; hai cách gần hoà thì chỉ số ô của cả thành phần có thể lệch một nấc).
    """
    names = sorted(images)
    objs = _pair_objects(names, pairs)
    A, label, U = component_frames(len(names), objs)
    lay = assign_layers(names, objs)
    pitch = {k: lay[k].pitch_deg for k in A}
    yaw = {k: YAW_SIGN * float(yaw_deg(A[k][:, 2], U[int(label[k])])[0]) for k in A}
    plan = common.capture_plan()
    Y = np.array([p[0] for p in plan])
    P = np.array([p[1] for p in plan])

    n_match = dict.fromkeys(range(len(names)), 0)
    for o in objs:
        n_match[o.i] += o.n_inliers
        n_match[o.j] += o.n_inliers

    comps = {}
    for k in A:
        comps.setdefault(int(label[k]), []).append(k)
    free = set(range(len(plan)))
    slot_of, conf = {}, {}
    for members in sorted(comps.values(), key=len, reverse=True):
        assign, psi, _, margin = _fit_component(members, free, pitch, yaw, Y, P)
        align = 1.0 if not np.isfinite(margin) else ALIGN_FLOOR + (1.0 - ALIGN_FLOOR) * min(1.0, margin / ALIGN_MARGIN)
        scale = 1.0 if len(members) >= 8 else min(1.0, (len(members) - 1) / 6.0) * 0.6 + 0.05
        for k, s in assign.items():
            dp = abs(pitch[k] - P[s])
            dy = 0.0 if (abs(pitch[k]) >= POLE_PITCH or abs(P[s]) >= POLE_PITCH) else float(_circ(yaw[k], Y[s] + psi))
            quality = max(0.0, 1.0 - 0.5 * max(dp, dy) / 10.0)
            slot_of[k], conf[k] = s, round(scale * quality * align * 0.95, 3)
            free.discard(s)
    left_imgs = [k for k in range(len(names)) if k not in slot_of]
    for k, s in zip(left_imgs, sorted(free)):             # không có thông tin: ô còn lại, độ tin cậy thấp
        slot_of[k], conf[k] = s, 0.05
    return {names[k]: {"slot": int(slot_of[k]), "yaw_deg": float(Y[slot_of[k]]), "pitch_deg": float(P[slot_of[k]]),
                       "confidence": float(conf[k]), "n_matches": int(n_match[k])} for k in range(len(names))}
