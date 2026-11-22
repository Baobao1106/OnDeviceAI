"""Cân độ sáng và màu giữa các ảnh. NGƯỜI PHỤ TRÁCH: C.

Mỗi ảnh có gain riêng 0,85–1,15 và lệch cân bằng trắng ±4% mỗi kênh. Chấm điểm chỉ bù một gain chung
cho cả ảnh ERP, nên sự khác biệt giữa các ảnh vẫn trừ điểm (đường nối lộ ra, PSNR và LPIPS giảm).

Cách làm (log-gain, bình phương tối thiểu):
  1. Thu nhỏ mọi lớp ERP xuống 1/SCALE và làm mờ nhẹ để sai số pose nhỏ không làm lệch tỉ lệ.
  2. Với mỗi cặp ảnh chồng lấn, chỉ lấy pixel mà cả hai đều ở phần "giữa ảnh" (trọng số >= CORE lần trọng số lớn
     nhất của ảnh đó), bỏ pixel quá tối, cháy sáng hoặc gần cạnh mạnh. Tỉ lệ mỗi kênh = trung vị của
     log(a) - log(b) trên các pixel đó (trung vị chịu được pixel lệch do pose sai).
  3. Giải hệ tuyến tính  log g_i - log g_j = -d_ij  (trọng số = căn số pixel chồng lấn), thêm ràng buộc mềm
     tổng log g = 0 và kéo nhẹ về 0 để ảnh không có cặp nào giữ nguyên gain 1.
     Cặp nào mặt nạ chặt quá ít pixel thì dùng mặt nạ lỏng (chỉ bỏ pixel tối hoặc cháy hẳn), trọng số giảm nửa.
  4. Cắt gain về [GMIN, GMAX].
Trả về mảng [N,3] (BGR) theo thứ tự `layers`.
"""
import cv2
import numpy as np

SCALE = 4          # thu nhỏ ERP 2048x1024 -> 512x256
CORE = 0.25        # pixel phải có trọng số >= CORE * max trọng số của ảnh
MIN_PIX = 200      # số pixel chồng lấn (mặt nạ chặt) tối thiểu; ít hơn thì dùng mặt nạ lỏng
MIN_PIX_LOOSE = 50
LOOSE_CORE = 0.05
LO, HI = 12.0, 240.0
PRIOR = 0.05       # kéo nhẹ log-gain về 0
GMIN, GMAX = 0.75, 1.33


def _prep(layers, weights):
    """Trả về log-ảnh nhỏ [N,h,w,3], mặt nạ chặt [N,h,w] và mặt nạ lỏng [N,h,w].

    Mặt nạ chặt: giữa ảnh, không tối/cháy, ít cạnh. Mặt nạ lỏng: chỉ cần có ảnh phủ và không tối/cháy hẳn,
    dùng cho cặp mà mặt nạ chặt còn quá ít pixel (ảnh nhiều chi tiết hoặc nhiều vùng cháy sáng)."""
    H, W = layers[0].shape[:2]
    w_, h_ = W // SCALE, H // SCALE
    logs, masks, loose = [], [], []
    for lay, wgt in zip(layers, weights):
        small = cv2.resize(lay, (w_, h_), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (3, 3), 0)
        ws = cv2.resize(wgt, (w_, h_), interpolation=cv2.INTER_AREA)
        wmax = float(ws.max()) if ws.size else 0.0
        ok = ws >= CORE * max(wmax, 1e-6)
        lo = (ws >= LOOSE_CORE * max(wmax, 1e-6)) & (small.min(axis=2) > 4.0) & (small.max(axis=2) < 250.0)
        ok &= (small.min(axis=2) > LO) & (small.max(axis=2) < HI)
        # bỏ pixel gần cạnh mạnh: ở đó sai số pose nhỏ cũng làm tỉ lệ lệch nhiều
        g = cv2.cvtColor(np.clip(small, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
        ok &= grad < 0.25 * np.maximum(g, 1.0) * 8  # gradient tương đối < ~25% mỗi pixel (Sobel ~x8)
        logs.append(np.log(np.maximum(small, 1.0)))
        masks.append(ok)
        loose.append(lo)
    return np.stack(logs), np.stack(masks), np.stack(loose)


def estimate_gains(layers, weights):
    """layers: list uint8 [H,W,3] (BGR), weights: list float32 [H,W]. Trả về [N,3] (BGR)."""
    n = len(layers)
    out = np.ones((n, 3), np.float32)
    if n < 2:
        return out
    logs, masks, loose = _prep(layers, weights)
    rows, rhs, wts = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            m = masks[i] & masks[j]
            cnt = int(m.sum())
            wfac = 1.0
            if cnt < MIN_PIX:
                m = loose[i] & loose[j]
                cnt = int(m.sum())
                wfac = 0.5          # cặp lỏng kém tin cậy hơn
                if cnt < MIN_PIX_LOOSE:
                    continue
            d = np.median(logs[i][m] - logs[j][m], axis=0)  # [3]: log a_i - log a_j
            rows.append((i, j))
            rhs.append(d)
            wts.append(wfac * np.sqrt(cnt))
    if not rows:
        return out
    k = len(rows)
    A = np.zeros((k + n + 1, n), np.float64)
    b = np.zeros((k + n + 1, 3), np.float64)
    wts = np.asarray(wts, np.float64)
    wts = wts / wts.mean()
    for r, ((i, j), d, w) in enumerate(zip(rows, rhs, wts)):
        # g_i * a_i = g_j * a_j  =>  log g_i - log g_j = -(log a_i - log a_j)
        A[r, i], A[r, j] = w, -w
        b[r] = -w * d
    A[k:k + n] = PRIOR * np.eye(n)
    A[k + n] = 10.0                     # tổng log g = 0
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    x -= x.mean(axis=0, keepdims=True)
    return np.clip(np.exp(x), GMIN, GMAX).astype(np.float32)
