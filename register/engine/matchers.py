"""Feature extraction + pair matching back-ends. Each matcher: extract(gray)->feat, match(fa, fb)->(pa, pb, score)."""
import cv2
import numpy as np

IMG_W, IMG_H = 864, 648


def vignette_map(w=IMG_W, h=IMG_H):
    """Vignetting model used by make_set.py (1 - 0.18 r^2); divided out before detection."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h) * h / w)
    return (1.0 - 0.18 * (xs ** 2 + ys ** 2)).astype(np.float32)


def preprocess(bgr, devignette=True, clahe=False):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if devignette:
        gray = np.clip(gray.astype(np.float32) / vignette_map(*gray.shape[::-1]), 0, 255).astype(np.uint8)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def locnorm_gray(bgr):
    """Ảnh xám đã bù vignetting, bỏ độ sáng trung bình cục bộ và chuẩn hóa độ tương phản cục bộ.
    Làm lộ chuyển sắc và đường nhạt trên tường trắng (thí nghiệm trên ảnh cô lập: SIFT/SuperPoint thường không thấy gì)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = np.clip(g / vignette_map(*g.shape[::-1]), 0, 255)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    mean = cv2.GaussianBlur(g, (0, 0), 12)
    std = np.sqrt(cv2.GaussianBlur((g - mean) ** 2, (0, 0), 12)) + 1.0
    return np.clip(128 + 40 * (g - mean) / std, 0, 255).astype(np.uint8)


class RootSift:
    name = "rootsift"

    def __init__(self, n_features=4000, ratio=0.8, contrast=0.04, device="cpu"):
        self.sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio = ratio
        self.device = device
        self.torch = None
        if device != "cpu":     # descriptor matching on the GPU (same ratio test + mutual check as the CPU path)
            import torch
            self.torch = torch

    def extract(self, gray):
        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None:
            return {"kp": np.zeros((0, 2), np.float32), "desc": np.zeros((0, 128), np.float32)}
        desc = desc / (desc.sum(1, keepdims=True) + 1e-7)
        feat = {"kp": np.array([k.pt for k in kps], np.float32), "desc": np.sqrt(desc).astype(np.float32)}
        if self.torch is not None:
            feat["desc_t"] = self.torch.from_numpy(feat["desc"]).to(self.device)
        return feat

    def _match_gpu(self, fa, fb):
        torch = self.torch
        with torch.no_grad():
            d = torch.cdist(fa["desc_t"], fb["desc_t"])
            two = d.topk(2, dim=1, largest=False)
            back = d.argmin(dim=0)
            ia = torch.arange(d.shape[0], device=d.device)
            ok = (two.values[:, 0] < self.ratio * two.values[:, 1]) & (back[two.indices[:, 0]] == ia)
            ia, ib = ia[ok], two.indices[ok, 0]
            score = 1.0 - two.values[ok, 0]
            ia, ib, score = ia.cpu().numpy(), ib.cpu().numpy(), score.cpu().numpy()
        return fa["kp"][ia], fb["kp"][ib], score.astype(np.float32)

    def match(self, fa, fb):
        empty = (np.zeros((0, 2), np.float32),) * 2 + (np.zeros(0, np.float32),)
        if len(fa["kp"]) < 2 or len(fb["kp"]) < 2:
            return empty
        if self.torch is not None and "desc_t" in fa:
            return self._match_gpu(fa, fb)
        fwd = self.bf.knnMatch(fa["desc"], fb["desc"], k=2)
        bwd = self.bf.match(fb["desc"], fa["desc"])
        back = {m.queryIdx: m.trainIdx for m in bwd}
        keep = [m for m, n in fwd if m.distance < self.ratio * n.distance and back.get(m.trainIdx) == m.queryIdx]
        if not keep:
            return empty
        ia = np.array([m.queryIdx for m in keep])
        ib = np.array([m.trainIdx for m in keep])
        score = np.array([1.0 - m.distance for m in keep], np.float32)
        return fa["kp"][ia], fb["kp"][ib], score


class SuperPointLightGlue:
    name = "superpoint_lightglue"

    def __init__(self, device="cpu", max_kp=2048, detection_threshold=0.0005, nms_radius=4,
                 filter_threshold=0.1, scale=1.0):
        import torch
        from lightglue import LightGlue, SuperPoint
        self.torch, self.device, self.scale = torch, device, float(scale)
        self.sp = SuperPoint(max_num_keypoints=int(max_kp), detection_threshold=float(detection_threshold),
                             nms_radius=int(nms_radius)).eval().to(device)
        self.lg = LightGlue(features="superpoint", filter_threshold=float(filter_threshold)).eval().to(device)

    def extract(self, gray):
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        t = self.torch.from_numpy(gray).float()[None, None].to(self.device) / 255.0
        with self.torch.no_grad():
            return self.sp.extract(t)

    def match(self, fa, fb):
        with self.torch.no_grad():
            out = self.lg({"image0": fa, "image1": fb})
        m = out["matches"][0].cpu().numpy()
        sc = out["scores"][0].cpu().numpy()
        ka = fa["keypoints"][0].cpu().numpy() / self.scale
        kb = fb["keypoints"][0].cpu().numpy() / self.scale
        if len(m) == 0:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        return ka[m[:, 0]], kb[m[:, 1]], sc.astype(np.float32)


class SuperPointLightGlueRot(SuperPointLightGlue):
    """SuperPoint is not rotation invariant: also extract on the image rotated by 90/180/270 deg and,
    per pair, keep the rotation of image B that yields most matches (keypoints mapped back to B's frame)."""
    name = "superpoint_lightglue_rot"

    def extract(self, gray):
        h, w = gray.shape
        feats = []
        for k in range(4):
            f = super().extract(np.ascontiguousarray(np.rot90(gray, k)))
            feats.append({"feat": f, "w": w, "h": h})
        return feats

    @staticmethod
    def _unrotate(kp, k, w, h):
        x, y = kp[:, 0], kp[:, 1]
        if k == 0:
            return kp
        if k == 1:
            return np.stack([w - 1 - y, x], 1)
        if k == 2:
            return np.stack([w - 1 - x, h - 1 - y], 1)
        return np.stack([y, h - 1 - x], 1)

    def match(self, fa, fb):
        best = None
        for k in range(4):
            a, b, sc = super().match(fa[0]["feat"], fb[k]["feat"])
            if best is None or len(a) > len(best[0]):
                best = (a, self._unrotate(b, k, fb[k]["w"], fb[k]["h"]).astype(np.float32), sc)
        return best


class LoFTRRot:
    """Dense detector-free matcher (kornia LoFTR, indoor weights) with 4-way rotation of image B.
    Images are resized to 640x480; the rotation with most confident matches is kept."""
    name = "loftr_rot"
    SIZE_SCALE = 640.0 / IMG_W

    def __init__(self, device="cpu", pretrained="indoor", conf=0.5):
        import torch
        from kornia.feature import LoFTR
        self.torch, self.device, self.conf = torch, device, float(conf)
        self.model = LoFTR(pretrained=pretrained).eval().to(device)

    def _tensor(self, gray):
        small = cv2.resize(gray, None, fx=self.SIZE_SCALE, fy=self.SIZE_SCALE, interpolation=cv2.INTER_AREA)
        return self.torch.from_numpy(small).float()[None, None].to(self.device) / 255.0

    def extract(self, gray):
        h, w = gray.shape
        return [{"img": self._tensor(np.ascontiguousarray(np.rot90(gray, k))), "w": w, "h": h} for k in range(4)]

    def match(self, fa, fb):
        best = None
        for k in range(4):
            with self.torch.no_grad():
                out = self.model({"image0": fa[0]["img"], "image1": fb[k]["img"]})
            keep = out["confidence"] > self.conf
            a = out["keypoints0"][keep].cpu().numpy() / self.SIZE_SCALE
            b = out["keypoints1"][keep].cpu().numpy() / self.SIZE_SCALE
            c = out["confidence"][keep].cpu().numpy()
            if best is None or len(a) > len(best[0]):
                b = SuperPointLightGlueRot._unrotate(b, k, fb[k]["w"], fb[k]["h"])
                best = (a.astype(np.float32), b.astype(np.float32), c.astype(np.float32))
        return best


def build(method, device="cpu", **kw):
    if method == "rootsift":
        return RootSift()
    if method == "rootsift_lowc":   # lower contrast threshold: more keypoints on flat walls
        return RootSift(n_features=8000, contrast=0.01, device=device)
    if method == "loftr_rot":
        return LoFTRRot(device=device, **kw)
    if method == "superpoint_lightglue_rot":
        return SuperPointLightGlueRot(device=device, **kw)
    if method == "superpoint_lightglue":
        return SuperPointLightGlue(device=device, **kw)
    raise ValueError(f"unknown method: {method}")
