import unittest

import numpy as np

import _paths  # noqa: F401
import common
from register.engine.matchers import locnorm_gray
from register.engine.pairs import PairEngine, PairResult, PipelineConfig

HFOV = 74.0
W, H = 864, 648


def project(R_cam2world, dirs):
    f, cx, cy = common.intrinsics(W, H, HFOV)
    c = dirs @ R_cam2world
    ok = c[:, 2] > 0.2
    x = c[:, 0] / np.where(ok, c[:, 2], 1) * f + cx
    y = -c[:, 1] / np.where(ok, c[:, 2], 1) * f + cy
    ok &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return np.stack([x, y], 1), ok


def make_engine(rots, matches_for):
    eng = PairEngine.__new__(PairEngine)
    eng.cfg = PipelineConfig(device="cpu", workers=1)
    eng.calls = []

    def extract_lowtex(bgr):
        eng.calls.append(1)
        return bgr

    def pooled(fi, fj):
        i, j = fi, fj
        rng = np.random.default_rng(10 * i + j)
        d = rng.normal(size=(3000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pi, oki = project(rots[i], d)
        pj, okj = project(rots[j], d)
        keep = oki & okj
        n = matches_for(i, j)
        return pi[keep][:n].astype(np.float64), pj[keep][:n].astype(np.float64)

    eng.extract_lowtex = extract_lowtex
    eng._pooled = pooled
    return eng


def accepted(i, j, rots):
    return PairResult(i, j, 50, 40, True, 1.0, HFOV, rots[i].T @ rots[j])


def rejected(i, j):
    return PairResult(i, j, 20, 2, False, 0.0, None, None)


class TestLowtexPass(unittest.TestCase):
    def setUp(self):
        self.rots = [common.pose_to_R(y, p, r) for y, p, r in
                     [(0, 0, 0), (30, 3, 2), (-25, -2, 1), (5, 4, -2)]]
        self.names = ["a", "b", "c", "d"]
        self.images = [0, 1, 2, 3]

    def pairs(self):
        return [accepted(0, 1, self.rots), accepted(0, 2, self.rots), accepted(1, 2, self.rots),
                rejected(0, 3), rejected(1, 3), rejected(2, 3)]

    def test_weak_image_gets_its_pairs_accepted_and_others_stay(self):
        eng = make_engine(self.rots, lambda i, j: 60)
        before = self.pairs()
        ip, cp = {}, {}
        out = eng._lowtex_pass(self.names, self.images, self.images, before, HFOV, None, None, None, ip, cp)
        by = {(p.i, p.j): p for p in out}
        for key in ((0, 3), (1, 3), (2, 3)):
            self.assertTrue(by[key].accepted and by[key].boosted, key)
            self.assertIn(key, ip)
            err = np.degrees(np.arccos(np.clip((np.trace(by[key].rotation.T @ self.rots[key[0]].T @
                                                             self.rots[key[1]]) - 1) / 2, -1, 1)))
            self.assertLess(err, 1.0)
        for key in ((0, 1), (0, 2), (1, 2)):
            self.assertIs(by[key], before[[(p.i, p.j) for p in before].index(key)])

    def test_nothing_to_do_when_every_image_has_two_pairs(self):
        eng = make_engine(self.rots, lambda i, j: 60)
        full = [accepted(0, 1, self.rots), accepted(0, 2, self.rots), accepted(1, 2, self.rots),
                accepted(0, 3, self.rots), accepted(1, 3, self.rots), accepted(2, 3, self.rots)]
        out = eng._lowtex_pass(self.names, self.images, self.images, full, HFOV, None, None, None, {}, {})
        self.assertEqual(out, full)
        self.assertEqual(eng.calls, [])

    def test_too_few_matches_stay_rejected(self):
        eng = make_engine(self.rots, lambda i, j: 6 if 3 in (i, j) else 60)
        out = eng._lowtex_pass(self.names, self.images, self.images, self.pairs(), HFOV, None, None, None, {}, {})
        self.assertTrue(all(not p.accepted for p in out if 3 in (p.i, p.j)))


class TestLocnorm(unittest.TestCase):
    def test_flat_wall_with_faint_gradient_gains_contrast(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, W)[None, :].repeat(H, 0)
        img = np.clip(120 + 3 * x + rng.normal(0, 0.4, (H, W)), 0, 255).astype(np.uint8)
        bgr = np.stack([img] * 3, -1)
        out = locnorm_gray(bgr)
        self.assertEqual(out.shape, (H, W))
        self.assertGreater(float(out.std()), 3 * float(img.std()))


if __name__ == "__main__":
    unittest.main()
