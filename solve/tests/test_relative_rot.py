import unittest

import numpy as np

import _paths  # noqa: F401
import _synth
from solve import relative_rot


def angle_between(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)))


class TestRelativeRotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches, cls.slots, cls.truth = _synth.make_scene(seed=1)

    def _pair(self):
        return max(self.matches["pairs"], key=lambda p: p["n_inliers"])

    def test_recovers_true_relative_rotation_at_true_fov(self):
        p = self._pair()
        R, n, err = relative_rot.relative_rotation(p, 74.0, self.matches["image_size"])
        want = self.truth["R"][p["i"]].T @ self.truth["R"][p["j"]]
        self.assertLess(angle_between(R, want), 0.3)
        self.assertGreater(n, 0.9 * p["n_inliers"])
        self.assertLess(err, 0.2)

    def test_wrong_fov_gives_larger_error(self):
        p = self._pair()
        size = self.matches["image_size"]
        good = relative_rot.relative_rotation(p, 74.0, size)[2]
        bad = relative_rot.relative_rotation(p, 78.5, size)[2]
        self.assertGreater(bad, good)

    def test_survives_outlier_matches(self):
        p = dict(self._pair())
        rng = np.random.default_rng(0)
        k = len(p["pts_i"]) // 3
        size = self.matches["image_size"]
        p["pts_i"] = p["pts_i"] + rng.uniform([0, 0], size, (k, 2)).tolist()
        p["pts_j"] = p["pts_j"] + rng.uniform([0, 0], size, (k, 2)).tolist()
        R, _, _ = relative_rot.relative_rotation(p, 74.0, size)
        want = self.truth["R"][p["i"]].T @ self.truth["R"][p["j"]]
        self.assertLess(angle_between(R, want), 0.5)

    def test_too_few_points_returns_identity_and_zero_inliers(self):
        p = {"pts_i": [[1, 1], [2, 2]], "pts_j": [[1, 1], [2, 2]]}
        R, n, _ = relative_rot.relative_rotation(p, 74.0, (864, 648))
        self.assertEqual(n, 0)
        np.testing.assert_allclose(R, np.eye(3))


if __name__ == "__main__":
    unittest.main()
